#!/usr/bin/env python
"""Equivalence smoke for generate_eval_batched.py — the GUARD on the left-pad +
mrope correctness risk.

Runs the SAME n ads (default 4) through BOTH paths in one process / one model
load (so the only difference is batch=1 vs batch=K), then asserts:

  (A) per-ad generated CoTs are identical (greedy is deterministic; if left-pad
      / mrope position_ids are wrong, the batched decode diverges and CoTs
      differ — this is the #1 failure mode and the primary assertion).
  (B) per-ad reasoned r_pred curves match within tol (head readout is per-ad in
      both, so this should be bit-identical up to numerics).
  (C) cross-ad reasoned SRCC matches within tol.

Exit 0 = batched path is equivalent. Exit 1 = divergence (do NOT trust batched
numbers; left-pad/posids handling is wrong for this transformers build).

Usage (on the eval box, ONE spare GPU — both eval GPUs are busy; this is tiny):
  python verification/smoke_batched_equiv.py \
    --checkpoint <ckpt> --base <base> --val-jsonl <val.jsonl> \
    --plugin examples/custom/qwen2_5_omni_retention/register.py \
    --attn-impl sdpa --max-new 600 --n 4 --batch-size 4
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
import numpy as np, torch


def import_plugin(p):
    spec = importlib.util.spec_from_file_location('retention_plugin', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--plugin', required=True)
    ap.add_argument('--base', default=None)
    ap.add_argument('--attn-impl', default='sdpa')
    ap.add_argument('--head-type', default='hazard')
    ap.add_argument('--max-length', type=int, default=32768)
    ap.add_argument('--max-new', type=int, default=600)
    ap.add_argument('--video-max-tokens', type=int, default=16384)
    ap.add_argument('--n', type=int, default=4, help='number of ads to compare')
    ap.add_argument('--batch-size', type=int, default=4, help='K for the batched path')
    ap.add_argument('--rtol', type=float, default=2e-3, help='r_pred / SRCC tolerance (numerics, not exactness)')
    args = ap.parse_args()

    os.environ['RETENTION_HEAD_TYPE'] = args.head_type
    os.environ.setdefault('MAX_PIXELS', '200704'); os.environ.setdefault('VIDEO_MAX_PIXELS', '200704')
    os.environ.setdefault('FPS_MAX_FRAMES', '60'); os.environ.setdefault('FPS', '1.0')
    os.environ.setdefault('VIDEO_MAX_TOKEN_NUM', str(args.video_max_tokens))

    import_plugin(args.plugin)
    # reuse the batched module's helpers so we test the SHIPPING code, not a copy
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location('genB', os.path.join(here, 'generate_eval_batched.py'))
    genB = importlib.util.module_from_spec(spec); spec.loader.exec_module(genB)

    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.template_inputs import TemplateInputs

    adapter_cfg = os.path.join(args.checkpoint, 'adapter_config.json')
    if os.path.exists(adapter_cfg):
        base = args.base or json.load(open(adapter_cfg))['base_model_name_or_path']
        model, proc = get_model_processor(base, torch_dtype=torch.bfloat16, attn_impl=args.attn_impl,
                                          model_kwargs={'device_map': 'cuda'}, model_type='qwen2_5_omni_retention')
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=False)
    else:
        model, proc = get_model_processor(args.checkpoint, torch_dtype=torch.bfloat16, attn_impl=args.attn_impl,
                                          model_kwargs={'device_map': 'cuda'}, model_type='qwen2_5_omni_retention')
    model.eval()

    tmpl_train = get_template(proc, max_length=args.max_length, template_type='qwen2_5_omni_retention',
                              remove_unused_columns=False); tmpl_train.set_mode('train')
    tmpl_gen = get_template(proc, max_length=args.max_length, template_type='qwen2_5_omni_retention',
                            remove_unused_columns=False); tmpl_gen.set_mode('transformers')

    rows = [json.loads(l) for l in open(args.val_jsonl) if (json.loads(l).get('R') or json.loads(l).get('R_true'))]
    # take the first n ads with T>=5 (same filter as the eval)
    sel = []
    for r in rows:
        Rt = np.array(r.get('R') or r.get('R_true'), float)
        if len(Rt) - 1 >= 5:
            sel.append(r)
        if len(sel) >= args.n: break
    assert len(sel) == args.n, f'only found {len(sel)} eligible ads, need {args.n}'
    TI = TemplateInputs

    # --- path 1: per-ad (batch=1, == generate_eval.py) ---
    cots_single = [genB.gen_cot_single(model, tmpl_gen, TI, proc, r, args.max_new) for r in sel]
    rgen_single = [genB.head_curve(model, tmpl_train, TI, r, c) for r, c in zip(sel, cots_single)]

    # --- path 2: batched (K) ---
    cots_batched = genB.gen_cot_batched(model, tmpl_gen, TI, proc, sel, args.max_new)
    rgen_batched = [genB.head_curve(model, tmpl_train, TI, r, c) for r, c in zip(sel, cots_batched)]

    ok = True
    print('\n========== EQUIVALENCE SMOKE ==========')
    for i in range(args.n):
        cot_match = cots_single[i] == cots_batched[i]
        rmax = float(np.max(np.abs(np.array(rgen_single[i]) - np.array(rgen_batched[i]))))
        rok = rmax <= args.rtol
        ok = ok and cot_match and rok
        print(f'ad{i}: CoT_match={cot_match}  r_pred_maxabsdiff={rmax:.2e} (<= {args.rtol}? {rok})')
        if not cot_match:
            # show the first divergence point for debugging left-pad/posids
            a, b = cots_single[i], cots_batched[i]
            j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            print(f'  single[:160]={a[:160]!r}')
            print(f'  batched[:160]={b[:160]!r}')
            print(f'  first diverge @char {j}')

    # cross-ad SRCC on the n ads, both ways (sanity; n=4 is small but must agree)
    def crossad(preds):
        Ts = [len(np.array(r.get('R') or r.get('R_true'), float)) - 1 for r in sel]
        per = []
        for t in range(1, 31):
            pv = [P[t] for P, T in zip(preds, Ts) if T >= t]
            tv = [np.array(r.get('R') or r.get('R_true'), float)[t] for r, T in zip(sel, Ts) if T >= t]
            rho = genB.spearman(pv, tv)
            if not np.isnan(rho): per.append(rho)
        return float(np.mean(per)) if per else float('nan')

    s1, s2 = crossad(rgen_single), crossad(rgen_batched)
    srcc_ok = (np.isnan(s1) and np.isnan(s2)) or abs(s1 - s2) <= args.rtol
    ok = ok and srcc_ok
    print(f'cross-ad SRCC: single={s1:.4f} batched={s2:.4f} (|diff|<= {args.rtol}? {srcc_ok})')
    print('=======================================')
    print('SMOKE PASS' if ok else 'SMOKE FAIL — batched path diverges; do NOT use batched numbers')
    sys.exit(0 if ok else 1)


if __name__ == '__main__': main()

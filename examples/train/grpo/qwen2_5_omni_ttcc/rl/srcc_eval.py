#!/usr/bin/env python
"""Cross-ad SRCC guard — the Goodhart verifier for the RL run.

Computes the TARGET metric (cross-ad Spearman rank correlation: at each second t,
Spearman across ads between predicted R(t) and true R(t); averaged over t in [t_lo,t_hi])
for a checkpoint, using the LEAK-FREE bypass forward (assistant overwritten to
'<cot></cot>', single forward, head reads the last token). attn_impl is PINNED so the
SFT baseline (ckpt-225) and every RL checkpoint are measured under the IDENTICAL kernel
(audit blocker #7 — h_anchor drifts across FA2/FA3/sdpa).

Usage:
  python srcc_eval.py --checkpoint <dir> --val-jsonl <val.jsonl> \
      --plugin <register.py> --attn-impl flash_attn --output srcc.json

Intended use: run on ckpt-225 once (the baseline), then on each RL checkpoint
(pulled from S3) on the FREE eval box, so the SRCC trajectory is tracked OUT of the
training loop (can't break the run; the cluster GPUs are busy with RL).
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
import numpy as np
import torch


def import_plugin(p):
    spec = importlib.util.spec_from_file_location('retention_plugin', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def spearman(x, y):
    """Spearman rho via Pearson on ranks (avg-rank ties). No scipy dependency."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    def rank(a):
        order = a.argsort(); r = np.empty(len(a), float); r[order] = np.arange(len(a))
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt); starts = csum - cnt
        avg = (starts + csum - 1) / 2.0
        return avg[inv]
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--plugin', required=True)
    ap.add_argument('--attn-impl', default='flash_attn')   # PINNED across baseline + RL ckpts
    ap.add_argument('--head-type', default='hazard')
    ap.add_argument('--max-length', type=int, default=32768)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--t-lo', type=int, default=1)
    ap.add_argument('--t-hi', type=int, default=30)
    ap.add_argument('--output', default=None)
    ap.add_argument('--dump-npz', default=None,
                    help='dump per-ad ad_ids/preds/trues/Ts for paired-bootstrap CI (paired_bootstrap.py)')
    args = ap.parse_args()

    os.environ['RETENTION_HEAD_TYPE'] = args.head_type
    import_plugin(args.plugin)
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.template_inputs import TemplateInputs
    from swift.template.base import MaxLengthError

    adapter_cfg = os.path.join(args.checkpoint, 'adapter_config.json')
    if os.path.exists(adapter_cfg):
        base = json.load(open(adapter_cfg))['base_model_name_or_path']
        model, proc = get_model_processor(base, torch_dtype=torch.bfloat16, attn_impl=args.attn_impl,
                                          model_kwargs={'device_map': 'cuda'}, model_type='qwen2_5_omni_retention')
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=False)
    else:
        model, proc = get_model_processor(args.checkpoint, torch_dtype=torch.bfloat16, attn_impl=args.attn_impl,
                                          model_kwargs={'device_map': 'cuda'}, model_type='qwen2_5_omni_retention')
    model.eval()
    template = get_template(proc, max_length=args.max_length, template_type='qwen2_5_omni_retention',
                            remove_unused_columns=False)
    template.set_mode('train')
    print(f'[srcc] loaded {args.checkpoint} attn={args.attn_impl}')

    rows = []
    with open(args.val_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if (r.get('R') or r.get('R_true')):
                rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break

    # per-ad predicted + true curves
    preds, trues, Ts, ids, skipped = [], [], [], [], 0
    for i, r in enumerate(rows):
        R_true = np.array(r.get('R') or r.get('R_true'), dtype=np.float64)
        T = len(R_true) - 1
        if T < 5:
            skipped += 1; continue
        row = dict(r); row['messages'] = list(r['messages'])
        row['messages'][-1] = {'role': 'assistant', 'content': '<cot></cot>'}   # bypass
        try:
            enc = template.encode(TemplateInputs.from_dict(row))
            batch = template.data_collator([enc])
            batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            with torch.no_grad():
                out = model(**batch)
            rp = getattr(out, 'r_pred', None)
            if rp is None:
                rp = model._retention_h_holder.r_pred
            R_pred = np.concatenate([[1.0], rp[0].float().cpu().numpy()])   # R(0..Tmax)
        except (MaxLengthError, Exception) as e:                            # noqa
            skipped += 1; continue
        preds.append(R_pred); trues.append(R_true); Ts.append(T); ids.append(str(r.get('ad_id', '')))
        if (i + 1) % 25 == 0:
            print(f'[srcc] {len(preds)} evaluated, {skipped} skipped')

    n = len(preds)
    # cross-ad SRCC: at each second t, Spearman across ads with T>=t
    per_t = {}
    for t in range(args.t_lo, args.t_hi + 1):
        pv, tv = [], []
        for P, Tr, T in zip(preds, trues, Ts):
            if T >= t:
                pv.append(P[t]); tv.append(Tr[t])
        rho = spearman(pv, tv)
        if not np.isnan(rho):
            per_t[t] = (rho, len(pv))
    srcc = float(np.mean([v[0] for v in per_t.values()])) if per_t else float('nan')

    print(f'\n[srcc] n_ads={n} skipped={skipped}')
    print(f'[srcc] CROSS-AD SRCC (t in [{args.t_lo},{args.t_hi}], avg) = {srcc:.4f}')
    print(f'[srcc] per-t (t: rho, n_ads): ' + ', '.join(f'{t}:{v[0]:.3f}' for t, v in sorted(per_t.items()) if t in (1, 5, 10, 15, 20, 25, 30)))
    if args.output:
        json.dump({'checkpoint': args.checkpoint, 'attn_impl': args.attn_impl, 'n_ads': n,
                   'skipped': skipped, 'srcc': srcc, 't_lo': args.t_lo, 't_hi': args.t_hi,
                   'per_t': {str(t): {'rho': v[0], 'n': v[1]} for t, v in per_t.items()}},
                  open(args.output, 'w'), indent=2)
        print(f'[srcc] wrote {args.output}')
    if args.dump_npz:
        np.savez(args.dump_npz, ad_ids=np.array(ids), Ts=np.array(Ts),
                 preds=np.array(preds, dtype=object), trues=np.array(trues, dtype=object))
        print(f'[srcc] dumped {len(ids)} per-ad curves -> {args.dump_npz}')


if __name__ == '__main__':
    main()

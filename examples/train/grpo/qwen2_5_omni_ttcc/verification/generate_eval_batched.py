#!/usr/bin/env python
"""BATCHED reasoned-retention eval — a faster drop-in for generate_eval.py.

Identical SEMANTICS to verification/generate_eval.py (same canonical video env,
same --attn-impl, same base+adapter load, same head-target kwarg strip, same
greedy decoding, same head readout) — but the autoregressive CoT generation
(the long pole, ~12-20s/ad at batch=1) is done K ads at a time in ONE
model.generate() call. Expected ~5-10x wall-clock cut on the generation step.

WHAT IS BATCHED, WHAT IS NOT
----------------------------
1. GENERATION (the bottleneck): K ads collated into one generate() call.
2. HEAD READOUT (cheap, a single forward): kept PER-AD (batch=1), bit-identical
   to generate_eval.head_curve() and to srcc_eval.py's bypass forward. The head
   reads a SINGLE token's hidden state at the last </cot> (register.py
   _locate_anchor_positions); batching it would right-pad and, for any row
   WITHOUT a </cot>, push the L-1 fallback anchor onto a PAD token
   (head_pg_trainer.py:19-20 documents exactly this hazard). Correctness over a
   tiny extra speedup: the head forward is not the long pole.

WHY LEFT-PAD + mrope IS HANDLED FOR US (the #1 correctness risk)
---------------------------------------------------------------
swift's collator left-pads input_ids/attention_mask in inference ('transformers')
mode (base.py:1817 `padding_side = ... if self.is_training else 'left'`) and, in
inference mode, does NOT precompute position_ids (qwen.py:933
`if not self.is_training: return {}`). So model.generate() receives input_ids +
attention_mask + stacked video tensors with position_ids=None.

Qwen2.5-Omni's thinker.forward then computes mrope position_ids itself
(modeling_qwen2_5_omni.py, the `if attention_mask is not None and position_ids
is None:` branch):
  - get_rope_index() masks padded positions per row before assigning mrope
    positions (`input_ids = input_ids[attention_mask[i]]`), so a left-pad does
    not corrupt the temporal/H/W rope coordinates.
  - the prefill caches rope_deltas; on transformers >= ~4.x with the left-pad
    fix it ALSO subtracts delta0 = (1 - attention_mask).sum(-1) so the per-row
    decode-step position arithmetic stays correct under left-pad.
  => We MUST pass attention_mask and MUST NOT pass position_ids. We do both.

  RESIDUAL VERSION RISK: the `rope_deltas - delta0` left-pad correction must be
  present in the transformers build on the eval box. It is in transformers 5.x
  (verified in the cached wheel) and in 4.51+. If the box runs an older build
  WITHOUT it, left-padded multimodal decode positions drift and batched CoTs
  diverge from batch=1. THE EQUIVALENCE SMOKE (n=4 both ways) IS THE GUARD —
  see research/BATCHED_EVAL_PLAN.md. Do not trust batched numbers until the
  smoke passes.

ALSO PRESERVED: ragged video token counts collate fine — pixel_values_videos is
torch.concat'd (variable rows) and video_grid_thw is concat'd row-wise
(base.py:_data_collator_mm_data), exactly as the model's get_video_features /
get_rope_index expect (they index videos by grid_thw, not by batch row).
"""
from __future__ import annotations
import argparse, importlib.util, json, os
import numpy as np, torch


def import_plugin(p):
    spec = importlib.util.spec_from_file_location('retention_plugin', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return np.nan
    def rk(a):
        u, inv, c = np.unique(a, return_inverse=True, return_counts=True)
        cs = np.cumsum(c); st = cs - c; return ((st + cs - 1) / 2.0)[inv]
    rx, ry = rk(x), rk(y); rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


# ---- head readout: PER-AD, identical to generate_eval.head_curve() ----------
_HEAD_TARGET_KWARGS = ('r_true', 'r_mask', 'ad_id', 'T', 'R', 'labels', 'loss_scale', 'channel')


def head_curve(model, template, TI, row, assistant):
    """Encode row with the given assistant content (train mode), forward, return
    head r_pred R(0..Tmax). Bit-identical to the batch=1 script's head_curve."""
    r = dict(row); r['messages'] = list(row['messages'])
    r['messages'][-1] = {'role': 'assistant', 'content': assistant}
    enc = template.encode(TI.from_dict(r)); batch = template.data_collator([enc])
    batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad(): out = model(**batch)
    rp = getattr(out, 'r_pred', None)
    if rp is None: rp = model._retention_h_holder.r_pred
    return np.concatenate([[1.0], rp[0].float().cpu().numpy()])


def _clean_cot(cot: str) -> str:
    """Identical post-processing to generate_eval.py: ensure a <cot>...</cot> wrap."""
    if '<cot>' not in cot: cot = '<cot>' + cot
    if '</cot>' not in cot:
        cot = cot.split('</cot>')[0] + '</cot>' if '</cot>' in cot else cot + '</cot>'
    return cot


# ---- per-ad generation (FALLBACK; identical to generate_eval.py) ------------
def gen_cot_single(model, tmpl_gen, TI, proc, row, max_new):
    rg = dict(row); rg['messages'] = list(row['messages'])
    rg['messages'][-1] = {'role': 'assistant', 'content': ''}
    enc = tmpl_gen.encode(TI.from_dict(rg)); batch = tmpl_gen.data_collator([enc])
    batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    for _k in _HEAD_TARGET_KWARGS:
        batch.pop(_k, None)
    with torch.no_grad():
        gen = model.generate(**batch, max_new_tokens=max_new, do_sample=False, num_beams=1)
    in_len = batch['input_ids'].shape[1]
    new = gen[0][in_len:]
    return _clean_cot(proc.tokenizer.decode(new, skip_special_tokens=True))


# ---- batched generation (the speedup) ---------------------------------------
def gen_cot_batched(model, tmpl_gen, TI, proc, rows, max_new):
    """Collate K rows into ONE generate() call. Returns list[str] of CoTs aligned
    to `rows`. The collator LEFT-pads in transformers mode (base.py:1817) and
    omits position_ids (qwen.py:933) so the model computes mrope itself with the
    left-pad delta0 correction. We slice each row's NEW tokens off the right end
    (generation is appended after the left-padded prompt block of width L).

    Encoding is per-row (multimodal encode is inherently per-sample); only the
    decode loop is amortized across the batch — that is where the 5-10x lives.
    """
    encs = []
    for r in rows:
        rg = dict(r); rg['messages'] = list(r['messages'])
        rg['messages'][-1] = {'role': 'assistant', 'content': ''}
        encs.append(tmpl_gen.encode(TI.from_dict(rg)))
    batch = tmpl_gen.data_collator(encs)              # LEFT-pads input_ids/attn; stacks video tensors
    batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    for _k in _HEAD_TARGET_KWARGS:
        batch.pop(_k, None)
    # SANITY: do NOT pass position_ids (must be absent so the model computes the
    # left-pad-corrected mrope itself). If the collator ever emits one in
    # inference mode, drop it — passing a pre-pad-shaped one would be wrong.
    batch.pop('position_ids', None)
    L = batch['input_ids'].shape[1]                  # padded prompt width (left-padded)
    with torch.no_grad():
        gen = model.generate(**batch, max_new_tokens=max_new, do_sample=False, num_beams=1)
    # gen: (K, L + n_new). With LEFT padding every row's prompt ends at column L-1,
    # so the generated tokens for EVERY row are exactly gen[i, L:]. (Right-pad would
    # need per-row offsets; left-pad makes the slice uniform — the whole point.)
    cots = []
    for i in range(gen.shape[0]):
        new = gen[i][L:]
        cots.append(_clean_cot(proc.tokenizer.decode(new, skip_special_tokens=True)))
    return cots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--plugin', required=True)
    ap.add_argument('--base', default=None,
                    help='original base model (processor + weights source). REQUIRED for chained RL adapters.')
    ap.add_argument('--attn-impl', default='flash_attn')
    ap.add_argument('--head-type', default='hazard')
    ap.add_argument('--max-length', type=int, default=32768)
    ap.add_argument('--max-new', type=int, default=600)
    ap.add_argument('--video-max-tokens', type=int, default=16384,
                    help='MATCH SFT (VIDEO_MAX_TOKEN_NUM). Must equal srcc_eval to keep bypass vs reasoned comparable.')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--t-lo', type=int, default=1)
    ap.add_argument('--t-hi', type=int, default=30)
    ap.add_argument('--output', default=None)
    ap.add_argument('--batch-size', type=int, default=8,
                    help='K ads per generate() call. 1 = per-ad fallback (== generate_eval.py). Start 8; '
                         'drop to 4 if OOM on long videos, raise toward 16 if VRAM is free. Generation VRAM '
                         'scales ~ K x (video tokens + prompt + max_new); long videos hit ~36k vid tok each.')
    ap.add_argument('--dump-cots', default=None,
                    help='optional path: write per-ad {ad_id, cot, R_gen, R_byp} jsonl for equivalence smoke diffing.')
    args = ap.parse_args()

    os.environ['RETENTION_HEAD_TYPE'] = args.head_type
    # CANONICAL video env -- MUST match srcc_eval.py + generate_eval.py exactly.
    os.environ.setdefault('MAX_PIXELS', '200704'); os.environ.setdefault('VIDEO_MAX_PIXELS', '200704')
    os.environ.setdefault('FPS_MAX_FRAMES', '60'); os.environ.setdefault('FPS', '1.0')
    os.environ.setdefault('VIDEO_MAX_TOKEN_NUM', str(args.video_max_tokens))
    print(f'[genB] CANONICAL video env: VIDEO_MAX_TOKEN_NUM={os.environ["VIDEO_MAX_TOKEN_NUM"]} '
          f'MAX_PIXELS={os.environ["MAX_PIXELS"]} FPS={os.environ["FPS"]} FPS_MAX_FRAMES={os.environ["FPS_MAX_FRAMES"]} '
          f'batch_size={args.batch_size}', flush=True)

    import_plugin(args.plugin)
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.template_inputs import TemplateInputs
    from swift.template.base import MaxLengthError

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
                              remove_unused_columns=False)
    tmpl_train.set_mode('train')
    tmpl_gen = get_template(proc, max_length=args.max_length, template_type='qwen2_5_omni_retention',
                            remove_unused_columns=False)
    tmpl_gen.set_mode('transformers')     # HF-inference encode; this is what triggers left-pad + no-precompute-posids
    print(f'[genB] loaded {args.checkpoint} attn={args.attn_impl}', flush=True)

    rows = [json.loads(l) for l in open(args.val_jsonl) if (json.loads(l).get('R') or json.loads(l).get('R_true'))]
    if args.limit: rows = rows[:args.limit]

    preds_gen, preds_byp, trues, Ts, ad_ids, skipped = [], [], [], [], [], 0
    dump = [] if args.dump_cots else None

    import time
    t_start = time.time()
    K = max(1, args.batch_size)

    def _flush_batch(buf):
        """buf: list of (idx_in_rows, row, R_true, T). Generate CoTs (batched if K>1),
        then per-ad head readout. Appends to the result lists; returns n_skipped_here."""
        nonlocal skipped
        if not buf: return
        rows_b = [b[1] for b in buf]
        try:
            if K > 1 and len(rows_b) > 1:
                cots = gen_cot_batched(model, tmpl_gen, TemplateInputs, proc, rows_b, args.max_new)
            else:
                cots = [gen_cot_single(model, tmpl_gen, TemplateInputs, proc, rows_b[0], args.max_new)]
        except (MaxLengthError, Exception) as e:                                   # noqa
            # A batched failure is ambiguous (could be one bad ad). Fall back to
            # per-ad so one pathological video doesn't drop the whole batch.
            print(f'[genB] batch generate failed ({type(e).__name__}: {str(e)[:120]}); '
                  f'retrying {len(rows_b)} ads per-ad', flush=True)
            cots = []
            for rb in rows_b:
                try:
                    cots.append(gen_cot_single(model, tmpl_gen, TemplateInputs, proc, rb, args.max_new))
                except (MaxLengthError, Exception) as e2:                          # noqa
                    cots.append(None)
        for (idx, row, R_true, T), cot in zip(buf, cots):
            if cot is None:
                skipped += 1; continue
            try:
                R_gen = head_curve(model, tmpl_train, TemplateInputs, row, cot)
                R_byp = head_curve(model, tmpl_train, TemplateInputs, row, '<cot></cot>')
            except (MaxLengthError, Exception) as e:                               # noqa
                skipped += 1
                if idx < 3: print(f'[genB] ad{idx} head-read skipped: {type(e).__name__}: {str(e)[:140]}')
                continue
            preds_gen.append(R_gen); preds_byp.append(R_byp); trues.append(R_true); Ts.append(T)
            ad_ids.append(str(row.get('ad_id', '')))
            if dump is not None:
                dump.append({'ad_id': str(row.get('ad_id', '')), 'cot': cot,
                             'R_gen': [float(x) for x in R_gen], 'R_byp': [float(x) for x in R_byp]})
            if idx < 3: print(f'[genB] ad{idx} cot[:120]={cot[:120]!r}')

    buf = []
    for i, r in enumerate(rows):
        R_true = np.array(r.get('R') or r.get('R_true'), float); T = len(R_true) - 1
        if T < 5: skipped += 1; continue
        buf.append((i, r, R_true, T))
        if len(buf) >= K:
            _flush_batch(buf); buf = []
            dt = time.time() - t_start
            print(f'[genB] {len(preds_gen)} done, {skipped} skipped, '
                  f'{dt/max(1,len(preds_gen)):.1f}s/ad avg', flush=True)
    _flush_batch(buf)

    def crossad(preds):
        per = []
        for t in range(args.t_lo, args.t_hi + 1):
            pv = [P[t] for P, T in zip(preds, Ts) if T >= t]
            tv = [Tr[t] for Tr, T in zip(trues, Ts) if T >= t]
            rho = spearman(pv, tv)
            if not np.isnan(rho): per.append(rho)
        return float(np.mean(per)) if per else float('nan')

    srcc_gen = crossad(preds_gen); srcc_byp = crossad(preds_byp)
    print(f'\n[genB] n={len(preds_gen)} skipped={skipped} batch_size={K}')
    print(f'[genB] BYPASS  (empty cot) cross-ad SRCC = {srcc_byp:.4f}')
    print(f'[genB] REASONED(gen cot)   cross-ad SRCC = {srcc_gen:.4f}')
    print(f'[genB] delta (reasoned - bypass) = {srcc_gen-srcc_byp:+.4f}   vs baseline 0.5142')
    if args.output:
        json.dump({'srcc_bypass': srcc_byp, 'srcc_reasoned': srcc_gen, 'n': len(preds_gen),
                   'skipped': skipped, 'batch_size': K}, open(args.output, 'w'), indent=2)
    if args.dump_cots and dump is not None:
        with open(args.dump_cots, 'w') as f:
            for d in dump: f.write(json.dumps(d) + '\n')
        print(f'[genB] dumped {len(dump)} per-ad CoTs -> {args.dump_cots}')


if __name__ == '__main__': main()

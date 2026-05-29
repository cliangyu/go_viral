#!/usr/bin/env python
"""Dump ckpt-225's h_anchor (last-token hidden the head reads) + true curve for each val ad,
for the HEAD-CEILING ORACLE (is val SRCC 0.514 the linear-readout ceiling of these features?).
Same loader/bypass forward as srcc_eval.py; just captures holder.last[:, -1] (== h_anchor) + R_true.
Output: npz with feats (n,d), rtrue (n,Tmax+1) padded, Ts (n,)."""
from __future__ import annotations
import argparse, importlib.util, json, os
import numpy as np, torch


def import_plugin(p):
    spec = importlib.util.spec_from_file_location('retention_plugin', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--plugin', required=True)
    ap.add_argument('--attn-impl', default='flash_attn')
    ap.add_argument('--head-type', default='hazard')
    ap.add_argument('--max-length', type=int, default=32768)
    ap.add_argument('--tmax', type=int, default=60)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    os.environ['RETENTION_HEAD_TYPE'] = args.head_type
    import_plugin(args.plugin)
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.template_inputs import TemplateInputs
    from swift.template.base import MaxLengthError

    model, proc = get_model_processor(args.checkpoint, torch_dtype=torch.bfloat16, attn_impl=args.attn_impl,
                                      model_kwargs={'device_map': 'cuda'}, model_type='qwen2_5_omni_retention')
    model.eval()
    template = get_template(proc, max_length=args.max_length, template_type='qwen2_5_omni_retention',
                            remove_unused_columns=False)
    template.set_mode('train')
    print(f'[dump] loaded {args.checkpoint} attn={args.attn_impl}')

    rows = []
    with open(args.val_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if (r.get('R') or r.get('R_true')):
                rows.append(r)

    Tmax = args.tmax
    feats, feats_mean, rtrue, Ts, skipped = [], [], [], [], 0
    for i, r in enumerate(rows):
        R_true = np.array(r.get('R') or r.get('R_true'), dtype=np.float64)
        T = len(R_true) - 1
        if T < 5:
            skipped += 1; continue
        row = dict(r); row['messages'] = list(r['messages'])
        row['messages'][-1] = {'role': 'assistant', 'content': '<cot></cot>'}
        try:
            enc = template.encode(TemplateInputs.from_dict(row))
            batch = template.data_collator([enc])
            batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            with torch.no_grad():
                _ = model(**batch)
            holder = model._retention_h_holder
            hl = holder.last[0].float()                              # (L, d) full sequence hiddens
            h_anchor = hl[-1].cpu().numpy()                          # (d,) last token = the head's actual input
            h_mean = hl.mean(0).cpu().numpy()                        # (d,) mean-pool over all tokens (richer-pooling probe)
        except (MaxLengthError, Exception) as e:                     # noqa
            skipped += 1; continue
        # pad/truncate true curve to Tmax+1 (carry last value beyond T)
        c = np.full(Tmax + 1, R_true[min(T, len(R_true) - 1)], dtype=np.float64)
        c[:min(len(R_true), Tmax + 1)] = R_true[:Tmax + 1]
        feats.append(h_anchor); feats_mean.append(h_mean); rtrue.append(c); Ts.append(T)
        if (i + 1) % 25 == 0:
            print(f'[dump] {len(feats)} done, {skipped} skipped')

    feats = np.stack(feats); feats_mean = np.stack(feats_mean); rtrue = np.stack(rtrue); Ts = np.array(Ts)
    np.savez(args.output, feats=feats.astype(np.float32), feats_mean=feats_mean.astype(np.float32),
             rtrue=rtrue.astype(np.float32), Ts=Ts)
    print(f'[dump] n={len(feats)} skipped={skipped} d={feats.shape[1]} -> {args.output}')


if __name__ == '__main__':
    main()

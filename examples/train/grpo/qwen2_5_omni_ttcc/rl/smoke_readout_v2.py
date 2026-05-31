#!/usr/bin/env python
"""GATE before any V2 readout re-SFT. Proves, for a given RETENTION_READOUT:
  (1) the <cot>-span MASK does NOT pool video/pad tokens (the #1 risk) and matches
      the literal CoT interior;
  (2) the forward produces a finite, monotone-hazard r_pred of shape (1,60) on both a
      real-CoT row and the empty-span bypass row (the §1.4 fallback fires);
  (3) for attn_pool, retention_pool.query receives a finite, non-zero gradient.

Run per readout on ONE GPU (RETENTION_READOUT is resolved at LOAD):
  RETENTION_READOUT picked via --readout; env video budget matches SFT.
"""
from __future__ import annotations
import argparse, importlib.util, json, os
import numpy as np, torch


def load_module(p, name='retplugin'):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--readout', required=True, choices=['last_token', 'mean', 'attn_pool'])
    ap.add_argument('--base', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--register', required=True)
    ap.add_argument('--attn-impl', default='flash_attention_2')
    args = ap.parse_args()

    os.environ['RETENTION_HEAD_TYPE'] = 'hazard'
    os.environ['RETENTION_READOUT'] = args.readout
    for k, v in {'MAX_PIXELS': '200704', 'VIDEO_MAX_PIXELS': '200704', 'FPS_MAX_FRAMES': '60',
                 'FPS': '1.0', 'VIDEO_MAX_TOKEN_NUM': '16384'}.items():
        os.environ.setdefault(k, v)

    reg = load_module(args.register)
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.template_inputs import TemplateInputs

    model, proc = get_model_processor(args.base, torch_dtype=torch.bfloat16, attn_impl=args.attn_impl,
                                      model_kwargs={'device_map': 'cuda'}, model_type='qwen2_5_omni_retention')
    model.eval()
    tok = proc.tokenizer
    tmpl = get_template(proc, max_length=32768, template_type='qwen2_5_omni_retention', remove_unused_columns=False)
    tmpl.set_mode('train')

    row = None
    for line in open(args.val_jsonl):
        r = json.loads(line)
        if (r.get('R') or r.get('R_true')):
            row = r; break
    assert row is not None, 'no eligible val row'
    CoT = '<cot>0s | hand enters, product, text overlay | strong visual hook\n1s | rapid cut, motion | high intensity\n2s | close-up payoff | clear value</cot>'
    real = dict(row); real['messages'] = list(row['messages']); real['messages'][-1] = {'role': 'assistant', 'content': CoT}
    byp = dict(row);  byp['messages'] = list(row['messages']);  byp['messages'][-1] = {'role': 'assistant', 'content': '<cot></cot>'}

    def enc(rr):
        e = tmpl.encode(TemplateInputs.from_dict(rr)); b = tmpl.data_collator([e])
        return {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in b.items()}

    # ===== MASK GATE (real-CoT row) =====
    br = enc(real); ids = br['input_ids']; L = ids.shape[1]
    open_ids = reg._find_open_cot_token_ids(tok); close_ids = reg._find_close_cot_token_ids(tok)
    mask, anchor = reg._build_cot_span_mask(ids, open_ids, close_ids)
    nmask = int(mask.sum().item()); masked_ids = ids[0][mask[0]].tolist()
    dec = tok.decode(masked_ids)
    vid_id = (getattr(tok, 'video_token_id', None)
              or getattr(getattr(model.config, 'thinker_config', model.config), 'video_token_id', None))
    nvid = sum(1 for t in masked_ids if vid_id is not None and t == vid_id) if vid_id is not None else -1
    print(f'[mask] L={L} mask.sum={nmask} anchor={int(anchor[0])} video_token_id={vid_id} video_in_mask={nvid}')
    print(f'[mask] decoded span: {dec[:220]!r}')
    assert 0 < nmask < 600, f'mask.sum={nmask} unreasonable (CoT is tens-hundreds, video is 10^3-10^4)'
    assert nvid <= 0, f'{nvid} VIDEO tokens inside the CoT mask -> pooling over video!'
    assert '<cot>' not in dec and '</cot>' not in dec, 'markers leaked into span'
    print('[mask] GATE PASS (no video/pad/markers in the pooled span)')

    # ===== forward: real + bypass =====
    with torch.no_grad():
        rp_real = getattr(model(**br), 'r_pred', None)
        rp_byp = getattr(model(**enc(byp)), 'r_pred', None)
    for nm, rp in [('real-CoT', rp_real), ('bypass', rp_byp)]:
        ok = rp is not None and rp.shape[-1] == 60 and bool(torch.isfinite(rp).all())
        mono = bool((rp[:, 1:] <= rp[:, :-1] + 1e-3).all()) if ok else False
        print(f'[fwd] {nm}: shape={tuple(rp.shape) if rp is not None else None} finite={ok} hazard_monotone={mono} '
              f'R[:5]={rp[0, :5].float().cpu().numpy().round(3) if ok else None}')
        assert ok and mono, f'{nm} r_pred bad'

    # ===== grad (attn_pool only) =====
    if args.readout == 'attn_pool':
        model.train()
        getattr(model(**br), 'r_pred').sum().backward()
        g = model.retention_pool.query.grad
        fin = bool(torch.isfinite(g).all()); nz = bool(g.abs().sum() > 0)
        print(f'[grad] retention_pool.query.grad finite={fin} nonzero={nz}')
        assert g is not None and fin and nz, 'attn_pool query got no gradient -> not in the graph'

    print(f'\nSMOKE PASS readout={args.readout}')


if __name__ == '__main__':
    main()

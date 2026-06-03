"""Delta-value gate, step 1: per-ad feature extraction (shardable, parallel-GPU).

For each val ad: (1) generate the CoT from video+prompt; (2) R_video = head curve from (video + CoT)
[the video-channel baseline]; (3) h_cot = pooled hidden state from a TEXT-ONLY re-encode of the CoT
(NO video) -> the CoT-only feature. Dump {ad_id, R_video, R_true, h_cot_mean, h_cot_last, T}.

The offline probe (delta_gate_probe.py) then fits lambda_cot on h_cot and computes
  Delta = SRCC(video + cot) - SRCC(video).

Model load / generation / video-env are copied verbatim from generate_eval.py (proven path).
Shard with --shard k --nshards N (processes ads[k::N]); run one per GPU for parallelism.
"""
import argparse, importlib.util, json, os
import numpy as np, torch


def import_plugin(p):
    spec = importlib.util.spec_from_file_location('reg_plugin', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def head_curve(model, template, TI, row, assistant):
    r = dict(row); r['messages'] = list(row['messages']); r['messages'][-1] = {'role': 'assistant', 'content': assistant}
    enc = template.encode(TI.from_dict(r)); batch = template.data_collator([enc])
    batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch)
    rp = getattr(out, 'r_pred', None)
    if rp is None:
        rp = model._retention_h_holder.r_pred
    return np.concatenate([[1.0], rp[0].float().cpu().numpy()])


def cot_only_feature(model, template, TI, row, cot):
    """TEXT-ONLY re-encode of the CoT (no video). Returns (h_mean, h_last) pooled over the
    captured final hidden state. holder.last is set by the lm_head pre-hook DURING the base
    forward (before any readout), so we get it even if the head's readout errors on this minimal input."""
    r = {'messages': [
        {'role': 'system', 'content': 'You forecast second-by-second short-video audience retention.'},
        {'role': 'user', 'content': 'Per-second reasoning about a short-video ad:'},
        {'role': 'assistant', 'content': cot}],
        'T': row.get('T'), 'R': row.get('R') or row.get('R_true')}
    enc = template.encode(TI.from_dict(r)); batch = template.data_collator([enc])
    batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    holder = model._retention_h_holder
    holder.last = None
    try:
        with torch.no_grad():
            model(**batch)
    except Exception:  # noqa: BLE001 -- readout may choke on the minimal text input; we only need holder.last
        pass
    h = holder.last
    if h is None:
        return None, None
    h = h[0].float()                                            # (L, d)
    am = batch.get('attention_mask')
    if am is not None:
        h = h[am[0].bool()]
    return h.mean(0).cpu().numpy(), h[-1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--val-jsonl', required=True); ap.add_argument('--plugin', required=True)
    ap.add_argument('--base', default=None); ap.add_argument('--attn-impl', default='flash_attn'); ap.add_argument('--head-type', default='hazard')
    ap.add_argument('--max-length', type=int, default=32768); ap.add_argument('--max-new', type=int, default=600)
    ap.add_argument('--video-max-tokens', type=int, default=16384)
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    os.environ['RETENTION_HEAD_TYPE'] = args.head_type
    os.environ.setdefault('MAX_PIXELS', '200704'); os.environ.setdefault('VIDEO_MAX_PIXELS', '200704')
    os.environ.setdefault('FPS_MAX_FRAMES', '60'); os.environ.setdefault('FPS', '1.0')
    os.environ.setdefault('VIDEO_MAX_TOKEN_NUM', str(args.video_max_tokens))
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
        # EXPLICIT retention-head restore: PEFT does NOT load the plain retention_head.* keys
        # from a LoRA adapter (they're stored as base_model.model.retention_head.*, not
        # modules_to_save.default.*), and register.py's _maybe_load only handles full-FT
        # model.safetensors. Without this the head runs at INIT (bias=-3) -> ~0 SRCC.
        from safetensors.torch import load_file as _lf
        _ad = os.path.join(args.checkpoint, 'adapter_model.safetensors')
        if os.path.exists(_ad):
            _sd = _lf(_ad)
            def _find(m):
                for p in ('base_model.model', 'base_model', 'model', ''):
                    o = m
                    ok = True
                    for a in (p.split('.') if p else []):
                        if hasattr(o, a):
                            o = getattr(o, a)
                        else:
                            ok = False; break
                    if ok and hasattr(o, 'retention_head'):
                        return o
                return None
            _host = _find(model)
            _dev = next(model.parameters()).device
            def _load_active(modu, sd):
                # modules_to_save=['retention_head'] -> PEFT wraps it; the ACTIVE forward path is
                # modules_to_save[active_adapter], NOT the plain module. The checkpoint stores PLAIN
                # keys (no modules_to_save.* prefix), so load them into the ACTIVE wrapper module.
                if hasattr(modu, 'modules_to_save'):
                    act = getattr(modu, 'active_adapter', 'default')
                    if isinstance(act, (list, tuple)):
                        act = act[0] if act else 'default'
                    tgt = modu.modules_to_save[act] if act in modu.modules_to_save else next(iter(modu.modules_to_save.values()))
                    m, u = tgt.load_state_dict(sd, strict=False)
                    if hasattr(modu, 'original_module'):
                        modu.original_module.load_state_dict(sd, strict=False)
                    return tgt, f'wrapper.modules_to_save[{act}]', m, u
                m, u = modu.load_state_dict(sd, strict=False)
                return modu, 'plain', m, u
            for _mod in ('retention_head', 'retention_pool'):
                if _host is None or not hasattr(_host, _mod):
                    continue
                _pre = f'base_model.model.{_mod}.'
                _msd = {k[len(_pre):]: v for k, v in _sd.items() if k.startswith(_pre)}
                if _msd:
                    _tgt, _path, _miss, _unexp = _load_active(getattr(_host, _mod), _msd)
                    getattr(_host, _mod).to(_dev)
                    if _mod == 'retention_head':
                        print(f'[gate] RESTORED retention_head -> {_path}: keys={list(_msd.keys())} '
                              f'bias[:3]={_tgt.linear.bias[:3].tolist()} (init=-3) missing={_miss} unexpected={_unexp}', flush=True)
    else:
        model, proc = get_model_processor(args.checkpoint, torch_dtype=torch.bfloat16, attn_impl=args.attn_impl,
                                          model_kwargs={'device_map': 'cuda'}, model_type='qwen2_5_omni_retention')
    model.eval()
    tmpl_train = get_template(proc, max_length=args.max_length, template_type='qwen2_5_omni_retention', remove_unused_columns=False)
    tmpl_train.set_mode('train')
    tmpl_gen = get_template(proc, max_length=args.max_length, template_type='qwen2_5_omni_retention', remove_unused_columns=False)
    tmpl_gen.set_mode('transformers')
    print(f'[gate] loaded {args.checkpoint} shard {args.shard}/{args.nshards}', flush=True)
    rows = [json.loads(l) for l in open(args.val_jsonl) if (json.loads(l).get('R') or json.loads(l).get('R_true'))]
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.shard::args.nshards]
    fout = open(args.output, 'w'); n_ok = n_skip = 0
    for i, r in enumerate(rows):
        R_true = np.array(r.get('R') or r.get('R_true'), float); T = len(R_true) - 1
        if T < 5:
            n_skip += 1; continue
        try:
            rg = dict(r); rg['messages'] = list(r['messages']); rg['messages'][-1] = {'role': 'assistant', 'content': ''}
            enc = tmpl_gen.encode(TemplateInputs.from_dict(rg)); batch = tmpl_gen.data_collator([enc])
            batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            for _k in ('r_true', 'r_mask', 'ad_id', 'T', 'R', 'labels', 'loss_scale', 'channel'):
                batch.pop(_k, None)
            with torch.no_grad():
                gen = model.generate(**batch, max_new_tokens=args.max_new, do_sample=False, num_beams=1)
            cot = proc.tokenizer.decode(gen[0][batch['input_ids'].shape[1]:], skip_special_tokens=True)
            if '<cot>' not in cot:
                cot = '<cot>' + cot
            if '</cot>' not in cot:
                cot = cot + '</cot>'
            R_video = head_curve(model, tmpl_train, TemplateInputs, r, cot)
            R_bypass = head_curve(model, tmpl_train, TemplateInputs, r, '<cot></cot>')  # empty-cot read = the 0.514 baseline
            h_mean, h_last = cot_only_feature(model, tmpl_train, TemplateInputs, r, cot)
            if h_mean is None:
                n_skip += 1; continue
        except (MaxLengthError, Exception) as e:  # noqa
            n_skip += 1
            if i < 3:
                print(f'[gate] ad{i} skip: {type(e).__name__}: {str(e)[:160]}', flush=True)
            continue
        fout.write(json.dumps({'ad_id': r.get('ad_id'), 'T': int(T),
                               'R_true': [float(x) for x in R_true],
                               'R_video': [float(x) for x in R_video],
                               'R_bypass': [float(x) for x in R_bypass],
                               'h_mean': [float(x) for x in h_mean],
                               'h_last': [float(x) for x in h_last]}) + '\n')
        fout.flush(); n_ok += 1
        if i < 2:
            print(f'[gate] ad{i} ok: cot[:90]={cot[:90]!r} R_video[3]={R_video[3]:.3f} R_true[3]={R_true[3]:.3f} d={len(h_mean)}', flush=True)
        if (i + 1) % 10 == 0:
            print(f'[gate] shard{args.shard}: {n_ok} ok, {n_skip} skip', flush=True)
    fout.close()
    print(f'[gate] DONE shard{args.shard}: {n_ok} ok, {n_skip} skip -> {args.output}', flush=True)


if __name__ == '__main__':
    main()

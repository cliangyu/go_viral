"""Precheck Part 2: does an RL gradient exist? For a few ads, generate G CoTs each,
apply the FROZEN head to each CoT's text-only (cot-only) hidden, and measure the
WITHIN-GROUP curve std (different CoTs of the SAME ad -> different head curves?).
std > 1e-4 => GRPO has a gradient to climb. Reuses delta_gate_extract's proven load."""
import argparse, importlib.util, json, os
import numpy as np, torch


def import_mod(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--plugin', required=True); ap.add_argument('--dge', required=True)
    ap.add_argument('--n-ads', type=int, default=5); ap.add_argument('--G', type=int, default=8)
    ap.add_argument('--video-max-tokens', type=int, default=8192)
    ap.add_argument('--temperature', type=float, default=1.0); ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    os.environ['RETENTION_HEAD_TYPE'] = 'hazard'
    os.environ.setdefault('MAX_PIXELS', '200704'); os.environ.setdefault('VIDEO_MAX_PIXELS', '200704')
    os.environ.setdefault('FPS_MAX_FRAMES', '60'); os.environ.setdefault('FPS', '1.0')
    os.environ.setdefault('VIDEO_MAX_TOKEN_NUM', str(args.video_max_tokens))
    os.environ.setdefault('RETENTION_READOUT', 'last_token')

    dge = import_mod(args.dge, 'dge')          # reuse delta_gate_extract's helpers
    dge.import_plugin(args.plugin)
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.template_inputs import TemplateInputs

    # ---- load model + faithful head (same path as delta_gate_extract.main) ----
    base = json.load(open(os.path.join(args.checkpoint, 'adapter_config.json')))['base_model_name_or_path']
    model, proc = get_model_processor(base, torch_dtype=torch.bfloat16, attn_impl='flash_attn',
                                      model_kwargs={'device_map': 'cuda'}, model_type='qwen2_5_omni_retention')
    from peft import PeftModel
    from safetensors.torch import load_file as _lf
    model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=False)
    _sd = _lf(os.path.join(args.checkpoint, 'adapter_model.safetensors'))
    # frozen head weights for manual application (consistent with precheck part 1)
    W = _sd['base_model.model.retention_head.linear.weight'].float().cpu().numpy()
    b = _sd['base_model.model.retention_head.linear.bias'].float().cpu().numpy()
    model.eval()
    tmpl_train = get_template(proc, max_length=32768, template_type='qwen2_5_omni_retention', remove_unused_columns=False)
    tmpl_train.set_mode('train')
    tmpl_gen = get_template(proc, max_length=32768, template_type='qwen2_5_omni_retention', remove_unused_columns=False)
    tmpl_gen.set_mode('transformers')

    def hcurve(h):
        z = W @ np.asarray(h, float) + b
        lam = np.logaddexp(0.0, z)
        return np.concatenate([[1.0], np.exp(-np.cumsum(lam))])

    from swift.template.base import MaxLengthError
    rows = [json.loads(l) for l in open(args.val_jsonl) if (json.loads(l).get('R') or json.loads(l).get('R_true'))]
    rows = [r for r in rows if len(r.get('R') or r.get('R_true')) - 1 >= 5]
    fout = open(args.output, 'w'); allstd = []; n_done = 0
    for ai, r in enumerate(rows):
        if n_done >= args.n_ads:
            break
        R_true = np.array(r.get('R') or r.get('R_true'), float)
        rg = dict(r); rg['messages'] = list(r['messages']); rg['messages'][-1] = {'role': 'assistant', 'content': ''}
        try:
            enc = tmpl_gen.encode(TemplateInputs.from_dict(rg)); batch = tmpl_gen.data_collator([enc])
        except MaxLengthError:
            continue                                   # skip over-length video ads (like delta_gate_extract)
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        for _k in ('r_true', 'r_mask', 'ad_id', 'T', 'R', 'labels', 'loss_scale', 'channel'):
            batch.pop(_k, None)
        cots, curves, cot_lens = [], [], []
        for g in range(args.G):
            with torch.no_grad():
                gen = model.generate(**batch, max_new_tokens=480, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p, num_beams=1)
            cot = proc.tokenizer.decode(gen[0][batch['input_ids'].shape[1]:], skip_special_tokens=True)
            if '<cot>' not in cot: cot = '<cot>' + cot
            if '</cot>' not in cot: cot = cot + '</cot>'
            h_mean, h_last = dge.cot_only_feature(model, tmpl_train, TemplateInputs, r, cot)
            if h_last is None:
                continue
            cots.append(cot); cot_lens.append(len(proc.tokenizer.encode(cot))); curves.append(hcurve(h_last))
        if len(curves) < 2:
            print(f'[smoke] ad{ai}: <2 valid gens', flush=True); continue
        M = np.stack([c[1:31] for c in curves])
        within_std = float(M.std(axis=0).mean())           # the RL-gradient signal
        scal = np.array([np.mean(c[1:31]) for c in curves])
        n_distinct_cot = len(set(cots))
        allstd.append(within_std)
        print(f'[smoke] ad{ai} ({r.get("ad_id")}): within-group curve std={within_std:.5f}  '
              f'scalar[min..max]=[{scal.min():.3f}..{scal.max():.3f}]  distinct_cots={n_distinct_cot}/{len(cots)}  '
              f'cot_len[min..max]=[{min(cot_lens)}..{max(cot_lens)}]', flush=True)
        fout.write(json.dumps({'ad_id': r.get('ad_id'), 'within_std': within_std,
                               'scalars': scal.tolist(), 'cot_lens': cot_lens,
                               'n_distinct': n_distinct_cot}) + '\n'); fout.flush()
        n_done += 1
    fout.close()
    if allstd:
        a = np.array(allstd)
        print(f'\n[smoke] WITHIN-GROUP curve std across {len(a)} ads: mean={a.mean():.5f} min={a.min():.5f} max={a.max():.5f}', flush=True)
        print(f'[smoke] GATE: {"PASS (>1e-4) -> RL gradient exists" if a.mean() > 1e-4 else "FAIL (~0) -> no gradient, do NOT launch"}', flush=True)


if __name__ == '__main__':
    main()

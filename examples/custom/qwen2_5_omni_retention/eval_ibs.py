#!/usr/bin/env python
# Copyright (c) Stanford TTCC. All rights reserved.
"""Phase-C eval: per-ad IBS from a retention-head checkpoint vs the train-mean
baseline (B_1).

Usage:
  python eval_ibs.py \
      --checkpoint /opt/dlami/nvme/ssm-out/phaseB_hazard_full/.../checkpoint-30 \
      --val-jsonl /home/ssm-user/work/data/ttcc_swift_v2cot/ttcc_test.jsonl \
      --limit 20 \
      --plugin examples/custom/qwen2_5_omni_retention/register.py

Outputs: per-ad IBS + aggregate mean, alongside the train-mean baseline IBS
computed from the val set's R_true values.

Inference protocol (teacher-forced):
  The retention head reads the final hidden state at the last </cot> token.
  For eval, we feed the assistant span verbatim (the distilled CoT body for
  with-CoT variants, the JSON-only body for no-CoT variants). This is
  equivalent to "model would have generated this exact CoT" and isolates the
  head's predictive quality from CoT-generation quality.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def import_plugin(plugin_path: str) -> None:
    """Import the retention plugin so model_type / template / loss are registered."""
    spec = importlib.util.spec_from_file_location('retention_plugin', plugin_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def load_val_rows(val_jsonl: str, limit: int | None) -> list[dict]:
    rows = []
    with open(val_jsonl) as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def compute_b1(val_rows: list[dict]) -> tuple[np.ndarray, int]:
    """Train-mean baseline B_1: per-second mean over val rows."""
    R_curves = []
    Ts = []
    for r in val_rows:
        R = r.get('R') or r.get('R_true')
        if R is None:
            continue
        R_curves.append(np.array(R, dtype=np.float32))
        Ts.append(len(R) - 1)
    T_max = max(Ts)
    # Pad to T_max+1, then per-second mean
    padded = np.full((len(R_curves), T_max + 1), np.nan)
    for i, R in enumerate(R_curves):
        padded[i, : len(R)] = R
    B1 = np.nanmean(padded, axis=0)
    return B1, T_max


def per_ad_ibs(R_pred: np.ndarray, R_true: np.ndarray, T_i: int) -> float:
    """IBS_i = (1/(T_i+1)) * sum_{t=0..T_i} (R_pred[t] - R_true[t])^2."""
    pred = R_pred[: T_i + 1]
    true = R_true[: T_i + 1]
    return float(((pred - true) ** 2).mean())


def _generate_cot_then_forward(model, processor, template, row, *,
                               max_new_tokens=600, bypass=False):
    """Autoregressive CoT generation followed by a single forward pass that
    triggers the retention head via the LM head pre-hook.

    Returns (r_pred_tensor, generated_cot_str_or_None).

    Two modes:
      bypass=False: generate <cot>...</cot> autoregressively (stop on </cot>),
        then run a single forward on (prompt + generated_cot) and read h[</cot>].
      bypass=True: skip generation entirely; insert "<cot></cot>" so the head
        anchors on the immediate </cot> with no intervening reasoning. Used as
        the CoT-bypass diagnostic (paired with bypass=False to compute ΔIBS).
    """
    import torch
    from swift.template.template_inputs import TemplateInputs

    # Encode the prompt-only side. Override the assistant span with a
    # generation-time prefix so the model continues from `<cot>`.
    row_for_gen = dict(row)
    row_for_gen['messages'] = list(row['messages'])
    if bypass:
        # Empty CoT: head reads the encoder state immediately after the open tag.
        row_for_gen['messages'][-1] = {'role': 'assistant', 'content': '<cot></cot>'}
        ti = TemplateInputs.from_dict(row_for_gen)
        enc = template.encode(ti)
        batch = template.data_collator([enc])
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        r_pred = getattr(out, 'r_pred', None)
        return r_pred, ''

    # bypass=False: use the chat template's normal "generation prompt" path —
    # encode with a `<cot>` open and let model.generate continue. The trick:
    # we set the assistant content to a literal `<cot>` so the template
    # appends `<cot>` and stops there; then generate up to a `</cot>` stop string.
    row_for_gen['messages'][-1] = {'role': 'assistant', 'content': '<cot>'}
    ti = TemplateInputs.from_dict(row_for_gen)
    enc = template.encode(ti)
    batch = template.data_collator([enc])
    # Move EVERYTHING to cuda for generation; HF generate handles its own
    # device placement.
    batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    # Generate until </cot>. Greedy for reproducibility; sampling is only for
    # V9 RL rollouts (different code path).
    gen_kwargs = {
        'max_new_tokens': max_new_tokens,
        'do_sample': False,
        'eos_token_id': processor.tokenizer.eos_token_id,
        'pad_token_id': processor.tokenizer.pad_token_id,
        'stop_strings': ['</cot>'],
        'tokenizer': processor.tokenizer,
    }
    # Drop r_true/r_mask before generate (it's a forward-only signal).
    gen_batch = {k: v for k, v in batch.items() if k not in ('r_true', 'r_mask')}
    with torch.no_grad():
        gen = model.generate(**gen_batch, **gen_kwargs)

    # Decode the generated CoT body for inspection.
    prompt_len = batch['input_ids'].shape[1]
    new_tokens = gen[0, prompt_len:]
    generated_cot = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Run a single forward over the FULL sequence (prompt + generated) to read
    # h[</cot>] via the retention head's pre-hook. r_true/r_mask must be
    # absent at this stage (no MSE target during inference); the head still
    # populates r_pred on outputs.
    full_ids = gen
    forward_inputs = dict(batch)
    forward_inputs['input_ids'] = full_ids
    # Adjust attention_mask if present.
    if 'attention_mask' in forward_inputs:
        forward_inputs['attention_mask'] = torch.ones_like(full_ids)
    # Drop r_true/r_mask (don't compute training loss here).
    forward_inputs.pop('r_true', None)
    forward_inputs.pop('r_mask', None)
    with torch.no_grad():
        out = model(**forward_inputs)
    r_pred = getattr(out, 'r_pred', None)
    if r_pred is None:
        # Fallback to holder.
        base = getattr(model, 'base_model', model)
        base = getattr(base, 'model', base)
        holder = getattr(base, '_retention_h_holder', None)
        r_pred = holder.r_pred if holder is not None else None
    return r_pred, generated_cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--plugin', required=True)
    ap.add_argument('--head-type', default='hazard', choices=['hazard', 'sigmoid'])
    ap.add_argument('--max-length', type=int, default=32768,
                    help='Sequence length cap. Defaults to 32768 to match the V7+ training '
                         'config; reduces eval drop rate ~12.5%% (at 24576) -> ~5%% (at 32768) '
                         'on the long-ad tail (per 40-ad token-distribution measurement).')
    ap.add_argument('--output', type=Path, default=None,
                    help='Optional: write per-ad results + summary as JSON (parent dirs auto-created).')
    ap.add_argument('--dump-curves', action='store_true',
                    help='Include per-ad R_pred and R_true arrays in the output JSON '
                         '(enables pure-render plotting + drop-localization analysis).')
    ap.add_argument('--generate-cot', action='store_true',
                    help='Autoregressively generate <cot>...</cot> before reading the '
                         'retention head. Required for V8+ CoT-trained models where the '
                         'head anchors on the last </cot> token of the MODEL OUTPUT, not '
                         'on a teacher-forced CoT. Off by default; default path is single-forward '
                         'teacher-forced eval matching V7. Slower per ad (~4x) since it does '
                         '1 generate + 1 forward instead of 1 forward.')
    ap.add_argument('--cot-max-new-tokens', type=int, default=600,
                    help='Cap on autoregressively generated CoT tokens (only when '
                         '--generate-cot is set). Median CoT in our training data is ~280 '
                         'tokens; 600 is comfortable headroom.')
    ap.add_argument('--cot-bypass', action='store_true',
                    help='Diagnostic: skip CoT generation; insert immediate `</cot>` to '
                         'force the head to read the encoder state without reasoning. '
                         'Use with --generate-cot to compare ΔIBS = bypass IBS - full IBS. '
                         'If ΔIBS ≈ 0, the CoT is decorative; if positive, CoT is doing work. '
                         'The proposal central diagnostic of the reasoning thesis.')
    ap.add_argument('--strip-assistant', action='store_true', default=True,
                    help='SECURITY: strip the assistant message content before encoding. '
                         'DEFAULT ON. The V7-era TTCC data prep stored the ground-truth R(t) '
                         'curve inside the assistant span as `Curve: {"R": [...]}`. If the '
                         'eval row is encoded WITH that span, the model sees the answer in '
                         'its own input and the retention head trivially echoes it via '
                         'h[anchor]. This flag clears the assistant span so the model must '
                         'predict R(t) from the user-side multimodal input alone. Set --no-strip-assistant '
                         'to reproduce the (leaky) historical V7 numbers.')
    ap.add_argument('--no-strip-assistant', dest='strip_assistant', action='store_false',
                    help='Disable assistant-stripping. Restores the historical (leaky) eval path.')
    ap.add_argument('--train-jsonl', default=None,
                    help='Optional: compute the B1 climatology baseline from this train JSONL '
                         'instead of from val_rows themselves. Without this, B1 is computed '
                         "from the val set you're evaluating on — a methodological leak that "
                         "inflates B1's apparent skill (and the model's win-rate vs B1).")
    args = ap.parse_args()

    # 1. Import plugin to register model_type / template / loss.
    import os
    os.environ['RETENTION_HEAD_TYPE'] = args.head_type
    import_plugin(args.plugin)

    # 2. Load val rows + compute B_1.
    val_rows = load_val_rows(args.val_jsonl, args.limit)
    print(f'[eval] loaded {len(val_rows)} val rows from {args.val_jsonl}')

    # Strip assistant message content (LEAK FIX). The historical V7 prep put
    # the ground-truth R(t) into the assistant span; encoding the full row
    # exposes the answer to the model. Default: strip; restore via --no-strip-assistant.
    if args.strip_assistant:
        n_stripped = 0
        for row in val_rows:
            msgs = row.get('messages', [])
            for m in msgs:
                if m.get('role') == 'assistant' and m.get('content'):
                    m['content'] = ''
                    n_stripped += 1
        print(f'[eval] LEAK FIX: stripped assistant content from {n_stripped} rows '
              '(restore historical behavior with --no-strip-assistant)')

    # B1 climatology — prefer train-derived to avoid val-leak inflation.
    if args.train_jsonl:
        train_rows = load_val_rows(args.train_jsonl, None)
        print(f'[eval] B1 computed from train: {len(train_rows)} rows')
        B1_curve, T_max = compute_b1(train_rows)
    else:
        print(f'[eval] WARNING: B1 computed from val_rows (val-leak). '
              'Pass --train-jsonl <train.jsonl> for a clean baseline.')
        B1_curve, T_max = compute_b1(val_rows)
    print(f'[eval] B_1 curve length: {len(B1_curve)} (T_max={T_max})')

    # 3. Load model + tokenizer via ms-swift.
    # LoRA checkpoint dir holds adapter_model.safetensors + adapter_config.json
    # but no preprocessor / tokenizer / image_processor — those live with the
    # base model. Detect LoRA via adapter_config.json, load base + processor
    # from base_model_name_or_path, then apply the adapter on top.
    from swift.model import get_model_processor
    from swift.template import get_template

    adapter_config_path = os.path.join(args.checkpoint, 'adapter_config.json')
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_path = adapter_cfg['base_model_name_or_path']
        print(f'[eval] LoRA adapter detected; base = {base_path}')
        model, processor = get_model_processor(
            base_path,
            torch_dtype=torch.bfloat16,
            model_kwargs={'device_map': 'cuda'},
            model_type='qwen2_5_omni_retention',
        )
        # Apply adapter on top (peft loads modules_to_save head too).
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=False)
        print(f'[eval] adapter loaded from {args.checkpoint}')
    else:
        # Full-FT checkpoint: directory contains model.safetensors* + processor files.
        model, processor = get_model_processor(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            model_kwargs={'device_map': 'cuda'},
            model_type='qwen2_5_omni_retention',
        )
    model.eval()
    # get_template signature: (processor, default_system=None, max_length=None, *, template_type=None, ...)
    # template_type is keyword-only; pass it explicitly because the registered name and the model_type collide.
    template = get_template(processor, max_length=args.max_length,
                            template_type='qwen2_5_omni_retention',
                            remove_unused_columns=False)
    template.set_mode('train')  # train mode keeps the assistant span supervised so the </cot> anchor is in-context

    print(f'[eval] model loaded from {args.checkpoint}; head_type={args.head_type}')

    # 4. Per-ad IBS for both the model and B_1.
    model_ibs = []
    b1_ibs = []
    ad_records = []
    skipped = 0
    for i, row in enumerate(val_rows):
        R_true = np.array(row.get('R') or row.get('R_true'), dtype=np.float32)
        T_i = len(R_true) - 1
        if T_i < 5:
            skipped += 1
            continue

        # Encode via the template; mimic the training-time path so the </cot>
        # anchor lands in the right place. template.encode expects
        # TemplateInputs (which wraps StdTemplateInputs in chosen/rejected/etc),
        # not bare StdTemplateInputs.
        #
        # Production val sets contain occasional bad rows (audio decode
        # failure, row over max_length, missing video file). Match training's
        # truncation_strategy=delete behaviour: skip the row, count it,
        # continue. Don't let one bad row sink the whole eval.
        from swift.template.template_inputs import TemplateInputs
        from swift.template.base import MaxLengthError
        ti = TemplateInputs.from_dict(row)
        try:
            enc = template.encode(ti)
        except MaxLengthError as e:
            print(f'[eval] ad {i:3d}: skip (max_length: {str(e)[:80]})')
            skipped += 1
            continue
        except Exception as e:
            print(f'[eval] ad {i:3d}: skip (encode {type(e).__name__}: {str(e)[:80]})')
            skipped += 1
            continue
        if not enc:
            skipped += 1
            continue

        # Move tensors to GPU and forward
        generated_cot = None
        try:
            if args.generate_cot:
                # V8+ path: autoregressively generate <cot>...</cot> then read the
                # retention head from the </cot> anchor in a final forward.
                r_pred, generated_cot = _generate_cot_then_forward(
                    model, processor, template, row,
                    max_new_tokens=args.cot_max_new_tokens,
                    bypass=args.cot_bypass,
                )
            else:
                # V7 path: single-forward teacher-forced eval.
                batch = template.data_collator([enc])
                batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                with torch.no_grad():
                    out = model(**batch)
                    r_pred = getattr(out, 'r_pred', None)
                    if r_pred is None:
                        holder = getattr(model, '_retention_h_holder', None)
                        r_pred = holder.r_pred if holder is not None else None
            if r_pred is None:
                print(f'[eval] ad {i:3d}: r_pred MISSING')
                skipped += 1
                continue
            # r_pred is (1, 60); take first :T_i+1, force R(0)=1
            R_pred = r_pred[0].float().cpu().numpy()
            R_pred_full = np.concatenate([[1.0], R_pred])[: T_i + 1]
        except Exception as e:
            print(f'[eval] ad {i:3d}: skip (forward {type(e).__name__}: {str(e)[:80]})')
            skipped += 1
            continue

        ibs_model = per_ad_ibs(R_pred_full, R_true, T_i)
        ibs_b1 = per_ad_ibs(B1_curve[: T_i + 1], R_true, T_i)
        model_ibs.append(ibs_model)
        b1_ibs.append(ibs_b1)
        record = {'idx': i, 'ad_id': row.get('ad_id'), 'T': T_i,
                  'ibs_model': ibs_model, 'ibs_b1': ibs_b1}
        if args.dump_curves:
            record['R_pred'] = R_pred_full[: T_i + 1].astype(float).tolist()
            record['R_true'] = R_true[: T_i + 1].astype(float).tolist()
        if generated_cot is not None:
            record['generated_cot'] = generated_cot
        ad_records.append(record)
        print(f'[eval] ad {i:3d} (T={T_i:2d}): IBS_model={ibs_model:.5f}  IBS_B1={ibs_b1:.5f}  Δ={ibs_model - ibs_b1:+.5f}')

    print()
    print(f'[summary] n_evaluated={len(model_ibs)}  n_skipped={skipped}')
    print(f'[summary] mean IBS (model) = {np.mean(model_ibs):.5f}')
    print(f'[summary] mean IBS (B_1)   = {np.mean(b1_ibs):.5f}')
    print(f'[summary] Δ = {np.mean(model_ibs) - np.mean(b1_ibs):+.5f}  '
          f"({'model wins' if np.mean(model_ibs) < np.mean(b1_ibs) else 'B_1 wins'})")

    if args.output is not None:
        import json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            'n_evaluated': len(model_ibs),
            'n_skipped': skipped,
            'mean_ibs_model': float(np.mean(model_ibs)) if model_ibs else None,
            'mean_ibs_b1': float(np.mean(b1_ibs)) if b1_ibs else None,
            'delta_mean_ibs': float(np.mean(model_ibs) - np.mean(b1_ibs)) if model_ibs else None,
            'checkpoint': args.checkpoint,
            'val_jsonl': args.val_jsonl,
            'head_type': args.head_type,
            'limit': args.limit,
            'max_length': args.max_length,
            'dump_curves': bool(args.dump_curves),
            'generate_cot': bool(args.generate_cot),
            'cot_bypass': bool(args.cot_bypass),
        }
        if args.dump_curves:
            summary['B1_curve'] = B1_curve.astype(float).tolist()
        args.output.write_text(json.dumps({'summary': summary, 'per_ad': ad_records}, indent=2))
        print(f'[eval] wrote {args.output}')


if __name__ == '__main__':
    main()

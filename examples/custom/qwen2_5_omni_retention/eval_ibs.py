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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--plugin', required=True)
    ap.add_argument('--head-type', default='hazard', choices=['hazard', 'sigmoid'])
    ap.add_argument('--max-length', type=int, default=24576)
    ap.add_argument('--output', type=Path, default=None,
                    help='Optional: write per-ad results + summary as JSON (parent dirs auto-created).')
    args = ap.parse_args()

    # 1. Import plugin to register model_type / template / loss.
    import os
    os.environ['RETENTION_HEAD_TYPE'] = args.head_type
    import_plugin(args.plugin)

    # 2. Load val rows + compute B_1.
    val_rows = load_val_rows(args.val_jsonl, args.limit)
    print(f'[eval] loaded {len(val_rows)} val rows from {args.val_jsonl}')
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
        try:
            batch = template.data_collator([enc])
            batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
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
        ad_records.append({'idx': i, 'ad_id': row.get('ad_id'), 'T': T_i,
                           'ibs_model': ibs_model, 'ibs_b1': ibs_b1})
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
        }
        args.output.write_text(json.dumps({'summary': summary, 'per_ad': ad_records}, indent=2))
        print(f'[eval] wrote {args.output}')


if __name__ == '__main__':
    main()

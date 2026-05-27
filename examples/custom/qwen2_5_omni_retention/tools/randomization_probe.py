#!/usr/bin/env python
"""Randomization probe — post-launch sanity check that the model is
actually using the video signal, not echoing a leak.

WHY THIS EXISTS
---------------
V7's training data put R(t) directly in the assistant span. The
retention head reading h[last token] trivially echoed it, and three
independent audits confirmed V7's backbone did not use video at all.
For V8 the assistant span has been cleaned to <cot>...</cot> only, and
the head is anchored at h[</cot>] — but the only way to *verify* the
model genuinely conditions on video is to perturb the video and check
that the prediction changes.

WHAT THIS PROBE DOES
--------------------
For N ads:
  1. Forward the model normally on (video, text)               -> R_orig
  2. Forward with the video tensor zeroed out                    -> R_zero
  3. Forward with another ad's video swapped in                  -> R_swap
  4. Report:
        L2(R_orig, R_zero)        — distance to the "no video" prediction
        L2(R_orig, R_swap)        — distance when the video is wrong
        L2(R_orig, R_orig_ad_B)   — natural between-ad variance
        ratio  swap_dist / between_ad_dist
        ratio  zero_dist / between_ad_dist

EXPECTED FOR A WORKING MODEL
----------------------------
- zero_dist / between_ad_dist >> 0.5   (zeroing video changes the answer materially)
- swap_dist  / between_ad_dist >> 0.5   (swapping video changes the answer materially)

V7's failing values were both <0.1: the model produced essentially the
same curve regardless of input video.

Usage
-----
  python randomization_probe.py \\
      --ckpt /opt/dlami/nvme/.../checkpoint-75 \\
      --val-jsonl /path/to/val_200_no_cot.jsonl \\
      --n-ads 20

Set --n-ads 5 for a fast smoke (~1 min); 20+ for a real read.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


def _l2(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="path to checkpoint dir (must have config.json + tokenizer).")
    ap.add_argument("--val-jsonl", required=True, type=Path,
                    help="val jsonl with leak-free assistant spans.")
    ap.add_argument("--n-ads", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        return 2
    if not args.val_jsonl.exists():
        print(f"ERROR: val jsonl not found: {args.val_jsonl}", file=sys.stderr)
        return 2

    # Heavy imports.
    import torch
    import importlib.util
    plugin_path = Path(__file__).resolve().parents[1] / "register.py"
    spec = importlib.util.spec_from_file_location("retention_plugin", plugin_path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.template_inputs import TemplateInputs

    import os
    os.environ.setdefault("RETENTION_HEAD_TYPE", "hazard")
    os.environ.setdefault("ENABLE_AUDIO_OUTPUT", "False")

    print(f"loading model from {args.ckpt} ...")
    model, processor = get_model_processor(
        str(args.ckpt),
        model_type="qwen2_5_omni_retention",
        torch_dtype=torch.bfloat16,
    )
    model = model.to(args.device).eval()

    template = get_template(
        processor,
        template_type="qwen2_5_omni_retention",
        truncation_strategy="right",
        max_length=49152,
    )

    rows = []
    with open(args.val_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[: args.n_ads * 2]               # need extras for swap pairs
    if len(rows) < 2:
        print("ERROR: need >=2 val rows for swap pairs", file=sys.stderr)
        return 2

    def _predict(row):
        enc = template.encode(TemplateInputs.from_dict(row))
        batch = template.data_collator([enc])
        batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        return out.r_pred[0].float().cpu().numpy()

    def _zero_video(row):
        r2 = copy.deepcopy(row)
        # Mark video path empty — template falls back to no-video processing
        # (Qwen-Omni still produces a multimodal embedding sequence but the
        # vision tokens carry zero info). Different ms-swift versions handle
        # this slightly differently; the safer alternative is to swap to a
        # 1-second black-frame MP4 prebuilt for this probe.
        r2["videos"] = []
        r2["audios"] = []
        return r2

    rng = np.random.RandomState(0)
    swap_indices = rng.permutation(len(rows))

    zero_dists, swap_dists, between_ad_dists = [], [], []
    for i in range(args.n_ads):
        row_a = rows[i]
        row_b = rows[(i + 1) % len(rows)]
        ad_a_b_for_swap = rows[swap_indices[i]]

        R_a = _predict(row_a)
        R_b = _predict(row_b)
        between_ad_dists.append(_l2(R_a, R_b))

        R_a_zero = _predict(_zero_video(row_a))
        zero_dists.append(_l2(R_a, R_a_zero))

        row_a_swap = copy.deepcopy(row_a)
        row_a_swap["videos"] = ad_a_b_for_swap["videos"]
        row_a_swap["audios"] = ad_a_b_for_swap["audios"]
        R_a_swap = _predict(row_a_swap)
        swap_dists.append(_l2(R_a, R_a_swap))

        if i % 5 == 4:
            print(f"  probed {i+1}/{args.n_ads}")

    z = float(np.median(zero_dists))
    s = float(np.median(swap_dists))
    b = float(np.median(between_ad_dists))
    print()
    print(f"median L2(R_orig, R_zero_video)  = {z:.4f}")
    print(f"median L2(R_orig, R_swapped_video) = {s:.4f}")
    print(f"median L2(R_orig_adA, R_orig_adB)  = {b:.4f}    (between-ad baseline)")
    print(f"  ratio zero/between = {z/max(b,1e-9):.2f}")
    print(f"  ratio swap/between = {s/max(b,1e-9):.2f}")
    print()
    if z/max(b,1e-9) < 0.3 or s/max(b,1e-9) < 0.3:
        print("FAIL: model output barely changes when video is removed/swapped.")
        print("      This is V7's failure mode. Investigate before continuing training.")
        return 1
    print("PASS: model output depends on video content.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

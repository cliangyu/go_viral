#!/usr/bin/env python
# Copyright (c) Stanford TTCC. All rights reserved.
"""Build ms-swift-format JSONL for TTCC retention-curve training/eval.

Streams the public HF dataset ``liangyuch/ttcc-v0_2_0`` and writes one
JSONL row per ad with the schema ms-swift expects, plus the per-ad
ground-truth fields used by the retention head (R, T) and ttcc-eval.

Adapted from cliangyu/ttcc-inference's modal_infer.py:53-91 preprocessor.

Usage:
  python scripts/data/build_ttcc_jsonl.py \
      --split test \
      --out /home/ssm-user/work/data/ttcc_swift/ttcc_test.jsonl
  python scripts/data/build_ttcc_jsonl.py \
      --split train --with-cot \
      --out /home/ssm-user/work/data/ttcc_swift_v2cot/ttcc_train_sft.jsonl

Each output row is:
  {
    "messages": [
      {"role": "system",    "content": <system prompt>},
      {"role": "user",      "content": <user prompt referring to the ad>},
      {"role": "assistant", "content": <answer; with or without <cot>...</cot>>}
    ],
    "videos": [<path to extracted mp4>],
    "audios": [<path to extracted mp4 — Qwen-Omni reads audio from same mp4>],
    "ad_id": <str>,
    "T": <int>,          # T_i, the per-ad retention horizon
    "R": [1.0, R(1), ..., R(T)]   # ground-truth retention curve, length T+1
  }

The R field is what the retention plugin's template reads. The messages
field is what ms-swift's tokenizer/template consumes. The same JSONL is
usable for both language-emission (V1/V2) and retention-head (V3/V4)
variants — the head plugin extracts R only when present.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from datasets import load_dataset


SYSTEM_PROMPT = (
    'You are an ad-retention forecaster. Given a short video advertisement '
    'and its metadata, predict the per-second viewer retention curve R(t) '
    'for t = 0, 1, ..., T seconds.'
)

USER_PROMPT_NO_COT = (
    'Predict the per-second retention curve for this ad. T = {T}. '
    'Return JSON: {{"R": [1.0, R(1), ..., R({T})]}} where each R(t) is in [0, 1].'
)

USER_PROMPT_WITH_COT = (
    'Predict the per-second retention curve for this ad. T = {T}. '
    'First reason about hook, pacing, audio cues, brand visibility, '
    'and call-to-action timing inside <cot>...</cot>. '
    'Then return JSON: {{"R": [1.0, R(1), ..., R({T})]}} where each R(t) is in [0, 1].'
)


def compute_T(duration_seconds: float, curve_len: int, t_max: int = 60) -> int:
    """T_i = min(round(duration), 60, len(curve) - 1).

    Matches ttcc-inference/src/ttcc_inference/data.py and ttcc-eval's
    ground-truth convention.
    """
    return int(min(round(duration_seconds), t_max, curve_len - 1))


def normalize_curve(raw: list[float], T: int) -> list[float]:
    """Peak-normalize 0..100 -> 0..1, take the first T+1 values, force
    R(0)=1, enforce monotone non-increasing via running min, clip to [0,1].
    """
    if not raw:
        return [1.0] + [0.0] * T
    peak = max(raw[:T + 1] + [1.0])
    R = [v / peak for v in raw[:T + 1]]
    while len(R) < T + 1:
        R.append(R[-1] if R else 0.0)
    R[0] = 1.0
    for i in range(1, len(R)):
        R[i] = min(max(R[i], 0.0), 1.0)
        if R[i] > R[i - 1]:
            R[i] = R[i - 1]
    return R


def build_row(record, video_dir: Path, with_cot: bool) -> dict | None:
    """Produce one ms-swift-format row from a HF dataset record.

    Returns None if the row should be skipped (missing curve, T < 5, etc.).
    """
    ad_id = str(record.get('ad_id'))
    duration = float(record.get('duration', 0))
    raw_curve = record.get('retention_curve') or record.get('curve')
    if not raw_curve:
        return None
    T = compute_T(duration, len(raw_curve))
    if T < 5 or T > 60:
        return None
    R = normalize_curve([float(x) for x in raw_curve], T)

    video_bytes = record.get('video_bytes') or record.get('video')
    mp4_path = video_dir / f'{ad_id}.mp4'
    if video_bytes is not None and not mp4_path.exists():
        if isinstance(video_bytes, dict) and 'bytes' in video_bytes:
            video_bytes = video_bytes['bytes']
        with open(mp4_path, 'wb') as f:
            f.write(video_bytes)
    if not mp4_path.exists():
        return None

    user_prompt = (USER_PROMPT_WITH_COT if with_cot else USER_PROMPT_NO_COT).format(T=T)

    # Assistant turn: for SFT we need a target; for inference we leave empty.
    # When --with-cot is set, the assistant turn carries a CoT span before
    # the JSON. Otherwise the assistant turn is just the JSON.
    R_json = json.dumps({'R': R})
    if with_cot:
        # CoT body is filled in by a separate distillation step (we don't
        # generate CoT here). Default to a placeholder; the with-cot
        # JSONL is the target of CoT-distillation pipeline output.
        assistant = f'<cot>(reasoning here)</cot>\n{R_json}'
    else:
        assistant = R_json

    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
            {'role': 'assistant', 'content': assistant},
        ],
        'videos': [str(mp4_path)],
        'audios': [str(mp4_path)],     # Qwen-Omni reads audio stream from same mp4
        'ad_id': ad_id,
        'T': T,
        'R': R,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf-dataset', default='liangyuch/ttcc-v0_2_0')
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--out', required=True, help='output JSONL path')
    ap.add_argument('--video-dir', default=None,
                    help='dir for extracted mp4s (default: sibling "videos/" of --out)')
    ap.add_argument('--with-cot', action='store_true',
                    help='emit user prompt + assistant span with <cot> reasoning')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap rows for quick smoke runs')
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir or (out_path.parent / 'videos'))
    video_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.hf_dataset, split=args.split, streaming=True)

    n_written = 0
    n_skipped = 0
    with open(out_path, 'w') as f:
        for i, record in enumerate(ds):
            if args.limit and n_written >= args.limit:
                break
            row = build_row(record, video_dir, args.with_cot)
            if row is None:
                n_skipped += 1
                continue
            f.write(json.dumps(row) + '\n')
            n_written += 1
            if n_written % 500 == 0:
                print(f'  {n_written} written, {n_skipped} skipped')

    print(f'done: {n_written} rows -> {out_path}; {n_skipped} skipped')
    print(f'  videos in {video_dir}')


if __name__ == '__main__':
    main()

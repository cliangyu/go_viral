#!/usr/bin/env python
"""Build the leak-free V8 training (or val/test) JSONL directly from the
public HuggingFace datasets — no local V7 / CoT artifacts required.

Why this exists
---------------
The HF-native path. For teammates running on Modal / fresh machines that
don't have access to the AWS S3 mirror of `cot_v6_train.jsonl` and the V7
jsonls, this script reconstructs the V8 JSONL from primary sources:

  - `liangyuch/ttcc-v0_2_0`  : video bytes + retention curves + per-ad metadata
  - `liangyuch/ttcc-cot`     : Gemini-distilled `<cot>...</cot>` traces

The two datasets are joined on `ad_id`. The output JSONL matches the
schema produced by `build_v8_train_jsonl.py` exactly, so downstream code
doesn't have to care which path produced the data.

Anti-leak invariants enforced row-by-row (drops violators):
  I1   assistant.content is exactly the `<cot>...</cot>` span; nothing else
  I3   videos[0] == audios[0]
  I4   len(R) == T + 1  and  R[0] == 1.0
  I5   R is monotone non-increasing within 1e-6
  I10  no R-value decimal (rounded to 2/3/4 dp) appears inside the CoT

Usage
-----
    # Train split (leak-free, with CoT):
    python build_v8_from_hf.py \\
        --split train \\
        --video-dir /vol/data/videos \\
        --out-jsonl /vol/data/ttcc_v8/ttcc_train_with_cot.jsonl

    # Val/test splits — no CoT in assistant span (inference-time prompt only):
    python build_v8_from_hf.py \\
        --split val --no-cot \\
        --video-dir /vol/data/videos \\
        --out-jsonl /vol/data/ttcc_v8/val_200_no_cot.jsonl \\
        --limit 200

When --no-cot is set, the assistant content is empty (the retention head
will read h[last input token]; the model is asked to predict the curve
without any CoT supervision target).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert in short-form video advertising. You forecast "
    "second-by-second audience retention curves. R(t) is the fraction "
    "of viewers still watching at second t, with R(0) = 1 by convention."
)
USER_PROMPT = (
    "This ad is {T} seconds long. "
    "Estimate the per-second retention curve."
)


def _compute_T(duration: float, curve_len: int, t_max: int = 60) -> int:
    return int(min(round(duration), t_max, curve_len - 1))


def _normalize_curve(raw: list[float], T: int) -> list[float]:
    if not raw:
        return [1.0] + [0.0] * T
    peak = max(raw[: T + 1] + [1.0])
    R = [v / peak for v in raw[: T + 1]]
    while len(R) < T + 1:
        R.append(R[-1] if R else 0.0)
    R[0] = 1.0
    for i in range(1, len(R)):
        R[i] = min(max(R[i], 0.0), 1.0)
        if R[i] > R[i - 1]:
            R[i] = R[i - 1]
    return R


def _check_no_R_leak(cot_span: str, R: list[float]) -> str | None:
    for r in R[1:]:
        for dp in (2, 3, 4):
            s = f"{r:.{dp}f}"
            if s in cot_span and s not in (
                "0.00", "0.000", "0.0000",
                "1.00", "1.000", "1.0000",
            ):
                return f"R={s} (dp={dp}) appears in CoT"
    return None


def _check_R_well_formed(R: list[float], T: int) -> str | None:
    if len(R) != T + 1:
        return f"len(R)={len(R)} != T+1={T+1}"
    if abs(R[0] - 1.0) > 1e-6:
        return f"R[0]={R[0]} (expected 1.0)"
    for i in range(1, len(R)):
        if R[i] > R[i - 1] + 1e-6:
            return f"R not monotone at i={i}"
        if R[i] < -1e-6 or R[i] > 1.0 + 1e-6:
            return f"R[{i}]={R[i]} outside [0,1]"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--video-dir", required=True, type=Path,
                    help="dir containing extracted <ad_id>.mp4 files; "
                         "rows referencing missing MP4s are dropped.")
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--no-cot", action="store_true",
                    help="emit empty assistant span (for val/test or "
                         "no-CoT-supervision training).")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows emitted (smoke testing).")
    ap.add_argument("--ttcc-dataset", default="liangyuch/ttcc-v0_2_0")
    ap.add_argument("--cot-dataset",  default="liangyuch/ttcc-cot")
    ap.add_argument("--streaming", action="store_true", default=True,
                    help="stream from HF instead of loading to RAM "
                         "(default: True for memory safety on 39K-row train).")
    args = ap.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not args.video_dir.exists():
        print(f"ERROR: --video-dir does not exist: {args.video_dir}", file=sys.stderr)
        print("       Run extract_videos_from_hf.py first.", file=sys.stderr)
        return 2

    from datasets import load_dataset                            # heavy import

    # --- Step 1: pull the CoT dataset (only train has cot; val/test will be empty).
    cot_by_ad: dict[str, str] = {}
    if not args.no_cot:
        print(f"loading CoT dataset {args.cot_dataset} (train split) ...")
        cot_ds = load_dataset(args.cot_dataset, split="train")
        for r in cot_ds:
            ad_id = str(r.get("ad_id") or "")
            cot   = r.get("cot") or ""
            if ad_id and cot:
                cot_by_ad[ad_id] = cot
        print(f"  {len(cot_by_ad)} ad_ids with non-empty CoT")

    # --- Step 2: stream the TTCC dataset for the requested split.
    print(f"streaming {args.ttcc_dataset} (split={args.split}) ...")
    ttcc = load_dataset(args.ttcc_dataset, split=args.split,
                        streaming=args.streaming)

    drops = {"no_video_file": 0, "no_curve": 0, "no_cot": 0,
             "T_out_of_range": 0, "R_malformed": 0, "leak_check_failed": 0}
    n_written = 0

    with open(args.out_jsonl, "w") as fout:
        for record in ttcc:
            if args.limit and n_written >= args.limit:
                break

            ad_id    = str(record.get("ad_id") or "")
            duration = float(record.get("duration") or 0)
            curve    = record.get("retention_curve")
            if not curve:
                drops["no_curve"] += 1
                continue
            T = _compute_T(duration, len(curve))
            if T < 5 or T > 60:
                drops["T_out_of_range"] += 1
                continue
            R = _normalize_curve([float(x) for x in curve], T)
            R_err = _check_R_well_formed(R, T)
            if R_err:
                drops["R_malformed"] += 1
                continue

            mp4 = args.video_dir / f"{ad_id}.mp4"
            if not mp4.exists():
                drops["no_video_file"] += 1
                continue

            if args.no_cot:
                assistant_span = ""
            else:
                cot = cot_by_ad.get(ad_id)
                if not cot:
                    drops["no_cot"] += 1
                    continue
                # Ensure exactly one <cot>...</cot> wrap. The HF dataset's
                # `cot` field already has markers; normalize to be safe.
                text = cot.strip()
                for marker in ("<cot>", "</cot>", "<COT>", "</COT>"):
                    text = text.replace(marker, "")
                assistant_span = f"<cot>{text.strip()}</cot>"
                leak = _check_no_R_leak(assistant_span, R)
                if leak:
                    drops["leak_check_failed"] += 1
                    continue

            out_row = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": USER_PROMPT.format(T=T)},
                    {"role": "assistant", "content": assistant_span},
                ],
                "videos": [str(mp4)],
                "audios": [str(mp4)],
                "ad_id":  ad_id,
                "T":      T,
                "R":      R,
            }
            fout.write(json.dumps(out_row) + "\n")
            n_written += 1
            if n_written % 1000 == 0:
                print(f"  ... wrote {n_written}")

    print()
    print(f"wrote {n_written} rows to {args.out_jsonl}")
    print(f"  drops: {drops}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

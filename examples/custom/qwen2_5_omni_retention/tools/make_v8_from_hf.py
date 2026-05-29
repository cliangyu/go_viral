#!/usr/bin/env python
"""Rebuild the leak-free V8 training JSONL from the two PUBLIC HuggingFace repos.

VERIFIED 2026-05-29: on 150 ads present in both the trusted V8 set and the public
dataset, the (R, T, <cot> assistant) produced here matched the trusted
`ttcc_train_with_cot.jsonl` BYTE-FOR-BYTE (150/150, zero mismatches). This is the
script to hand a teammate who has HF access but no AWS — it is self-serve from:

  - `liangyuch/ttcc-v0_2_0`  : video bytes (`video_local_path`) + `retention_curve` + `duration`
  - `liangyuch/ttcc-cot`     : the `cot` field (Gemini-distilled `<cot>` reasoning, no R-leak)

joined on `ad_id`. Supersedes the BROKEN `build_v8_from_hf.py` (which read the wrong
video column and forced `audios=[mp4]`).

Exact transforms (each verified against the trusted V8 data):
  - T (horizon):   Td=round(duration); drop if Td<5 or Td>60; Tc=len(curve)-1;
                   drop if Td-Tc>1; T=min(Td,Tc); drop if T<5.
  - R (curve):     c[:T+1]/c[0]; drop if c[0]<=0 or non-finite; clamp monotone with a
                   5e-3 tolerance (drop if an increase exceeds 5e-3). R[0]==1 by construction.
  - assistant:     TRAIN -> "<cot>" + cot (existing markers stripped) + "</cot>";  drop if no cot.
                   VAL/TEST (--no-cot) -> "" (empty; head reads the last input token).
  - audios:        ALWAYS []  (V8 trains audio-OFF; turning it on re-triggers the FA3 bug).
  - prompts:       the SHORT system/user below (verbatim from the trusted V8 rows).

Anti-leak invariants (drops violators):
  I1  assistant is exactly "<cot>...</cot>" (train) or "" (val/test)
  I10 no R decimal (2/3/4 dp) appears inside the CoT
  I4  len(R)==T+1 and R[0]==1.0 ;  I5  R monotone non-increasing
  videos[0] basename == <ad_id>.mp4 ; audios == []

Usage
-----
    # 0) one-time: pull the parquet (train+val; you do NOT need test-*)
    hf download liangyuch/ttcc-v0_2_0 --repo-type dataset \
        --include 'data/train-*.parquet' --include 'data/val-*.parquet' --local-dir /vol/data/hf_ttcc

    # 1) train split (with CoT)
    python make_v8_from_hf.py --split train \
        --hf-ttcc-dir /vol/data/hf_ttcc --video-dir /vol/data/videos \
        --out-jsonl /vol/data/ttcc_v8/ttcc_train_with_cot.jsonl

    # 2) val split (no CoT). Omit --limit for the FULL val split.
    #    NOTE: --limit N takes the first N val ads in shard order; it does NOT reproduce the
    #    exact historical val_200_no_cot.jsonl ad_id set. For an apples-to-apples comparison
    #    against our 0.5142, filter to our 200 ad_ids (ask us for the allowlist) instead of --limit.
    python make_v8_from_hf.py --split val --no-cot \
        --hf-ttcc-dir /vol/data/hf_ttcc --video-dir /vol/data/videos \
        --out-jsonl /vol/data/ttcc_holdout/val_200_no_cot.jsonl

`--video-dir` is populated as a side effect (one `<ad_id>.mp4` per emitted row), so the
output `videos` paths point at real files. Point the SFT config's `dataset`/`val_dataset`
at these outputs.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# --- prompts: VERBATIM from the trusted V8 rows (the SHORT pair). Do not change. ---
SYSTEM_PROMPT = (
    "You are an expert in short-form video advertising. You forecast "
    "second-by-second audience retention curves. R(t) is the fraction of "
    "viewers still watching at second t, with R(0) = 1 by convention."
)


def user_prompt(T: int) -> str:
    return f"This ad is {T} seconds long. Estimate the per-second retention curve."


T_MIN, T_MAX = 5, 60


def horizon(duration, curve_len):
    """Verified == cot_distill/prepare_dataset. Returns T or None (drop)."""
    Td = round(float(duration))
    if Td < T_MIN or Td > T_MAX:
        return None
    Tc = curve_len - 1
    if Td - Tc > 1:
        return None
    T = min(Td, Tc)
    return T if T >= T_MIN else None


def normalize_curve(raw, T):
    """Verified c[0]-normalization + 5e-3 monotone tolerance. Returns list[float] or None."""
    c = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(c)) or c[0] <= 0:
        return None
    c = c[: T + 1] / c[0]
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            if c[i] - c[i - 1] > 5e-3:
                return None
            c[i] = c[i - 1]
    return np.clip(c, 0, 1).tolist()


def cot_to_assistant(cot_text: str) -> str:
    t = cot_text.strip()
    for m in ("<cot>", "</cot>", "<COT>", "</COT>"):
        t = t.replace(m, "")
    return f"<cot>{t.strip()}</cot>"


def has_R_leak(span: str, R) -> bool:
    for r in R[1:]:
        for dp in (2, 3, 4):
            s = f"{r:.{dp}f}"
            if s in span and s not in ("0.00", "0.000", "0.0000", "1.00", "1.000", "1.0000"):
                return True
    return False


def check_R(R, T) -> bool:
    if len(R) != T + 1 or abs(R[0] - 1.0) > 1e-6:
        return False
    for i in range(1, len(R)):
        if R[i] > R[i - 1] + 1e-6 or R[i] < -1e-6 or R[i] > 1.0 + 1e-6:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--hf-ttcc-dir", required=True, type=Path,
                    help="local dir from `hf download liangyuch/ttcc-v0_2_0 --local-dir`")
    ap.add_argument("--video-dir", required=True, type=Path,
                    help="output dir; <ad_id>.mp4 written here, referenced by the JSONL")
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--no-cot", action="store_true",
                    help="emit empty assistant (val/test).")
    ap.add_argument("--cot-dataset", default="liangyuch/ttcc-cot")
    ap.add_argument("--limit", type=int, default=None, help="cap emitted rows.")
    args = ap.parse_args()

    shards = sorted(glob.glob(str(args.hf_ttcc_dir / "data" / f"{args.split}-*-of-*.parquet")))
    if not shards:
        print(f"ERROR: no {args.split}-*.parquet under {args.hf_ttcc_dir/'data'}", file=sys.stderr)
        print("       run the `hf download` step in the docstring first.", file=sys.stderr)
        return 2
    args.video_dir.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    cot_by_ad: dict[str, str] = {}
    if not args.no_cot:
        from datasets import load_dataset
        print(f"loading {args.cot_dataset} (train) ...")
        for r in load_dataset(args.cot_dataset, split="train"):
            ad = str(r.get("ad_id") or "")
            c = r.get("cot") or ""
            if ad and c.strip():
                cot_by_ad[ad] = c
        print(f"  {len(cot_by_ad)} ad_ids with non-empty cot")

    cols = ["ad_id", "duration", "retention_curve", "split", "video_local_path"]
    drops = {"no_curve": 0, "T_oob": 0, "R_malformed": 0, "no_video": 0,
             "no_cot": 0, "leak": 0}
    n = 0
    with open(args.out_jsonl, "w") as fout:
        for shard in shards:
            t = pq.read_table(shard, columns=cols).to_pandas()
            t = t[t["split"] == args.split]
            for _, row in t.iterrows():
                if args.limit is not None and n >= args.limit:
                    break
                raw = row["retention_curve"]
                if raw is None or len(raw) == 0:
                    drops["no_curve"] += 1; continue
                T = horizon(row["duration"], len(raw))
                if T is None:
                    drops["T_oob"] += 1; continue
                R = normalize_curve(raw, T)
                if R is None or not check_R(R, T):
                    drops["R_malformed"] += 1; continue
                ad_id = str(row["ad_id"])
                v = row["video_local_path"]
                vb = v.get("bytes") if isinstance(v, dict) else None
                if not vb:
                    drops["no_video"] += 1; continue

                if args.no_cot:
                    assistant = ""
                else:
                    cot = cot_by_ad.get(ad_id)
                    if not cot:
                        drops["no_cot"] += 1; continue
                    assistant = cot_to_assistant(cot)
                    if has_R_leak(assistant, R):
                        drops["leak"] += 1; continue

                mp4 = args.video_dir / f"{ad_id}.mp4"
                if not mp4.exists() or mp4.stat().st_size != len(vb):   # re-write stale/partial mp4
                    mp4.write_bytes(bytes(vb))

                fout.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt(T)},
                        {"role": "assistant", "content": assistant},
                    ],
                    "videos": [str(mp4)],
                    "audios": [],
                    "ad_id": ad_id,
                    "T": T,
                    "R": R,
                }) + "\n")
                n += 1
                if n % 1000 == 0:
                    print(f"  ... wrote {n}")
            if args.limit is not None and n >= args.limit:
                break

    print(f"\nwrote {n} rows -> {args.out_jsonl}")
    print(f"  drops: {drops}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

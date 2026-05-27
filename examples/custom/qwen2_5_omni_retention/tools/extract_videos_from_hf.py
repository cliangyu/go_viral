#!/usr/bin/env python
"""Extract MP4 videos from the HuggingFace dataset `liangyuch/ttcc-v0_2_0`
(or any HF dataset with `ad_id` + `video_bytes` columns) into a local
directory, one file per ad named `<ad_id>.mp4`.

This is the deterministic video-staging step used by both the AWS H100
launch path and the Modal launch path (different `--out-dir`, same logic).

Usage
-----
    # Pull the parquet shards first (one-time):
    huggingface-cli download liangyuch/ttcc-v0_2_0 \\
        --repo-type dataset \\
        --local-dir /vol/data/hf_ttcc

    # Then extract MP4s:
    python extract_videos_from_hf.py \\
        --hf-dir /vol/data/hf_ttcc \\
        --out-dir /vol/data/videos \\
        --split train

  Or, in streaming mode (no local parquet, slower, no random access):
    python extract_videos_from_hf.py \\
        --hf-dataset liangyuch/ttcc-v0_2_0 \\
        --out-dir /vol/data/videos \\
        --split train --streaming

The script is idempotent: existing `<ad_id>.mp4` files are not
overwritten. Set --force to overwrite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path,
                    help="local dir containing the HF parquet shards "
                         "(from `huggingface-cli download ... --local-dir`).")
    ap.add_argument("--hf-dataset", default=None,
                    help="HF dataset id; used with --streaming when --hf-dir not provided.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="output directory; <ad_id>.mp4 files are written here.")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--streaming", action="store_true",
                    help="stream from HF instead of loading from --hf-dir.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows extracted (for smoke testing).")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing MP4s.")
    args = ap.parse_args()

    if not args.hf_dir and not args.hf_dataset:
        print("ERROR: must provide --hf-dir or --hf-dataset", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset                # heavy import, do it late
    if args.streaming and args.hf_dataset:
        ds = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    elif args.hf_dir:
        # The HF parquet snapshot lays out shards like:
        #   <hf-dir>/data/train-*-of-*.parquet
        # `load_dataset` auto-discovers this layout when given the local dir.
        ds = load_dataset(str(args.hf_dir), split=args.split)
    else:
        ds = load_dataset(args.hf_dataset, split=args.split)

    n_written = 0
    n_skipped_existing = 0
    n_skipped_no_bytes = 0

    for i, record in enumerate(ds):
        if args.limit and n_written >= args.limit:
            break
        ad_id = str(record.get("ad_id") or "")
        if not ad_id:
            continue

        out_path = args.out_dir / f"{ad_id}.mp4"
        if out_path.exists() and not args.force:
            n_skipped_existing += 1
            continue

        # video_bytes may be a raw bytes blob or a {"bytes": <bytes>} dict
        # depending on the dataset version.
        video_bytes = record.get("video_bytes") or record.get("video")
        if isinstance(video_bytes, dict) and "bytes" in video_bytes:
            video_bytes = video_bytes["bytes"]
        if not video_bytes:
            n_skipped_no_bytes += 1
            continue

        with open(out_path, "wb") as f:
            f.write(video_bytes)
        n_written += 1
        if n_written % 500 == 0:
            print(f"  ... {n_written} extracted, {n_skipped_existing} existing, {n_skipped_no_bytes} missing-bytes")

    print()
    print(f"wrote      : {n_written} MP4s -> {args.out_dir}")
    print(f"skipped existing : {n_skipped_existing}")
    print(f"skipped no-bytes : {n_skipped_no_bytes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Prepare TTCC dataset for ms-swift training.

This is a thin wrapper over ttcc-eval's preprocess pipeline so our training
data and ttcc-eval's test data use *byte-identical* preprocessing rules.
Doing it any other way risks training on ground truth that's been munged
differently from what eval expects (different T_i, different normalization,
different drop list).

Pipeline:
  1. ttcc_eval.data.load_ground_truth  → raw (ad_id, duration, retention_curve, split)
  2. ttcc_eval.preprocess.preprocess   → CleanGroundTruth: T_i, peak-normed curve, R(0)=1
  3. Split by the dataset's `split` column ("train"/"val"/"test"), NOT by our own shuffle.
  4. For each kept ad: pull mp4 bytes from the same parquet (second pass), write to disk.
  5. ffmpeg-extract 16 kHz mono audio.
  6. Emit ms-swift JSONL with _meta.{duration_s, retention_curve} = the clean values.

Audio cap policy: training pipeline filters to T_i <= 30 (Whisper truncates
audio past that). Eval pipeline keeps the full T_i <= 60 — Liangyu's inference
will produce R_hat for the same set ttcc-eval considers, and our train/val/test
JSONL writes carry the full clean curve. We just downselect training rows.

Run:
  python cs224r_project/data/prep_ttcc.py --out_dir cs224r_project/data
  python cs224r_project/data/prep_ttcc.py --train_max_duration 60   # if you want to keep long ads in train
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tqdm import tqdm


USER_PROMPT = (
    "<video><audio>Predict the per-second retention curve. "
    "Reason about hook, pacing, and audio cues. "
    "Wrap reasoning in <cot>...</cot>."
)

_DEFAULT_REPO_ID = "liangyuch/ttcc-v0_1_0"
_DEFAULT_N_SHARDS = 17


def extract_audio(mp4_path: str, wav_path: str) -> bool:
    if os.path.exists(wav_path):
        return True
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", mp4_path,
        "-ar", "16000", "-ac", "1",
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def write_video_bytes(video_struct, dst_mp4: Path) -> bool:
    """The HF Video column stores {'bytes': ..., 'path': ...}. Prefer bytes
    (works on Modal where the path is inside the shard). Falls back to path."""
    if video_struct is None:
        return False
    if isinstance(video_struct, dict):
        b = video_struct.get("bytes")
        if b is not None:
            dst_mp4.write_bytes(bytes(b))
            return True
        p = video_struct.get("path")
        if p and os.path.exists(p):
            try:
                os.symlink(p, dst_mp4)
            except OSError:
                shutil.copy2(p, dst_mp4)
            return True
    elif hasattr(video_struct, "path") and os.path.exists(video_struct.path):
        try:
            os.symlink(video_struct.path, dst_mp4)
        except OSError:
            shutil.copy2(video_struct.path, dst_mp4)
        return True
    elif isinstance(video_struct, str) and os.path.exists(video_struct):
        try:
            os.symlink(video_struct, dst_mp4)
        except OSError:
            shutil.copy2(video_struct, dst_mp4)
        return True
    return False


def make_jsonl_row(ad_id: str, video_path: str, audio_path: str,
                   duration_s: int, retention_curve: list, cot: str = "") -> dict:
    return {
        "messages": [
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": f"<cot>{cot}</cot>"},
        ],
        "videos": [video_path],
        "audios": [audio_path],
        "_meta": {
            "ad_id": ad_id,
            "duration_s": int(duration_s),
            "retention_curve": [float(x) for x in retention_curve],
        },
    }


def _read_video_bytes_for_ads(
    repo_id: str,
    n_shards: int,
    wanted_ad_ids: set[str],
    out_video_dir: Path,
) -> dict[str, Path]:
    """Second pass through the HF parquet shards to materialize mp4 files.

    Pulls only the {ad_id, video_local_path} columns to keep memory bounded.
    Returns ad_id → local mp4 path.
    """
    fs = HfFileSystem()
    written: dict[str, Path] = {}
    cols = ["ad_id", "video_local_path"]
    for shard in range(n_shards):
        path = f"datasets/{repo_id}/data/train-{shard:05d}-of-{n_shards:05d}.parquet"
        try:
            with fs.open(path, "rb") as f:
                t = pq.read_table(f, columns=cols)
        except FileNotFoundError:
            # Smaller datasets (e.g. smoke set) may have fewer shards; stop gracefully.
            break
        df = t.to_pandas()
        for _, row in df.iterrows():
            ad_id = str(row["ad_id"])
            if ad_id not in wanted_ad_ids or ad_id in written:
                continue
            dst = out_video_dir / f"{ad_id}.mp4"
            if write_video_bytes(row["video_local_path"], dst):
                written[ad_id] = dst
        if len(written) >= len(wanted_ad_ids):
            break
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", default=_DEFAULT_REPO_ID)
    ap.add_argument("--n_shards", type=int, default=_DEFAULT_N_SHARDS)
    ap.add_argument("--out_dir", default="cs224r_project/data")
    ap.add_argument("--train_max_duration", type=int, default=60,
                    help="Cap training T_i; val/test always keep full clean T_i. "
                         "Set to 30 for the strict Whisper audio cap, 60 to match eval.")
    args = ap.parse_args()

    # Local import so the script's --help works even without ttcc-eval installed,
    # and the error message is clear when it's missing.
    try:
        from ttcc_eval.data import load_ground_truth
        from ttcc_eval.preprocess import preprocess
    except ImportError as e:
        raise SystemExit(
            "ttcc-eval is required. Install it with:\n"
            "  pip install 'ttcc-eval @ git+https://github.com/cliangyu/ttcc-eval.git@main'\n"
            f"original error: {e}"
        ) from e

    out_dir = Path(args.out_dir).resolve()
    raw_cache_dir = out_dir / "raw"
    audio_dir = out_dir / "audios"
    video_dir = out_dir / "videos"
    split_dir = out_dir / "splits"
    for d in (raw_cache_dir, audio_dir, video_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.hf_dataset} via ttcc-eval")
    gt = load_ground_truth(
        cache_path=raw_cache_dir / "ttcc_v0_1_0_curves.parquet",
        repo_id=args.hf_dataset,
        n_shards=args.n_shards,
    )
    print(f"  raw rows: {len(gt)}")

    print("Running ttcc-eval preprocess (peak-normalize, monotone smooth, T_i cap)")
    clean, report = preprocess(gt)
    print(f"  kept: {report.n_kept}/{report.n_input}; "
          f"smoothed: {report.n_monotonicity_smoothed}; "
          f"dropped: {report.dropped}")

    # Build {ad_id: (T_i, curve, split)} for every clean ad.
    clean_lookup: dict[str, tuple[int, list[float], str]] = {}
    for i in range(len(clean.ad_id)):
        ad_id = str(clean.ad_id[i])
        clean_lookup[ad_id] = (
            int(clean.T[i]),
            [float(x) for x in clean.curves[i]],
            str(clean.split[i]),
        )

    # Materialize mp4s for every clean ad (training filter applies later, but
    # eval needs all of val/test too).
    print(f"Pulling mp4 bytes for {len(clean_lookup)} clean ads")
    ad_to_video = _read_video_bytes_for_ads(
        repo_id=args.hf_dataset,
        n_shards=args.n_shards,
        wanted_ad_ids=set(clean_lookup.keys()),
        out_video_dir=video_dir,
    )
    print(f"  wrote {len(ad_to_video)} mp4 files")

    # Extract audio for every kept ad.
    print("Extracting audio (ffmpeg)")
    ad_to_audio: dict[str, Path] = {}
    audio_fail = 0
    for ad_id, mp4_path in tqdm(ad_to_video.items(), desc="ffmpeg"):
        wav_path = audio_dir / f"{ad_id}.wav"
        if extract_audio(str(mp4_path), str(wav_path)):
            ad_to_audio[ad_id] = wav_path
        else:
            audio_fail += 1
    if audio_fail:
        print(f"  audio_fail: {audio_fail}")

    # Write JSONL splits using the dataset's own split column.
    counts: dict[str, dict[str, int]] = {
        s: {"kept": 0, "no_video": 0, "no_audio": 0, "filtered_long": 0}
        for s in ("train", "val", "test")
    }
    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for ad_id, (T_i, curve, split) in clean_lookup.items():
        if split not in rows_by_split:
            continue
        if ad_id not in ad_to_video:
            counts[split]["no_video"] += 1
            continue
        if ad_id not in ad_to_audio:
            counts[split]["no_audio"] += 1
            continue
        # Training-only filter (val/test pass through unchanged).
        if split == "train" and T_i > args.train_max_duration:
            counts[split]["filtered_long"] += 1
            continue
        rows_by_split[split].append(make_jsonl_row(
            ad_id=ad_id,
            video_path=str(ad_to_video[ad_id]),
            audio_path=str(ad_to_audio[ad_id]),
            duration_s=T_i,
            retention_curve=curve,
        ))
        counts[split]["kept"] += 1

    for split, rows in rows_by_split.items():
        path = split_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {path}: {len(rows)} rows")

    manifest = {
        "hf_dataset": args.hf_dataset,
        "split_source": "dataset_column",
        "train_max_duration": args.train_max_duration,
        "preprocess_report": report.to_dict(),
        "counts": counts,
        "ad_ids": {
            split: [r["_meta"]["ad_id"] for r in rows]
            for split, rows in rows_by_split.items()
        },
    }
    with open(split_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {split_dir / 'split_manifest.json'}")

    if counts["test"]["kept"] == 0:
        print("WARNING: no test rows kept. The smoke set may not have a test "
              "split — check split_manifest.json:preprocess_report.")


if __name__ == "__main__":
    main()

"""prepare_dataset.py — Build SFT/GRPO JSONL from ttcc-v0_2_0 parquet shards.

Single source of truth for TTCC training data preparation. Supports two modes:
- CoT mode (default): SFT assistant target = Content/Drops/Reasoning + Curve.
  Requires --cot-jsonl with distillation outputs; only rows with CoT seeds enter SFT.
- no-CoT mode (--no-cot): SFT assistant target = Curve only.
  Does not require CoT distillation; all rows passing quality filters enter SFT.

Validation (default on, --no-validate to skip):
- Probes each mp4 with ffprobe to record video/audio stream presence + duration.
- Sets `audios: [mp4]` only if audio stream exists, else `audios: []`
  (the canonical Qwen2.5-Omni pattern per HF docs).
- Drops rows with corrupt mp4 or no video stream.

Outputs:
- ttcc_train_sft.jsonl   — SFT training rows
- ttcc_train_grpo.jsonl  — GRPO seed rows (no assistant turn)
- ttcc_val.jsonl         — val split (read from val-*.parquet)
- ttcc_test.jsonl        — test split (read from test-*.parquet)
- prep_stats.json        — counts + drop reasons + audio-coverage %

Usage:
    # CoT mode (original behavior)
    python prepare_dataset.py --cot-jsonl path/to/cot.jsonl \
        --out-dir /home/ssm-user/work/data/ttcc_swift_v2cot

    # no-CoT mode (curve-only target, no CoT distill needed)
    python prepare_dataset.py --no-cot \
        --out-dir /home/ssm-user/work/data/ttcc_swift_v2cot_nocot
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq

WORK = Path("/home/ssm-user/work")
SYSTEM_PROMPT = (
    "You are an expert in short-form video advertising. You forecast "
    "second-by-second audience retention curves. R(t) is the fraction of "
    "viewers still watching at second t, with R(0) = 1 by definition. "
    "R(t) is monotone non-increasing. Use the video and audio content to "
    "estimate where viewers drop off."
)
T_MIN, T_MAX = 5, 60


def user_text(T: int) -> str:
    return (
        f"This ad is {T} seconds long. Watch and listen to it, then write your "
        f"analysis on three labeled lines and finish with the JSON curve.\n"
        f"Content: <one sentence describing the ad>.\n"
        f"Drops: <one or two sentences naming SPECIFIC seconds where retention "
        f"falls fastest, with reasons tied to what happens on screen or audio>.\n"
        f"Reasoning: <one sentence summarizing the overall shape>.\n"
        f"Curve: {{\"R\": [1.0, R(1), R(2), ..., R({T})]}}\n"
        f"Rules: the Curve line MUST be a valid JSON object exactly of the form "
        f"{{\"R\": [...]}}, exactly {T+1} numbers in R, R(0) = 1.0, every value "
        f"in [0, 1], monotone non-increasing."
    )


def horizon(d, L):
    """Return (T, drop_reason). Rejects videos shorter than T_MIN or longer than T_MAX."""
    Td = round(float(d))
    if Td < T_MIN:
        return None, "too_short"
    if Td > T_MAX:
        return None, "too_long"
    Tc = L - 1
    if Td - Tc > 1:
        return None, "length_mismatch"
    T = min(Td, Tc)
    if T < T_MIN:
        return None, "clamped_too_short"
    return T, None


# ---- validation -----------------------------------------------------------


def _probe_streams(mp4: str) -> dict:
    """One ffprobe call: returns has_video, has_audio, video_codec, audio_codec, duration."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,codec_name:format=duration",
             "-of", "json", mp4],
            capture_output=True, text=True, timeout=10,
        )
        info = json.loads(r.stdout)
    except Exception as e:
        return {"path": mp4, "ok": False, "error": f"ffprobe_fail:{e}"}
    streams = info.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "path": mp4,
        "ok": bool(vstream),
        "has_video": bool(vstream),
        "has_audio": bool(astream),
        "video_codec": vstream.get("codec_name") if vstream else None,
        "audio_codec": astream.get("codec_name") if astream else None,
        "duration_s": float(info.get("format", {}).get("duration", 0)) or None,
        "error": None if vstream else "no_video_stream",
    }


def validate_mp4s(paths: list[str], workers: int = 48) -> dict[str, dict]:
    """Parallel ffprobe validation; returns {path: probe_result}."""
    with mp.Pool(workers) as p:
        results = list(p.imap_unordered(_probe_streams, paths, chunksize=20))
    return {r["path"]: r for r in results}


# ---- dataset construction -------------------------------------------------


def extract_split(
    split_name: str,
    data_root: Path,
    videos_dir: Path,
    shard_glob: str,
) -> tuple[list[dict], Counter]:
    """Read parquet shards matching shard_glob, filter by split + curve quality,
    extract mp4 bytes. Returns (rows, drops_counter)."""
    videos_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    drops: Counter = Counter()
    cols = ["ad_id", "duration", "retention_curve", "split", "video_local_path"]
    for shard in sorted((data_root / "data").glob(shard_glob)):
        t = pq.read_table(shard, columns=cols).to_pandas()
        t = t[t["split"] == split_name]
        for _, row in t.iterrows():
            raw = row["retention_curve"]
            if raw is None or len(raw) == 0:
                drops["null_curve"] += 1
                continue
            c = np.asarray(raw, dtype=np.float64)
            if not np.all(np.isfinite(c)) or c[0] <= 0:
                drops["bad_curve"] += 1
                continue
            T, dr = horizon(row["duration"], len(c))
            if T is None:
                drops[dr] += 1
                continue
            c = c[:T + 1] / c[0]
            ok = True
            for i in range(1, len(c)):
                if c[i] > c[i - 1]:
                    if c[i] - c[i - 1] > 5e-3:
                        ok = False
                        break
                    c[i] = c[i - 1]
            if not ok:
                drops["nonmonotone"] += 1
                continue
            ad_id = str(row["ad_id"])
            v = row["video_local_path"]
            if v is None or v.get("bytes") is None:
                drops["no_video_bytes"] += 1
                continue
            mp4 = videos_dir / f"{ad_id}.mp4"
            if not mp4.exists():
                mp4.write_bytes(bytes(v["bytes"]))
            rows.append({
                "ad_id": ad_id, "T": int(T),
                "R": np.clip(c, 0, 1).tolist(),
                "mp4": str(mp4),
            })
    return rows, drops


def build_row(r: dict, with_assistant: bool, cot: dict | None,
              probe: dict | None) -> dict:
    """Assemble a JSONL row.
    - with_assistant=False: GRPO seed (no assistant turn).
    - with_assistant=True + cot: SFT row with Content/Drops/Reasoning + Curve.
    - with_assistant=True + cot=None: SFT row with Curve-only target (no-CoT mode).
    - probe controls audios field: if probe says no audio → audios=[].
    """
    T, R, mp4 = r["T"], r["R"], r["mp4"]
    has_audio = (probe is None) or probe.get("has_audio", True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text(T)},
    ]
    if with_assistant:
        R_str = "[" + ", ".join(f"{x:.4f}" for x in R) + "]"
        if cot is not None and "raw" in cot:
            assistant = cot["raw"].strip() + f"\nCurve: {{\"R\": {R_str}}}"
        else:
            assistant = f"Curve: {{\"R\": {R_str}}}"
        messages.append({"role": "assistant", "content": assistant})
    return {
        "ad_id": r["ad_id"], "messages": messages,
        "videos": [mp4],
        "audios": [mp4] if has_audio else [],
        "T": T, "R_true": R,
    }


def write_jsonl(rows: list[dict], out: Path) -> int:
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


# ---- main -----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Build TTCC training JSONLs.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=WORK / "data/ttcc")
    ap.add_argument("--no-cot", action="store_true",
                    help="Skip CoT requirement; assistant target is Curve only.")
    ap.add_argument("--cot-jsonl", type=Path, default=None,
                    help="Path to cot_distill_thinking.jsonl (required unless --no-cot).")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip ffprobe validation (default: validate ON).")
    ap.add_argument("--validate-workers", type=int, default=48)
    args = ap.parse_args()

    if not args.no_cot and args.cot_jsonl is None:
        ap.error("--cot-jsonl is required unless --no-cot is set")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_root = WORK / "data/videos"

    # Load CoT seeds (CoT mode only)
    cots: dict[str, dict] = {}
    if not args.no_cot:
        with open(args.cot_jsonl) as f:
            for line in f:
                row = json.loads(line)
                cots[row["ad_id"]] = row
        print(f"[CoT mode] loaded {len(cots)} CoT seeds", flush=True)
    else:
        print("[no-CoT mode] assistant target = Curve only; all valid rows enter SFT",
              flush=True)

    stats: dict[str, Any] = {"mode": "no-cot" if args.no_cot else "cot", "splits": {}}

    for split, shard_glob, subdir in [
        ("train", "train-*-of-*.parquet", "train"),
        ("val",   "val-*-of-*.parquet",   ""),
        ("test",  "test-*-of-*.parquet",  ""),
    ]:
        sub = videos_root / subdir if subdir else videos_root
        rows, drops = extract_split(split, args.data_root, sub, shard_glob)
        print(f"[{split}] extracted {len(rows)} rows  drops={dict(drops)}", flush=True)

        # Validate each mp4 (default on)
        probes: dict[str, dict] = {}
        if not args.no_validate:
            paths = [r["mp4"] for r in rows]
            print(f"[{split}] validating {len(paths)} mp4s with ffprobe "
                  f"({args.validate_workers} workers)...", flush=True)
            probes = validate_mp4s(paths, workers=args.validate_workers)
            n_no_video = sum(1 for p in probes.values() if not p.get("has_video"))
            n_no_audio = sum(1 for p in probes.values() if not p.get("has_audio"))
            n_ok = sum(1 for p in probes.values() if p.get("ok"))
            print(f"[{split}] validation: {n_ok}/{len(paths)} ok, "
                  f"{n_no_video} no_video, {n_no_audio} no_audio", flush=True)
            # Drop rows that failed video validation
            rows = [r for r in rows if probes.get(r["mp4"], {}).get("has_video")]
            drops["no_video_stream"] = n_no_video

        # Train split: write both SFT and GRPO
        if split == "train":
            n_grpo = write_jsonl(
                [build_row(r, with_assistant=False, cot=None,
                           probe=probes.get(r["mp4"])) for r in rows],
                args.out_dir / "ttcc_train_grpo.jsonl",
            )
            sft_rows = []
            for r in rows:
                cot = cots.get(r["ad_id"])
                if not args.no_cot and cot is None:
                    continue  # CoT mode: skip rows without CoT seed
                sft_rows.append(build_row(r, with_assistant=True, cot=cot,
                                          probe=probes.get(r["mp4"])))
            n_sft = write_jsonl(sft_rows, args.out_dir / "ttcc_train_sft.jsonl")
            print(f"[train] wrote GRPO: {n_grpo}, SFT: {n_sft}", flush=True)
            stats["splits"]["train"] = {
                "extracted": len(rows), "sft": n_sft, "grpo": n_grpo,
                "drops": dict(drops),
                "audio_coverage_pct": (100 * sum(1 for r in rows if probes.get(r["mp4"], {}).get("has_audio", True)) / len(rows)) if rows else None,
            }
        else:
            n = write_jsonl(
                [build_row(r, with_assistant=True, cot=cots.get(r["ad_id"]),
                           probe=probes.get(r["mp4"])) for r in rows],
                args.out_dir / f"ttcc_{split}.jsonl",
            )
            print(f"[{split}] wrote {n} rows", flush=True)
            stats["splits"][split] = {
                "rows": n, "drops": dict(drops),
                "audio_coverage_pct": (100 * sum(1 for r in rows if probes.get(r["mp4"], {}).get("has_audio", True)) / len(rows)) if rows else None,
            }

    stats_path = args.out_dir / "prep_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[stats] -> {stats_path}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

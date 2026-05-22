"""Convert TTCC train split + CoT distillation JSONL into ms-swift's
multimodal training format.

Output: JSONL with one row per train ad. Each row has the ms-swift fields:
    {
      "messages": [
        {"role": "system", "content": <system prompt>},
        {"role": "user",   "content": "<video><audio>\\nThis ad is N seconds...
                                         output a JSON object {\"R\": [...]}"},
      ],
      "videos":   [<absolute mp4 path>],
      "audios":   [<absolute mp4 path>],  # same file; Qwen-Omni handles
      "T":        int,
      "R_true":   list[float],  # length T+1; the GT curve for the IBS reward
      "cot_seed": "<Content + Drops + Reasoning from teacher>"   # used only for SFT (--dataset_columns)
    }

Two outputs:
  - <out>/ttcc_train_grpo.jsonl  : for GRPO (R_true is the reward signal)
  - <out>/ttcc_train_sft.jsonl   : same rows but with an assistant message
                                    pre-filled with cot_seed + the R curve,
                                    suitable for SFT seeding before GRPO.

Usage:
    python prepare_dataset.py \\
        --cot-jsonl /home/ssm-user/work/work-out/cot_distill_thinking.jsonl \\
        --out-dir /home/ssm-user/work/data/ttcc_swift
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load CoT distillation outputs
    cots: dict[str, dict] = {}
    with open(args.cot_jsonl) as f:
        for line in f:
            row = json.loads(line)
            cots[row["ad_id"]] = row
    print(f"CoTs loaded: {len(cots)}")

    # Iterate train manifests
    T_MIN, T_MAX = 5, 60
    def horizon(d, L):
        Td = round(float(d))
        if Td < T_MIN: return None
        Tc = L - 1
        if min(Td, T_MAX) - Tc > 1: return None
        T = min(Td, T_MAX, Tc)
        return T if T >= T_MIN else None

    DATA = WORK / "data/ttcc"
    VIDEOS = WORK / "data/videos/train"
    VIDEOS.mkdir(parents=True, exist_ok=True)

    train_rows = []
    for shard in sorted((DATA / "data").glob("train-*-of-*.parquet")):
        t = pq.read_table(
            shard,
            columns=["ad_id", "duration", "retention_curve", "split", "video_local_path"],
        ).to_pandas()
        t = t[t["split"] == "train"]
        for _, row in t.iterrows():
            raw = row["retention_curve"]
            if raw is None or len(raw) == 0:
                continue
            c = np.asarray(raw, dtype=np.float64)
            if not np.all(np.isfinite(c)) or c[0] <= 0:
                continue
            T = horizon(row["duration"], len(c))
            if T is None:
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
                continue
            ad_id = str(row["ad_id"])
            v = row["video_local_path"]
            if v is None or v.get("bytes") is None:
                continue
            mp4 = VIDEOS / f"{ad_id}.mp4"
            if not mp4.exists():
                mp4.write_bytes(bytes(v["bytes"]))
            train_rows.append(
                {"ad_id": ad_id, "T": T, "R": np.clip(c, 0, 1).tolist(), "mp4": str(mp4)}
            )
    print(f"train rows after preprocess: {len(train_rows)}")
    print(f"with CoT seeds: {sum(1 for r in train_rows if r['ad_id'] in cots)}")

    out_grpo = args.out_dir / "ttcc_train_grpo.jsonl"
    out_sft = args.out_dir / "ttcc_train_sft.jsonl"
    n_grpo = n_sft = 0
    with open(out_grpo, "w") as fg, open(out_sft, "w") as fs:
        for r in train_rows:
            ad = r["ad_id"]
            T, R, mp4 = r["T"], r["R"], r["mp4"]
            user_msg = user_text(T)
            base = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "videos": [mp4],
                "audios": [mp4],
                "T": T,
                "R_true": R,
            }
            fg.write(json.dumps(base) + "\n")
            n_grpo += 1

            cot = cots.get(ad)
            if cot is not None and "raw" in cot:
                # Assemble assistant target: CoT raw + JSON curve appended.
                R_str = "[" + ", ".join(f"{x:.4f}" for x in R) + "]"
                assistant = cot["raw"].strip() + f"\nCurve: {{\"R\": {R_str}}}"
                sft = dict(base)
                sft["messages"] = base["messages"] + [
                    {"role": "assistant", "content": assistant}
                ]
                fs.write(json.dumps(sft) + "\n")
                n_sft += 1

    print(f"wrote GRPO dataset: {out_grpo}  ({n_grpo} rows)")
    print(f"wrote SFT  dataset: {out_sft}   ({n_sft} rows)")

    # Build a JSONL for any held-out split (val + test). Same filters as train.
    DATA = WORK / "data/ttcc"
    VIDEOS_HOLDOUT = WORK / "data/videos"

    def build_holdout_jsonl(split_name: str, out_path):
        rows = []
        for shard in sorted((DATA / "data").glob("train-*-of-*.parquet")):
            t = pq.read_table(
                shard,
                columns=["ad_id", "duration", "retention_curve", "split", "video_local_path"],
            ).to_pandas()
            t = t[t["split"] == split_name]
            for _, row in t.iterrows():
                raw = row["retention_curve"]
                if raw is None or len(raw) == 0:
                    continue
                c = np.asarray(raw, dtype=np.float64)
                if not np.all(np.isfinite(c)) or c[0] <= 0:
                    continue
                T = horizon(row["duration"], len(c))
                if T is None:
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
                    continue
                ad_id = str(row["ad_id"])
                mp4 = VIDEOS_HOLDOUT / f"{ad_id}.mp4"
                if not mp4.exists():
                    v = row["video_local_path"]
                    if v is not None and v.get("bytes") is not None:
                        mp4.write_bytes(bytes(v["bytes"]))
                rows.append(
                    {"ad_id": ad_id, "T": T, "R": np.clip(c, 0, 1).tolist(), "mp4": str(mp4)}
                )

        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps({
                    "ad_id": r["ad_id"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text(r["T"])},
                    ],
                    "videos": [r["mp4"]],
                    "audios": [r["mp4"]],
                    "T": r["T"],
                    "R_true": r["R"],
                }) + "\n")
        return len(rows)

    # val: model-selection split (~104 raw ads pre-filter)
    out_val = args.out_dir / "ttcc_val.jsonl"
    n_val = build_holdout_jsonl("val", out_val)
    print(f"wrote VAL  dataset: {out_val}    ({n_val} rows)")

    # test: final evaluation split; do NOT touch during training
    out_test = args.out_dir / "ttcc_test.jsonl"
    n_test = build_holdout_jsonl("test", out_test)
    print(f"wrote TEST dataset: {out_test}   ({n_test} rows)")


if __name__ == "__main__":
    main()

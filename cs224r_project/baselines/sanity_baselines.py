"""
Plumbing-level sanity baselines (B0, B1, B2) per ttcc-eval docs/05_roadmap.md.

  B0_constant_05    R̂(t) = 0.5  ∀ i, t        (catches scale/length bugs)
  B1_mean_train_curve   R̂(t) = mean over train ads of R(t)  (climatology)
  B2_uniform_decay  R̂(t) = max(0, 1 − t / T_i)             (per-ad linear)

Forces R̂(0) = 1.0 as required by the predictions contract.

Run:
  python sanity_baselines.py \\
      --train_jsonl /vol/data/splits/train.jsonl \\
      --test_jsonl  /vol/data/splits/test.jsonl \\
      --out_dir     /vol/runs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

T_MAX = 60


def load_test(test_jsonl: Path) -> list[tuple[str, int]]:
    out = []
    with open(test_jsonl) as f:
        for line in f:
            row = json.loads(line)
            out.append((str(row["_meta"]["ad_id"]), int(row["_meta"]["duration_s"])))
    return out


def per_second_mean(train_jsonl: Path, T_max: int = T_MAX) -> np.ndarray:
    """R̄_train(t) = mean over train ads with T >= t of R(t)."""
    sums = np.zeros(T_max + 1, dtype=np.float64)
    counts = np.zeros(T_max + 1, dtype=np.int64)
    with open(train_jsonl) as f:
        for line in f:
            row = json.loads(line)
            curve = row["_meta"]["retention_curve"]
            T_i = int(row["_meta"]["duration_s"])
            for t in range(min(T_i + 1, T_max + 1)):
                sums[t] += float(curve[t])
                counts[t] += 1
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    mean[0] = 1.0
    # Enforce monotone non-increasing.
    mean = np.minimum.accumulate(mean)
    return mean


def write_submission(rows: list[dict], out_path: Path, method: str, seed: int = 0):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "ad_id":  pa.array([r["ad_id"] for r in rows], type=pa.string()),
        "R_hat":  pa.array([r["R_hat"] for r in rows], type=pa.list_(pa.float64())),
        "method": pa.array([method] * len(rows), type=pa.string()),
        "seed":   pa.array([seed]   * len(rows), type=pa.int64()),
    })
    pq.write_table(table, out_path)
    print(f"wrote {out_path}: {len(rows)} rows")


def make_b0(test_ads: list[tuple[str, int]]) -> list[dict]:
    """R̂_i(t) = 0.5 for t >= 1, R̂_i(0) = 1.0."""
    rows = []
    for ad_id, T_i in test_ads:
        curve = [1.0] + [0.5] * T_i        # length T_i + 1
        rows.append({"ad_id": ad_id, "R_hat": curve})
    return rows


def make_b1(test_ads: list[tuple[str, int]], mean_curve: np.ndarray) -> list[dict]:
    """R̂_i(t) = R̄_train(t) for t in 0..T_i."""
    rows = []
    for ad_id, T_i in test_ads:
        curve = mean_curve[: T_i + 1].copy()
        curve = np.minimum.accumulate(curve).clip(0.0, 1.0)
        curve[0] = 1.0
        rows.append({"ad_id": ad_id, "R_hat": curve.tolist()})
    return rows


def make_b2(test_ads: list[tuple[str, int]]) -> list[dict]:
    """R̂_i(t) = max(0, 1 − t / T_i), per-ad linear decay."""
    rows = []
    for ad_id, T_i in test_ads:
        ts = np.arange(T_i + 1, dtype=np.float64)
        curve = np.clip(1.0 - ts / max(T_i, 1), 0.0, 1.0)
        curve[0] = 1.0
        rows.append({"ad_id": ad_id, "R_hat": curve.tolist()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--test_jsonl",  required=True)
    ap.add_argument("--out_dir",     required=True,
                    help="Each baseline writes <out_dir>/<name>/submission.parquet")
    args = ap.parse_args()

    test_ads = load_test(Path(args.test_jsonl))
    print(f"test ads: {len(test_ads)}")
    mean_curve = per_second_mean(Path(args.train_jsonl))
    print(f"mean_curve[:5] = {mean_curve[:5].tolist()}")
    print(f"mean_curve[30] = {mean_curve[30]:.4f}, mean_curve[60] = {mean_curve[60]:.4f}")

    out_dir = Path(args.out_dir)

    for name, builder in [
        ("B0_constant_05", lambda: make_b0(test_ads)),
        ("B1_mean_train_curve", lambda: make_b1(test_ads, mean_curve)),
        ("B2_uniform_decay", lambda: make_b2(test_ads)),
    ]:
        rows = builder()
        write_submission(
            rows,
            out_dir / name / "submission.parquet",
            method=name, seed=0,
        )


if __name__ == "__main__":
    main()

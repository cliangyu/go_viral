"""
Content-blind constant-curve baseline (audit check #1).

For each test ad, predict R_hat(t) = mean over training ads of R(t),
truncated to the test ad's T_i + 1. This is the "marginal-mean" baseline:
the prediction depends only on t, not on the video / audio / brand.

If the trained methods do not beat this baseline on IBS, they have learned
nothing per-ad that the data couldn't give us by averaging.

Run on Modal:
  modal run cs224r_project/modal/modal_constant_baseline.py
or locally:
  python cs224r_project/baselines/constant_curve_baseline.py \
      --train_jsonl /vol/data/splits/train.jsonl \
      --test_jsonl  /vol/data/splits/test.jsonl \
      --out_submission /vol/runs/constant_mean/submission.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def per_second_mean(train_jsonl: Path, T_max: int = 60) -> np.ndarray:
    """Compute mean R(t) over training ads for each second t = 0..T_max."""
    sums = np.zeros(T_max + 1, dtype=np.float64)
    counts = np.zeros(T_max + 1, dtype=np.int64)
    with open(train_jsonl) as f:
        for line in f:
            row = json.loads(line)
            curve = row["_meta"]["retention_curve"]
            T_i = int(row["_meta"]["duration_s"])
            # curve has length T_i + 1; index t = 0..T_i.
            for t in range(min(T_i + 1, T_max + 1)):
                sums[t] += float(curve[t])
                counts[t] += 1
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return mean  # shape (T_max + 1,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_submission", required=True)
    ap.add_argument("--method", default="constant_mean")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_path = Path(args.train_jsonl)
    test_path = Path(args.test_jsonl)
    out_path = Path(args.out_submission)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Computing per-second mean R(t) from {train_path}")
    mean_curve = per_second_mean(train_path, T_max=60)
    # Force the t=0 anchor to 1.0 (ttcc-eval validates this).
    mean_curve[0] = 1.0
    print(f"  mean_curve[:5] = {mean_curve[:5].tolist()}")
    print(f"  mean_curve[5:11] = {mean_curve[5:11].tolist()}")
    print(f"  mean_curve[30] = {mean_curve[30]:.4f}, "
          f"mean_curve[60] = {mean_curve[60]:.4f}")

    ad_ids: list[str] = []
    R_hats: list[list[float]] = []
    n_skipped = 0
    with open(test_path) as f:
        for line in f:
            row = json.loads(line)
            ad_id = str(row["_meta"]["ad_id"])
            T_i = int(row["_meta"]["duration_s"])
            curve = mean_curve[: T_i + 1].copy()
            # Sanity: monotone non-increasing? mean curve usually is, but
            # the t=0 force above could create a tiny rise to t=1. Clip
            # via running min.
            curve = np.minimum.accumulate(curve)
            curve = np.clip(curve, 0.0, 1.0)
            if curve[0] != 1.0:
                curve[0] = 1.0
            if not np.all(np.isfinite(curve)):
                n_skipped += 1
                continue
            ad_ids.append(ad_id)
            R_hats.append(curve.tolist())

    print(f"\nwriting {len(ad_ids)} rows (skipped {n_skipped})")
    table = pa.table({
        "ad_id":  pa.array(ad_ids, type=pa.string()),
        "R_hat":  pa.array(R_hats, type=pa.list_(pa.float64())),
        "method": pa.array([args.method] * len(ad_ids), type=pa.string()),
        "seed":   pa.array([args.seed] * len(ad_ids), type=pa.int64()),
    })
    pq.write_table(table, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

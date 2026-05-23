"""
Convert our diagnostic test_preds.parquet (includes R_true, hat_at_3, etc.)
into a submission parquet that satisfies ttcc-eval's contract.

Submission contract (ttcc-eval/docs/04_predictions_contract.md):
  Columns:
    ad_id   : string
    R_hat   : list<float64>, length EXACTLY T_i + 1, includes R_hat(0) = 1.0
    method  : string (optional but recommended)
    seed    : int    (optional but recommended)

Run:
  python cs224r_project/eval/write_submission.py \\
      --diagnostic_parquet cs224r_project/runs/sft_mse/seed42/test_preds.parquet \\
      --out_parquet        cs224r_project/runs/sft_mse/seed42/submission.parquet \\
      --method sft_mse --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic_parquet", required=True,
                    help="Output of make_test_preds.py (has ad_id, R_hat, R_true, duration, ...)")
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--method", required=True,
                    help="e.g. 'sft_mse', 'sft_hazard_cot', 'grpo'")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.diagnostic_parquet)
    if "ad_id" not in df.columns or "R_hat" not in df.columns:
        raise SystemExit(f"diagnostic parquet missing ad_id or R_hat: {list(df.columns)}")
    if "duration" not in df.columns:
        raise SystemExit("diagnostic parquet missing 'duration' column (need it for length check)")

    n = len(df)
    print(f"loaded {n} diagnostic rows from {args.diagnostic_parquet}")

    # Validate every row before writing — ttcc-eval will reject malformed files.
    bad_len = 0
    bad_finite = 0
    bad_anchor = 0
    kept_ids: list[str] = []
    kept_R: list[list[float]] = []
    for _, row in df.iterrows():
        ad_id = str(row["ad_id"])
        T_i = int(row["duration"])
        R = np.asarray(row["R_hat"], dtype=np.float64)
        if len(R) != T_i + 1:
            bad_len += 1
            continue
        if not np.all(np.isfinite(R)):
            bad_finite += 1
            continue
        if abs(float(R[0]) - 1.0) > 1e-6:
            bad_anchor += 1
            continue
        kept_ids.append(ad_id)
        kept_R.append([float(x) for x in R])

    print(f"validated: kept={len(kept_ids)}, "
          f"bad_length={bad_len}, bad_finite={bad_finite}, bad_anchor={bad_anchor}")
    if not kept_ids:
        raise SystemExit("no valid rows; refusing to write empty submission")

    table = pa.table({
        "ad_id":  pa.array(kept_ids, type=pa.string()),
        "R_hat":  pa.array(kept_R, type=pa.list_(pa.float64())),
        "method": pa.array([args.method] * len(kept_ids), type=pa.string()),
        "seed":   pa.array([args.seed] * len(kept_ids), type=pa.int64()),
    })
    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    print(f"wrote {out}  ({len(kept_ids)} rows)")


if __name__ == "__main__":
    main()

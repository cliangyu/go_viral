"""
Diagnostic A: plot R_hat curves from B1, SFT-CoT, and ground truth for
a handful of test ads.

If SFT-CoT looks ~identical to B1 across ads → method learned only marginal
mean. If SFT-CoT shapes vary per ad → method is making per-ad predictions
(may still be lost in noise floor, but the wiring is doing something).

Run:
  python eval/plot_sample_curves.py \\
      --gt /vol/data/splits/test.jsonl \\
      --b1 /vol/runs/B1_mean_train_curve/submission.parquet \\
      --sft_cot /vol/runs/sft_hazard_cot/seed42_a0.1_strict/submission.parquet \\
      --sft_mse /vol/runs/sft_mse/seed42_strict/submission.parquet \\
      --out_dir cs224r_project/runs/diagnostics/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_preds(path: str) -> dict[str, list[float]]:
    t = pq.read_table(path)
    return dict(zip(t["ad_id"].to_pylist(), t["R_hat"].to_pylist()))


def load_gt(path: str) -> dict[str, tuple[int, list[float]]]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            m = r["_meta"]
            out[str(m["ad_id"])] = (int(m["duration_s"]), list(m["retention_curve"]))
    return out


def ascii_curve(curve: list[float], width: int = 60) -> str:
    """Render a small ASCII visualization of a 0-1 curve.
    Each character represents R(t) bucketed into 10 rows."""
    rows = 10
    # Each curve point becomes one column.
    cols = len(curve)
    if cols == 0:
        return ""
    # Resample to `width` columns.
    if cols > width:
        idx = np.linspace(0, cols - 1, width).round().astype(int)
        sampled = [curve[i] for i in idx]
    else:
        sampled = curve
    out_lines = []
    for r in range(rows, 0, -1):
        thresh = r / rows
        line = "".join("█" if c >= thresh else " " for c in sampled)
        out_lines.append(f"{thresh:.1f} |{line}")
    out_lines.append("    " + "-" * len(sampled))
    return "\n".join(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--b1", required=True, help="B1 submission parquet")
    ap.add_argument("--sft_cot", required=True)
    ap.add_argument("--sft_mse", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=5,
                    help="How many ads to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gt = load_gt(args.gt)
    b1 = load_preds(args.b1)
    sft_cot = load_preds(args.sft_cot)
    sft_mse = load_preds(args.sft_mse)

    common = sorted(set(gt) & set(b1) & set(sft_cot) & set(sft_mse))
    print(f"common ads: {len(common)}")

    rng = np.random.default_rng(args.seed)
    sample = sorted(rng.choice(common, size=min(args.n, len(common)), replace=False).tolist())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also dump CSV-like text for downstream analysis.
    summary_lines = []
    summary_lines.append("ad_id\tT\tIBS_b1\tIBS_sft_mse\tIBS_sft_cot")

    for ad_id in sample:
        T_i, R_true = gt[ad_id]
        R_b1 = list(b1[ad_id])
        R_mse = list(sft_mse[ad_id])
        R_cot = list(sft_cot[ad_id])

        ibs = lambda a, b: float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
        summary_lines.append(
            f"{ad_id}\t{T_i}\t{ibs(R_b1, R_true):.4f}\t"
            f"{ibs(R_mse, R_true):.4f}\t{ibs(R_cot, R_true):.4f}"
        )

        print("=" * 76)
        print(f"ad_id={ad_id}  T_i={T_i}  "
              f"IBS: B1={ibs(R_b1, R_true):.4f}  "
              f"SFT-MSE={ibs(R_mse, R_true):.4f}  "
              f"SFT-CoT={ibs(R_cot, R_true):.4f}")
        print()
        print("Ground truth:")
        print(ascii_curve(R_true))
        print()
        print("B1 (constant train-mean):")
        print(ascii_curve(R_b1))
        print()
        print("SFT-MSE-42:")
        print(ascii_curve(R_mse))
        print()
        print("SFT-CoT-42:")
        print(ascii_curve(R_cot))
        print()
        # Also: numeric values at a few seconds.
        ts = [0, 1, 3, 5, 10, min(T_i, 30), T_i]
        ts = sorted(set(t for t in ts if t <= T_i))
        print(f"{'t':>5s} | {'R_true':>8s} | {'B1':>8s} | {'SFT-MSE':>8s} | {'SFT-CoT':>8s}")
        for t in ts:
            print(f"{t:5d} | {R_true[t]:8.4f} | {R_b1[t]:8.4f} | {R_mse[t]:8.4f} | {R_cot[t]:8.4f}")
        print()

    summary_path = out_dir / "sample_ad_summary.tsv"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nwrote summary to {summary_path}")


if __name__ == "__main__":
    main()

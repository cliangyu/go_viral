"""
Failure-mode diagnostics:
  - Distribution of R_true(1) over test set (how many fast/slow ads?)
  - Distribution of R_hat(1) per method (is model collapsing predictions?)
  - Per-second mean prediction across all test ads (mode collapse signature:
    all predictions ~ marginal mean)
  - Per-ad variance of (R_hat across t) — low variance means flat output

Run:
  python eval/diagnose_failure_mode.py \\
      --gt /vol/data/splits/test.jsonl \\
      --preds "/vol/.../B1:B1" "/vol/.../sft_mse_v2_43:MSE-43" "/vol/.../sft_cot_v2_43:CoT-43"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_preds(path: Path) -> dict[str, list[float]]:
    t = pq.read_table(path)
    return dict(zip(t["ad_id"].to_pylist(), t["R_hat"].to_pylist()))


def load_gt(path: Path) -> dict[str, tuple[int, list[float]]]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            m = r["_meta"]
            out[str(m["ad_id"])] = (int(m["duration_s"]),
                                    [float(x) for x in m["retention_curve"]])
    return out


def bucket(val, edges, labels):
    for i, e in enumerate(edges):
        if val <= e:
            return labels[i]
    return labels[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", nargs="+", required=True,
                    help="parquet:label per method")
    args = ap.parse_args()

    gt = load_gt(Path(args.gt))
    ad_ids = sorted(gt.keys())
    print(f"n test ads: {len(ad_ids)}")

    # ---- Ground truth distribution: R_true(1) buckets ----
    r1_true = np.array([gt[a][1][1] for a in ad_ids])
    print(f"\nR_true(1) summary: mean={r1_true.mean():.3f}  "
          f"median={np.median(r1_true):.3f}  std={r1_true.std():.3f}  "
          f"range=[{r1_true.min():.3f}, {r1_true.max():.3f}]")
    edges = [0.2, 0.4, 0.6]
    labels = ["fast-drop (<0.2)", "medium-drop (0.2-0.4)",
              "slow-drop (0.4-0.6)", "very-slow (>0.6)"]
    counts = {l: 0 for l in labels}
    for r in r1_true:
        counts[bucket(r, edges, labels)] += 1
    print("Ground truth R(1) bucketed:")
    for l in labels:
        n = counts[l]
        pct = 100 * n / len(ad_ids)
        bar = "█" * int(pct / 2)
        print(f"  {l:30s}  n={n:>3d}  {pct:5.1f}%  {bar}")

    print("\n" + "=" * 78)

    # ---- Per-method diagnostics ----
    for spec in args.preds:
        path, label = spec.rsplit(":", 1)
        preds = load_preds(Path(path))
        print(f"\n--- {label} ---")

        # 1) R_hat(1) distribution
        r1_hat = np.array([preds[a][1] for a in ad_ids if a in preds])
        print(f"R_hat(1) summary: mean={r1_hat.mean():.3f}  "
              f"median={np.median(r1_hat):.3f}  std={r1_hat.std():.3f}  "
              f"range=[{r1_hat.min():.3f}, {r1_hat.max():.3f}]")
        counts_hat = {l: 0 for l in labels}
        for r in r1_hat:
            counts_hat[bucket(r, edges, labels)] += 1
        print("R_hat(1) bucketed:")
        for l in labels:
            n = counts_hat[l]
            pct = 100 * n / len(r1_hat)
            bar = "█" * int(pct / 2)
            print(f"  {l:30s}  n={n:>3d}  {pct:5.1f}%  {bar}")

        # 2) Per-ad shape variance (low = output is ~flat)
        per_ad_var = []
        for a in ad_ids:
            if a not in preds:
                continue
            T_i = gt[a][0]
            r = np.array(preds[a][: T_i + 1])
            per_ad_var.append(r.var())
        per_ad_var = np.array(per_ad_var)
        print(f"Per-ad variance of R_hat: mean={per_ad_var.mean():.4f}  "
              f"median={np.median(per_ad_var):.4f}")

        # 3) Mode-collapse signature: cosine similarity of predictions
        # Stack all R_hat(0..60), truncate each to min length, compute pair sim
        vecs = []
        min_T = 60
        for a in ad_ids:
            if a not in preds: continue
            T_i = gt[a][0]
            min_T = min(min_T, T_i)
            vecs.append(preds[a])
        # take prefix of length 6 (R(0..5)) which all ads have
        prefix = np.array([v[:6] for v in vecs])
        # center each row by removing R(0)=1 (which is always 1)
        prefix_c = prefix[:, 1:]  # R(1..5)
        # mean and dispersion across ads
        prefix_mean = prefix_c.mean(axis=0)
        prefix_std = prefix_c.std(axis=0)
        print(f"R_hat(1..5) mean: " + " ".join(f"{v:.3f}" for v in prefix_mean))
        print(f"R_hat(1..5) std:  " + " ".join(f"{v:.3f}" for v in prefix_std))
        # If std ≈ 0 → mode collapse on early seconds
        # If std comparable to R_true(1..5) std, predictions are diverse
        true_prefix = np.array([gt[a][1][:6] for a in ad_ids])[:, 1:]
        print(f"R_true(1..5) std: " + " ".join(f"{v:.3f}" for v in true_prefix.std(axis=0)))

        # 4) Calibration: per-ad bias = mean(R_hat - R_true) over t in 1..T_i
        biases = []
        for a in ad_ids:
            if a not in preds: continue
            T_i = gt[a][0]
            err = (np.array(preds[a][1: T_i + 1])
                   - np.array(gt[a][1][1: T_i + 1]))
            biases.append(err.mean())
        biases = np.array(biases)
        print(f"Per-ad bias (R_hat - R_true) over t>=1: "
              f"mean={biases.mean():+.4f}  median={np.median(biases):+.4f}  "
              f"std={biases.std():.4f}")
        n_over = (biases > 0).sum()
        n_under = (biases < 0).sum()
        print(f"  bias > 0 (over-predicting):  {n_over:>3d} ads "
              f"({100*n_over/len(biases):.1f}%)")
        print(f"  bias < 0 (under-predicting): {n_under:>3d} ads "
              f"({100*n_under/len(biases):.1f}%)")


if __name__ == "__main__":
    main()

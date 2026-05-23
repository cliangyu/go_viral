"""
Diagnostic A (matplotlib version): produce PNG plots comparing
ground truth vs B1 vs SFT-MSE vs SFT-CoT for sampled test ads.

Run:
  python plot_curves_png.py \\
      --gt /vol/data/splits/test.jsonl \\
      --b1 /vol/runs/B1_mean_train_curve/submission.parquet \\
      --sft_cot /vol/runs/sft_hazard_cot/seed42_a0.1_strict/submission.parquet \\
      --sft_mse /vol/runs/sft_mse/seed42_strict/submission.parquet \\
      --out_dir /vol/runs/diagnostics/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def plot_one(ad_id: str, T_i: int, R_true, R_b1, R_mse, R_cot, out_path: Path):
    ts = np.arange(T_i + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(ts, R_true[: T_i + 1], "k-",  linewidth=2.5, label="R_true",   zorder=5)
    ax.plot(ts, R_b1[: T_i + 1],   "--",  linewidth=1.8, label="B1 (train-mean)",
            color="#888")
    ax.plot(ts, R_mse[: T_i + 1],  ":",   linewidth=1.6, label="SFT-MSE-42",
            color="#1f77b4")
    ax.plot(ts, R_cot[: T_i + 1],  "-",   linewidth=1.8, label="SFT-CoT-42",
            color="#d62728")

    def ibs(a, b):
        a = np.asarray(a[: T_i + 1]); b = np.asarray(b[: T_i + 1])
        return float(np.mean((a - b) ** 2))

    ax.set_xlim(0, T_i)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("t (seconds)")
    ax.set_ylabel("R(t)")
    ax.set_title(
        f"ad {ad_id}   T={T_i}s\n"
        f"IBS:  B1={ibs(R_b1, R_true):.4f}  "
        f"SFT-MSE={ibs(R_mse, R_true):.4f}  "
        f"SFT-CoT={ibs(R_cot, R_true):.4f}",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--b1", required=True)
    ap.add_argument("--sft_cot", required=True)
    ap.add_argument("--sft_mse", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy",
                    choices=["random", "by_T", "all", "hardest_for_cot",
                             "easiest_for_cot", "biggest_cot_vs_b1_win",
                             "biggest_cot_vs_b1_loss"],
                    default="by_T",
                    help="random | by_T | all | hardest_for_cot (highest SFT-CoT IBS) "
                         "| easiest_for_cot | biggest_cot_vs_b1_win (most negative ΔIBS) "
                         "| biggest_cot_vs_b1_loss")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    b1 = load_preds(args.b1)
    sft_cot = load_preds(args.sft_cot)
    sft_mse = load_preds(args.sft_mse)

    common = sorted(set(gt) & set(b1) & set(sft_cot) & set(sft_mse))
    print(f"common ads: {len(common)}")

    def per_ad_ibs(preds_dict, ad):
        T_i, true = gt[ad]
        a = np.asarray(preds_dict[ad][: T_i + 1]); b = np.asarray(true[: T_i + 1])
        return float(np.mean((a - b) ** 2))

    if args.strategy == "by_T":
        # Bucket ads by T_i, pick roughly equal counts across buckets.
        buckets: dict[int, list[str]] = {}
        for ad in common:
            T_i = gt[ad][0]
            bucket = (T_i // 10) * 10  # 0-9, 10-19, ..., 60
            buckets.setdefault(bucket, []).append(ad)
        rng = np.random.default_rng(args.seed)
        sample = []
        for bucket in sorted(buckets):
            ads = buckets[bucket]
            rng.shuffle(ads)
            sample.append(ads[0])
        if len(sample) > args.n:
            sample = sample[: args.n]
        elif len(sample) < args.n:
            extras = rng.choice([a for a in common if a not in sample],
                                size=args.n - len(sample), replace=False)
            sample += extras.tolist()
        sample.sort(key=lambda a: gt[a][0])
    elif args.strategy == "all":
        sample = sorted(common, key=lambda a: gt[a][0])
        args.n = len(sample)
    elif args.strategy == "hardest_for_cot":
        sample = sorted(common, key=lambda a: per_ad_ibs(sft_cot, a), reverse=True)
        sample = sample[: args.n]
    elif args.strategy == "easiest_for_cot":
        sample = sorted(common, key=lambda a: per_ad_ibs(sft_cot, a))
        sample = sample[: args.n]
    elif args.strategy == "biggest_cot_vs_b1_win":
        sample = sorted(common, key=lambda a: per_ad_ibs(sft_cot, a) - per_ad_ibs(b1, a))
        sample = sample[: args.n]
    elif args.strategy == "biggest_cot_vs_b1_loss":
        sample = sorted(common, key=lambda a: per_ad_ibs(sft_cot, a) - per_ad_ibs(b1, a), reverse=True)
        sample = sample[: args.n]
    else:
        rng = np.random.default_rng(args.seed)
        sample = sorted(
            rng.choice(common, size=min(args.n, len(common)), replace=False).tolist()
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-ad single-panel PNGs.
    for ad_id in sample:
        T_i, R_true = gt[ad_id]
        out_path = out_dir / f"ad_{ad_id}_T{T_i}.png"
        plot_one(ad_id, T_i, R_true, b1[ad_id], sft_mse[ad_id], sft_cot[ad_id], out_path)

    # Multi-panel grid (one figure).
    n = len(sample)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows), squeeze=False)
    for idx, ad_id in enumerate(sample):
        ax = axes[idx // cols][idx % cols]
        T_i, R_true = gt[ad_id]
        ts = np.arange(T_i + 1)
        ax.plot(ts, R_true[: T_i + 1], "k-", lw=2.5, label="R_true", zorder=5)
        ax.plot(ts, b1[ad_id][: T_i + 1],      "--", lw=1.5, label="B1", color="#888")
        ax.plot(ts, sft_mse[ad_id][: T_i + 1], ":",  lw=1.4, label="SFT-MSE",
                color="#1f77b4")
        ax.plot(ts, sft_cot[ad_id][: T_i + 1], "-",  lw=1.6, label="SFT-CoT",
                color="#d62728")
        def ibs(a, b):
            a = np.asarray(a[: T_i + 1]); b = np.asarray(b[: T_i + 1])
            return float(np.mean((a - b) ** 2))
        ax.set_title(
            f"ad …{ad_id[-6:]}  T={T_i}s   "
            f"IBS: B1={ibs(b1[ad_id], R_true):.3f}  "
            f"SFT-MSE={ibs(sft_mse[ad_id], R_true):.3f}  "
            f"SFT-CoT={ibs(sft_cot[ad_id], R_true):.3f}",
            fontsize=9,
        )
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, T_i)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        if idx // cols == rows - 1:
            ax.set_xlabel("t (s)")
        if idx % cols == 0:
            ax.set_ylabel("R(t)")
    # Hide empty subplots.
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    plt.tight_layout()
    grid_path = out_dir / "sample_curves_grid.png"
    fig.savefig(grid_path, dpi=130)
    plt.close(fig)
    print(f"wrote {grid_path}")


if __name__ == "__main__":
    main()

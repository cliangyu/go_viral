"""
Aggregate ttcc-eval JSON reports across seeds (§9 cross-seed aggregation).

Reads one or more report JSONs (produced by `ttcc-eval evaluate`) and prints
a per-metric summary: per-seed point + CI, then the cross-seed mean ± std.
The per-seed CIs come straight from the BCa bootstrap; the cross-seed
columns are the seed-noise diagnostic the milestone §9 calls for.

Usage:
  python cs224r_project/eval/aggregate_reports.py \\
      cs224r_project/runs/reports/sft_mse_seed42.json \\
      cs224r_project/runs/reports/sft_mse_seed43.json \\
      cs224r_project/runs/reports/sft_mse_seed44.json

  # Or on Modal volume after `modal volume get`:
  python cs224r_project/eval/aggregate_reports.py reports/sft_mse_seed*.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = [
    # Primary proper-scoring-rule + diagnostics (docs/07 revision).
    ("ibs",                 "IBS↓"),
    ("calibration_slope",   "calib_slope"),
    ("auc_spearman",        "AUC_ρ"),
    # Legacy milestone-§1 rank metrics (kept as diagnostics).
    ("hook_spearman",       "ρ_hook"),
    ("completion_spearman", "ρ_comp"),
    ("shape_spearman_mean", "ρ̄_shape"),
    ("hook_mae",            "MAE@3"),
    ("completion_mae",      "MAE@T"),
]

# Metrics where lower is better — used for the "all CIs better than baseline"
# annotation; for rank metrics we test against 0 (the random-chance threshold).
LOWER_IS_BETTER = {"ibs", "hook_mae", "completion_mae"}


def load_report(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    args = ap.parse_args()

    reports = [load_report(p) for p in args.reports]
    # Each report has either {"raw": {...}} (modal_eval wrapper) or the raw payload directly.
    payloads = []
    for r, p in zip(reports, args.reports):
        if "raw" in r and isinstance(r["raw"], dict):
            payload = r["raw"]
        else:
            payload = r
        payloads.append((p, payload))

    print(f"Loaded {len(payloads)} report(s):")
    for path, p in payloads:
        meta = p.get("predictions_meta", {})
        print(f"  {path.name}  method={meta.get('method')}  seed={meta.get('seed')}  "
              f"n={p.get('n_ads_evaluated')}")
    print()

    # Header.
    seed_labels = [
        str(p.get("predictions_meta", {}).get("seed", path.stem))
        for path, p in payloads
    ]
    col_w = 22
    header = ["metric".ljust(10)] + [f"seed {s}".center(col_w) for s in seed_labels]
    header += ["mean".center(col_w), "std".center(10), "all CIs > 0?".center(14)]
    print("  ".join(header))
    print("-" * (len(" ".join(header)) + 30))

    for key, label in METRICS:
        cells = [label.ljust(10)]
        points = []
        all_excl_zero = True
        for _, payload in payloads:
            m = payload["metrics"][key]
            point = m["point"]
            lo = m["lo"]
            hi = m["hi"]
            points.append(point)
            cells.append(f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]".center(col_w))
            # Spearman metrics meaningful vs 0; MAE not.
            if key.endswith("_spearman") or key.endswith("_spearman_mean"):
                if not (lo > 0 or hi < 0):
                    all_excl_zero = False
        mean = statistics.fmean(points)
        std = statistics.pstdev(points) if len(points) > 1 else 0.0
        cells.append(f"{mean:+.4f}".center(col_w))
        cells.append(f"{std:.4f}".center(10))
        flag = ("yes" if all_excl_zero else "no") if (
            key.endswith("_spearman") or key.endswith("_spearman_mean")
        ) else "n/a"
        cells.append(flag.center(14))
        print("  ".join(cells))

    print()
    print("Notes:")
    print(" - per-seed CIs are 95% BCa from ttcc-eval (B=10,000).")
    print(" - mean / std across seeds is a seed-noise diagnostic, not a CI.")
    print(" - 'all CIs > 0?' flags rank metrics that are significant in every seed.")


if __name__ == "__main__":
    main()

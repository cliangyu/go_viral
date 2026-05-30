#!/usr/bin/env python
"""report_eval.py -- the VISUALIZATION layer.

Third stage of the decoupled pipeline:
    srcc_eval.py (prediction) -> retention_metrics.py (algorithm) -> THIS (viz)

Reads ONLY the .npz dumps + the metrics.json that retention_metrics.py emits.
Renders a markdown comparison table + a 2x2 PNG. NO torch, NO model, NO metric
math of its own -- it imports retention_metrics for any numbers it needs, so the
algorithm stays the single source of truth.

Usage:
  python report_eval.py --metrics metrics.json \
      --dump baseline:dump_ckpt225_full_fps1.npz \
      --dump nodeA_reshape_sft:dump_nodeA300_full.npz \
      --dump nodeB_proper_rl:dump_nodeB75_full.npz \
      --out-md report.md --out-png report.png
"""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import retention_metrics as rm


def mean_curves(by_id, ads, t_hi=30):
    """Mean predicted + mean true curve over ads, t=0..t_hi (for calibration view)."""
    P = np.full((len(ads), t_hi + 1), np.nan); Tr = np.full((len(ads), t_hi + 1), np.nan)
    for i, ad in enumerate(ads):
        p, q, T = by_id[ad]
        hi = min(t_hi, T)
        P[i, :hi + 1] = p[:hi + 1]; Tr[i, :hi + 1] = q[:hi + 1]
    return np.nanmean(P, axis=0), np.nanmean(Tr, axis=0)


def write_md(metrics, out_md):
    b = metrics['baseline']; b1 = metrics['b1_climatology_ibs_full']
    lines = ['# Baseline vs RL — cross-ad SRCC + IBS', '',
             'Two metrics, same leak-free bypass eval, same val set (n=1332), fps=1.0.',
             '- **SRCC** (cross-ad ranking, t∈[1,30], higher better)',
             '- **IBS** (Integrated Brier Score = mean_t (R_pred−R_true)², lower better)',
             '', '| ckpt | SRCC | ΔSRCC (95% CI) | IBS_full | ΔIBS (95% CI) | calibration |',
             '|---|---|---|---|---|---|',
             f"| **baseline ckpt-225 (SFT)** | {b['srcc']:.4f} | — | {b['ibs_full']:.5f} | — | beats climatology ({b1:.5f}) |"]
    for c in metrics['candidates']:
        ps, pi = c['paired_srcc_vs_baseline'], c['paired_ibs_vs_baseline']
        s_ci = f"{ps['delta']:+.4f} [{ps['delta_ci'][0]:+.4f}, {ps['delta_ci'][1]:+.4f}]"
        i_ci = f"{pi['delta']:+.5f} [{pi['delta_ci'][0]:+.5f}, {pi['delta_ci'][1]:+.5f}]"
        s_verdict = 'beats' if ps['beats_baseline'] else ('null' if ps['delta_ci'][0] <= 0 <= ps['delta_ci'][1] else 'worse')
        i_verdict = 'better' if pi['beats_baseline'] else ('null' if pi['delta_ci'][0] <= 0 <= pi['delta_ci'][1] else 'WORSE')
        cal = 'worse than climatology' if c['ibs_full'] > b1 else 'beats climatology'
        lines.append(f"| {c['label']} | {c['srcc']:.4f} ({s_verdict}) | {s_ci} | "
                     f"{c['ibs_full']:.5f} ({i_verdict}) | {i_ci} | {cal} |")
    lines += ['', '**Read:** SRCC null for both RL runs (CIs include 0); IBS significantly '
              'WORSE for both (CIs entirely positive). The SRCC-only view hid real '
              'calibration damage — which is why both metrics are reported. climatology '
              f'B₁ IBS = {b1:.5f} (val-derived reference).']
    with open(out_md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[viz] wrote {out_md}')


def make_fig(metrics, dumps, out_png):
    labels = ['baseline'] + [c['label'] for c in metrics['candidates']]
    colors = {'baseline': 'k', metrics['candidates'][0]['label']: 'tab:red',
              metrics['candidates'][1]['label']: 'tab:blue'} if len(metrics['candidates']) == 2 \
        else {l: c for l, c in zip(labels, ['k', 'tab:red', 'tab:blue', 'tab:green'])}
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # (1) per-t cross-ad SRCC
    allm = [metrics['baseline']] + metrics['candidates']
    for m, lab in zip(allm, labels):
        ts = sorted(int(t) for t in m['per_t_srcc']); ys = [m['per_t_srcc'][str(t)] for t in ts]
        ax[0, 0].plot(ts, ys, marker='.', label=lab, color=colors[lab])
    ax[0, 0].set(title='(1) cross-ad SRCC per second (higher=better)', xlabel='t (s)', ylabel='Spearman ρ')
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    # (2) IBS bars (full-curve) + climatology
    bar_labels = labels + ['climatology B₁']
    bar_vals = [metrics['baseline']['ibs_full']] + [c['ibs_full'] for c in metrics['candidates']] + [metrics['b1_climatology_ibs_full']]
    bar_cols = [colors[l] for l in labels] + ['gray']
    ax[0, 1].bar(range(len(bar_vals)), bar_vals, color=bar_cols)
    ax[0, 1].set_xticks(range(len(bar_vals))); ax[0, 1].set_xticklabels(bar_labels, rotation=20, ha='right', fontsize=8)
    ax[0, 1].axhline(metrics['baseline']['ibs_full'], ls='--', color='k', lw=0.8)
    ax[0, 1].set(title='(2) IBS full-curve (lower=better)', ylabel='IBS')
    for i, v in enumerate(bar_vals):
        ax[0, 1].text(i, v, f'{v:.5f}', ha='center', va='bottom', fontsize=8)

    # (3) mean retention curves (calibration view)
    for lab, npz in dumps.items():
        by = rm.load_dump(npz); ads = sorted(by)
        mp, mt = mean_curves(by, ads, t_hi=30)
        ax[1, 0].plot(range(len(mp)), mp, color=colors.get(lab, 'C0'), label=f'{lab} pred')
    # one shared TRUE curve (same val set)
    by0 = rm.load_dump(dumps['baseline']); _, mt = mean_curves(by0, sorted(by0), t_hi=30)
    ax[1, 0].plot(range(len(mt)), mt, color='green', ls='--', lw=2, label='TRUE (mean)')
    ax[1, 0].set(title='(3) mean retention curve — calibration', xlabel='t (s)', ylabel='R(t)')
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # (4) paired deltas vs baseline (forest)
    yy = []; ax4 = ax[1, 1]
    rows = []
    for c in metrics['candidates']:
        rows.append((f"{c['label']}\nΔSRCC", c['paired_srcc_vs_baseline']['delta'],
                     c['paired_srcc_vs_baseline']['delta_ci'], 'tab:purple'))
    for i, (name, d, ci, col) in enumerate(rows):
        ax4.errorbar(d, i, xerr=[[d - ci[0]], [ci[1] - d]], fmt='o', color=col, capsize=4)
        ax4.text(d, i + 0.12, name, fontsize=8, ha='center')
    ax4.axvline(0, color='k', lw=1)
    ax4.set(title='(4) paired ΔSRCC vs baseline (CI crosses 0 = null)', xlabel='ΔSRCC', yticks=[])
    ax4.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    print(f'[viz] wrote {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--dump', action='append', default=[], help='label:npz (repeatable)')
    ap.add_argument('--out-md', default='report.md')
    ap.add_argument('--out-png', default='report.png')
    args = ap.parse_args()
    metrics = json.load(open(args.metrics))
    dumps = dict(d.split(':', 1) for d in args.dump)
    write_md(metrics, args.out_md)
    if dumps:
        make_fig(metrics, dumps, args.out_png)


if __name__ == '__main__':
    main()

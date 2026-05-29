#!/usr/bin/env python
"""Paired-bootstrap CI for the cross-ad SRCC difference between two checkpoints.

Fixes the morning-RL "unreadable result" problem: a bare point SRCC on n~158 has a
95% CI of ~+/-0.11, so single-point comparisons (RL vs baseline) were dominated by
noise. This scores BOTH checkpoints on the SAME ads and bootstraps the PAIRED delta,
so we gate any "X beats Y" claim on `delta_ci_lo > 0`.

Inputs are the per-ad dumps from `srcc_eval.py --dump-npz` (must be on the SAME val set):
  baseline.npz, candidate.npz  each with {ad_ids, Ts, preds(obj R(0..T)), trues(obj R(0..T))}

Usage:
  python paired_bootstrap.py --baseline ckpt225.npz --candidate rl_step550.npz \
      --label rl_proper_step550 --ledger /opt/dlami/nvme/v8_eval/rl_ledger.jsonl

Note (cluster caveat): this resamples at the AD level. If the val set contains repeats
from the same advertiser/campaign, the cross-ad ranks are not i.i.d. and this CI is
optimistic — add a cluster_id and pass --cluster-col to cluster the resample. (TODO when
a cluster_id is available; standard ad-level bootstrap for now.)
"""
from __future__ import annotations
import argparse, json
import numpy as np


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    def rank(a):
        order = a.argsort(); r = np.empty(len(a), float); r[order] = np.arange(len(a))
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt); starts = csum - cnt
        return ((starts + csum - 1) / 2.0)[inv]
    rx, ry = rank(x), rank(y); rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def load(npz):
    z = np.load(npz, allow_pickle=True)
    ids = [str(a) for a in z['ad_ids']]
    return {ad: (np.asarray(z['preds'][i], float), np.asarray(z['trues'][i], float), int(z['Ts'][i]))
            for i, ad in enumerate(ids)}


def srcc_on(idx, Pcols, Tcols, valid, t_range):
    """cross-ad SRCC for a (possibly resampled) index array `idx` into the common-ad arrays."""
    rhos = []
    for ti, t in enumerate(t_range):
        m = valid[ti][idx]
        if m.sum() < 3:
            continue
        rho = spearman(Pcols[ti][idx][m], Tcols[ti][idx][m])
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--t-lo', type=int, default=1)
    ap.add_argument('--t-hi', type=int, default=30)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ledger', default=None)
    args = ap.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    common = sorted(set(base) & set(cand))
    n = len(common)
    assert n >= 10, f'only {n} common ads'
    t_range = list(range(args.t_lo, args.t_hi + 1))

    # Precompute per-t aligned columns over `common` (+ validity mask T>=t), for both models.
    def cols(by_id):
        P = np.zeros((len(t_range), n)); valid = np.zeros((len(t_range), n), bool)
        for ti, t in enumerate(t_range):
            for j, ad in enumerate(common):
                Pc, Tr, T = by_id[ad]
                if T >= t and t < len(Pc):
                    P[ti, j] = Pc[t]; valid[ti, j] = True
        return P, valid
    bP, bvalid = cols(base)
    Ttrue = np.zeros((len(t_range), n))   # true curve from baseline dump (same ads/true)
    for ti, t in enumerate(t_range):
        for j, ad in enumerate(common):
            _, Tr, T = base[ad]
            if T >= t and t < len(Tr):
                Ttrue[ti, j] = Tr[t]
    cP, _ = cols(cand)

    idx_all = np.arange(n)
    s_base = srcc_on(idx_all, bP, Ttrue, bvalid, t_range)
    s_cand = srcc_on(idx_all, cP, Ttrue, bvalid, t_range)
    delta = s_cand - s_base

    rng = np.random.default_rng(args.seed)
    dboot, cboot = [], []
    for _ in range(args.n_boot):
        bi = rng.integers(0, n, n)
        sb = srcc_on(bi, bP, Ttrue, bvalid, t_range)
        sc = srcc_on(bi, cP, Ttrue, bvalid, t_range)
        if not (np.isnan(sb) or np.isnan(sc)):
            dboot.append(sc - sb); cboot.append(sc)
    dlo, dhi = np.percentile(dboot, [2.5, 97.5])
    clo, chi = np.percentile(cboot, [2.5, 97.5])
    sig = bool(dlo > 0)

    rec = {'label': args.label, 'n_common': n, 'n_boot': len(dboot),
           'srcc_base': round(s_base, 4), 'srcc_cand': round(s_cand, 4),
           'delta': round(delta, 4), 'delta_ci': [round(dlo, 4), round(dhi, 4)],
           'cand_ci': [round(clo, 4), round(chi, 4)], 'significant_gt_baseline': sig}
    print(json.dumps(rec, indent=2))
    print(f"\n>>> {args.label}: SRCC {s_cand:.4f} (95% CI [{clo:.4f},{chi:.4f}]) vs baseline {s_base:.4f}")
    print(f">>> paired Δ = {delta:+.4f}  95% CI [{dlo:+.4f},{dhi:+.4f}]  -> "
          f"{'SIGNIFICANT improvement' if sig else 'NOT distinguishable from baseline (CI includes 0)'}")
    if args.ledger:
        with open(args.ledger, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f">>> appended to {args.ledger}")


if __name__ == '__main__':
    main()

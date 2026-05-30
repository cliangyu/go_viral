#!/usr/bin/env python
"""retention_metrics.py -- the ALGORITHM layer (single source of truth).

DECOUPLING SEAM (Leon's SE rule: prediction / algorithm / visualization are 3
separate concerns):

    prediction          algorithm              visualization
    ----------          ---------              -------------
    srcc_eval.py   -->  retention_metrics.py  -->  report_eval.py
    (GPU forward)       (THIS FILE)                (plots/tables)
    dumps .npz          pure numpy on .npz         reads metrics

This module is PURE numpy: NO torch, NO model load, NO matplotlib. It runs
anywhere (laptop, no GPU) on the per-ad dumps that srcc_eval.py --dump-npz writes
({ad_ids, Ts, preds(obj R(0..T)), trues(obj R(0..T))}).

It is the ONE place that computes BOTH project metrics, off the SAME loader and
SAME spearman, so they can never silently diverge:

  * cross-ad SRCC  -- RANKING quality. At each second t, Spearman across ads
                      between predicted R(t) and true R(t); averaged over t in
                      [t_lo, t_hi]. "Do we order ads correctly?"
  * IBS  (Integrated Brier Score) -- CURVE CALIBRATION. per ad,
                      mean_t (R_pred[t] - R_true[t])^2; averaged over ads.
                      "Is the predicted curve numerically close to the truth?"
                      Compared against the train-mean climatology B_1.

They answer DIFFERENT questions. An RL run that optimizes a cross-ad ranking
reward can move SRCC without moving IBS (or vice-versa) -- which is exactly why
both must be reported together.

CLI (algorithm only -- emits JSON, no plots):
  python retention_metrics.py \
      --baseline dump_ckpt225_full_fps1.npz \
      --candidate dump_nodeA300_full.npz --label nodeA_reshape \
      --candidate dump_nodeB75_full.npz  --label nodeB_proper_rl \
      --out metrics.json
"""
from __future__ import annotations
import argparse, json
import numpy as np


# ----------------------------------------------------------------------------- spearman (canonical)
def spearman(x, y):
    """Spearman rho via Pearson on average ranks. The ONE canonical impl;
    srcc_eval.py / srcc_guard.py / paired_bootstrap.py each inline a byte-identical
    copy (technical debt -- they should import this). Kept verbatim so results match."""
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


# ----------------------------------------------------------------------------- dump loader
def load_dump(npz):
    """npz from srcc_eval.py --dump-npz -> {ad_id: (pred R(0..T), true R(0..T), T)}."""
    z = np.load(npz, allow_pickle=True)
    ids = [str(a) for a in z['ad_ids']]
    return {ad: (np.asarray(z['preds'][i], float),
                 np.asarray(z['trues'][i], float),
                 int(z['Ts'][i])) for i, ad in enumerate(ids)}


# ----------------------------------------------------------------------------- metric 1: cross-ad SRCC
def cross_ad_srcc(by_id, ads, t_lo=1, t_hi=30):
    """Per-second cross-ad Spearman averaged over t in [t_lo, t_hi]. Returns
    (srcc, per_t={t: (rho, n_ads)})."""
    per_t = {}
    for t in range(t_lo, t_hi + 1):
        pv, tv = [], []
        for ad in ads:
            P, Tr, T = by_id[ad]
            if T >= t and t < len(P) and t < len(Tr):
                pv.append(P[t]); tv.append(Tr[t])
        rho = spearman(pv, tv)
        if not np.isnan(rho):
            per_t[t] = (rho, len(pv))
    srcc = float(np.mean([v[0] for v in per_t.values()])) if per_t else float('nan')
    return srcc, per_t


# ----------------------------------------------------------------------------- metric 2: IBS
def per_ad_ibs(pred, true, T, t_lo=0, t_hi=None):
    """IBS_i = mean_{t in [t_lo, min(t_hi,T)]} (pred[t] - true[t])^2.
    Default t_lo=0,t_hi=None -> full curve [0..T], matching eval_ibs.py."""
    hi = T if t_hi is None else min(t_hi, T)
    if hi < t_lo:
        return np.nan
    p = pred[t_lo:hi + 1]; q = true[t_lo:hi + 1]
    n = min(len(p), len(q))
    if n == 0:
        return np.nan
    return float(((p[:n] - q[:n]) ** 2).mean())


def mean_ibs(by_id, ads, t_lo=0, t_hi=None):
    """Mean per-ad IBS over `ads`. Returns (mean, {ad: ibs})."""
    per_ad = {}
    for ad in ads:
        P, Tr, T = by_id[ad]
        v = per_ad_ibs(P, Tr, T, t_lo, t_hi)
        if not np.isnan(v):
            per_ad[ad] = v
    m = float(np.mean(list(per_ad.values()))) if per_ad else float('nan')
    return m, per_ad


def b1_climatology(by_id, ads):
    """Train-mean baseline B_1: per-second mean of the TRUE curves over `ads`.
    NOTE: derived from the eval ads themselves (val-leak) -- a reference, not a
    clean held-out climatology. For a leak-free B_1 pass train curves. Flagged
    in the report."""
    Tmax = max(by_id[ad][2] for ad in ads)
    acc = np.full((len(ads), Tmax + 1), np.nan)
    for i, ad in enumerate(ads):
        _, Tr, T = by_id[ad]
        acc[i, :len(Tr)] = Tr
    return np.nanmean(acc, axis=0)


def b1_ibs(by_id, ads, B1, t_lo=0, t_hi=None):
    """Mean per-ad IBS of the climatology curve B_1 (broadcast to every ad)."""
    per_ad = {}
    for ad in ads:
        _, Tr, T = by_id[ad]
        per_ad[ad] = per_ad_ibs(B1[:len(Tr)], Tr, T, t_lo, t_hi)
    return float(np.mean([v for v in per_ad.values() if not np.isnan(v)])), per_ad


# ----------------------------------------------------------------------------- paired bootstrap (both metrics)
def paired_bootstrap_srcc(base, cand, common, t_lo=1, t_hi=30, n_boot=2000, seed=0):
    """Paired-bootstrap CI on the cross-ad SRCC delta (cand - base) over common ads.
    Higher SRCC is better -> cand beats base iff delta_ci_lo > 0."""
    t_range = list(range(t_lo, t_hi + 1))
    n = len(common)

    def cols(by_id):
        P = np.zeros((len(t_range), n)); valid = np.zeros((len(t_range), n), bool)
        for ti, t in enumerate(t_range):
            for j, ad in enumerate(common):
                Pc, Tr, T = by_id[ad]
                if T >= t and t < len(Pc):
                    P[ti, j] = Pc[t]; valid[ti, j] = True
        return P, valid
    bP, bvalid = cols(base); cP, _ = cols(cand)
    Ttrue = np.zeros((len(t_range), n))
    for ti, t in enumerate(t_range):
        for j, ad in enumerate(common):
            _, Tr, T = base[ad]
            if T >= t and t < len(Tr):
                Ttrue[ti, j] = Tr[t]

    def srcc_idx(idx, P):
        rhos = []
        for ti in range(len(t_range)):
            m = bvalid[ti][idx]
            if m.sum() < 3:
                continue
            rho = spearman(P[ti][idx][m], Ttrue[ti][idx][m])
            if not np.isnan(rho):
                rhos.append(rho)
        return float(np.mean(rhos)) if rhos else np.nan

    idx_all = np.arange(n)
    s_base = srcc_idx(idx_all, bP); s_cand = srcc_idx(idx_all, cP)
    rng = np.random.default_rng(seed)
    dboot = []
    for _ in range(n_boot):
        bi = rng.integers(0, n, n)
        sb = srcc_idx(bi, bP); sc = srcc_idx(bi, cP)
        if not (np.isnan(sb) or np.isnan(sc)):
            dboot.append(sc - sb)
    dlo, dhi = np.percentile(dboot, [2.5, 97.5])
    return {'srcc_base': round(s_base, 4), 'srcc_cand': round(s_cand, 4),
            'delta': round(s_cand - s_base, 4), 'delta_ci': [round(dlo, 4), round(dhi, 4)],
            'beats_baseline': bool(dlo > 0), 'n_common': n, 'n_boot': len(dboot)}


def paired_bootstrap_ibs(base, cand, common, t_lo=0, t_hi=None, n_boot=2000, seed=0):
    """Paired-bootstrap CI on the mean-IBS delta (cand - base) over common ads.
    LOWER IBS is better -> cand beats base iff delta_ci_hi < 0."""
    ib = np.array([per_ad_ibs(*base[ad], t_lo=t_lo, t_hi=t_hi) for ad in common])
    ic = np.array([per_ad_ibs(*cand[ad], t_lo=t_lo, t_hi=t_hi) for ad in common])
    ok = ~(np.isnan(ib) | np.isnan(ic))
    ib, ic = ib[ok], ic[ok]
    n = len(ib)
    base_m, cand_m = float(ib.mean()), float(ic.mean())
    rng = np.random.default_rng(seed)
    dboot = []
    for _ in range(n_boot):
        bi = rng.integers(0, n, n)
        dboot.append(float(ic[bi].mean() - ib[bi].mean()))
    dlo, dhi = np.percentile(dboot, [2.5, 97.5])
    return {'ibs_base': round(base_m, 5), 'ibs_cand': round(cand_m, 5),
            'delta': round(cand_m - base_m, 5), 'delta_ci': [round(dlo, 5), round(dhi, 5)],
            'beats_baseline': bool(dhi < 0), 'n_common': n, 'n_boot': len(dboot)}


# ----------------------------------------------------------------------------- top-level: compute everything
def compute_all(baseline_npz, candidates, t_lo=1, t_hi=30, ibs_full=True, n_boot=2000, seed=0):
    """candidates: list of (label, npz). Returns a results dict (JSON-able) with
    point SRCC+IBS for baseline and each candidate, the B_1 climatology IBS, and
    paired deltas vs baseline for both metrics."""
    base = load_dump(baseline_npz)
    base_ads = sorted(base)
    # IBS horizon: full curve [0,T] (canonical) + a [1,30] variant matching SRCC.
    ibs_hi = None if ibs_full else t_hi
    B1 = b1_climatology(base, base_ads)
    b1_m, _ = b1_ibs(base, base_ads, B1, t_lo=0, t_hi=ibs_hi)

    def point(by_id, ads):
        srcc, per_t = cross_ad_srcc(by_id, ads, t_lo, t_hi)
        ibs_curve, _ = mean_ibs(by_id, ads, t_lo=0, t_hi=ibs_hi)          # full-curve IBS
        ibs_win, _ = mean_ibs(by_id, ads, t_lo=t_lo, t_hi=t_hi)          # [1,30]-window IBS
        return {'srcc': round(srcc, 4),
                'ibs_full': round(ibs_curve, 5), 'ibs_t1_30': round(ibs_win, 5),
                'n_ads': len(ads),
                'per_t_srcc': {str(t): round(v[0], 4) for t, v in per_t.items()}}

    out = {'baseline': {'npz': baseline_npz, **point(base, base_ads)},
           'b1_climatology_ibs_full': round(b1_m, 5),
           'b1_note': 'val-derived (per-second mean of eval trues); reference, not leak-free',
           'candidates': []}

    for label, npz in candidates:
        cand = load_dump(npz)
        common = sorted(set(base) & set(cand))
        pb_srcc = paired_bootstrap_srcc(base, cand, common, t_lo, t_hi, n_boot, seed)
        pb_ibs = paired_bootstrap_ibs(base, cand, common, t_lo=0, t_hi=ibs_hi, n_boot=n_boot, seed=seed)
        out['candidates'].append({'label': label, 'npz': npz, **point(cand, sorted(cand)),
                                  'paired_srcc_vs_baseline': pb_srcc,
                                  'paired_ibs_vs_baseline': pb_ibs})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--candidate', action='append', default=[], help='npz (repeatable)')
    ap.add_argument('--label', action='append', default=[], help='label per --candidate (repeatable)')
    ap.add_argument('--t-lo', type=int, default=1)
    ap.add_argument('--t-hi', type=int, default=30)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    assert len(args.candidate) == len(args.label), 'need one --label per --candidate'
    res = compute_all(args.baseline, list(zip(args.label, args.candidate)),
                      t_lo=args.t_lo, t_hi=args.t_hi, n_boot=args.n_boot, seed=args.seed)
    txt = json.dumps(res, indent=2)
    print(txt)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(txt)
        print(f'[metrics] wrote {args.out}')


if __name__ == '__main__':
    main()

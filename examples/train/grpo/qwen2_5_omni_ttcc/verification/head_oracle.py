#!/usr/bin/env python
"""HEAD-CEILING ORACLE: given ckpt-225's h_anchor features (feature_dump.py npz), what is the
MAX cross-ad SRCC a linear readout of those features can reach? Per-second ridge h->R(t), K-fold
CV (generalizable ceiling) + fit-on-all (overfit upper bound). Decision: CV-SRCC ~ baseline 0.514
=> 0.514 is the linear-feature ceiling (win must come from FEATURES). CV-SRCC >> 0.514 => the SFT
head under-extracts existing signal (a better head/optimizer/reward can chase it). Pure numpy."""
from __future__ import annotations
import argparse, numpy as np

def spearman(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3: return np.nan
    def rk(a):
        u,inv,c=np.unique(a,return_inverse=True,return_counts=True)
        cs=np.cumsum(c); st=cs-c; avg=(st+cs-1)/2.0; return avg[inv]
    rx,ry=rk(x),rk(y); rx-=rx.mean(); ry-=ry.mean()
    d=np.sqrt((rx*rx).sum()*(ry*ry).sum()); return float((rx*ry).sum()/d) if d>0 else np.nan

def ridge_fit(X, y, lam):
    # X: (m,d) already standardized + bias col; closed form (X^T X + lam I)^-1 X^T y
    d=X.shape[1]; A=X.T@X + lam*np.eye(d); return np.linalg.solve(A, X.T@y)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--key', default='feats')   # 'feats' (last-token) or 'feats_mean' (mean-pool)
    ap.add_argument('--t-lo', type=int, default=1); ap.add_argument('--t-hi', type=int, default=30)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--baseline', type=float, default=0.5142)
    args=ap.parse_args()
    z=np.load(args.npz); F=z[args.key].astype(np.float64); R=z['rtrue'].astype(np.float64); Ts=z['Ts']
    n,d=F.shape
    print(f'[oracle] key={args.key} n={n} d={d}  baseline SFT val SRCC={args.baseline}')
    # standardize features (per-dim), add bias
    mu=F.mean(0); sd=F.std(0)+1e-6; Fs=(F-mu)/sd
    Fb=np.concatenate([Fs, np.ones((n,1))],1)
    rng=np.random.RandomState(0); perm=rng.permutation(n)
    folds=np.array_split(perm, args.folds)
    lams=[1.0,10.0,100.0,1000.0,10000.0]
    print(f"{'lambda':>8} | {'CV-SRCC':>8} | {'fit-all-SRCC':>12}")
    best=(-1,None)
    for lam in lams:
        # ---- K-fold CV: pooled held-out preds per t ----
        heldout={t:([],[]) for t in range(args.t_lo,args.t_hi+1)}
        for fi in range(args.folds):
            te=folds[fi]; tr=np.concatenate([folds[j] for j in range(args.folds) if j!=fi])
            for t in range(args.t_lo,args.t_hi+1):
                trm=tr[Ts[tr]>=t]; tem=te[Ts[te]>=t]
                if len(trm)<10 or len(tem)<3: continue
                w=ridge_fit(Fb[trm], R[trm,t], lam); pred=Fb[tem]@w
                heldout[t][0].extend(pred.tolist()); heldout[t][1].extend(R[tem,t].tolist())
        cv=[spearman(p,tt) for t,(p,tt) in heldout.items() if len(p)>=3]
        cv=np.nanmean([c for c in cv if not np.isnan(c)])
        # ---- fit-on-all (overfit upper bound) ----
        fa=[]
        for t in range(args.t_lo,args.t_hi+1):
            m=np.where(Ts>=t)[0]
            if len(m)<3: continue
            w=ridge_fit(Fb[m], R[m,t], lam); pred=Fb[m]@w
            r=spearman(pred, R[m,t])
            if not np.isnan(r): fa.append(r)
        fa=np.mean(fa)
        print(f"{lam:>8.0f} | {cv:>+8.4f} | {fa:>+12.4f}")
        if cv>best[0]: best=(cv,lam)
    print(f"\n[oracle] BEST CV-SRCC = {best[0]:+.4f} (lambda={best[1]})  vs baseline {args.baseline}")
    delta=best[0]-args.baseline
    if best[0] >= args.baseline+0.05:
        print(f"[oracle] VERDICT: HEADROOM (+{delta:.3f}) — features hold cross-ad signal the SFT head under-extracts; a better head/optimizer/reward can chase it.")
    elif best[0] >= args.baseline-0.03:
        print(f"[oracle] VERDICT: AT CEILING (delta {delta:+.3f}) — 0.514 ~ the linear-feature ceiling; the win must come from FEATURES (audio-on / partial unfreeze / richer pooling), not reward tuning.")
    else:
        print(f"[oracle] VERDICT: ridge below baseline (delta {delta:+.3f}) — hazard head + soft-Spearman likely needed to match; interpret with care.")

if __name__=='__main__':
    main()

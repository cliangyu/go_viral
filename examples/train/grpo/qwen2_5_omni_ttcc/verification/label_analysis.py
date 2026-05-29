#!/usr/bin/env python
"""GPU-FREE analysis of the retention data: is 0.514 a TASK ceiling (intrinsic
predictability / label noise) or a MODEL ceiling? Pure math on the R curves.

(1) Trivial-predictor cross-ad SRCC: rank ads by metadata (video length T) or
    privileged early-retention (R(1), R(3)) and see how much of the per-second
    cross-ad ranking that ALONE explains. If a trivial feature ~ the SFT 0.514,
    the task is near-trivial+noise and content-RL has little to add.
(2) Rank STABILITY across time: Spearman of the ad-ranking at t vs t+1. High =
    structured/low-noise ranking; low = noisy. Also t=1 ranking vs t=30 ranking.
(3) Internal-consistency ceiling: how well does the ranking at early seconds
    predict the ranking at later seconds (an upper bound on what's learnable if
    the curve shape were perfectly known early)."""
import json, sys, numpy as np

def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3: return np.nan
    def rk(a):
        u,inv,c=np.unique(a,return_inverse=True,return_counts=True); cs=np.cumsum(c); st=cs-c; return ((st+cs-1)/2.0)[inv]
    rx,ry=rk(x),rk(y); rx-=rx.mean(); ry-=ry.mean()
    d=np.sqrt((rx*rx).sum()*(ry*ry).sum()); return float((rx*ry).sum()/d) if d>0 else np.nan

path=sys.argv[1] if len(sys.argv)>1 else '/opt/dlami/nvme/v8_eval/data/val_present.jsonl'
T_LO,T_HI=1,30
rows=[json.loads(l) for l in open(path)]
ads=[]
for r in rows:
    R=r.get('R') or r.get('R_true')
    if not R: continue
    R=np.asarray(R,float); T=int(r.get('T',len(R)-1))
    if T<5: continue
    ads.append((R,T))
n=len(ads); print(f"[data] {path}  n_ads={n}")

def crossad_srcc_of(predict):
    """predict: ad-> scalar score used to rank at every t (privileged/trivial). Returns mean SRCC over t."""
    per=[]
    for t in range(T_LO,T_HI+1):
        idx=[i for i,(R,T) in enumerate(ads) if T>=t and len(R)>t]
        if len(idx)<3: continue
        sc=[predict(ads[i]) for i in idx]; tv=[ads[i][0][t] for i in idx]
        rho=spearman(sc,tv)
        if not np.isnan(rho): per.append(rho)
    return float(np.mean(per)) if per else float('nan')

print("\n=== (1) TRIVIAL / PRIVILEGED predictor cross-ad SRCC (avg over t in [1,30]) ===")
print(f"  {'video length  -T':22s}: {crossad_srcc_of(lambda a: -a[1]):+.4f}   (metadata only)")
print(f"  {'video length  +T':22s}: {crossad_srcc_of(lambda a: a[1]):+.4f}")
print(f"  {'R(1) early retention':22s}: {crossad_srcc_of(lambda a: a[0][1] if len(a[0])>1 else a[0][-1]):+.4f}   (privileged: uses observed R(1))")
print(f"  {'R(3) early retention':22s}: {crossad_srcc_of(lambda a: a[0][min(3,a[1])]):+.4f}   (privileged)")
print(f"  {'mean R[1:min(T,30)]':22s}: {crossad_srcc_of(lambda a: float(np.mean(a[0][1:min(a[1],30)+1]))):+.4f}   (privileged: overall level)")
print(f"  {'SFT ckpt-225 (ref)':22s}: 0.5142   (model, from video)")

print("\n=== (2) RANK STABILITY across time (Spearman of ad-ranking at t vs t+gap) ===")
for gap in (1,5,10):
    per=[]
    for t in range(T_LO,T_HI-gap+1):
        idx=[i for i,(R,T) in enumerate(ads) if T>=t+gap and len(R)>t+gap]
        if len(idx)<3: continue
        a=[ads[i][0][t] for i in idx]; b=[ads[i][0][t+gap] for i in idx]
        rho=spearman(a,b)
        if not np.isnan(rho): per.append(rho)
    print(f"  rank(t) vs rank(t+{gap:2d}): mean Spearman = {np.mean(per):+.4f}   (high=>structured/low-noise)")

print("\n=== (3) does early ranking predict late ranking? (privileged ceiling) ===")
# rank ads by R at t=5; how well does it predict true ranking at each later t?
for anchor in (1,5,10):
    print(f"  rank-by-R({anchor:2d}) -> cross-ad SRCC over t in [1,30] = {crossad_srcc_of(lambda a,k=anchor: a[0][min(k,a[1])]):+.4f}")

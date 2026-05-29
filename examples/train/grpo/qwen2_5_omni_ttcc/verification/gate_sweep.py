"""Sweep beta/sigma for the cross-ad concordance reward: cosine vs SRCC direction + within-group
spread (signal strength). Synthetic/CPU. Tells us which way to tune if the live signal is too weak.
(Synthetic within_grp_std is smaller than the live run's — read the RELATIVE trend across beta/sigma,
and require cosine to stay strongly positive.)"""
from __future__ import annotations
import os, sys
import numpy as np, torch
_HERE=os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0,_HERE)
import reinforce_core as RC, cross_ad_rank_reward as CARR

torch.manual_seed(0); np.random.seed(0)
N,d,T,G,SEEDS = 40,48,30,32,24
T_LO,T_HI = 1,30
H=torch.randn(N,d,dtype=torch.float64)
decay=0.02+0.06*(np.arange(N)/N)
true_curves=[(np.exp(-decay[i]*np.arange(T+1))).tolist() for i in range(N)]
for c in true_curves: c[0]=1.0
W=(0.05*torch.randn(T,d,dtype=torch.float64)).requires_grad_(True)

def policy_curves(Wp): return RC.curve_from_hazards(H@Wp.t())
def soft_rank(x,a=20.0):
    dd=x.unsqueeze(1)-x.unsqueeze(0)
    return torch.sigmoid(a*dd).sum(dim=1)
def pear(a,b):
    a=a-a.mean(); b=b-b.mean(); return (a*b).sum()/(a.norm()*b.norm()+1e-12)
def soft_srcc(Wp):
    R=policy_curves(Wp); Rt=torch.tensor(true_curves,dtype=torch.float64)
    return torch.stack([pear(soft_rank(R[:,t]),soft_rank(Rt[:,t])) for t in range(T_LO,T_HI+1)]).mean()

W.grad=None; (-soft_srcc(W)).backward(); g_srcc=W.grad.detach().clone().flatten()

def reinforce_grad(beta,sigma):
    acc=torch.zeros_like(W).flatten(); wgs=[]; rmeans=[]; deads=[]
    for s in range(SEEDS):
        torch.manual_seed(1000+s)
        mu=H@W.t()
        with torch.no_grad(): meanc=RC.curve_from_hazards(mu).numpy()
        eps=torch.randn(N,G,T,dtype=torch.float64); z=(mu.unsqueeze(1)+sigma*eps); zd=z.detach()
        with torch.no_grad(): curves=RC.curve_from_hazards(zd.reshape(N*G,T)).numpy().reshape(N,G,T+1)
        rew=np.empty((N,G))
        for i in range(N):
            bp=[meanc[j] for j in range(N) if j!=i]; bt=[true_curves[j] for j in range(N) if j!=i]
            rew[i]=CARR.crossad_rank_reward([curves[i,g] for g in range(G)],true_curves[i],bp,bt,T_LO,T_HI,beta)
        rt=torch.tensor(rew,dtype=torch.float64)
        adv=(rt-rt.mean(dim=1,keepdim=True))/(rt.std(dim=1,keepdim=True)+1e-8)
        logp=RC.gaussian_logp(zd.reshape(N*G,T),mu.unsqueeze(1).expand(N,G,T).reshape(N*G,T),sigma).reshape(N,G)
        W.grad=None; (-(logp*adv.detach()).mean()).backward(); acc=acc+W.grad.detach().clone().flatten()
        if s<4:
            wgs.append(float(rt.std(dim=1).mean())); rmeans.append(float(rt.mean())); deads.append(float((rt.std(dim=1)<1e-4).float().mean()))
    return acc/SEEDS, np.mean(wgs), np.mean(rmeans), np.mean(deads)

def cos(a,b): return float((a@b)/(a.norm()*b.norm()+1e-12))
print(f"{'beta':>5} {'sigma':>6} | {'cosine':>8} {'within_std':>11} {'reward':>7} {'dead':>5}")
for beta in (5,10,20,40,80):
    for sigma in (0.2,0.3,0.5):
        g,wg,rm,dd=reinforce_grad(beta,sigma)
        print(f"{beta:>5} {sigma:>6} | {cos(g,g_srcc):>+8.3f} {wg:>11.4f} {rm:>7.3f} {dd:>5.2f}")

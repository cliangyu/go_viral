"""Operating-point gate: does the cross-ad concordance reward's REINFORCE gradient point toward
SRCC *when SRCC is already ~0.5* (ckpt-225's regime)? The original gate tested only a random init
(SRCC~0.04). run-2 collapsed 0.514->0.487->0.387, so test the direction AT the operating point.

Ascend a synthetic head W on soft-SRCC until hard cross-ad SRCC ~= target, then measure
cos(reward-gradient, SRCC-direction) + within-group spread there, for several beta."""
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
Rt_t=torch.tensor(true_curves,dtype=torch.float64)

def policy_curves(Wp): return RC.curve_from_hazards(H@Wp.t())
def soft_rank(x,a=20.0): return torch.sigmoid(a*(x.unsqueeze(1)-x.unsqueeze(0))).sum(dim=1)
def pear(a,b):
    a=a-a.mean(); b=b-b.mean(); return (a*b).sum()/(a.norm()*b.norm()+1e-12)
def soft_srcc(Wp):
    R=policy_curves(Wp)
    return torch.stack([pear(soft_rank(R[:,t]),soft_rank(Rt_t[:,t])) for t in range(T_LO,T_HI+1)]).mean()
def hard_srcc(Wp):
    with torch.no_grad(): R=policy_curves(Wp).numpy()
    Rt=np.array(true_curves); rhos=[]
    for t in range(T_LO,T_HI+1):
        ra=np.argsort(np.argsort(R[:,t])).astype(float); rb=np.argsort(np.argsort(Rt[:,t])).astype(float)
        ra-=ra.mean(); rb-=rb.mean(); rhos.append((ra*rb).sum()/(np.linalg.norm(ra)*np.linalg.norm(rb)+1e-12))
    return float(np.mean(rhos))

def srcc_grad(Wp):
    Wp.grad=None; (-soft_srcc(Wp)).backward(); return Wp.grad.detach().clone().flatten()

def reward_grad(Wp,beta,sigma):
    acc=torch.zeros(Wp.numel(),dtype=torch.float64); wg=[]
    for s in range(SEEDS):
        torch.manual_seed(7000+s)
        mu=H@Wp.t()
        with torch.no_grad(): meanc=RC.curve_from_hazards(mu).numpy()
        eps=torch.randn(N,G,T,dtype=torch.float64); z=(mu.unsqueeze(1)+sigma*eps); zd=z.detach()
        with torch.no_grad(): curves=RC.curve_from_hazards(zd.reshape(N*G,T)).numpy().reshape(N,G,T+1)
        rew=np.empty((N,G))
        for i in range(N):
            bp=[meanc[j] for j in range(N) if j!=i]; bt=[true_curves[j] for j in range(N) if j!=i]
            rew[i]=CARR.crossad_rank_reward([curves[i,g] for g in range(G)],true_curves[i],bp,bt,T_LO,T_HI,beta)
        rtt=torch.tensor(rew,dtype=torch.float64)
        adv=(rtt-rtt.mean(dim=1,keepdim=True))/(rtt.std(dim=1,keepdim=True)+1e-8)
        logp=RC.gaussian_logp(zd.reshape(N*G,T),mu.unsqueeze(1).expand(N,G,T).reshape(N*G,T),sigma).reshape(N,G)
        Wp.grad=None; (-(logp*adv.detach()).mean()).backward(); acc=acc+Wp.grad.detach().clone().flatten()
        if s<6: wg.append(float(rtt.std(dim=1).mean()))
    return acc/SEEDS, float(np.mean(wg))
def cos(a,b): return float((a@b)/(a.norm()*b.norm()+1e-12))

# ascend W on soft-SRCC to several operating points, measure reward-grad cosine there
W=(0.05*torch.randn(T,d,dtype=torch.float64)).requires_grad_(True)
opt=torch.optim.Adam([W],lr=0.004)
targets=[0.35,0.42,0.48,0.54,0.60]; ti=0
print(f"{'SRCC':>6} | {'beta':>4} {'sigma':>5} | {'cos(reward,SRCC)':>16} {'within_std':>11}")
for it in range(4000):
    opt.zero_grad(); loss=-soft_srcc(W); loss.backward(); opt.step()
    if ti<len(targets) and hard_srcc(W)>=targets[ti]:
        sr=hard_srcc(W); gs=srcc_grad(W)
        for beta in (10.0,50.0):
            gr,wgv=reward_grad(W,beta,0.3)
            print(f"{sr:>6.3f} | {beta:>4.0f} {0.3:>5.1f} | {cos(gr,gs):>+16.3f} {wgv:>11.4f}")
        ti+=1
    if ti>=len(targets): break
print("done")

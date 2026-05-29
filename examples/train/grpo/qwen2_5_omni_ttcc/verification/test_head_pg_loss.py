"""Unit test for head_pg_compute_loss.head_pg_loss — verify the compute_loss core
composes correctly (grad flows to mu_z, reward detached, within-group-std reported,
loss finite) using a FAKE mu_z + the real reward/CDF. No model, no GPU. Gate before swift."""
import math, numpy as np, torch
import cross_ad_reward as CAR
import head_pg_compute_loss as H

torch.manual_seed(0); np.random.seed(0)
fails=[]
def check(n,c,x=""):
    print(f"  [{'PASS' if c else 'FAIL'}] {n} {x}");
    if not c: fails.append(n)

Tmax=20
def curve(decay):
    lam=np.full(Tmax,decay); R=np.concatenate([[1.0],np.exp(-np.cumsum(lam))]); return R.tolist()
pop=[curve(d) for d in np.random.uniform(0.05,0.5,40)]
cdf=CAR.build_cdf(pop, Tmax)

def reward_fn(R_hat, R_true, cdf, t_lo=1, t_hi=30):
    return [CAR.r_rank(rh, rt, cdf, t_lo, t_hi) for rh, rt in zip(R_hat, R_true)]

B,G=3,8
mu_z=torch.randn(B,Tmax,requires_grad=True)
mu_ref=torch.zeros(B,Tmax)
R_true_list=[curve(d) for d in [0.1,0.25,0.45]]

loss,m=H.head_pg_loss(mu_z,R_true_list,cdf,G=G,sigma=0.7,kl_coef=0.01,mu_ref=mu_ref,reward_fn=reward_fn)
print("  metrics:",{k:round(v,4) for k,v in m.items()})
loss.backward()
check("loss finite", torch.isfinite(loss).item(), f"(={float(loss):.4f})")
check("grad flows to mu_z", mu_z.grad is not None and torch.isfinite(mu_z.grad).all().item())
check("grad nonzero (signal exists)", float(mu_z.grad.abs().sum())>0)
check("within-group reward std reported >=0", m["reward_within_group_std"]>=0, f"(={m['reward_within_group_std']:.4f})")
check("reward_mean in [0,1]", 0<=m["reward_mean"]<=1, f"(={m['reward_mean']:.4f})")
check("kl >= 0", m["kl"]>=0, f"(={m['kl']:.4f})")

# reward-detach check: loss grad must equal grad with rewards as constants
mu2=mu_z.detach().clone().requires_grad_(True)
loss2,_=H.head_pg_loss(mu2,R_true_list,cdf,G=G,sigma=0.7,kl_coef=0.0,mu_ref=None,reward_fn=reward_fn)
# reproducible: same seed -> same eps? randn differs per call; just assert finite grad (detach correctness covered by reinforce_core T4)
loss2.backward()
check("no-KL variant grad finite", mu2.grad is not None and torch.isfinite(mu2.grad).all().item())

print()
if fails: print(f"==== {len(fails)} FAILED: {fails} ===="); raise SystemExit(1)
print("==== HEAD-PG COMPUTE-LOSS CORE PASSED ====")

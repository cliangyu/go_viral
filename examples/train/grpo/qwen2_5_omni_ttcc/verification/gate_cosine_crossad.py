"""COSINE GATE for run-2's cross-ad ranking reward (must pass before any cluster compute).

Question: does the new reward's REINFORCE gradient on the head W point in the
SRCC-improving direction?  run-1's per-ad R_rank gave cos ~ -0.01 (orthogonal) -> the
run empirically went DOWN (ckpt-25/50/75 all <= baseline). The new cross-ad-concordance
reward must give a STRONGLY POSITIVE cosine, or the design is wrong and we iterate
BEFORE spending GPUs.

Synthetic, CPU, ~seconds. No model, no data, no cluster.
- N synthetic ads: hidden h_i in R^d, head W in R^{Txd}, mu_z_i = W h_i,
  R_i = curve_from_hazards(mu_z_i). True curves have a clear cross-ad order.
- g_srcc   = d/dW [ soft cross-ad Spearman( R(W), R_true ) ]   (the IDEAL ranking direction)
- g_new    = E[ REINFORCE grad ] using crossad_rank_reward (reward detached, grad via logp)
- g_old    = E[ REINFORCE grad ] using run-1's per-ad r_rank (control; expect cos~0)
PASS iff cos(g_new, g_srcc) is strongly positive (>0.3) AND clearly beats cos(g_old, g_srcc).
"""
from __future__ import annotations
import os, sys, math
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import reinforce_core as RC
import cross_ad_reward as CAR            # run-1 reward (control)
import cross_ad_rank_reward as CARR      # run-2 reward (under test)

torch.manual_seed(0); np.random.seed(0)

N   = 40      # ads
d   = 48      # hidden dim
T   = 30      # seconds (t_hi=30 fits exactly)
G   = 32      # rollouts/ad
SIGMA = 0.2
BETA  = 10.0
T_LO, T_HI = 1, 30
SEEDS = 24    # average the REINFORCE estimator over this many rollout draws


# ---- synthetic population: each ad i has decay_i increasing in i => clear cross-ad order ----
H = torch.randn(N, d, dtype=torch.float64)
decay = 0.02 + 0.06 * (np.arange(N) / N)          # ad 0 retains best ... ad N-1 worst
true_curves = []                                   # R_true_i(0..T)
for i in range(N):
    t = np.arange(T + 1)
    c = np.exp(-decay[i] * t); c[0] = 1.0
    true_curves.append(c.tolist())
# CDF for the run-1 r_rank control (built from the true population, per-second)
cdf = CAR.build_cdf(true_curves, T_max=T)

# head W: init so the model is NOT already perfectly ordered (room to move both grads)
W = (0.05 * torch.randn(T, d, dtype=torch.float64)).requires_grad_(True)


def policy_mean_curves(Wp):
    """mu_z_i = W h_i ; R_i = curve_from_hazards(mu_z). Returns (N, T+1) torch, grad-connected."""
    mu = H @ Wp.t()                                # (N, T)
    return RC.curve_from_hazards(mu)               # (N, T+1)


def soft_rank(x, alpha=20.0):
    """Differentiable rank: rank_i ~ 0.5 + sum_j sigmoid(alpha (x_i - x_j))."""
    d_ = x.unsqueeze(1) - x.unsqueeze(0)           # (n,n)
    return 0.5 + torch.sigmoid(alpha * d_).sum(dim=1) - torch.sigmoid(torch.zeros(1, dtype=x.dtype))
    # (the -sigmoid(0)=−0.5 removes the i==j self term; constant, drops out of correlation)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


def soft_srcc(Wp):
    """mean_t soft-Spearman across the N ads at second t (differentiable in W)."""
    R = policy_mean_curves(Wp)                     # (N, T+1)
    Rt = torch.tensor([tc for tc in true_curves], dtype=torch.float64)  # (N, T+1)
    vals = []
    for t in range(T_LO, T_HI + 1):
        pr = soft_rank(R[:, t]); tr = soft_rank(Rt[:, t])
        vals.append(pearson(pr, tr))
    return torch.stack(vals).mean()


def hard_srcc(Wp):
    with torch.no_grad():
        R = policy_mean_curves(Wp).numpy()
    Rt = np.array(true_curves)
    rhos = []
    for t in range(T_LO, T_HI + 1):
        a = R[:, t]; b = Rt[:, t]
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        ra = ra - ra.mean(); rb = rb - rb.mean()
        rhos.append((ra * rb).sum() / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))
    return float(np.mean(rhos))


# ---------- g_srcc : the ideal ranking-improvement direction ----------
W.grad = None
loss_srcc = -soft_srcc(W)            # maximize SRCC => minimize -SRCC
loss_srcc.backward()
g_srcc = W.grad.detach().clone().flatten()


def reinforce_grad(reward_kind):
    """E[ REINFORCE grad on W ] using the given reward. reward detached; grad flows via logp."""
    acc = torch.zeros_like(W).flatten()
    for s in range(SEEDS):
        torch.manual_seed(1000 + s)
        mu = H @ W.t()                               # (N, T) grad-connected
        with torch.no_grad():
            mean_curves = RC.curve_from_hazards(mu).numpy()   # (N, T+1) reference preds (detached)
        # sample G rollouts per ad
        eps = torch.randn(N, G, T, dtype=torch.float64)
        z = (mu.unsqueeze(1) + SIGMA * eps)          # (N, G, T) grad via mu
        z_det = z.detach()
        with torch.no_grad():
            curves = RC.curve_from_hazards(z_det.reshape(N * G, T)).numpy().reshape(N, G, T + 1)
        # rewards (N, G), detached
        rew = np.empty((N, G), dtype=np.float64)
        for i in range(N):
            if reward_kind == 'new':
                # buffer = all OTHER ads' mean preds + true curves
                buf_pred = [mean_curves[j] for j in range(N) if j != i]
                buf_true = [true_curves[j] for j in range(N) if j != i]
                rew[i] = CARR.crossad_rank_reward(
                    [curves[i, g] for g in range(G)], true_curves[i],
                    buf_pred, buf_true, t_lo=T_LO, t_hi=T_HI, beta=BETA)
            else:  # 'old' run-1 per-ad percentile match vs fixed CDF
                rew[i] = np.array([CAR.r_rank(curves[i, g].tolist(), true_curves[i], cdf, T_LO, T_HI)
                                   for g in range(G)])
        rew_t = torch.tensor(rew, dtype=torch.float64)
        adv = (rew_t - rew_t.mean(dim=1, keepdim=True)) / (rew_t.std(dim=1, keepdim=True) + 1e-8)
        # logp of each rollout under N(mu, sigma)  (grad via mu = W H)
        logp = RC.gaussian_logp(z_det.reshape(N * G, T),
                                mu.unsqueeze(1).expand(N, G, T).reshape(N * G, T), SIGMA).reshape(N, G)
        loss = -(logp * adv.detach()).mean()
        W.grad = None
        loss.backward()
        acc = acc + W.grad.detach().clone().flatten()
        # report within-group spread (signal) once
        if s == 0:
            reinforce_grad._wg = float(rew_t.std(dim=1).mean())
            reinforce_grad._dead = float((rew_t.std(dim=1) < 1e-4).float().mean())
            reinforce_grad._rmean = float(rew_t.mean())
    return acc / SEEDS


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


print(f"[gate] N={N} d={d} T={T} G={G} sigma={SIGMA} beta={BETA} seeds={SEEDS}")
print(f"[gate] initial hard cross-ad SRCC of policy = {hard_srcc(W):.4f}  (room to improve to 1.0)")

g_new = reinforce_grad('new')
wg_new, dead_new, rmean_new = reinforce_grad._wg, reinforce_grad._dead, reinforce_grad._rmean
g_old = reinforce_grad('old')
wg_old, dead_old, rmean_old = reinforce_grad._wg, reinforce_grad._dead, reinforce_grad._rmean

c_new = cos(g_new, g_srcc)
c_old = cos(g_old, g_srcc)
print()
print(f"  run-1 reward (per-ad r_rank): reward~{rmean_old:.3f}  within_grp_std={wg_old:.4f}  dead_frac={dead_old:.2f}")
print(f"     cos(g_old, g_srcc) = {c_old:+.4f}   <- expect ~0 (orthogonal; reproduces the failure)")
print(f"  run-2 reward (cross-ad concordance): reward~{rmean_new:.3f}  within_grp_std={wg_new:.4f}  dead_frac={dead_new:.2f}")
print(f"     cos(g_new, g_srcc) = {c_new:+.4f}   <- want STRONGLY POSITIVE (>0.3)")
print()
ok = (c_new > 0.30) and (c_new > c_old + 0.25)
print(f"[gate] {'PASS' if ok else 'FAIL'}: new reward {'aligns with' if ok else 'does NOT align with'} the SRCC direction "
      f"(c_new={c_new:+.3f}, c_old={c_old:+.3f})")
sys.exit(0 if ok else 1)

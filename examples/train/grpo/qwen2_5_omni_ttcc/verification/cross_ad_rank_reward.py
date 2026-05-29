"""Cross-ad RANKING reward for head-PG run-2 (fixes Bug A + Bug B of run-1).

run-1 used R_rank = per-ad percentile-MATCH vs a FIXED CDF (cross_ad_reward.r_rank).
That is a *calibration* reward: it sits at its no-skill floor (~0.78 = constant-median
predictor) and, crucially, the within-ad advantage `(r-mean)/std` subtracts the ad's
level, so the gradient never compares ad A to ad B -> cross-ad SRCC cannot move
(empirically: ckpt-25/50/75 all <= baseline 0.514; V8 cos ~ -0.01).

THE FIX (this module): each rollout of ad A is scored by its *pairwise concordance
against a reference set of OTHER ads* (a detached running buffer of recent ads'
policy-mean curves + true curves). The reward now DEPENDS on cross-ad order, so the
within-ad advantage encodes "which rollout ranks A correctly vs the population" -- a
genuine cross-ad ranking gradient through the EXISTING REINFORCE path (no trainer-core
surgery). This is BPR/RankNet/C-index applied cross-ad, used directly (REINFORCE needs
no differentiable reward). reward = a smooth surrogate of the per-second cross-ad
Spearman the eval (srcc_eval.py) measures -> reward == eval metric (RLVR best practice).

reward_{A,g} = mean_{t in [t_lo,t_hi]} mean_{B in buffer, valid} sigma( beta *
                 (R_Ag(t) - R_B(t)) * sign(R_true_A(t) - R_true_B(t)) )

- valid (t,B): both A and B have a value at second t (len > t) AND sign(...) != 0 (drop ties).
- bounded [0,1]; 0.5 = chance; ->1 as A is ordered correctly vs the whole buffer.
- pure numpy (curves arrive detached as lists, exactly like cross_ad_reward.reward_fn).
"""
from __future__ import annotations
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def crossad_rank_reward(R_hat_rollouts, R_true_self, buf_pred, buf_true,
                        t_lo=1, t_hi=30, beta=10.0, margin=0.0):
    """Rewards (G,) for one ad's G rollout curves vs a reference buffer of other ads.

    R_hat_rollouts : list[G] of curves (each list/array, R(0..T_A))  -- this ad's rollouts
    R_true_self    : curve (R(0..T_A))                               -- this ad's TRUE curve
    buf_pred       : list[M] of curves                               -- buffer ads' policy-mean preds
    buf_true       : list[M] of curves                               -- buffer ads' TRUE curves
    Returns np.ndarray (G,) in [0,1]. If the buffer is empty / no valid pairs -> 0.5 (chance).
    """
    G = len(R_hat_rollouts)
    if not buf_pred:
        return np.full(G, 0.5, dtype=np.float64)
    self_true = np.asarray(R_true_self, dtype=np.float64)
    T_self = len(self_true) - 1
    hi = min(t_hi, T_self)
    # Precompute, per buffer ad B and per second t in [t_lo,hi] valid for BOTH A and B:
    #   s_Bt = sign(R_true_A(t) - R_true_B(t))   (the target order; 0 -> dropped)
    #   p_Bt = R_B(t)                            (B's predicted value at t)
    # then reward_g(t,B) = sigmoid(beta * (R_Ag(t) - p_Bt) * s_Bt).
    cols = []          # list of (t, p_Bt, s_Bt) valid comparisons
    for B_pred, B_true in zip(buf_pred, buf_true):
        B_pred = np.asarray(B_pred, dtype=np.float64)
        B_true = np.asarray(B_true, dtype=np.float64)
        T_B = len(B_true) - 1
        hiB = min(hi, T_B, len(B_pred) - 1)
        for t in range(t_lo, hiB + 1):
            gap = self_true[t] - B_true[t]
            if abs(gap) < margin:
                continue                       # near-tie (|gap|<margin): noisy order -> drop.
                                               # margin=0 keeps all (run-2/3); margin>0 = large-margin
                                               # LtR -> generalizes (Lan et al. 2009), fixes over-opt.
            s = np.sign(gap)
            if s == 0.0:
                continue                       # exact tie -> undefined order, drop
            cols.append((t, B_pred[t], s))
    if not cols:
        return np.full(G, 0.5, dtype=np.float64)
    ts = np.array([c[0] for c in cols], dtype=np.int64)        # (K,)
    p = np.array([c[1] for c in cols], dtype=np.float64)       # (K,)
    s = np.array([c[2] for c in cols], dtype=np.float64)       # (K,)
    out = np.empty(G, dtype=np.float64)
    for g in range(G):
        rg = np.asarray(R_hat_rollouts[g], dtype=np.float64)
        # R_Ag(t) for each comparison's t (t <= T_self guaranteed by hi)
        a = rg[ts]                                              # (K,)
        c = _sigmoid(beta * (a - p) * s)                        # (K,) smooth concordance
        out[g] = float(c.mean())
    return out


def curve_summary(curve, t_lo=1, t_hi=30):
    """A single scalar order-key for buffer/eval convenience: mean retention over [t_lo,t_hi]
    (== area-under-retention, monotone in 'this ad retains better'). Not used by the reward
    above (which is per-second), but handy for diagnostics / a cheaper variant."""
    c = np.asarray(curve, dtype=np.float64)
    hi = min(t_hi, len(c) - 1)
    if hi < t_lo:
        return float(c[-1])
    return float(c[t_lo:hi + 1].mean())

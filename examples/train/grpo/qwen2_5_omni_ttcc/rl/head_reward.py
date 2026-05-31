"""Head-channel reward adapter for CoT-GRPO.

CoT-GRPO's reward must read the retention HEAD conditioned on the generated CoT,
NOT parse a curve out of the text. `cross_ad_reward.composite` takes a TEXT string
and runs `parse_curve` (the abandoned text channel). This adapter is the same math
(`r_rank` cross-ad percentile vs the fixed train CDF + `r_acc` = 1-IBS) but its input
is the head-produced curve R_hat (a numpy array of length T+1), supplied by the
CoT-GRPO trainer subclass after it forwards the policy through the head.

Reward per rollout:  r = beta * R_rank + alpha * R_acc        (+ gamma * R_fmt, applied
separately by the `ttcc_cot_format` ORM on the completion string; KL by the trainer).

R_rank is CROSS-AD by construction: it scores R_hat's percentile against a *shared*
train-population CDF, so the within-group spread encodes cross-ad rank. A per-ad reward
would cancel under GRPO's group-mean-subtracted advantage and could not move cross-ad
SRCC -- do not replace r_rank with a per-ad calibration term.

Pure numpy; no torch, no model. Unit-testable offline.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# co-locate the math import: cross_ad_reward.py lives in ../verification/
_VERI = os.path.join(os.path.dirname(__file__), "..", "verification")
if _VERI not in sys.path:
    sys.path.insert(0, _VERI)

from cross_ad_reward import r_rank, r_acc, build_cdf, percentile  # noqa: E402,F401

# Default weights (COT_RL_PLAN.md): rank is primary; IBS is a calibration anchor so the
# rank objective does not rot calibration the way head-PG did (IBS doubled there).
BETA = 1.0     # R_rank (cross-ad, primary)
ALPHA = 0.25   # R_acc = 1 - IBS (calibration anchor)


def head_reward(
    R_hat,
    R_true,
    cdf,
    *,
    beta: float = BETA,
    alpha: float = ALPHA,
    t_lo: int = 1,
    t_hi: int = 30,
):
    """Per-rollout head-channel reward.

    Args:
        R_hat:  head-produced curve, list/array length T+1 (R_hat[0] should be 1.0).
        R_true: ground-truth curve for THIS ad, length T+1.
        cdf:    dict {t: sorted np.array of train values at second t} (train-only, leak-safe).
        beta, alpha: term weights.
        t_lo, t_hi:  discriminative band for the rank term.

    Returns:
        (reward: float, components: dict) -- components = {'r_rank','r_acc','beta','alpha'}.
    """
    R_hat = np.asarray(R_hat, dtype=np.float64)
    R_true = np.asarray(R_true, dtype=np.float64)
    T = len(R_true) - 1
    rk = r_rank(R_hat, R_true, cdf, t_lo=t_lo, t_hi=t_hi)
    ac = r_acc(R_hat, R_true, T)
    reward = beta * rk + alpha * ac
    return float(reward), {"r_rank": float(rk), "r_acc": float(ac), "beta": beta, "alpha": alpha}


def head_reward_batch(R_hats, R_trues, cdf, **kw):
    """Vectorized convenience: lists of curves -> (rewards list, components list).

    The CoT-GRPO trainer calls this on the B*G rollouts of a step (each R_hat read
    from the head after forwarding the policy on prompt+CoT_g)."""
    rewards, comps = [], []
    for R_hat, R_true in zip(R_hats, R_trues):
        r, c = head_reward(R_hat, R_true, cdf, **kw)
        rewards.append(r)
        comps.append(c)
    return rewards, comps

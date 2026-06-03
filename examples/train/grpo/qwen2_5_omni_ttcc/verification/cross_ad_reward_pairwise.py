"""Production: batch SOFT PAIRWISE cross-ad rank reward (a soft Kendall's-tau).

Replaces the percentile-MATCH `r_rank` with a reward that is the actual rank metric the eval
scores. Pure-python (no numpy/torch) so it runs identically on laptop and box -- no kernel/version
risk. Wire into `CoTGRPOTrainer._compute_rewards_per_func`, which already holds every rollout's R_hat.

Per rollout i of ad a (over band [t_lo,t_hi]):
    reward_i = mean_{b != a} mean_t  sigma( (R_hat_a^i(t) - rep_b(t)) * sign(R_true_a(t) - R_true_b(t)) / tau )
where rep_b = mean predicted curve of ad b's rollouts (stable cross-ad anchor; the true order comes
from R_true so the target is fixed). Variable length handled by t only running where BOTH curves reach.
"""
from __future__ import annotations
import math


def _mean_curve(curves):
    T = max(len(c) for c in curves)
    out = []
    for t in range(T):
        vals = [c[t] for c in curves if len(c) > t]
        out.append(sum(vals) / len(vals) if vals else 0.0)
    return out


def r_pair_soft_batch(R_hats, R_trues, ad_ids, *, t_lo=1, t_hi=30, tau=0.05):
    """Per-rollout soft pairwise cross-ad rank reward.

    Args:
        R_hats:  list of predicted curves (one per rollout in the step), each list[float] len T+1.
        R_trues: list of true curves, same length/order as R_hats (rollouts of one ad share it).
        ad_ids:  list of ad identifiers; rollouts of the same ad share an id.
    Returns:
        list[float] per-rollout rewards in [0,1], aligned to the input order.
    """
    groups = {}
    for idx, a in enumerate(ad_ids):
        groups.setdefault(a, []).append(idx)
    ads = list(groups.keys())
    rep = {a: _mean_curve([R_hats[i] for i in groups[a]]) for a in ads}
    true_of = {a: R_trues[groups[a][0]] for a in ads}

    rewards = [0.5] * len(R_hats)
    for a in ads:
        Rt_a = true_of[a]
        others = [b for b in ads if b != a]
        for i in groups[a]:
            Rh_a = R_hats[i]
            terms = []
            for b in others:
                Rh_b, Rt_b = rep[b], true_of[b]
                hi = min(t_hi, len(Rh_a) - 1, len(Rh_b) - 1, len(Rt_a) - 1, len(Rt_b) - 1)
                for t in range(t_lo, hi + 1):
                    st = 1.0 if Rt_a[t] > Rt_b[t] else (-1.0 if Rt_a[t] < Rt_b[t] else 0.0)
                    if st == 0.0:
                        continue
                    terms.append(1.0 / (1.0 + math.exp(-(Rh_a[t] - Rh_b[t]) * st / tau)))
            if terms:
                rewards[i] = sum(terms) / len(terms)
    return rewards


def _selftest():
    # 3 ads, true order A>B>C by retention. 2 rollouts each.
    A_t = [1.0, 0.80, 0.65, 0.55]
    B_t = [1.0, 0.70, 0.55, 0.45]
    C_t = [1.0, 0.60, 0.45, 0.35]
    # rollouts: ad A has one curve that RANKS RIGHT (high) and one that ranks WRONG (too low, below C)
    A_good = [1.0, 0.81, 0.66, 0.56]
    A_bad  = [1.0, 0.55, 0.40, 0.30]      # predicts A as the lowest -> wrong order
    B_p    = [1.0, 0.70, 0.55, 0.45]
    C_p    = [1.0, 0.60, 0.45, 0.35]
    R_hats = [A_good, A_bad, B_p, B_p, C_p, C_p]
    R_trues= [A_t,    A_t,   B_t, B_t, C_t, C_t]
    ids    = ["A",    "A",   "B", "B", "C", "C"]
    r = r_pair_soft_batch(R_hats, R_trues, ids, t_hi=3)
    print("rewards:", [f"{x:.3f}" for x in r])
    assert r[0] > r[1], f"correct-order rollout should beat wrong-order: {r[0]:.3f} vs {r[1]:.3f}"
    assert 0.0 <= min(r) and max(r) <= 1.0
    print(f"PASS: A_good {r[0]:.3f} > A_bad {r[1]:.3f}; reward correctly rewards correct cross-ad order.")


if __name__ == "__main__":
    _selftest()

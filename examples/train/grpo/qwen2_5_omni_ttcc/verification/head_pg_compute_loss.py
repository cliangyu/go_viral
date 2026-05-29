"""Head-PG compute_loss core (the novel logic of the custom Trainer).

Design (NORTH_STAR §10): one forward per ad -> h_anchor -> mu_z = head.linear(h_anchor)
-> sample G hazard-noised z' -> curves -> cross_ad_reward -> within-ad advantage
-> REINFORCE loss + KL. G is memory-free (shared forward). Reward/advantage DETACHED.

This is written framework-light so it can be unit-tested with a fake model/batch,
then dropped into a Seq2SeqTrainer subclass's compute_loss (swift) which provides
the real forward + distributed + optimizer. Functions reuse reinforce_core + cross_ad_reward.
"""
from __future__ import annotations
import torch
import reinforce_core as RC


def head_pg_loss(mu_z, R_true_list, cdf, *, G, sigma, kl_coef, mu_ref=None,
                 reward_fn, t_lo=1, t_hi=30):
    """
    mu_z:        (B, Tmax) hazard logits from head.linear(h_anchor), requires_grad.
    R_true_list: list of B true curves (lists) — for the reward.
    cdf:         train-population CDF (dict t -> sorted np array) for percentile reward.
    G:           rollouts per ad. sigma: exploration std. kl_coef: KL weight.
    mu_ref:      (B, Tmax) reference hazards (frozen SFT) for KL; None -> no KL.
    reward_fn:   callable(R_hat_list, R_true_list, cdf) -> tensor (B,) of rewards in [0,1].
    Returns (loss, metrics dict).
    """
    B, Tmax = mu_z.shape
    device = mu_z.device
    # Expand each ad into G rollouts: sample z' = mu + sigma*eps  (eps detached -> action)
    mu_rep = mu_z.unsqueeze(1).expand(B, G, Tmax)                  # (B,G,Tmax) shares the ONE forward graph
    eps = torch.randn(B, G, Tmax, device=device)
    z_samp = (mu_rep + sigma * eps).detach()                      # action, detached
    # Curves from sampled hazards (detached -> for reward only)
    curves = RC.curve_from_hazards(z_samp.view(B * G, Tmax))      # (B*G, Tmax+1)
    # Reward per rollout (cross-ad percentile vs train CDF). Build R_hat lists.
    R_hat = [curves[i].tolist() for i in range(B * G)]
    R_true_rep = [R_true_list[b] for b in range(B) for _ in range(G)]
    rewards = reward_fn(R_hat, R_true_rep, cdf, t_lo=t_lo, t_hi=t_hi)  # (B*G,)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    # Within-ad-group advantage (group = one ad's G rollouts)
    adv = RC.group_advantage(rewards, group_size=G)               # (B*G,) detached returns
    # REINFORCE: logpi(z'|mu) summed over hazards, * advantage. mu_rep carries the grad.
    logp = RC.gaussian_logp(z_samp.view(B * G, Tmax), mu_rep.reshape(B * G, Tmax), sigma)  # (B*G,)
    pg = -(logp * adv.detach()).mean()
    kl = torch.tensor(0.0, device=device)
    if mu_ref is not None and kl_coef > 0:
        kl = RC.kl_to_ref(mu_z, mu_ref, sigma)
    loss = pg + kl_coef * kl
    metrics = {
        "pg_loss": float(pg.detach()),
        "kl": float(kl.detach()),
        "reward_mean": float(rewards.mean()),
        "reward_within_group_std": float(rewards.view(B, G).std(dim=1).mean()),  # the make-or-break signal
        "adv_abs_mean": float(adv.abs().mean()),
    }
    return loss, metrics

"""Core head-PG (REINFORCE) math, isolated for verification (per RL_DESIGN/NORTH_STAR).

Policy: per-second hazard logits z ~ N(mu_z, sigma^2) (diagonal Gaussian on the
UNCONSTRAINED hazards). Curve = exp(-cumsum(softplus(z))) — monotone by construction.
Loss = -(logpi(z'; mu_z, sigma) * A.detach()).mean()  + kl_coef * KL(pi||ref).
Reward & advantage & sampled action are DETACHED — gradient flows only through mu_z.

Kept deliberately minimal; verified by test_reinforce_core.py before any swift integration.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def sample_hazards(mu_z: torch.Tensor, sigma: float):
    """z' = mu_z + sigma*eps. Returns (z_sample[detached as action], eps)."""
    eps = torch.randn_like(mu_z)
    z = mu_z + sigma * eps
    return z, eps


def curve_from_hazards(z: torch.Tensor) -> torch.Tensor:
    """Hazards -> monotone curve. R(t)=exp(-cumsum(softplus(z))). Prepend R(0)=1."""
    lam = F.softplus(z)                                  # (..., T) >= 0
    R = torch.exp(-torch.cumsum(lam, dim=-1))            # (..., T) in (0,1]
    one = torch.ones(*R.shape[:-1], 1, dtype=R.dtype, device=R.device)
    return torch.cat([one, R], dim=-1)                   # (..., T+1)


def gaussian_logp(z: torch.Tensor, mu_z: torch.Tensor, sigma: float) -> torch.Tensor:
    """Sum-over-dims diagonal-Gaussian log prob of z under N(mu_z, sigma^2). Shape (...,)."""
    var = sigma * sigma
    return (-0.5 * ((z - mu_z) ** 2) / var - 0.5 * math.log(2 * math.pi * var)).sum(dim=-1)


def group_advantage(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """GRPO-style within-group normalized advantage. rewards: (N*G,) grouped by ad (G each)."""
    r = rewards.view(-1, group_size)
    a = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-8)
    return a.view(-1)


def reinforce_loss(mu_z: torch.Tensor, z_sample: torch.Tensor, advantages: torch.Tensor,
                   sigma: float) -> torch.Tensor:
    """Policy-gradient loss. z_sample & advantages MUST be detached (actions/returns).
    Gradient flows only through mu_z via logpi."""
    logp = gaussian_logp(z_sample.detach(), mu_z, sigma)          # (B,)
    return -(logp * advantages.detach()).mean()


def kl_to_ref(mu_z: torch.Tensor, mu_ref: torch.Tensor, sigma: float) -> torch.Tensor:
    """KL(N(mu_z,sigma) || N(mu_ref,sigma)) for fixed diagonal sigma = sum (mu-mu_ref)^2/(2 sigma^2)."""
    return ((mu_z - mu_ref.detach()) ** 2 / (2 * sigma * sigma)).sum(dim=-1).mean()

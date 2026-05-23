"""
RetentionVLM wrapper (§2 of the milestone report).

Qwen2.5-Omni-3B Thinker trunk + softplus hazard head reading the hidden state
at the closing CoT marker </cot>. The retention curve is recovered by
exponentiated cumulative hazard:
    R_hat(t) = exp(-sum_{s<=t} lambda_hat(s)).

The hazard head outputs a fixed-length 60-vector; the loss is masked to per-ad T_i.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2_5OmniForConditionalGeneration


T_MAX = 60
CLOSE_COT = "</cot>"


def find_close_cot_token_ids(tokenizer) -> list[int]:
    """Tokenize the literal '</cot>' string and return its token id sequence."""
    ids = tokenizer.encode(CLOSE_COT, add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer produced empty ids for {CLOSE_COT!r}")
    return ids


def locate_close_cot_positions(input_ids: torch.Tensor,
                               close_ids: list[int]) -> torch.Tensor:
    """For each row in input_ids, return the index of the LAST token of the
    LAST '</cot>' occurrence. Falls back to the last non-pad token (caller's
    responsibility to pass attention_mask if needed).

    input_ids: (B, L)
    close_ids: list of ints, length k

    Returns: (B,) long tensor of positions.
    """
    B, L = input_ids.shape
    k = len(close_ids)
    close_t = torch.tensor(close_ids, device=input_ids.device, dtype=input_ids.dtype)
    # Sliding-window equality: (B, L - k + 1) bool, True where the k-gram matches.
    if L < k:
        return torch.full((B,), L - 1, device=input_ids.device, dtype=torch.long)
    windows = input_ids.unfold(dimension=1, size=k, step=1)         # (B, L-k+1, k)
    match = (windows == close_t).all(dim=-1)                        # (B, L-k+1)
    positions = torch.full((B,), -1, device=input_ids.device, dtype=torch.long)
    for b in range(B):
        idxs = match[b].nonzero(as_tuple=False).flatten()
        if idxs.numel() > 0:
            positions[b] = idxs[-1].item() + (k - 1)                # end of the </cot> span
    # Fallback for rows with no match: last token.
    positions = torch.where(positions >= 0, positions,
                            torch.full_like(positions, L - 1))
    return positions


class RetentionVLM(nn.Module):
    """Trunk + hazard head. The LM head stays on the trunk for CoT cross-entropy."""

    T_max = T_MAX

    def __init__(self, model_id_or_trunk, tokenizer=None):
        """Accepts either a HF model id (str) or a pre-built trunk module.
        Inference scripts pass a pre-built trunk to avoid re-downloading the
        base weights every call."""
        super().__init__()
        if isinstance(model_id_or_trunk, str):
            self.trunk = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_id_or_trunk, torch_dtype=torch.bfloat16,
            )
            model_id = model_id_or_trunk
        else:
            self.trunk = model_id_or_trunk
            model_id = "Qwen/Qwen2.5-Omni-3B"  # for AutoTokenizer fallback
        if hasattr(self.trunk, "disable_talker"):
            self.trunk.disable_talker()
        # In Qwen2.5-Omni the LM hidden size is at thinker.text_config.hidden_size.
        thinker = getattr(self.trunk, "thinker", self.trunk)
        thinker_cfg = thinker.config
        d = getattr(getattr(thinker_cfg, "text_config", thinker_cfg), "hidden_size",
                    getattr(thinker_cfg, "hidden_size", None))
        if d is None:
            raise RuntimeError(f"could not locate hidden_size on {type(thinker_cfg).__name__}")
        # Keep the hazard head in fp32: it's tiny (d*60 params) and the
        # downstream cumsum/exp for R_hat needs the precision to avoid bf16
        # rounding error accumulating over T_i seconds.
        self.hazard_head = nn.Linear(d, T_MAX, dtype=torch.float32)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._close_cot_ids = find_close_cot_token_ids(tokenizer)

    def forward(self, mm_inputs: dict):
        """
        Returns:
          logits:     (B, L, V) for CoT cross-entropy.
          lam_hat:    (B, T_max) >= 0, hazard at </cot> position.
        """
        # The outer Omni wrapper has no implemented forward; route directly
        # into the thinker (the LM-with-MM-encoders submodule).
        out = self.trunk.thinker(**mm_inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]                                   # (B, L, d)
        input_ids = mm_inputs["input_ids"]
        eoc_idx = locate_close_cot_positions(input_ids, self._close_cot_ids)
        h_eoc = h[torch.arange(h.size(0), device=h.device), eoc_idx]   # (B, d)
        lam_hat = F.softplus(self.hazard_head(h_eoc.float()))       # (B, T_max), fp32, >= 0
        return out.logits, lam_hat


def recover_curve(lam_hat: torch.Tensor, T_i: int) -> torch.Tensor:
    """lam_hat: (60,) or (B, 60); returns R_hat of shape (T_i+1,) per row.

    R_hat(0) = 1; R_hat(t) = exp(-sum_{s=1..t} lam_hat[s-1]).
    """
    if lam_hat.dim() == 1:
        lam = lam_hat[:T_i].float()
        tail = torch.exp(-torch.cumsum(lam, dim=0))
        return torch.cat([torch.ones(1, device=lam.device), tail])
    raise ValueError("recover_curve expects a 1-D hazard vector; batch externally.")


def masked_hazard_log_mse(lam_hat: torch.Tensor,
                          lam_true: torch.Tensor,
                          T_i: torch.Tensor,
                          eps_hat: float = 1e-6,
                          floor_true: float = 1e-3,
                          strict_spec: bool = False) -> torch.Tensor:
    """Per-batch masked log-hazard MSE (§3).

    Two modes:
      - strict_spec=False (default): mask = (t < T_i) AND (lam_true > floor_true).
        The second condition drops seconds where retention is flat (lam_true ~ 0)
        to avoid pushing the model toward log(eps) ~ -13.8 on stable segments.
        Empirically helps training stability. NOT in the spec.
      - strict_spec=True: mask = (t < T_i) only, as written in exp_plan §3.
        L_hazard_i = sum((log lam_hat - log lam_true_safe)^2 * mask) / T_i.
        lam_true is floor-clamped to eps_hat before log.

    lam_hat:  (B, 60)  softplus output, > 0.
    lam_true: (B, 60)  ground-truth hazards (only first T_i entries valid).
    T_i:      (B,)     long tensor, per-ad duration.
    """
    B, T = lam_hat.shape
    arange = torch.arange(T, device=lam_hat.device).unsqueeze(0).expand(B, T)
    duration_mask = arange < T_i.unsqueeze(1)
    if strict_spec:
        mask = duration_mask.to(lam_hat.dtype)
        denom = T_i.to(lam_hat.dtype).clamp(min=1.0)
    else:
        informative = lam_true > floor_true
        mask = (duration_mask & informative).to(lam_hat.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
    lam_hat_safe = lam_hat.clamp(min=eps_hat)
    lam_true_safe = lam_true.clamp(min=eps_hat)
    sq = (torch.log(lam_hat_safe) - torch.log(lam_true_safe)) ** 2
    per_ad = (sq * mask).sum(dim=1) / denom
    return per_ad.mean()


def hazards_from_retention(R_true: torch.Tensor, T_i: int,
                           eps: float = 1e-6) -> torch.Tensor:
    """Convert a retention curve R(0..T_i) to ground-truth hazards of shape (60,).

    lam[t] = log R[t-1] - log R[t], for t = 1..T_i. Positions t >= T_i are
    zero. Positions where R is flat (lam ~ 0) are also kept as their true
    near-zero value; masked_hazard_log_mse drops them via floor_true.
    """
    lam = torch.zeros(T_MAX, dtype=torch.float32)
    R = R_true[:T_i + 1].float().clamp(min=eps)
    lam_t = torch.log(R[:-1]) - torch.log(R[1:])                    # (T_i,)
    lam_t = lam_t.clamp(min=0.0)                                    # numerical: hazards non-neg
    lam[:T_i] = lam_t
    return lam

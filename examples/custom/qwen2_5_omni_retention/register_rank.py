# Copyright (c) ModelScope Contributors. All rights reserved.
"""Retention-curve head plugin for Qwen2.5-Omni in ms-swift.

Registers a model_type ``qwen2_5_omni_retention`` that wraps Qwen2.5-Omni-3B
with a small head producing a 60-dim per-second retention curve R(t). The
head reads the hidden state at a configurable anchor position (the last
``</cot>`` token, falling back to the last input token).

Two head architectures are selectable by env var or YAML ``head_type``:

  hazard   : softplus(Linear(h)) -> lambda(t); R(t) = exp(-cumsum(lambda)).
             Monotone non-increasing by construction. Matches §6 of the
             milestone (SFT-Hazard+CoT). Reference: DeepHit (Lee et al.,
             AAAI 2018); SurvTRACE (Wang & Sun, CHIL 2022); pycox.

  sigmoid  : sigmoid(Linear(h)) per second; no monotone prior. Matches §5
             of the milestone (SFT-MSE / per-second sigmoid).

Both heads keep their final Linear in fp32 because the downstream cumsum
plus exp for R(t) accumulates bf16 rounding error over up to 60 steps.

Loading: ``swift sft --external_plugins examples/custom/qwen2_5_omni_retention/register.py``
Selection: ``--model_type qwen2_5_omni_retention --loss_type retention_loss``
Head choice: ``--head_type {hazard,sigmoid}`` exposed via the
``RETENTION_HEAD_TYPE`` env var or in YAML.

Architecture provenance: the hazard-head design was first prototyped in
wanjia/main:cs224r_project/baselines/retention_vlm.py against raw HF
Trainer. This file is the ms-swift-native port of that architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional

# =====================================================================
# CRITICAL: transformers v4.56.x + FA3 + Qwen2.5-Omni TMRoPE bug
# =====================================================================
# Qwen2.5-Omni uses 3D position_ids shape [3, batch, seq] (time/H/W).
# transformers' _is_packed_sequence() indexes position_ids.shape[1]
# blindly and falsely returns True -> prepare_fa_kwargs_from_position_ids
# generates a wrong-sized cu_seqlens -> FA3's strict 1D check raises
# RuntimeError("cu_seqlens_q must have 1 dimensions, got 3").
#
# FA2 silently allowed the OOB read (this is why V7's FA2 run did not
# crash -- but the attention values were mathematically incorrect).
#
# The fix is from transformers PR #44911 (closed unmerged):
#   if position_ids.dim() > 2: return False
#
# We monkey-patch on plugin import so it applies before any model load.
# See INCIDENT_2026-05-28_AUDIO_OOB.md and FLASH_ATTENTION_INSTALL.md.
try:
    import transformers.modeling_flash_attention_utils as _fa_utils
    _orig_is_packed = _fa_utils._is_packed_sequence
    def _patched_is_packed(position_ids, batch_size):
        if position_ids is not None and position_ids.dim() > 2:
            return False
        return _orig_is_packed(position_ids, batch_size)
    _fa_utils._is_packed_sequence = _patched_is_packed
except (ImportError, AttributeError):
    pass


# =====================================================================
# Wire RetentionLoss sub-losses into swift's custom_metrics pipeline so
# wandb/tensorboard see loss_curve / loss_cot / cot_alpha separately.
#
# Why this exists:
#   RetentionLoss returns total = loss_curve + alpha * loss_cot. swift's
#   trainer logs only `loss` (the total) by default. Without this patch,
#   we cannot see which term is driving the total down — bad for
#   diagnosing whether retention head is learning vs LM head is drifting.
#
# How this works (matches swift's own aux_loss pattern at
# swift/trainers/seq2seq_trainer.py:141-142):
#   1. RetentionLoss already stashes loss_curve/loss_cot/cot_alpha on the
#      model holder (see line ~688 below).
#   2. After every compute_loss, we read them off the holder and call
#      self.custom_metrics[mode][name].update(value).
#   3. swift's log() at trainers/mixin.py:980-985 auto-merges these into
#      the logs dict via compute_custom_metrics() and routes them to
#      wandb/tensorboard with prefix='' for train, 'eval_' for eval.
#
# This is the same pipeline transformers uses for mixture-of-experts
# auxiliary loss. No callback registration, no new infrastructure.
# =====================================================================
try:
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
    _orig_compute_loss = Seq2SeqTrainer.compute_loss

    def _compute_loss_with_retention_metrics(self, model, inputs,
                                             return_outputs=False,
                                             num_items_in_batch=None):
        result = _orig_compute_loss(self, model, inputs,
                                    return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
        try:
            base = self.accelerator.unwrap_model(model)
            for attr in ('base_model', 'model'):
                inner = getattr(base, attr, None)
                if inner is not None and getattr(inner, '_retention_h_holder', None) is not None:
                    base = inner
                    break
            holder = getattr(base, '_retention_h_holder', None)
            if holder is not None:
                mode = 'train' if self.model.training else 'eval'
                if holder.loss_curve is not None:
                    self.custom_metrics[mode]['loss_curve'].update(holder.loss_curve)
                if holder.loss_cot is not None:
                    self.custom_metrics[mode]['loss_cot'].update(holder.loss_cot)
                if holder.cot_alpha is not None:
                    self.custom_metrics[mode]['cot_alpha'].update(holder.cot_alpha)
        except (AttributeError, KeyError):
            # Holder/model structure changed; metrics won't log but training continues.
            pass
        return result

    Seq2SeqTrainer.compute_loss = _compute_loss_with_retention_metrics
except (ImportError, AttributeError):
    pass


from swift.loss import BaseLoss, loss_map
from swift.model import (Model, ModelGroup, ModelLoader, ModelMeta, MultiModelKeys,
                         register_model, register_model_arch)
from swift.template import register_template
from swift.template.templates.qwen import Qwen2_5OmniTemplate, QwenTemplateMeta
from swift.utils import get_env_args, get_logger

logger = get_logger()

T_MAX = 60
CLOSE_COT = '</cot>'
OPEN_COT = '<cot>'
RETENTION_RANK_LAMBDA = get_env_args('RETENTION_RANK_LAMBDA', str, '0.0')
RETENTION_RANK_LOSS   = get_env_args('RETENTION_RANK_LOSS',   str, 'none')   # 'none'|'pairwise'|'pointwise'|'both'
RETENTION_RANK_BETA   = get_env_args('RETENTION_RANK_BETA',   str, '10.0')
RANK_GRADLOG_EVERY    = get_env_args('RANK_GRADLOG_EVERY',    str, '0')


def _retention_rank_enabled() -> bool:
    return RETENTION_RANK_LOSS != 'none' and float(RETENTION_RANK_LAMBDA) > 0


# ---- Helpers ------------------------------------------------------------

def _find_close_cot_token_ids(tokenizer) -> list[int]:
    """Tokenize the literal ``</cot>`` string and return its token id sequence.

    The sequence may be 1 or 2 ids depending on tokenizer merges; we match
    on the full span at inference time.
    """
    ids = tokenizer.encode(CLOSE_COT, add_special_tokens=False)
    if not ids:
        raise ValueError(f'tokenizer produced empty ids for {CLOSE_COT!r}')
    return ids


def _locate_anchor_positions(input_ids: torch.Tensor,
                             close_ids: list[int]) -> torch.Tensor:
    """Per row in input_ids, return the index of the last token of the LAST
    ``</cot>`` occurrence. Rows without ``</cot>`` fall back to the last
    position (works for no-CoT variants which use last-input-token anchor).

    input_ids : (B, L)
    close_ids : list[int], length k
    returns   : (B,) long
    """
    B, L = input_ids.shape
    k = len(close_ids)
    if L < k:
        return torch.full((B,), L - 1, device=input_ids.device, dtype=torch.long)
    close_t = torch.tensor(close_ids, device=input_ids.device, dtype=input_ids.dtype)
    windows = input_ids.unfold(dimension=1, size=k, step=1)         # (B, L-k+1, k)
    match = (windows == close_t).all(dim=-1)                         # (B, L-k+1)
    positions = torch.full((B,), -1, device=input_ids.device, dtype=torch.long)
    for b in range(B):
        idxs = match[b].nonzero(as_tuple=False).flatten()
        if idxs.numel() > 0:
            positions[b] = idxs[-1].item() + (k - 1)
    positions = torch.where(positions >= 0, positions,
                            torch.full_like(positions, L - 1))
    return positions


def _find_open_cot_token_ids(tokenizer) -> list[int]:
    """Token id-sequence for the literal '<cot>' open marker (1+ ids; BPE)."""
    ids = tokenizer.encode(OPEN_COT, add_special_tokens=False)
    if not ids:
        raise ValueError(f'tokenizer produced empty ids for {OPEN_COT!r}')
    return ids


def _locate_open_positions(input_ids, open_ids, close_first_idx):
    """Per row, index of the LAST id of the last '<cot>' window whose end is
    strictly before the matching '</cot>' first id. Returns -1 if none found."""
    B, L = input_ids.shape
    k = len(open_ids)
    out = torch.full((B,), -1, device=input_ids.device, dtype=torch.long)
    if L < k:
        return out
    open_t = torch.tensor(open_ids, device=input_ids.device, dtype=input_ids.dtype)
    windows = input_ids.unfold(dimension=1, size=k, step=1)   # (B, L-k+1, k)
    match = (windows == open_t).all(dim=-1)                    # (B, L-k+1)
    for b in range(B):
        idxs = match[b].nonzero(as_tuple=False).flatten()
        if idxs.numel() == 0:
            continue
        end_idx = idxs + (k - 1)                               # last id of each open window
        valid = end_idx[end_idx < int(close_first_idx[b].item())]
        if valid.numel() > 0:
            out[b] = valid[-1]
    return out


def _build_cot_span_mask(input_ids, open_ids, close_ids):
    """Boolean mask (B, L), True ONLY on tokens strictly between the last
    '<cot>' and the matching '</cot>' (the generated reasoning span). Excludes
    prompt, video placeholders, the markers, padding, scaffolding.

    Returns (mask, anchor_idx) where anchor_idx is the legacy last-token anchor
    (last id of '</cot>', or L-1) for the empty-span fallback."""
    B, L = input_ids.shape
    anchor_idx = _locate_anchor_positions(input_ids, close_ids)            # (B,) last id of '</cot>' (or L-1)
    kc = len(close_ids)
    close_first = (anchor_idx - (kc - 1)).clamp(min=0)                     # first id of '</cot>'
    open_last = _locate_open_positions(input_ids, open_ids, close_first)   # (B,) or -1
    pos = torch.arange(L, device=input_ids.device).unsqueeze(0)           # (1, L)
    have_span = open_last >= 0
    lo = (open_last + 1).clamp(min=0).unsqueeze(1)                         # span start (inclusive)
    hi = close_first.unsqueeze(1)                                          # span end (exclusive)
    mask = (pos >= lo) & (pos < hi) & have_span.unsqueeze(1)               # (B, L) bool
    return mask, anchor_idx


# ---- Retention head -----------------------------------------------------

class RetentionHead(nn.Module):
    """Small head producing R(t) of length T_MAX from a single hidden state.

    head_type:
      hazard  : softplus(Linear(h)) -> lambda(t); R(t) = exp(-cumsum(lam)).
      sigmoid : sigmoid(Linear(h)) per second; no monotone prior.
    """

    def __init__(self, hidden_size: int, head_type: str = 'hazard',
                 t_max: int = T_MAX):
        super().__init__()
        if head_type not in ('hazard', 'sigmoid'):
            raise ValueError(f'head_type must be hazard or sigmoid, got {head_type!r}')
        self.head_type = head_type
        self.t_max = t_max
        # fp32 head: bf16 rounding error compounds over the cumsum.
        self.linear = nn.Linear(hidden_size, t_max, dtype=torch.float32)
        # Init the bias so the head starts in a sensible regime, not vanished.
        # For hazard: softplus(-3) ~= 0.05  ->  cumsum at T=60 ~= 3  ->  R(60) ~= 0.05
        # instead of the default exp(-42) ~= 0 that triggers the eps clamp and
        # makes the log-hazard MSE / grad_norm explode at step 0. This is the
        # standard trick for monotone hazard heads at init.
        if head_type == 'hazard':
            nn.init.constant_(self.linear.bias, -3.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h : (B, hidden_size)
        # DeepSpeed's bf16 mode casts our Linear's weights to bf16 even though
        # we constructed it as fp32 (Linear(..., dtype=torch.float32)). Match
        # the actual weight dtype for the matmul, then upcast to fp32 *after*
        # so the downstream cumsum + exp stays numerically stable.
        w_dtype = self.linear.weight.dtype
        z = self.linear(h.to(w_dtype)).float()
        # Expose the pre-activation logits z (= the policy mean mu_z for the hazard head) so the
        # RL trainer can read mu_z DIRECTLY rather than reconstructing it via the ill-conditioned
        # softplus^{-1}(R) inversion (head_pg_trainer Bug A: the old inversion zeroed the gradient
        # at flat seconds via clamp_min and its 1/lam Jacobian blew up as R(t)->R(t-1)).
        self._last_z = z
        if self.head_type == 'hazard':
            lam = F.softplus(z)                                       # (B, T) >= 0
            return torch.exp(-torch.cumsum(lam, dim=-1))              # (B, T) in (0, 1]
        # sigmoid
        return torch.sigmoid(z)


class RetentionRankHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, m):
        return torch.sigmoid(self.a.float() * m.float() + self.b.float())


# ---- Widened readout (V2): learned attention-pool over the CoT span ------

class CoTAttnPool(nn.Module):
    """k=1 learned cross-attention pool over the masked CoT span (NV-Embed
    latent-attention / PMA k=1). One learnable seed query attends over the
    CoT-span hidden states -> one pooled vector of size d. fp32 (feeds the
    head's cumsum/exp chain), 8 heads, ~16.79M params at d=2048."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, hidden_size, dtype=torch.float32))
        nn.init.normal_(self.query, std=0.02)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,
                                         batch_first=True, dtype=torch.float32)

    def forward(self, h_last: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h_last: (B, L, d); mask: (B, L) bool, True on CoT-span tokens.
        w_dtype = self.mha.in_proj_weight.dtype
        h = h_last.to(w_dtype)
        q = self.query.to(w_dtype).expand(h.size(0), -1, -1)           # (B, 1, d)
        key_padding_mask = ~mask                                       # True = IGNORE (non-CoT-span)
        pooled, _ = self.mha(q, h, h, key_padding_mask=key_padding_mask,
                             need_weights=False)                       # (B, 1, d)
        return pooled[:, 0, :].float()                                 # (B, d)


# ---- Model wrapper (subclass of the stock model) ------------------------

class _RetentionWrapperState:
    """Mutable per-instance state stored on the wrapped model.

    We avoid PretrainedConfig pollution; anchor token ids and head are
    attached as Python attrs at load time.
    """


class _HiddenStateHolder:
    """Per-instance side-channel state used by the retention wrapper.

    - ``last``: final-layer hidden state captured by a forward_pre_hook on
      ``model.thinker.lm_head``. Bypasses ``output_hidden_states``, which
      has issues with transformers >=4.5 ``@capture_outputs`` + gradient
      checkpointing.
    - ``input_ids``: raw token ids captured by a forward_pre_hook on
      ``model``. ms-swift's post_encode_hook converts ``input_ids`` to
      ``inputs_embeds`` before the wrapped forward runs, so the wrapper
      can no longer find the anchor token from ``kwargs['input_ids']``.
      Our pre_hook is registered at model-load time (before swift's),
      so PyTorch fires it first and we get the pre-conversion value.
    """
    __slots__ = ('last', 'input_ids', 'r_true', 'r_mask', 'r_pred', 'z',
                 'loss_curve', 'loss_cot', 'cot_alpha')
    __slots__ = __slots__ + ('p_rank', 'loss_rank')

    def __init__(self):
        self.last = None
        self.input_ids = None
        self.r_true = None
        self.r_mask = None
        self.p_rank = None
        self.r_pred = None
        self.z = None
        self.loss_curve = None
        self.loss_cot = None
        self.cot_alpha = None
        self.loss_rank = None


def _make_lm_head_capture_hook(holder: '_HiddenStateHolder'):
    """forward_pre_hook on lm_head: input[0] is (B, L, d) — the final hidden state."""
    def _hook(module, args, kwargs):
        h = args[0] if args else kwargs.get('input')
        holder.last = h
    return _hook


def _make_retention_forward(original_forward, head: RetentionHead,
                            open_ids, anchor_ids, holder,
                            readout='last_token', attn_pool=None):
    """Patch the base model's forward so it also computes r_pred.

    readout: 'last_token' (default, byte-identical to V1) | 'mean' | 'attn_pool'.
    The hazard/sigmoid head is unchanged; only its INPUT vector h changes."""

    def forward(self, *args, r_true=None, r_mask=None, **kwargs):
        # The lm_head forward_pre_hook captures the final hidden state into
        # holder.last during this call. No need to ask for output_hidden_states.
        holder.last = None
        out = original_forward(*args, **kwargs)
        h_last = holder.last
        if h_last is None:
            raise RuntimeError(
                'RetentionWrapper: lm_head forward_pre_hook did not fire; '
                'check that thinker.lm_head is the right capture point.')
        # input_ids / r_true / r_mask were stashed by our template's
        # _post_encode before swift's pre_forward_hook stripped them.
        input_ids = holder.input_ids if holder.input_ids is not None else kwargs.get('input_ids')
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if r_true is None:
            r_true = holder.r_true
        if r_mask is None:
            r_mask = holder.r_mask
        B = h_last.size(0)
        ar = torch.arange(B, device=h_last.device)
        if readout == 'last_token':
            # EXACT V1 behavior — default; opt-in required to change.
            anchor_idx = _locate_anchor_positions(input_ids, anchor_ids)
            h_in = h_last[ar, anchor_idx]                              # (B, d)
        else:
            mask, anchor_idx = _build_cot_span_mask(input_ids, open_ids, anchor_ids)
            anchor_h = h_last[ar, anchor_idx]                          # (B, d) empty-span fallback
            empty = ~mask.any(dim=1)                                   # rows with no CoT span (e.g. '<cot></cot>')
            if readout == 'mean':
                denom = mask.float().sum(1).clamp(min=1.0).unsqueeze(1)
                h_in = (h_last.float() * mask.float().unsqueeze(-1)).sum(1) / denom
            elif readout == 'attn_pool':
                # empty-span rows would NaN; give them a 1-hot mask at the anchor so
                # MHA is well-defined, then overwrite with the anchor read below.
                safe_mask = mask.clone()
                safe_mask[empty, anchor_idx[empty]] = True
                h_in = attn_pool(h_last, safe_mask)                    # (B, d)
            else:
                raise ValueError(f'unknown RETENTION_READOUT={readout!r}')
            if empty.any():
                h_in = torch.where(empty.unsqueeze(1), anchor_h.to(h_in.dtype), h_in)
        r_pred = head(h_in)                                            # (B, T)
        # Set on out for the loss to read. Some downstream transforms (e.g.
        # DeepSpeed/DDP wrappers) may drop arbitrary attrs, so we also stash
        # on the holder as a fallback. RetentionLoss reads via getattr first
        # and falls back to holder.r_pred if missing.
        out.r_pred = r_pred
        out.mu_z = getattr(head, '_last_z', None)                     # pre-softplus z = mu_z (Bug A: read directly, no inversion)
        out.r_true = r_true
        out.r_mask = r_mask
        holder.r_pred = r_pred
        holder.z = getattr(head, '_last_z', None)
        # r_true / r_mask are already in holder from _post_encode; refresh
        # only if the caller passed explicit overrides.
        if r_true is not None:
            holder.r_true = r_true
        if r_mask is not None:
            holder.r_mask = r_mask
        return out

    return forward


def _maybe_load_retention_head_from_dir(head: 'RetentionHead', model_dir: str) -> None:
    """Restore retention_head.* weights from a saved checkpoint dir, if present.

    Handles both layouts produced by HF Trainer under full-FT:
      - sharded: model.safetensors.index.json + model-00001-of-NN.safetensors
      - single:  model.safetensors

    Silent no-op if neither file exists (initial training run, not a reload).
    Silent no-op if the safetensors contains no retention_head.* keys (e.g.
    base Qwen checkpoint with no head trained).
    """
    import json
    import os
    from safetensors.torch import load_file

    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    single_path = os.path.join(model_dir, 'model.safetensors')

    head_state: dict = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx.get('weight_map', {})
        shards_to_read = {weight_map[k] for k in weight_map
                          if k.startswith('retention_head.')}
        for shard in shards_to_read:
            sd = load_file(os.path.join(model_dir, shard))
            for k, v in sd.items():
                if k.startswith('retention_head.'):
                    head_state[k[len('retention_head.'):]] = v
    elif os.path.exists(single_path):
        sd = load_file(single_path)
        for k, v in sd.items():
            if k.startswith('retention_head.'):
                head_state[k[len('retention_head.'):]] = v
    if not head_state:
        return                                                # initial training run

    missing, unexpected = head.load_state_dict(head_state, strict=False)
    logger.info(f'Restored retention_head from checkpoint: '
                f'{sorted(head_state.keys())} '
                f'(missing={list(missing)}, unexpected={list(unexpected)})')


# ---- ModelLoader: wires the head onto the base model on load -----------

class Qwen2_5OmniRetentionLoader(ModelLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        from transformers import Qwen2_5OmniConfig
        config = Qwen2_5OmniConfig.from_pretrained(model_dir, trust_remote_code=True)
        # Disable Talker (~833 M params) unless explicitly enabled.
        enable_audio_output = get_env_args('ENABLE_AUDIO_OUTPUT', bool, False)
        config.enable_audio_output = enable_audio_output
        return config

    def get_processor(self, model_dir: str, config: PretrainedConfig):
        from qwen_omni_utils import vision_process
        from transformers import Qwen2_5OmniProcessor
        from swift.model.models.qwen import patch_qwen_vl_utils
        processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)
        patch_qwen_vl_utils(vision_process)
        return processor

    def get_model(self, model_dir: str, config: PretrainedConfig, processor,
                  model_kwargs) -> PreTrainedModel:
        from transformers import Qwen2_5OmniForConditionalGeneration
        from swift.model.utils import use_submodel_func
        from swift.model.patcher import patch_get_input_embeddings
        from transformers import AutoTokenizer

        self.auto_model_cls = self.auto_model_cls or Qwen2_5OmniForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)

        # Route model.forward / generate through the thinker (LM-with-MM-encoders).
        use_submodel_func(model, 'thinker')
        patch_get_input_embeddings(model.thinker.visual, 'patch_embed')
        model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
        if hasattr(model.config, 'talker_config') and model.config.talker_config is not None:
            model.config.talker_config.pad_token_id = None

        # Attach the retention head.
        head_type = get_env_args('RETENTION_HEAD_TYPE', str, 'hazard')
        thinker_cfg = model.thinker.config
        d = getattr(getattr(thinker_cfg, 'text_config', thinker_cfg), 'hidden_size',
                    getattr(thinker_cfg, 'hidden_size', None))
        if d is None:
            raise RuntimeError('Could not locate hidden_size on thinker config.')
        head = RetentionHead(hidden_size=d, head_type=head_type)

        # Register the head as a submodule so it participates in
        # save_pretrained / state_dict / DDP wrapping.
        model.retention_head = head

        # V2 widened readout: RETENTION_READOUT in {last_token, mean, attn_pool}.
        # Default last_token == byte-identical V1. attn_pool adds the CoTAttnPool
        # submodule (~16.8M params, trained from init via modules_to_save:[...,retention_pool]).
        readout = get_env_args('RETENTION_READOUT', str, 'last_token')
        attn_pool = None
        if readout == 'attn_pool':
            attn_pool = CoTAttnPool(hidden_size=d, num_heads=8)
            model.retention_pool = attn_pool
        if _retention_rank_enabled():
            model.retention_rank = RetentionRankHead()

        # If this is a resume / reload (full-FT checkpoint), the trained
        # retention_head.* weights live in the safetensors but were just
        # dropped by HF's from_pretrained as UNEXPECTED keys (because the
        # base Qwen2_5OmniForConditionalGeneration class doesn't declare
        # retention_head). Restore them now into the freshly-attached head.
        #
        # LoRA checkpoints are handled by peft's modules_to_save path, not
        # here — they go through adapter_model.safetensors with prefixed
        # keys like base_model.model.retention_head.* and are restored by
        # set_peft_model_state_dict downstream of this loader.
        _maybe_load_retention_head_from_dir(head, model_dir)
        # Move the head to the model's device/dtype. RetentionHead is
        # constructed on CPU in fp32 (intentional for cumsum/exp stability),
        # but the rest of the model may already be on CUDA via
        # model_kwargs={'device_map': 'cuda'}. Training under DeepSpeed
        # would move everything during initialize(), but inference paths
        # (eval, RL rollouts) call forward directly and need this here.
        try:
            target_device = next(model.parameters()).device
            head.to(target_device)
            if attn_pool is not None:
                attn_pool.to(target_device)
            if _retention_rank_enabled():
                model.retention_rank.to(target_device)
        except StopIteration:
            pass

        # Resolve the </cot> anchor token ids once at load time.
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        anchor_ids = _find_close_cot_token_ids(tokenizer)
        open_ids = _find_open_cot_token_ids(tokenizer)

        # Install lm_head forward_pre_hook to capture the final hidden state
        # into a per-instance holder. This sidesteps output_hidden_states which
        # has issues with @capture_outputs + gradient checkpointing in
        # transformers >=4.5.
        #
        # The holder is also used by Qwen2_5OmniRetentionTemplate._post_encode
        # to stash input_ids / r_true / r_mask before swift's pre_forward_hook
        # strips them from kwargs. Lookup path: model._retention_h_holder.
        holder = _HiddenStateHolder()
        model._retention_h_holder = holder
        model.thinker.lm_head.register_forward_pre_hook(
            _make_lm_head_capture_hook(holder), with_kwargs=True)

        # Patch forward to also compute r_pred and pass-through r_true / r_mask.
        # Capture the *bound* instance method that use_submodel_func attached on
        # line 197 above. type(model).forward would resolve to
        # nn.Module._forward_unimplemented here because
        # Qwen2_5OmniForConditionalGeneration has no class-level forward — the
        # working forward lives on the instance as a routed delegate to
        # model.thinker.forward.
        original_forward = model.forward
        new_forward = _make_retention_forward(original_forward, head, open_ids, anchor_ids,
                                              holder, readout=readout, attn_pool=attn_pool)
        # Bind as instance method so we don't affect other instances.
        import types
        model.forward = types.MethodType(new_forward, model)

        logger.info(f'Attached RetentionHead (type={head_type}, d={d}, T={T_MAX}, '
                    f'readout={readout}) anchor={anchor_ids} open={open_ids}'
                    + (' pool=CoTAttnPool(8h,~16.8M)' if attn_pool is not None else ''))
        return model


# ---- Architecture (per-submodule freeze/lora targets) ------------------

register_model_arch(
    MultiModelKeys(
        'qwen2_5_omni_retention',
        # Mirror the stock my_qwen2_5_omni split: --freeze_vit / --freeze_aligner
        # / --freeze_llm act through these prefixes; LoRA targets the same.
        language_model=['thinker.model', 'thinker.lm_head'],
        vision_tower=['thinker.audio_tower', 'thinker.visual'],
        aligner=['thinker.audio_tower.proj', 'thinker.visual.merger'],
        # Talker + vocoder are not trained; the retention head is treated as
        # part of language_model targets implicitly (it's `retention_head`,
        # outside the trunk; we let LoRA's all-linear target_modules find it).
        generator=['talker', 'token2wav'],
    ))


# ---- Model registration ------------------------------------------------

register_model(
    ModelMeta(
        'qwen2_5_omni_retention',
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-Omni-3B', 'Qwen/Qwen2.5-Omni-3B'),
                Model('Qwen/Qwen2.5-Omni-7B', 'Qwen/Qwen2.5-Omni-7B'),
            ]),
        ],
        Qwen2_5OmniRetentionLoader,
        template='qwen2_5_omni_retention',
        is_multimodal=True,
        model_arch='qwen2_5_omni_retention',
        architectures=['Qwen2_5OmniForConditionalGeneration'],
        requires=['transformers>=4.50', 'qwen_omni_utils', 'decord'],
        tags=['vision', 'video', 'audio', 'retention-curve'],
        additional_saved_files=['spk_dict.pt'],
    ))


# ---- Template (data collator that carries r_true / r_mask) -------------

class Qwen2_5OmniRetentionTemplate(Qwen2_5OmniTemplate):
    """Inherits the stock Qwen2.5-Omni template (all multimodal handling:
    audio/video token expansion, mrope position ids, padding-free, etc.)
    and adds two batch tensors:

      r_true : (B, T_MAX)  float, NaN-padded for t >= T_i
      r_mask : (B, T_MAX)  bool,  True for t < T_i

    Source rows are expected to carry an ``R`` field: list of floats of
    length T_i + 1 (R[0] == 1 by convention, R[1..T_i] are the per-second
    retention values).
    """

    def _encode(self, inputs):
        enc = super()._encode(inputs)
        if _retention_rank_enabled():
            pct = None
            if hasattr(inputs, 'extra') and isinstance(inputs.extra, dict):
                pct = inputs.extra.get('pct')
            elif isinstance(inputs, dict):
                pct = inputs.get('pct')
            if pct is not None:
                enc['p_rank'] = torch.tensor(pct, dtype=torch.float32)
        R = None
        # Accept either 'R' (build_ttcc_jsonl schema) or 'R_true' (Wanjia's
        # legacy v2cot schema). Same semantics: the per-second retention curve.
        if hasattr(inputs, 'extra') and isinstance(inputs.extra, dict):
            R = inputs.extra.get('R') or inputs.extra.get('R_true')
        elif isinstance(inputs, dict):
            R = inputs.get('R') or inputs.get('R_true')
        if R is None:
            return enc                                            # inference path
        T_i = max(0, len(R) - 1)
        r_true = torch.full((T_MAX,), float('nan'))
        r_mask = torch.zeros(T_MAX, dtype=torch.bool)
        if T_i > 0:
            tail = torch.tensor(R[1:T_i + 1], dtype=torch.float32)
            r_true[:T_i] = tail
            r_mask[:T_i] = True
        enc['r_true'] = r_true
        enc['r_mask'] = r_mask
        return enc

    def _data_collator(self, batch, *, padding_to=None):
        # Super handles position_ids, packed_seq_params, padding_free,
        # multimodal mm_data, etc. We stack our per-sample tensors on top —
        # they pass through to model.forward as kwargs and get captured by
        # the retention patch.
        #
        # _encode adds r_true / r_mask to each sample dict, but swift's
        # encoding pipeline collapses keys to a stable schema before
        # collation, so r_true may be missing from batch[i]. The 'R' field
        # from the raw dataset row survives (remove_unused_columns=False),
        # so derive r_true / r_mask from R here as a fallback.
        res = super()._data_collator(batch, padding_to=padding_to)
        # Source for R per sample, in priority order:
        #   1. batch[i]['r_true']  — populated by _encode (usually stripped)
        #   2. batch[i]['R']       — raw column when remove_unused_columns=False
        #   3. batch[i]['_extra_kwargs']['R']  — swift stashes original row
        #      extras here when remove_unused_columns=False; this is the
        #      reliable path on multimodal templates because _encode's
        #      non-canonical keys get filtered.
        if batch and 'r_true' in batch[0]:
            res['r_true'] = torch.stack([b['r_true'] for b in batch])
            res['r_mask'] = torch.stack([b['r_mask'] for b in batch])
        else:
            R_list = []
            for b in batch:
                R = b.get('R') or b.get('R_true')
                if R is None:
                    ek = b.get('_extra_kwargs') or {}
                    if isinstance(ek, dict):
                        R = ek.get('R') or ek.get('R_true')
                R_list.append(R)
            if any(R is not None for R in R_list):
                r_trues, r_masks = [], []
                for R in R_list:
                    rt = torch.full((T_MAX,), float('nan'))
                    rm = torch.zeros(T_MAX, dtype=torch.bool)
                    if R is not None:
                        T_i = max(0, len(R) - 1)
                        if T_i > 0:
                            rt[:T_i] = torch.tensor(R[1:T_i + 1], dtype=torch.float32)
                            rm[:T_i] = True
                    r_trues.append(rt)
                    r_masks.append(rm)
                res['r_true'] = torch.stack(r_trues)
                res['r_mask'] = torch.stack(r_masks)
        if _retention_rank_enabled():
            if batch and all('p_rank' in b for b in batch):
                res['p_rank'] = torch.stack([b['p_rank'] for b in batch])
            else:
                pct_list = []
                for b in batch:
                    pct = b.get('pct')
                    if pct is None:
                        ek = b.get('_extra_kwargs') or {}
                        if isinstance(ek, dict):
                            pct = ek.get('pct')
                    pct_list.append(pct)
                if pct_list and all(pct is not None for pct in pct_list):
                    res['p_rank'] = torch.stack([
                        torch.as_tensor(pct, dtype=torch.float32) for pct in pct_list
                    ])
        return res

    def _post_encode(self, model, inputs):
        # ms-swift's pre_forward_hook calls _post_encode, then keeps only a
        # canonical set of kwargs (input_ids/attention_mask/labels/position_ids/
        # output_hidden_states/logits_to_keep/...). r_true, r_mask and the
        # original input_ids would be dropped before our wrapped forward runs.
        # Stash them into the model's per-instance holder so the wrapper can
        # read them back. Then proceed with the parent's MM encoding.
        holder = getattr(model, '_retention_h_holder', None)
        if holder is None:
            base = getattr(model, 'base_model', None)
            if base is not None:
                holder = getattr(getattr(base, 'model', base), '_retention_h_holder', None)
        if holder is not None:
            holder.input_ids = inputs.get('input_ids')
            if 'r_true' in inputs:
                holder.r_true = inputs['r_true']
                holder.r_mask = inputs['r_mask']
            if _retention_rank_enabled():
                holder.p_rank = None
                if 'p_rank' in inputs:
                    holder.p_rank = inputs['p_rank']
        return super()._post_encode(model, inputs)


# Reuse the stock Qwen template meta (chat-template strings, stop words,
# default system, agent_template='hermes', etc.) — swap only template_cls.
register_template(
    QwenTemplateMeta(
        'qwen2_5_omni_retention',
        template_cls=Qwen2_5OmniRetentionTemplate,
    ))


# ---- Loss --------------------------------------------------------------

def _log_hazard_mse(r_pred: torch.Tensor, r_true: torch.Tensor,
                    r_mask: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """Discrete-time log-hazard MSE.

    Converts both predicted and true R(t) curves to per-second hazards via
    lam(t) = log R(t-1) - log R(t) and computes masked MSE on log-hazards.
    Follows the discrete-time formulation in DeepHit (Lee et al., 2018) and
    matches wanjia milestone §3.
    """
    R_pred = r_pred.clamp(min=eps)
    R_prev_pred = torch.cat([torch.ones_like(R_pred[:, :1]), R_pred[:, :-1]], dim=1)
    lam_pred = (R_prev_pred.log() - R_pred.log()).clamp(min=eps)

    R_true_safe = torch.where(r_mask, r_true, torch.ones_like(r_true)).clamp(min=eps)
    R_prev_true = torch.cat([torch.ones_like(R_true_safe[:, :1]),
                             R_true_safe[:, :-1]], dim=1)
    lam_true = (R_prev_true.log() - R_true_safe.log()).clamp(min=eps)

    sq = (lam_pred.log() - lam_true.log()) ** 2
    denom = r_mask.float().sum(dim=1).clamp(min=1.0)
    per_ad = (sq * r_mask.float()).sum(dim=1) / denom
    return per_ad.mean()


def _masked_mse(r_pred: torch.Tensor, r_true: torch.Tensor,
                r_mask: torch.Tensor) -> torch.Tensor:
    """Masked per-second MSE on R(t). Used by the sigmoid head."""
    diff = (r_pred - torch.nan_to_num(r_true, nan=0.0)) ** 2
    denom = r_mask.float().sum(dim=1).clamp(min=1.0)
    per_ad = (diff * r_mask.float()).sum(dim=1) / denom
    return per_ad.mean()


class RetentionLoss(BaseLoss):
    """Loss for retention-curve heads — plain masked MSE on R(t) for both
    hazard and sigmoid variants.

    Reads r_pred from the model output (stashed by the patched forward) and
    r_true / r_mask similarly. If labels are present (with-CoT variants),
    optionally adds an LM cross-entropy term gated by RETENTION_COT_ALPHA.

    Why plain MSE on R(t) and not log-hazard MSE:
      The eval metric is IBS = mean MSE on R. The hazard architecture
      (softplus -> cumsum -> exp) enforces monotonicity structurally, so the
      loss is free to be anything; matching the eval metric is the natural
      choice. Log-hazard MSE (wanjia milestone Sec.3) suffers a ~120x
      per-position loss inflation on flat-tail positions where
      lam_true ~= 0 gets clamped to eps -> log(eps) ~= -13.8; this noise
      then dominates the gradient and clipping rules training (observed
      grad_norm 25K vs 2K when we switched). Switching to plain MSE
      reduced reported loss ~60x and grad_norm ~10x on V7 production vs
      V6 (both 8-GPU full-FT, same data). See HEAD_COMPARISON.md and the
      loss-choice fragment in docs/ for the full Feynman walkthrough.

    Known residual risk: the gradient ∂L/∂z(s) carries a factor R_pred(t),
    which is small for tail positions, so tail-region z values learn more
    slowly than mid-curve ones. Bounded contribution to IBS (~15% of
    wanjia's best result IBS=0.0076 in the worst case). Mitigation if
    observed in eval: weighted MSE upweighting tail residuals.
    """

    def __call__(self, outputs, labels, *, num_items_in_batch=None,
                 loss_scale=None, trainer=None, **kwargs) -> torch.Tensor:
        r_pred = getattr(outputs, 'r_pred', None)
        r_true = getattr(outputs, 'r_true', None)
        r_mask = getattr(outputs, 'r_mask', None)
        # Fallback: read from the model's _retention_h_holder when the
        # outputs object lost the attrs (e.g. DDP/DeepSpeed wrappers).
        if (r_pred is None or r_true is None or r_mask is None) and trainer is not None:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            base = getattr(unwrapped, 'base_model', unwrapped)
            base = getattr(base, 'model', base)
            holder = getattr(base, '_retention_h_holder', None)
            if holder is not None:
                if r_pred is None:
                    r_pred = holder.r_pred
                if r_true is None:
                    r_true = holder.r_true
                if r_mask is None:
                    r_mask = holder.r_mask
        if r_pred is None or r_true is None or r_mask is None:
            keys = list(outputs.keys()) if hasattr(outputs, 'keys') else dir(outputs)
            raise RuntimeError(
                'RetentionLoss requires r_pred/r_true/r_mask on the model '
                'output (or holder). The retention plugin must be loaded and '
                f'Qwen2_5OmniRetentionTemplate must be active. Output keys: {keys[:20]}; '
                f'r_pred={r_pred is not None}, r_true={r_true is not None}, r_mask={r_mask is not None}.')

        # Both heads use plain MSE on R(t). This matches the IBS eval metric
        # exactly. The monotonicity prior (for hazard) lives in the architecture
        # (softplus → cumsum → exp), not the loss; using MSE here doesn't
        # weaken the prior. Avoids log-hazard MSE's ~1000x loss inflation on
        # flat-tail positions where lam_true clamps to eps. See docs/loss-choice.md.
        head_type = get_env_args('RETENTION_HEAD_TYPE', str, 'hazard')
        loss_curve = _masked_mse(r_pred, r_true, r_mask)

        loss_rank = None
        rank_kind = RETENTION_RANK_LOSS
        lam_rank = float(RETENTION_RANK_LAMBDA) if rank_kind != 'none' else 0.0
        # TRAIN-ONLY: skip the rank block (and its all_gather) during eval -> eval_loss stays
        # curve+cot (comparable to prod) and a ragged val batch cannot deadlock the gather.
        _rank_training = (trainer is not None) and bool(getattr(getattr(trainer, 'model', None), 'training', False))
        if lam_rank > 0 and rank_kind != 'none' and _rank_training:
            if rank_kind not in {'pairwise', 'pointwise', 'both'}:
                raise ValueError(f'RETENTION_RANK_LOSS must be none|pairwise|pointwise|both, got {rank_kind!r}')
            # per-ad masked means, ALIGNED to the eval window t in [1,30] (indices 0..29)
            w = r_mask.float()[:, :30]
            denom = w.sum(1).clamp(min=1.0)
            m_pred = (r_pred[:, :30] * w).sum(1) / denom                  # (B,) grad-carrying
            m_true = (torch.nan_to_num(r_true, nan=0.0)[:, :30] * w).sum(1) / denom  # (B,) detach later

            # differentiable all-gather across DP ranks
            try:
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    world = torch.distributed.get_world_size()
                    rank  = torch.distributed.get_rank()
                    gp = [torch.zeros_like(m_pred) for _ in range(world)]
                    torch.distributed.all_gather(gp, m_pred.contiguous())
                    gp[rank] = m_pred   # restore grad on own slice
                    all_mp = torch.cat(gp)
                    gt = [torch.zeros_like(m_true) for _ in range(world)]
                    torch.distributed.all_gather(gt, m_true.contiguous())
                    all_mt = torch.cat(gt)
                else:
                    all_mp = m_pred
                    all_mt = m_true
            except Exception as _gather_err:
                logger.warning_once(f'[RetentionRankLoss] all_gather failed ({_gather_err}); falling back to local batch')
                all_mp = m_pred
                all_mt = m_true

            all_mt_d = all_mt.detach()

            loss_pair  = m_pred.new_tensor(0.0)
            loss_point = m_pred.new_tensor(0.0)

            # pairwise Bradley-Terry
            if rank_kind in {'pairwise', 'both'}:
                beta = float(RETENTION_RANK_BETA)
                gt_mask = all_mt_d[:, None] > all_mt_d[None, :]   # (N,N) bool
                if gt_mask.any():
                    diff = all_mp[:, None] - all_mp[None, :]       # (N,N)
                    loss_pair = -F.logsigmoid(beta * diff)[gt_mask].mean()

            # pointwise
            if rank_kind in {'pointwise', 'both'}:
                unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                rank_model = getattr(unwrapped, 'base_model', unwrapped)
                rank_model = getattr(rank_model, 'model', rank_model)
                holder2   = getattr(rank_model, '_retention_h_holder', None)
                p_rank    = getattr(holder2, 'p_rank', None) if holder2 else None
                if p_rank is not None:
                    p_local = p_rank.float().to(m_pred.device).detach().view(-1)
                    if p_local.numel() != m_pred.numel():
                        p_local = torch.full_like(m_pred, float('nan'))
                else:
                    p_local = torch.full_like(m_pred, float('nan'))
                try:
                    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                        world = torch.distributed.get_world_size()
                        gpct = [torch.full_like(p_local, float('nan')) for _ in range(world)]
                        torch.distributed.all_gather(gpct, p_local.contiguous())
                        P = torch.cat(gpct)
                    else:
                        P = p_local
                except Exception as _pct_err:
                    logger.warning_once(f'[RetentionRankLoss] pct all_gather failed ({_pct_err}); using batch-relative fallback')
                    P = torch.full_like(all_mt_d, float('nan'))
                if P.numel() != all_mp.numel() or not torch.isfinite(P).all():
                    # batch-relative percentile fallback
                    N = all_mt_d.shape[0]
                    order = all_mt_d.argsort()
                    ranks = torch.zeros_like(all_mt_d)
                    ranks[order] = torch.arange(N, dtype=all_mt_d.dtype, device=all_mt_d.device)
                    P = (ranks + 0.5) / max(N, 1)
                s = rank_model.retention_rank(all_mp.float())
                loss_point = F.mse_loss(s, P.detach())

            loss_rank = loss_pair + loss_point

            # log sub-losses
            try:
                unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                rank_model = getattr(unwrapped, 'base_model', unwrapped)
                rank_model = getattr(rank_model, 'model', rank_model)
                holder_out = getattr(rank_model, '_retention_h_holder', None)
                if holder_out is not None:
                    object.__setattr__(holder_out, 'loss_rank', loss_rank.detach())
                mode = 'train' if trainer.model.training else 'eval'
                trainer.custom_metrics[mode]['loss_rank'].update(loss_rank.detach().float().item())
                trainer.custom_metrics[mode]['rank_std_mpred'].update(all_mp.std(unbiased=False).detach().float().item())
                if rank_kind in {'pointwise', 'both'}:
                    trainer.custom_metrics[mode]['rank_std_s'].update(s.std(unbiased=False).detach().float().item())
            except Exception as _log_err:
                logger.warning_once(f'[RetentionRankLoss] logging failed: {_log_err}')

            # optional grad-norm ratio probe
            gradlog_every = int(RANK_GRADLOG_EVERY)
            if gradlog_every > 0:
                try:
                    step = getattr(trainer, 'state', None)
                    step = getattr(step, 'global_step', 0) if step else 0
                    if step % gradlog_every == 0:
                        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                        rank_model = getattr(unwrapped, 'base_model', unwrapped)
                        rank_model = getattr(rank_model, 'model', rank_model)
                        head_params = list(rank_model.retention_head.linear.parameters())
                        curve_grads = [g.norm() for g in torch.autograd.grad(
                            loss_curve, head_params, retain_graph=True, allow_unused=True) if g is not None]
                        rank_grads = [g.norm() for g in torch.autograd.grad(
                            lam_rank * loss_rank, head_params, retain_graph=True, allow_unused=True) if g is not None]
                        if curve_grads and rank_grads:
                            gn_curve = torch.stack(curve_grads).norm()
                            gn_rank  = torch.stack(rank_grads).norm()
                            ratio = (gn_rank / gn_curve.clamp(min=1e-8)).item()
                            mode = 'train' if trainer.model.training else 'eval'
                            trainer.custom_metrics[mode]['rank_grad_ratio'].update(ratio)
                        else:
                            logger.warning_once('[RetentionRankLoss] grad probe found no retention_head gradients')
                except Exception as _grad_err:
                    logger.warning_once(f'[RetentionRankLoss] grad probe failed: {_grad_err}')

        alpha = float(get_env_args('RETENTION_COT_ALPHA', str, '0.0'))
        loss_cot = None
        if alpha > 0 and labels is not None and getattr(outputs, 'logits', None) is not None:
            logits = outputs.logits
            # CRITICAL SHIFT (fix 2026-05-30): ms-swift hands the loss RAW (unshifted)
            # labels. Every correct consumer shifts to the next-token target — the
            # canonical per_token_loss_func uses torch.roll(labels,-1) (trainers/utils.py),
            # channel-loss masks on roll(labels,-1) (seq2seq_trainer.py), and the
            # label_smoother path passes shift_labels=True. The previous code compared
            # logits[i] to labels[i] (NO shift): since SFT response tokens are BOTH input
            # and label, h[i] already contains labels[i], so this is a trivial COPY task
            # that drives loss_cot->~0 while DESTROYING genuine next-token prediction
            # (root cause of the token_acc collapse 0.50->~0.005, full-FT AND LoRA, that
            # alpha-tuning never fixed because the objective itself was malformed). roll
            # wraps labels[0] (=-100, prompt/BOS) into the last position -> ignored,
            # matching swift's own convention exactly.
            shift_labels = torch.roll(labels, shifts=-1, dims=-1)
            loss_cot = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total = loss_curve + alpha * loss_cot
        else:
            total = loss_curve
        if loss_rank is not None:
            # DDP/ZeRO averages param grads by world_size, but the gather-with-grad restore makes
            # each rank backprop only its OWN ad's slice -> the rank-loss grad is ~1/world too small.
            # Multiply by world_size so the nominal lambda is the effective lambda (topology-independent).
            _rs = torch.distributed.get_world_size() if (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1) else 1
            total = total + lam_rank * _rs * loss_rank
        # Stash component losses for the trainer to log. Reading by callbacks/
        # _maybe_log_save_evaluate keyed by `outputs.loss_curve` / `outputs.loss_cot`.
        try:
            object.__setattr__(outputs, 'loss_curve', loss_curve.detach())
            if loss_rank is not None:
                object.__setattr__(outputs, 'loss_rank', loss_rank.detach())
            if loss_cot is not None:
                object.__setattr__(outputs, 'loss_cot', loss_cot.detach())
                object.__setattr__(outputs, 'cot_alpha', alpha)
        except (AttributeError, TypeError):
            # Some output container types reject attribute setting; skip silently.
            pass
        # Also stash on the model holder as a robust fallback (Output objects
        # are sometimes discarded by DDP/DeepSpeed wrappers before logging).
        if trainer is not None:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            base = getattr(unwrapped, 'base_model', unwrapped)
            base = getattr(base, 'model', base)
            holder = getattr(base, '_retention_h_holder', None)
            if holder is not None:
                holder.loss_curve = float(loss_curve.detach().item())
                if loss_rank is not None:
                    holder.loss_rank = float(loss_rank.detach().item())
                if loss_cot is not None:
                    holder.loss_cot = float(loss_cot.detach().item())
                    holder.cot_alpha = float(alpha)
            # --- log component losses to swift's custom_metrics (same path swift uses for
            # aux_loss / token_acc). Direct write from inside the loss, which already holds the
            # live trainer -> fires from step 1, no monkeypatch, no callback. LOUD on failure
            # (the 'stash without callback = silent gap' lesson: a logging failure must be visible).
            try:
                mode = 'train' if trainer.model.training else 'eval'
                cm = trainer.custom_metrics[mode]
                cm['loss_curve'].update(float(loss_curve.detach().item()))
                if loss_cot is not None:
                    cm['loss_cot'].update(float(loss_cot.detach().item()))
                    cm['cot_alpha'].update(float(alpha))
            except Exception as e:  # noqa: BLE001 -- observability must be loud, not swallowed
                from swift.utils import get_logger
                logger.warning_once(f'RetentionLoss: component-metric logging failed: {e!r}')
        return total


loss_map['retention_loss'] = RetentionLoss

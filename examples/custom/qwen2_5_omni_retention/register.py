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

from swift.loss import BaseLoss, loss_map
from swift.model import (Model, ModelGroup, ModelLoader, ModelMeta, MultiModelKeys,
                         register_model, register_model_arch)
from swift.template import Template, TemplateMeta, register_template
from swift.utils import get_env_args, get_logger

logger = get_logger()

T_MAX = 60
CLOSE_COT = '</cot>'


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

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h : (B, hidden_size)
        z = self.linear(h.float())
        if self.head_type == 'hazard':
            lam = F.softplus(z)                                       # (B, T) >= 0
            return torch.exp(-torch.cumsum(lam, dim=-1))              # (B, T) in (0, 1]
        # sigmoid
        return torch.sigmoid(z)


# ---- Model wrapper (subclass of the stock model) ------------------------

class _RetentionWrapperState:
    """Mutable per-instance state stored on the wrapped model.

    We avoid PretrainedConfig pollution; anchor token ids and head are
    attached as Python attrs at load time.
    """


def _make_retention_forward(original_forward, head: RetentionHead,
                            anchor_ids: list[int]):
    """Patch the base model's forward so it also computes r_pred."""

    def forward(self, *args, r_true=None, r_mask=None, **kwargs):
        # Force hidden states on so we can read the anchor position.
        kwargs.setdefault('output_hidden_states', True)
        out = original_forward(*args, **kwargs)
        # Outputs from Qwen2.5-Omni: last_hidden_state OR hidden_states[-1].
        h_last = None
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            h_last = out.hidden_states[-1]
        elif hasattr(out, 'last_hidden_state'):
            h_last = out.last_hidden_state
        if h_last is None:
            raise RuntimeError(
                'RetentionWrapper: model forward did not return hidden states; '
                'output_hidden_states=True must be honored by the backbone.')
        input_ids = kwargs.get('input_ids')
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        anchor_idx = _locate_anchor_positions(input_ids, anchor_ids)
        h_anchor = h_last[torch.arange(h_last.size(0), device=h_last.device),
                          anchor_idx]                                  # (B, d)
        r_pred = head(h_anchor)                                        # (B, T)
        out.r_pred = r_pred
        out.r_true = r_true
        out.r_mask = r_mask
        return out

    return forward


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

        # Resolve the </cot> anchor token ids once at load time.
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        anchor_ids = _find_close_cot_token_ids(tokenizer)

        # Patch forward to also compute r_pred and pass-through r_true / r_mask.
        original_forward = type(model).forward
        new_forward = _make_retention_forward(original_forward, head, anchor_ids)
        # Bind as instance method so we don't affect other instances.
        import types
        model.forward = types.MethodType(new_forward, model)

        logger.info(f'Attached RetentionHead (type={head_type}, d={d}, T={T_MAX}) '
                    f'with anchor ids {anchor_ids}')
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

class Qwen2_5OmniRetentionTemplate(Template):
    """Inherits the stock Qwen2.5-Omni template and adds two batch tensors:

      r_true : (B, T_MAX)  float, NaN-padded for t >= T_i
      r_mask : (B, T_MAX)  bool,  True for t < T_i

    Source rows are expected to carry an ``R`` field: list of floats of
    length T_i + 1 (R[0] == 1 by convention, R[1..T_i] are the per-second
    retention values).
    """

    def _encode(self, inputs):
        enc = super()._encode(inputs)
        R = None
        if hasattr(inputs, 'extra') and isinstance(inputs.extra, dict):
            R = inputs.extra.get('R')
        elif isinstance(inputs, dict):
            R = inputs.get('R')
        if R is None:
            # No retention target on this row (e.g. inference). Leave absent.
            return enc
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
        res = super()._data_collator(batch, padding_to=padding_to)
        if 'r_true' in batch[0]:
            res['r_true'] = torch.stack([b['r_true'] for b in batch])
            res['r_mask'] = torch.stack([b['r_mask'] for b in batch])
        return res


register_template(
    TemplateMeta(
        'qwen2_5_omni_retention',
        prefix=[''],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|im_end|>'],
        system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
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
    """Loss for retention-curve heads.

    Reads r_pred from the model output (stashed by the patched forward) and
    r_true / r_mask similarly. If labels are present (with-CoT variants),
    optionally adds an LM cross-entropy term gated by RETENTION_COT_ALPHA.
    """

    def __call__(self, outputs, labels, *, num_items_in_batch=None,
                 loss_scale=None, **kwargs) -> torch.Tensor:
        r_pred = getattr(outputs, 'r_pred', None)
        r_true = getattr(outputs, 'r_true', None)
        r_mask = getattr(outputs, 'r_mask', None)
        if r_pred is None or r_true is None or r_mask is None:
            raise RuntimeError(
                'RetentionLoss requires r_pred/r_true/r_mask on the model '
                'output. The retention plugin must be loaded and the '
                'Qwen2_5OmniRetentionTemplate must be active.')

        head_type = get_env_args('RETENTION_HEAD_TYPE', str, 'hazard')
        if head_type == 'hazard':
            loss_curve = _log_hazard_mse(r_pred, r_true, r_mask)
        else:
            loss_curve = _masked_mse(r_pred, r_true, r_mask)

        alpha = float(get_env_args('RETENTION_COT_ALPHA', str, '0.0'))
        if alpha > 0 and labels is not None and getattr(outputs, 'logits', None) is not None:
            logits = outputs.logits
            loss_cot = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss_curve + alpha * loss_cot
        return loss_curve


loss_map['retention_loss'] = RetentionLoss

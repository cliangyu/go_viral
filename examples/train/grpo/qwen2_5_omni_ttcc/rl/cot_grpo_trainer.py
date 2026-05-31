"""CoT-GRPO trainer: subclass of the OFFICIAL swift GRPOTrainer.

We change ONE thing: how the reward is computed. Everything else -- multimodal
generation of the G CoT rollouts, per-token logprobs, group-relative advantage,
KL to the SFT ref, vLLM colocate, DeepSpeed/torchrun multi-node -- is the upstream
GRPOTrainer, untouched.

The head reward reads the retention HEAD conditioned on each generated CoT (NOT a
text-parsed curve). For each rollout we re-encode (prompt + generated <cot>...</cot>)
in train mode, forward the policy, and read the head's r_pred off the model output /
holder -- exactly the proven path in verification/generate_eval.py:head_curve. Then
`head_reward` turns R_hat into the scalar (cross-ad rank vs the train CDF + IBS anchor).

References (per Leon): official swift GRPOTrainer (base); generate_eval.py + register.py
(CoT-SFT head readout); head_pg_trainer.py / train_head_pg.py (RL entry + unwrap pattern).

Injected by the entry (train_cot_grpo.py) after construction:
  self._cot_cdf            : dict {t:int -> sorted np.array}  (train-only CDF)
The head readout re-uses self.template with a save/restore to 'train' mode (so the
generated CoT is encoded as in-sequence context, not masked as a to-generate span).
Env knobs: COT_BETA (rank, default 1.0), COT_ALPHA (IBS anchor, default 0.25),
           COT_T_LO/COT_T_HI (band, default 1/30).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # head_reward (-> cross_ad_reward)

from swift.rlhf_trainers import GRPOTrainer
from swift.utils import get_logger

from head_reward import head_reward

logger = get_logger()


class CoTGRPOTrainer(GRPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cot_cdf = None             # injected by the entry
        self._cot_head_col = None        # resolved on first reward call
        self._cot_beta = float(os.environ.get('COT_BETA', '1.0'))
        self._cot_alpha = float(os.environ.get('COT_ALPHA', '0.25'))
        self._cot_t_lo = int(os.environ.get('COT_T_LO', '1'))
        self._cot_t_hi = int(os.environ.get('COT_T_HI', '30'))
        self._cot_step = 0
        logger.info(f'[cot-grpo] head reward: beta={self._cot_beta} alpha={self._cot_alpha} '
                    f't=[{self._cot_t_lo},{self._cot_t_hi}] G={getattr(self, "num_generations", "?")}')

    # ---- the holder/base unwrap (head_pg_trainer pattern) ----
    def _base(self):
        m = self.accelerator.unwrap_model(self.model)
        m = getattr(m, 'base_model', m)
        m = getattr(m, 'model', m)
        return m

    def _head_curve(self, inp) -> np.ndarray | None:
        """Re-encode (prompt + this rollout's generated CoT) in train mode, forward the
        policy, read the head r_pred -> R(0..T). Mirrors generate_eval.py:head_curve.
        Returns None on encode/forward failure (-> reward floored to 0 by the caller)."""
        from swift.template.template_inputs import TemplateInputs
        tmpl = self.template
        row = {k: inp[k] for k in ('messages', 'videos', 'audios', 'images', 'ad_id', 'T', 'R', 'R_true')
               if k in inp and inp[k] is not None}
        # inp['messages'][-1] already holds the generated CoT (swift put it there post-rollout).
        prev_mode = getattr(tmpl, 'mode', None)
        try:
            tmpl.set_mode('train')  # encode the generated CoT as in-sequence context (head reads last token)
            enc = tmpl.encode(TemplateInputs.from_dict(row))
            batch = tmpl.data_collator([enc])
            dev = self.accelerator.device
            batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch.pop('labels', None)
            with torch.no_grad():
                out = self.model(**batch)
            rp = getattr(out, 'r_pred', None)
            if rp is None:
                rp = self._base()._retention_h_holder.r_pred
            rp = rp[0].float().cpu().numpy()
            return np.concatenate([[1.0], rp])
        except Exception as e:  # noqa: BLE001 -- truncation / encode / OOM -> floored reward
            if self._cot_step < 3:
                logger.warning(f'[cot-grpo] head_curve failed ({type(e).__name__}: {str(e)[:120]}) -> reward 0')
            return None
        finally:
            if prev_mode is not None:
                tmpl.set_mode(prev_mode)

    def _compute_rewards_per_func(self, inputs):
        rewards = super()._compute_rewards_per_func(inputs)  # (N, K); ttcc_cot_format etc. already filled
        if self._cot_head_template is None or self._cot_cdf is None:
            raise RuntimeError('[cot-grpo] head template / CDF not injected by the entry')
        if self._cot_head_col is None:
            names = list(self.reward_func_names)
            self._cot_head_col = names.index('ttcc_head') if 'ttcc_head' in names else 0
            logger.info(f'[cot-grpo] head-reward column = {self._cot_head_col} ({names})')
        col = self._cot_head_col

        R_hats = []
        for i, inp in enumerate(inputs):
            R_hat = self._head_curve(inp)
            R_true = inp.get('R') or inp.get('R_true')
            if R_hat is None or not R_true:
                r = 0.0
            else:
                r, _ = head_reward(R_hat, R_true, self._cot_cdf,
                                   beta=self._cot_beta, alpha=self._cot_alpha,
                                   t_lo=self._cot_t_lo, t_hi=self._cot_t_hi)
            R_hats.append(R_hat)
            rewards[i, col] = r

        self._log_curve_std(R_hats)
        self._cot_step += 1
        return rewards

    def _log_curve_std(self, R_hats):
        """R1 make-or-break: do different CoTs give different head curves? Log the mean
        within-group std of R_hat over the band. std~0 => no advantage => no learning."""
        G = int(getattr(self, 'num_generations', 1) or 1)
        valid = [(i, R) for i, R in enumerate(R_hats) if R is not None]
        if G < 2 or len(valid) < G:
            return
        lo, hi = self._cot_t_lo, self._cot_t_hi
        stds = []
        n = len(R_hats)
        for g0 in range(0, n - n % G, G):
            grp = [R_hats[g0 + j] for j in range(G) if R_hats[g0 + j] is not None]
            if len(grp) < 2:
                continue
            tmax = min(len(R) for R in grp) - 1
            h = min(hi, tmax)
            if h <= lo:
                continue
            M = np.stack([R[lo:h + 1] for R in grp])         # (g, band)
            stds.append(float(M.std(axis=0).mean()))         # std across rollouts, mean over t
        if stds:
            cstd = float(np.mean(stds))
            try:
                mode = 'train' if self.model.training else 'eval'
                self._metrics[mode]['cot_curve_std'].append(cstd)  # -> wandb if available
            except Exception:  # noqa: BLE001
                pass
            if self._cot_step % 5 == 0:
                logger.info(f'[cot-grpo] step~{self._cot_step} within-group curve std={cstd:.4f} '
                            f'(R1: must be >0 or no gradient)')

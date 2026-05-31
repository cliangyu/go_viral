"""CoT-GRPO trainer: subclass of the OFFICIAL swift GRPOTrainer.

We change ONE thing: how the reward is computed. Everything else -- multimodal
generation of the G CoT rollouts, per-token logprobs, group-relative advantage,
KL to the SFT ref, DeepSpeed/torchrun multi-node -- is the upstream GRPOTrainer.

The head reward reads the retention HEAD conditioned on each generated CoT (NOT a
text-parsed curve). For each rollout we re-encode (prompt + generated <cot>...</cot>)
in train mode, forward the policy, and read the head's r_pred off the model output /
holder -- the path proven in verification/generate_eval.py:head_curve. Then
`head_reward` turns R_hat into the scalar (cross-ad rank vs the train CDF + IBS anchor).

References (per Leon): official swift GRPOTrainer (base); generate_eval.py + register.py
(CoT-SFT head readout); head_pg_trainer.py / train_head_pg.py (RL entry + unwrap pattern).

Injected by the entry (train_cot_grpo.py) after construction:
  self._cot_cdf : dict {t:int -> sorted np.array}  (train-only CDF)
Env knobs: COT_BETA (rank, 1.0), COT_ALPHA (IBS anchor, 0.25), COT_T_LO/COT_T_HI (1/30).

Preflight-audit hardening (2026-05-31): column resolved by registered class (not a
name swift never stores); completion coerced to str (use_vllm=false -> TransformersEngine
may hand token_ids); head-forward failures COUNTED + step-0 abort (a systematic r_pred
non-attach must NOT masquerade as the R1 'dead-channel' signal); warm-start made visible
via cross-ad curve std at step 0 (a near-zero cross-ad std => the head is untrained/dead).
"""
from __future__ import annotations

import copy
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
        self._cot_cdf = None             # injected by the entry (train CDF)
        self._cot_head_col = None        # resolved on first reward call
        self._cot_beta = float(os.environ.get('COT_BETA', '1.0'))
        self._cot_alpha = float(os.environ.get('COT_ALPHA', '0.25'))
        self._cot_t_lo = int(os.environ.get('COT_T_LO', '1'))
        self._cot_t_hi = int(os.environ.get('COT_T_HI', '30'))
        self._cot_step = 0
        logger.info(f'[cot-grpo] head reward: beta={self._cot_beta} alpha={self._cot_alpha} '
                    f't=[{self._cot_t_lo},{self._cot_t_hi}] G={getattr(self, "num_generations", "?")}')

    # ---- holder/base unwrap (head_pg_trainer pattern) ----
    def _base(self):
        m = self.accelerator.unwrap_model(self.model)
        m = getattr(m, 'base_model', m)
        m = getattr(m, 'model', m)
        return m

    def _completion_str(self, msg_content) -> str:
        """use_vllm=false -> TransformersEngine may deliver token_ids/dict, not text.
        Coerce to a decoded string (mirrors swift grpo_trainer logging)."""
        if isinstance(msg_content, str):
            return msg_content
        if isinstance(msg_content, dict):
            msg_content = msg_content.get('token_ids', [])
        if isinstance(msg_content, list):
            try:
                return self.processing_class.decode(msg_content)
            except Exception:  # noqa: BLE001
                return ''
        return str(msg_content)

    def _head_curve(self, inp) -> np.ndarray | None:
        """Re-encode (prompt + this rollout's generated CoT) in train mode, forward the
        policy, read the head r_pred -> R(0..T). Mirrors generate_eval.py:head_curve.
        Returns None on encode/forward/non-finite failure (caller counts + may abort)."""
        from swift.template.template_inputs import TemplateInputs
        tmpl = self.template
        row = {k: inp[k] for k in ('videos', 'audios', 'images', 'ad_id', 'T', 'R', 'R_true')
               if k in inp and inp[k] is not None}
        msgs = copy.deepcopy(inp['messages'])              # last turn already holds the generated CoT
        msgs[-1]['content'] = self._completion_str(msgs[-1].get('content'))
        row['messages'] = msgs
        prev_mode = getattr(tmpl, 'mode', None)
        try:
            tmpl.set_mode('train')  # encode the CoT as in-sequence context (head reads the last token)
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
            if not np.isfinite(rp).all():
                raise ValueError('r_pred non-finite')
            return np.concatenate([[1.0], rp])
        except Exception as e:  # noqa: BLE001
            if self._cot_step < 3 or self._cot_step % 20 == 0:
                logger.warning(f'[cot-grpo] head_curve failed ({type(e).__name__}: {str(e)[:140]})')
            return None
        finally:
            if prev_mode is not None:
                tmpl.set_mode(prev_mode)

    def _resolve_head_col(self):
        """Resolve the head-reward column by the CLASS swift actually registered
        (reward_func_names holds __class__.__name__, NOT the orms key 'ttcc_head')."""
        from swift.rewards import orms
        head_cls = orms.get('ttcc_head')
        cands = [i for i, f in enumerate(self.reward_funcs)
                 if (head_cls is not None and type(f) is head_cls) or getattr(f, 'name', None) == 'ttcc_head']
        assert len(cands) == 1, (
            f'[cot-grpo] ttcc_head column not uniquely resolved: cands={cands} '
            f'names={list(self.reward_func_names)}')
        col = cands[0]
        try:
            w = float(self.reward_weights[col])
            assert abs(w - 1.0) < 1e-6, f'[cot-grpo] head column weight={w}, expected 1.0'
        except (TypeError, IndexError):
            pass
        logger.info(f'[cot-grpo] head-reward column={col} names={list(self.reward_func_names)}')
        return col

    def _compute_rewards_per_func(self, inputs):
        rewards = super()._compute_rewards_per_func(inputs)  # ttcc_cot_format column already filled
        if self._cot_cdf is None:
            raise RuntimeError('[cot-grpo] train CDF not injected by the entry (trainer._cot_cdf)')
        if self._cot_head_col is None:
            self._cot_head_col = self._resolve_head_col()
        col = self._cot_head_col

        R_hats = []
        for i, inp in enumerate(inputs):
            R_hat = self._head_curve(inp)
            R_true = inp.get('R')
            if R_true is None:
                R_true = inp.get('R_true')
            if R_hat is None or R_true is None or len(R_true) < 2:
                r = 0.0
            else:
                r, _ = head_reward(R_hat, R_true, self._cot_cdf,
                                   beta=self._cot_beta, alpha=self._cot_alpha,
                                   t_lo=self._cot_t_lo, t_hi=self._cot_t_hi)
            R_hats.append(R_hat)
            rewards[i, col] = r

        self._audit_head_health(R_hats, inputs)
        self._log_curve_std(R_hats)
        self._cot_step += 1
        return rewards

    def _audit_head_health(self, R_hats, inputs):
        """Fail LOUD instead of silently flooring to 0: a systematic head-forward failure
        (r_pred non-attach under the wrapped policy, OOM) is otherwise indistinguishable
        from the legitimate 'CoTs don't move the head' R1 signal."""
        n = len(R_hats)
        n_fail = sum(R is None for R in R_hats)
        fail_frac = n_fail / max(1, n)
        try:
            mode = 'train' if self.model.training else 'eval'
            self._metrics[mode]['cot_head_fail_frac'].append(fail_frac)
        except Exception:  # noqa: BLE001
            pass
        if self._cot_step == 0:
            if fail_frac > 0.5:
                raise RuntimeError(
                    f'[cot-grpo] STEP-0 ABORT: {n_fail}/{n} head forwards failed (r_pred not reachable '
                    f'under the wrapped policy). Fix the head readout before scaling -- do NOT train on a '
                    f'zero reward column. (See preflight audit: r_pred-attachment is the step-0 verifier.)')
            # warm-start visibility: cross-AD curve std. ~0 => head is untrained/dead (same curve for all ads).
            valid = [R for R in R_hats if R is not None]
            if len(valid) >= 2:
                lo, hi = self._cot_t_lo, self._cot_t_hi
                tmax = min(len(R) for R in valid) - 1
                h = min(hi, tmax)
                if h > lo:
                    M = np.stack([R[lo:h + 1] for R in valid])
                    cross_ad_std = float(M.std(axis=0).mean())
                    logger.info(f'[cot-grpo] STEP-0 warm-start check: cross-ad curve std={cross_ad_std:.5f}, '
                                f'sample R_hat[10]={[round(float(R[min(10, len(R)-1)]), 3) for R in valid[:4]]}')
                    if cross_ad_std < 1e-4:
                        logger.warning('[cot-grpo] STEP-0 WARNING: cross-ad curve std ~ 0 -> the head gives the '
                                       'SAME curve for every ad. The trained head likely was NOT warm-started '
                                       '(PEFT modules_to_save closure). Verify the warm-start before trusting RL.')

    def _log_curve_std(self, R_hats):
        """R1 make-or-break: do different CoTs (same ad) give different head curves?
        Mean within-group std of R_hat over the band. ~0 => no advantage => no learning."""
        if getattr(self, 'dynamic_num_samples', False):
            return  # group stride != G under dynamic sampling; the simple grouping would be wrong
        G = int(getattr(self, 'num_generations', 1) or 1)
        if G < 2 or len([R for R in R_hats if R is not None]) < G:
            return
        lo, hi = self._cot_t_lo, self._cot_t_hi
        n = len(R_hats)
        stds = []
        for g0 in range(0, n - n % G, G):
            grp = [R_hats[g0 + j] for j in range(G) if R_hats[g0 + j] is not None]
            if len(grp) < 2:
                continue
            tmax = min(len(R) for R in grp) - 1
            h = min(hi, tmax)
            if h <= lo:
                continue
            M = np.stack([R[lo:h + 1] for R in grp])
            stds.append(float(M.std(axis=0).mean()))
        if stds:
            cstd = float(np.mean(stds))
            try:
                mode = 'train' if self.model.training else 'eval'
                self._metrics[mode]['cot_curve_std'].append(cstd)
            except Exception:  # noqa: BLE001
                pass
            if self._cot_step % 5 == 0:
                logger.info(f'[cot-grpo] step~{self._cot_step} within-group curve std={cstd:.4f} '
                            f'(R1: must be >0 or no gradient)')

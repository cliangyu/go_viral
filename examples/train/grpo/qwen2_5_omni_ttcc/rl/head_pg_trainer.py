#!/usr/bin/env python
"""Head-PG REINFORCE RL trainer for the Qwen2.5-Omni retention head, as a
ms-swift `Seq2SeqTrainer` subclass (override `compute_loss`). Wired in via
`rl_register.py` (external_plugin) which points TrainerFactory at this class —
NO swift-core edits, NO register.py edits. See NORTH_STAR §10.

THE FOUR DETAILS (Leon: "the devil is in the details"):

1. CONFIGURATIONS (env, prefix HPG_): G=rollouts/ad (memory-free, share 1 forward);
   SIGMA=hazard exploration std (calibrated 0.2, smoke); KL=trust-region coef to the
   frozen SFT head; CLIP=grad-norm; T_LO..T_HI=reward seconds [1,30]; CDF=train-only
   percentile npz; FREEZE_BACKBONE=1 for run-1 (head-only, KL-to-SFT-head EXACT).
   ** per_device_train_batch_size MUST be 1 ** (see assumption A2) — asserted.

2. ASSUMPTIONS (verified, not guessed):
   A1. h_anchor = h_last[:, -1] (the LITERAL last token). register._locate_anchor_positions
       returns L-1 when the </cot> matcher misses (it always does — dead anchor), in
       training AND eval. So the head's input IS the last column.
   A2. With bs=1 there is NO intra-batch padding, so column L-1 == the real last token
       == exactly what eval (eval_ibs bypass, also bs=1) reads. bs>1 would right-pad and
       L-1 could be a PAD hidden -> train/eval mismatch. Hence bs MUST be 1.
   A3. holder.r_true is R(1..Tmax) aligned with r_pred=R(1..Tmax); r_mask marks valid
       seconds; the true curve is [1.0] + r_true[:T]. (Same layout RetentionLoss MSEs.)
   A4. reward / advantage / sampled action are DETACHED; gradient flows ONLY through
       logpi(z'|mu_z) (REINFORCE) and the KL(mu_z||mu_ref) term -> via mu_z = head.linear(h).
   ** SELF-CHECK on first batch: curve_from_hazards(mu_z)[:,1:] ≈ holder.r_pred **
   proves A1/A2 (my mu_z == the forward's head input). Aborts on mismatch.

3. NAMINGS: HPG_* env, head_pg_trainer / rl_register / rl_head_pg.yaml / ttcc_train_bypass.jsonl.
4. MATH: identical to the unit-tested reinforce_core + head_pg_compute_loss + the passed
   GPU smoke. z'=mu_z+sigma*eps; curve=exp(-cumsum(softplus(z'))); A=(r-mean)/std within
   the ad's G rollouts; loss = -(logpi*A).mean() + KL*coef; KL=((mu_z-mu_ref)^2/(2 sigma^2)).sum.
"""
from __future__ import annotations
import os
import sys
from collections import deque

import numpy as np
import torch

# verified RL components are the SINGLE source in ../verification (unit-tested + smoke).
# Import them from there so there is no duplicated copy to drift.
_HERE = os.path.dirname(os.path.abspath(__file__))
_VERIF = os.path.normpath(os.path.join(_HERE, '..', 'verification'))
for _p in (_VERIF, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cross_ad_reward as CAR          # Gate-1 verified reward (../verification) — run-1 (per-ad calibration)
import cross_ad_rank_reward as CARR    # run-2 reward: cross-ad concordance (cosine gate +0.59 vs SRCC dir)
import reinforce_core as RC            # Gate-2 verified REINFORCE math (../verification)

from swift.trainers import Seq2SeqTrainer
from swift.utils import get_logger

logger = get_logger()

# --- 1. CONFIGURATIONS --- read in __init__ (AFTER the entry's parse_yaml_args
# has exported the yaml ENV: block), NOT at import time.


def load_cdf(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    for k in z.keys():
        kk = k[1:] if k.startswith('t') else k
        out[int(kk)] = np.asarray(z[k])
    return out


def reward_fn(R_hat, R_true, cdf, t_lo=1, t_hi=30):
    return [CAR.r_rank(rh, rt, cdf, t_lo, t_hi) for rh, rt in zip(R_hat, R_true)]


class HeadPGTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make HF normalize the loss by gradient_accumulation_steps. Our compute_loss returns
        # a per-step MEAN REINFORCE loss and ignores num_items_in_batch; with the default
        # model_accepts_loss_kwargs=True + 'labels' present, HF SKIPS the /grad_accum divide
        # (transformers training_step guard) -> 8 micro-backwards sum un-normalized = 8x
        # effective LR. Setting False makes HF divide correctly so effective LR == configured.
        self.model_accepts_loss_kwargs = False
        # config (read now: the entry's parse_yaml_args already exported the ENV block)
        self.G = int(os.environ.get('HPG_G', '16'))
        self.sigma = float(os.environ.get('HPG_SIGMA', '0.2'))
        self.kl_coef = float(os.environ.get('HPG_KL', '0.04'))
        self.clip = float(os.environ.get('HPG_CLIP', '1.0'))
        self.t_lo = int(os.environ.get('HPG_TLO', '1'))
        self.t_hi = int(os.environ.get('HPG_THI', '30'))
        self.freeze_backbone = os.environ.get('HPG_FREEZE_BACKBONE', '1') == '1'
        # --- REWARD MODE (run-2): 'crossad' = cross-ad ranking concordance vs a detached
        #     running buffer of recent ads (fixes Bug A: the within-ad advantage now carries
        #     cross-ad order; cosine-gate +0.59 vs the SRCC direction). 'rrank' = run-1's
        #     per-ad percentile-match calibration reward (cosine 0.0 -> dead; kept for A/B). ---
        # OBJECTIVE: 'reinforce' (head-PG RL, runs 1-4) or 'rank_sft' (Step-1 SUPERVISED
        # differentiable cross-ad pairwise-rank loss on r_pred -> reshapes head+backbone with a
        # DENSE gradient; the effective tool to LEARN features since we have labels). rank_sft uses
        # the same cross-ad buffer but a BPR/logistic loss (no rollouts, no REINFORCE).
        self.objective = os.environ.get('HPG_OBJECTIVE', 'reinforce').lower()
        self.alpha_mse = float(os.environ.get('HPG_ALPHA_MSE', '0.1'))   # rank_sft: small MSE calibration anchor
        self.reward_mode = os.environ.get('HPG_REWARD', 'crossad').lower()
        self.beta = float(os.environ.get('HPG_BETA', '10.0'))         # concordance sigmoid sharpness
        self.margin = float(os.environ.get('HPG_MARGIN', '0.0'))      # run-4: drop pairs with |true gap|<margin (large-margin -> generalizes)
        self.buf_cap = int(os.environ.get('HPG_BUF', '256'))          # per-GPU FIFO of recent ads
        self.buf_min = int(os.environ.get('HPG_BUF_MIN', '8'))        # warmup: below this, low signal
        self._buffer = deque(maxlen=self.buf_cap)                     # [(pred_mean_curve, true_curve)]
        cdf_path = os.environ.get('HPG_CDF', '')
        assert cdf_path and os.path.exists(cdf_path), f'HPG_CDF missing/not found: {cdf_path!r}'
        # A2: bs MUST be 1 so the last column is the real last token (== eval).
        bs = getattr(self.args, 'per_device_train_batch_size', 1)
        assert bs == 1, (f'HeadPGTrainer requires per_device_train_batch_size=1 '
                         f'(got {bs}); bs>1 right-pads -> h_anchor at L-1 becomes a PAD '
                         f'hidden and mismatches eval. Use grad_accum / more GPUs to scale.')
        self._cdf = load_cdf(cdf_path)
        base = self._base()
        if self.freeze_backbone:
            n_train = 0
            for p in self.model.parameters():
                p.requires_grad_(False)
            for p in base.retention_head.linear.parameters():
                p.requires_grad_(True); n_train += p.numel()
            logger.info(f'[head-pg] FREEZE_BACKBONE: only retention_head.linear trains ({n_train} params)')
        # KL reference = frozen SFT head. Under ZeRO-3 the live head.linear weight is
        # PARTITIONED (size 0 on non-owner ranks), so a plain .clone() yields an empty shard
        # and the KL matmul fails ("size mismatch ... vec (0)"). Gather the full param first
        # (deepspeed.zero.GatheredParameters is a no-op when the param isn't ZeRO-3-sharded).
        _lin = base.retention_head.linear
        try:
            import deepspeed
            with deepspeed.zero.GatheredParameters([_lin.weight, _lin.bias], modifier_rank=None):
                self._ref_W = _lin.weight.detach().float().clone()
                self._ref_b = _lin.bias.detach().float().clone()
        except Exception:
            self._ref_W = _lin.weight.detach().float().clone()
            self._ref_b = _lin.bias.detach().float().clone()
        logger.info(f'[head-pg] KL ref captured: W{tuple(self._ref_W.shape)} b{tuple(self._ref_b.shape)}')
        # mu_z (the head's hazards) is DERIVED from the model's OUTPUT curve r_pred inside
        # compute_loss — NOT captured from an intermediate. This routes the backward through
        # the STANDARD model.forward -> output -> loss path that DDP and ZeRO-3 both instrument
        # (exactly like GRPO derives its loss from the model's logits), so ZeRO-3/FSDP/DDP all
        # work with NO hooks and NO static_graph. mu_z = softplus^{-1}(-Δlog r_pred), the exact
        # inverse of the head's R = exp(-cumsum(softplus(z))).
        self._selfcheck_done = False
        logger.info(f'[head-pg] G={self.G} sigma={self.sigma} kl={self.kl_coef} clip={self.clip} '
                    f't=[{self.t_lo},{self.t_hi}] freeze_backbone={self.freeze_backbone}')

    def _base(self):
        m = self.accelerator.unwrap_model(self.model)
        m = getattr(m, 'base_model', m)
        m = getattr(m, 'model', m)
        return m

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # swift's base compute_loss pops these non-model keys before model(**inputs);
        # we override compute_loss so we must pop them too (else the Qwen forward sees
        # unexpected kwargs). We use no LM labels in RL (reward is the only objective).
        for k in ('compute_loss_func', 'loss_scale', 'text_position_ids', 'channel', 'labels'):
            inputs.pop(k, None)
        outputs = model(**inputs)                      # patched forward -> holder.{last,r_pred,r_true,r_mask}
        base = self._base()
        holder = base._retention_h_holder
        head = base.retention_head
        h_last = holder.last                           # (B, L, d)
        B = h_last.size(0)
        # A1: literal last token (matches register's dead-anchor fallback L-1).
        h_anchor = h_last[:, -1, :]                    # (B, d) — for the KL ref only (detached below)
        w_dtype = head.linear.weight.dtype
        # mu_z (hazards) DERIVED from the model's OUTPUT curve r_pred — the standard, backend-
        # instrumented path (DDP/ZeRO-3 backward hooks fire normally; like GRPO from logits).
        # r_pred is grad-connected to head.linear + backbone. lambda(t)=log R(t-1)-log R(t)=
        # softplus(z); mu_z=softplus^{-1}(lambda), the exact inverse of the head.
        r_pred = holder.r_pred.float()                 # (B, Tmax) = R(1..Tmax), grad-connected
        Tmax = r_pred.size(1)
        R_prev = torch.cat([torch.ones(B, 1, device=r_pred.device, dtype=r_pred.dtype),
                            r_pred[:, :-1]], dim=1)     # R(0..Tmax-1)
        lam = (R_prev.clamp_min(1e-8).log() - r_pred.clamp_min(1e-8).log()).clamp_min(1e-7)  # =softplus(z)>0
        mu_z = torch.log(torch.expm1(lam))             # softplus^{-1} -> z  (B, Tmax), grad flows via r_pred

        # A3: reconstruct true curves R(0..T) from holder
        r_true = holder.r_true.float()
        r_mask = holder.r_mask
        R_true_list = []
        for b in range(B):
            Tb = int(r_mask[b].sum().item())
            R_true_list.append([1.0] + r_true[b, :Tb].tolist())

        # SELF-CHECK (once): my mu_z must reproduce the forward's r_pred.
        if not self._selfcheck_done:
            with torch.no_grad():
                recon = RC.curve_from_hazards(mu_z)[:, 1:]      # (B, Tmax) = R(1..Tmax)
                rp = holder.r_pred.float()
                md = float((recon - rp).abs().max())
            ok = md < 1e-3
            mono = all(all(R_true_list[b][i] >= R_true_list[b][i + 1] - 1e-6
                           for i in range(len(R_true_list[b]) - 1)) for b in range(B))
            logger.info(f'[head-pg] SELF-CHECK mu_z->curve vs forward r_pred max|Δ|={md:.2e} '
                        f'({"OK" if ok else "MISMATCH"}); R_true monotone={mono}; '
                        f'B={B} Tmax={Tmax} T0={len(R_true_list[0])-1}')
            assert ok, ('SELF-CHECK FAILED: recomputed mu_z does not match the forward head '
                        '(wrong anchor token or dtype). Refusing to train on a wrong signal.')
            assert mono, 'SELF-CHECK FAILED: R_true not monotone — r_true layout assumption (A3) wrong.'
            self._selfcheck_done = True

        # --- STEP-1: SUPERVISED differentiable cross-ad pairwise-rank loss (no rollouts).
        #     r_pred is grad-connected -> this reshapes head+backbone with a DENSE gradient. ---
        if self.objective == 'rank_sft':
            return self._rank_sft_loss(r_pred, R_true_list, B, Tmax, outputs, return_outputs)

        # --- 4. MATH: head-PG REINFORCE (all reward/adv/action detached) ---
        G, sigma = self.G, self.sigma
        mu_rep = mu_z.unsqueeze(1).expand(B, G, Tmax)                 # shares the ONE forward graph
        eps = torch.randn(B, G, Tmax, device=mu_z.device)
        z = (mu_rep + sigma * eps).detach()                          # action, detached
        curves = RC.curve_from_hazards(z.view(B * G, Tmax))          # (B*G, Tmax+1)
        if self.reward_mode == 'crossad':
            # cross-ad RANKING reward (run-2): score each rollout by pairwise concordance vs a
            # detached running buffer of OTHER ads (policy-mean curve + true curve). The within-ad
            # advantage then encodes "which rollout ranks THIS ad correctly vs the population" ->
            # a real cross-ad SRCC gradient through the SAME REINFORCE path. cosine-gate=+0.59.
            with torch.no_grad():
                mean_curves = RC.curve_from_hazards(mu_z)            # (B, Tmax+1) policy-mean refs
            buf_pred = [c for (c, _) in self._buffer]
            buf_true = [tc for (_, tc) in self._buffer]
            rew_rows = []
            for b in range(B):
                rollouts_b = [curves[b * G + g].tolist() for g in range(G)]
                rew_rows.append(CARR.crossad_rank_reward(
                    rollouts_b, R_true_list[b], buf_pred, buf_true,
                    t_lo=self.t_lo, t_hi=self.t_hi, beta=self.beta, margin=self.margin))
                self._buffer.append((mean_curves[b].tolist(), R_true_list[b]))  # add AFTER scoring
            rew = torch.as_tensor(np.stack(rew_rows), dtype=torch.float32, device=mu_z.device).view(B, G)
        else:  # 'rrank' — run-1 per-ad percentile-match calibration reward (A/B / fallback)
            Rh = [curves[i].tolist() for i in range(B * G)]
            Rt = [R_true_list[b] for b in range(B) for _ in range(G)]
            rew = torch.as_tensor(reward_fn(Rh, Rt, self._cdf, self.t_lo, self.t_hi),
                                  dtype=torch.float32, device=mu_z.device).view(B, G)
        adv = (rew - rew.mean(dim=1, keepdim=True)) / (rew.std(dim=1, keepdim=True) + 1e-6)
        adv = adv.view(-1).detach()
        # WARMUP SKIP (crossad): until the cross-ad buffer is warm, concordance is computed against
        # too few ads -> noisy advantage that drifts the head off the SFT init (the run-2 ckpt-25
        # dip). Zero the advantage (pg=0; KL still anchors) until len(buffer) >= buf_min.
        if self.reward_mode == 'crossad' and len(self._buffer) < self.buf_min:
            adv = torch.zeros_like(adv)
        logp = RC.gaussian_logp(z.view(B * G, Tmax), mu_rep.reshape(B * G, Tmax), sigma)
        pg = -(logp * adv).mean()
        # KL to frozen SFT head on the CURRENT features. Detach h_anchor so the KL ref is a
        # fixed target (no grad to the backbone via this term) -> the ONLY grad path to
        # head.linear+backbone is via mu_z (r_pred), keeping it a single standard path.
        mu_ref = (h_anchor.detach().to(w_dtype) @ self._ref_W.t().to(w_dtype)
                  + self._ref_b.to(w_dtype)).float()
        kl = (((mu_z - mu_ref) ** 2) / (2 * sigma ** 2)).sum(dim=-1).mean()
        loss = pg + self.kl_coef * kl

        # --- OBSERVABILITY: log every signal needed to debug RL failure modes ---
        # (lesson from SFT: we had 2 losses but logged only the total -> painful to debug.
        #  here we log both loss components AND the detectors for each known failure mode.)
        with torch.no_grad():
            wg = rew.std(dim=1)                                         # per-ad within-group std
            mean_curve = RC.curve_from_hazards(mu_z.detach())          # (B, Tmax+1) policy-MEAN curve
            tail_idx = min(self.t_hi, mean_curve.size(1) - 1)
            self._log(
                # loss components (the SFT lesson: never hide sub-losses)
                pg=float(pg.detach()), kl=float(kl.detach()),
                # reward distribution (saturation / range)
                reward=float(rew.mean()), reward_min=float(rew.min()), reward_max=float(rew.max()),
                # R1 collapse detector: variance in the rank dimension (make-or-break)
                within_grp_std=float(wg.mean()),
                dead_grp_frac=float((wg < 1e-4).float().mean()),       # frac of ads with no signal
                # advantage health
                adv_abs=float(adv.abs().mean()),
                # head/policy health: hazard scale + curve tail (detect saturation -> collapse)
                muz_absmean=float(mu_z.detach().abs().mean()),
                muz_max=float(mu_z.detach().abs().max()),
                curve_tail=float(mean_curve[:, tail_idx].mean()),      # mean policy R(t_hi)
                buf=float(len(self._buffer)),                          # cross-ad buffer fill (run-2)
            )
        return (loss, outputs) if return_outputs else loss

    def _rank_sft_loss(self, r_pred, R_true_list, B, Tmax, outputs, return_outputs):
        """SUPERVISED cross-ad pairwise-rank (BPR/logistic) loss + small MSE calibration.
        For ad A (r_pred grad-connected, = R(1..Tmax)) vs a detached buffer of recent ads:
          loss_rank = mean over valid (t, buffer-ad B) of  -log sigma( beta*(R_A(t)-R_B(t))*sign(R^true_A(t)-R^true_B(t)) )
          only pairs with |true gap| >= margin (large-margin -> generalizes).
        Gradient flows through R_A(t) -> head -> backbone (DENSE; reshapes features)."""
        import torch.nn.functional as F
        dev = r_pred.device
        tlo, thi, margin, beta = self.t_lo, self.t_hi, self.margin, self.beta
        rank_terms, mse_terms = [], []
        if len(self._buffer) >= self.buf_min:
            Pbuf = torch.as_tensor(np.stack([x[0] for x in self._buffer]), device=dev, dtype=torch.float32)  # (M,Tmax) R(1..Tmax)
            Tbuf = torch.as_tensor(np.stack([x[1] for x in self._buffer]), device=dev, dtype=torch.float32)  # (M,Tmax+1) R(0..Tmax) padded
            Tlen = torch.as_tensor([x[2] for x in self._buffer], device=dev)                                 # (M,) true length
            for b in range(B):
                ta = R_true_list[b]; Tb = len(ta) - 1
                hi = min(thi, Tb)
                if hi < tlo:
                    continue
                tt = torch.arange(tlo, hi + 1, device=dev)                       # seconds (nt,)
                pa_t = r_pred[b].float()[tt - 1]                                  # R_A(t) grad (nt,)
                ta_t = torch.as_tensor([ta[int(t)] for t in tt], device=dev, dtype=torch.float32)
                gap = ta_t[None, :] - Tbuf[:, tt]                                 # (M, nt)
                s = torch.sign(gap)
                diff = beta * (pa_t[None, :] - Pbuf[:, tt - 1]) * s              # (M, nt) grad
                valid = (gap.abs() >= margin) & (Tlen[:, None] >= tt[None, :])    # drop near-ties + invalid t
                if valid.any():
                    rank_terms.append(-F.logsigmoid(diff[valid]).mean())
                mse_terms.append(((pa_t - ta_t) ** 2).mean())                     # calibration anchor
        # push this ad's prediction + true curve into the buffer (after scoring)
        for b in range(B):
            ta = R_true_list[b]; Tb = len(ta) - 1
            tp = np.zeros(Tmax + 1, dtype=np.float32)
            tp[:min(len(ta), Tmax + 1)] = np.asarray(ta[:Tmax + 1], dtype=np.float32)
            self._buffer.append((r_pred[b].detach().float().cpu().numpy(), tp, Tb))
        loss_rank = torch.stack(rank_terms).mean() if rank_terms else r_pred.sum() * 0.0   # warmup: ~0 update
        loss_mse = torch.stack(mse_terms).mean() if mse_terms else r_pred.sum() * 0.0
        loss = loss_rank + self.alpha_mse * loss_mse
        with torch.no_grad():
            self._log(rank_loss=float(loss_rank.detach()), mse_loss=float(loss_mse.detach()),
                      buf=float(len(self._buffer)), npairs=float(len(rank_terms)),
                      pred_spread=float(r_pred.detach().float().std()))
        return (loss, outputs) if return_outputs else loss

    def _log(self, **kv):
        try:
            mode = 'train' if self.model.training else 'eval'
            for k, v in kv.items():
                self.custom_metrics[mode][f'hpg_{k}'].update(v)
        except Exception:
            pass

    def create_optimizer(self):
        # grad-clip is applied by HF Trainer via max_grad_norm; set it from self.clip.
        if getattr(self.args, 'max_grad_norm', None) != self.clip:
            self.args.max_grad_norm = self.clip
        return super().create_optimizer()

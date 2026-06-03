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
# soft pairwise (Kendall) cross-ad RANK reward + the IBS-acc anchor. cross_ad_reward.py and
# cross_ad_reward_pairwise.py live in ../verification (on sys.path via head_reward's import).
from cross_ad_reward_pairwise import r_pair_soft_batch
from cross_ad_reward import r_acc as _r_acc

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
        # soft pairwise rank reward (replaces the percentile-MATCH r_rank when enabled)
        self._cot_use_pairwise = os.environ.get('COT_USE_PAIRWISE', '0') == '1'
        self._cot_tau = float(os.environ.get('COT_TAU', '0.02'))
        # BOTTLENECK: read the head off a TEXT-ONLY re-encode of the CoT (no video) -> the reward
        # depends on the CoT text ALONE (the cot_only_feature path; precheck: last-token, SRCC 0.17).
        self._cot_bottleneck = os.environ.get('COT_BOTTLENECK', '0') == '1'
        self._cot_pool = os.environ.get('COT_BOTTLENECK_POOL', 'last')   # last|mean (precheck: mean is dead)
        # ALL-GATHER the cross-ad reward inputs across ranks so each rollout ranks vs ALL ads in the
        # global step (decouples per-device generation memory from cross-ad signal). Inert at world_size=1.
        self._cot_allgather = os.environ.get('COT_ALLGATHER', '1') == '1'
        self._cot_global_nads = None     # set by _pairwise_rewards (gather path) = GLOBAL distinct ads
        self._head_wb_cache = None
        self._cot_step = 0
        logger.info(f'[cot-grpo] head reward: beta={self._cot_beta} alpha={self._cot_alpha} '
                    f't=[{self._cot_t_lo},{self._cot_t_hi}] G={getattr(self, "num_generations", "?")} '
                    f'pairwise={self._cot_use_pairwise} tau={self._cot_tau} '
                    f'BOTTLENECK={self._cot_bottleneck} pool={self._cot_pool}')

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

    def _head_wb(self):
        """Frozen retention-head linear weight/bias for MANUAL application to the cot-only hidden.
        The model's built-in readout (out.r_pred) can choke on the minimal text-only input, so we
        apply softplus/cumsum/exp by hand (exactly the precheck path). Head is in modules_to_save ->
        reach the ACTIVE wrapper module. Cached; head gets no gradient under GRPO (reward is no_grad)."""
        if self._head_wb_cache is None:
            head = getattr(self._base(), 'retention_head', None)
            if head is None:
                raise RuntimeError('[cot-grpo] BOTTLENECK: base model has no retention_head')
            if hasattr(head, 'modules_to_save'):
                act = getattr(head, 'active_adapter', 'default')
                if isinstance(act, (list, tuple)):
                    act = act[0] if act else 'default'
                lin = (head.modules_to_save[act] if act in head.modules_to_save
                       else next(iter(head.modules_to_save.values()))).linear
            else:
                lin = head.linear
            W = lin.weight.detach().float().cpu().numpy()
            b = lin.bias.detach().float().cpu().numpy()
            self._head_wb_cache = (W, b)
            import numpy as _np
            logger.info(f'[cot-grpo] BOTTLENECK head W{W.shape} Wstd={float(_np.std(W)):.4f} '
                        f'bias[:3]={b[:3].tolist()} (NOTE bias=-3 is the TRAINED value too; the real '
                        f'warm-start check = cross-ad curve std in _audit_head_health, not bias)')
        return self._head_wb_cache

    def _cot_only_curve(self, inp) -> np.ndarray | None:
        """BOTTLENECK readout: re-encode ONLY the generated CoT as text (NO video/audio/image),
        read model._retention_h_holder.last (set by the lm_head pre-hook during the base forward),
        pool it, and apply the frozen head by hand -> R(0..T). The curve is a deterministic function
        of the CoT TEXT alone, so RL can only raise the reward by making the CoT verbalize retention."""
        from swift.template.template_inputs import TemplateInputs
        cot = self._completion_str(inp['messages'][-1].get('content'))
        if '<cot>' not in cot:
            cot = '<cot>' + cot
        if '</cot>' not in cot:
            cot = cot + '</cot>'
        row = {'messages': [
            {'role': 'system', 'content': 'You forecast second-by-second short-video audience retention.'},
            {'role': 'user', 'content': 'Per-second reasoning about a short-video ad:'},
            {'role': 'assistant', 'content': cot}],
            'T': inp.get('T'),
            'R': inp.get('R') if inp.get('R') is not None else inp.get('R_true')}   # explicit (R may be np array)
        tmpl = self.template
        prev_mode = getattr(tmpl, 'mode', None)
        try:
            tmpl.set_mode('train')
            enc = tmpl.encode(TemplateInputs.from_dict(row))
            batch = tmpl.data_collator([enc])
            dev = self.accelerator.device
            batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch.pop('labels', None)
            holder = self._base()._retention_h_holder
            holder.last = None
            with torch.no_grad():
                try:
                    self.model(**batch)
                except Exception:  # noqa: BLE001 -- readout may error on minimal text; holder.last is set pre-readout
                    pass
            h = holder.last
            if h is None:
                return None
            h = h[0].float()
            am = batch.get('attention_mask')
            if am is not None:
                h = h[am[0].bool()]
            if self._cot_pool == 'attn_pool':
                # B2 widened readout: run the trained CoTAttnPool over the CoT-span hiddens
                # (reads the WHOLE reasoning, not just the last token) -> CoT content becomes
                # load-bearing -> within-group reward variance -> a live gradient. Same frozen head W,b.
                pool = getattr(self._base(), 'retention_pool', None)
                if pool is None:
                    hv = h[-1].cpu().numpy()                         # fallback: pool not attached
                else:
                    pdtype = next(pool.parameters()).dtype
                    h_seq = h.unsqueeze(0).to(pdtype)                # (1, L, d)
                    m = torch.ones(h_seq.shape[:2], dtype=torch.bool, device=h_seq.device)
                    with torch.no_grad():
                        pooled = pool(h_seq, m)                      # (1, d)
                    hv = pooled[0].float().cpu().numpy()
            else:
                hv = (h.mean(0) if self._cot_pool == 'mean' else h[-1]).cpu().numpy()
            W, b = self._head_wb()
            z = W @ hv + b
            lam = np.logaddexp(0.0, z)                          # softplus
            return np.concatenate([[1.0], np.exp(-np.cumsum(lam))])
        except Exception as e:  # noqa: BLE001
            if self._cot_step < 3 or self._cot_step % 20 == 0:
                logger.warning(f'[cot-grpo] cot_only_curve failed ({type(e).__name__}: {str(e)[:140]})')
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

        R_hats, R_trues, ad_ids, cot_lens = [], [], [], []
        for inp in inputs:
            R_hats.append(self._cot_only_curve(inp) if self._cot_bottleneck else self._head_curve(inp))
            rt = inp.get('R')
            if rt is None:
                rt = inp.get('R_true')
            R_trues.append(rt)
            ad_ids.append(inp.get('ad_id'))
            cot_lens.append(len(self._completion_str(inp['messages'][-1].get('content'))))
        n_ads = len(set(str(a) for a in ad_ids))           # LOCAL distinct ads on this rank
        self._cot_global_nads = None                       # reset; gather path sets the GLOBAL count
        pair = self._pairwise_rewards(R_hats, R_trues, ad_ids) if self._cot_use_pairwise else None
        n_ads_eff = self._cot_global_nads if self._cot_global_nads is not None else n_ads  # GLOBAL when gathered

        pair_vals, racc_vals = [], []   # fine-grained reward sub-terms for observability
        for i in range(len(inputs)):
            R_hat, R_true = R_hats[i], R_trues[i]
            if R_hat is None or R_true is None or len(R_true) < 2:
                r = 0.0
            elif pair is not None and pair[i] is not None:
                # soft pairwise cross-ad RANK (replaces percentile-match r_rank) + IBS-acc anchor
                racc_i = _r_acc(R_hat, R_true, len(R_true) - 1)
                r = self._cot_beta * pair[i] + self._cot_alpha * racc_i
                pair_vals.append(float(pair[i])); racc_vals.append(float(racc_i))
            else:
                r, _ = head_reward(R_hat, R_true, self._cot_cdf,
                                   beta=self._cot_beta, alpha=self._cot_alpha,
                                   t_lo=self._cot_t_lo, t_hi=self._cot_t_hi)
            rewards[i, col] = r

        # --- dynamics watch: cross-ad availability + length-Goodhart ---
        rcol = rewards[:, col].detach().float().cpu().numpy()
        try:
            mode = 'train' if self.model.training else 'eval'
            self._metrics[mode]['cot_n_ads'].append(float(n_ads_eff))
            self._metrics[mode]['cot_len_mean'].append(float(np.mean(cot_lens)))
            if pair_vals:   # fine-grained reward observability: the MOVABLE signal vs the calib floor
                self._metrics[mode]['cot_r_pair'].append(float(np.mean(pair_vals)))
                self._metrics[mode]['cot_r_pair_centered'].append(float(2.0 * np.mean(pair_vals) - 1.0))
            if racc_vals:
                self._metrics[mode]['cot_r_acc'].append(float(np.mean(racc_vals)))
            if len(cot_lens) >= 3 and np.std(rcol) > 0 and np.std(cot_lens) > 0:
                rl = np.argsort(np.argsort(rcol)).astype(float)
                cl = np.argsort(np.argsort(cot_lens)).astype(float)
                self._metrics[mode]['cot_len_reward_corr'].append(float(np.corrcoef(rl, cl)[0, 1]))
        except Exception:  # noqa: BLE001
            pass
        if self._cot_step == 0 and n_ads_eff < 2:
            logger.warning(f'[cot-grpo] STEP-0 WARNING: only {n_ads_eff} distinct ad(s) in the (global) reward '
                           f'pool -> pairwise has no cross-ad pairs, caller falls back to the saturated '
                           f'percentile head_reward (weak gradient). Raise generation_batch_size or enable COT_ALLGATHER.')
        if self._cot_step % 5 == 0:
            logger.info(f'[cot-grpo] step~{self._cot_step} n_ads={n_ads_eff}(global) local={n_ads} '
                        f'cot_len_mean={np.mean(cot_lens):.0f} reward[min..max]=[{rcol.min():.3f}..{rcol.max():.3f}]')
        self._audit_head_health(R_hats, inputs)
        self._log_curve_std(R_hats)
        self._cot_step += 1
        return rewards

    def _pairwise_rewards(self, R_hats, R_trues, ad_ids):
        """Batch soft-pairwise (Kendall) cross-ad RANK reward, per rollout.
        Each ad's R_hat is scored against the OTHER ads' mean predicted curve (sign from the TRUE
        order), summed over t in [t_lo,t_hi] -> a soft Kendall-tau. Within-group variance for ad a
        comes from a's R_hat varying across its rollouts. Returns a list aligned to R_hats; None for
        rollouts with a failed/short curve (caller floors those to 0)."""
        valid = [i for i in range(len(R_hats))
                 if R_hats[i] is not None and R_trues[i] is not None and len(R_trues[i]) >= 2]
        out = [None] * len(R_hats)
        vh = [[float(x) for x in R_hats[i]] for i in valid]
        vt = [[float(x) for x in R_trues[i]] for i in valid]
        va = [str(ad_ids[i]) for i in valid]

        # --- ALL-GATHER: pool the cross-ad inputs across ALL ranks so each rollout ranks vs the GLOBAL
        #     ad set (not just this device's 2 ads). EVERY rank MUST reach all_gather_object (no early
        #     return before it) or the collective hangs. ---
        import torch.distributed as dist
        if (self._cot_allgather and dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            ws, rk = dist.get_world_size(), dist.get_rank()
            gathered = [None] * ws
            # NO try/except around the collective (Codex review 2026-06-02): all_gather_object is a
            # COLLECTIVE; a per-rank catch-and-fallback-to-local would DESYNC ranks and DEADLOCK the
            # next collective for hours. The entry condition above is uniform across ranks (same env +
            # world_size), so all ranks enter together; let any failure PROPAGATE (visible, restartable
            # crash) rather than hang. r_pair_soft_batch is deterministic on the gathered set (same on
            # every rank), so it is left un-caught too — a throw crashes all ranks together, not one.
            dist.all_gather_object(gathered, list(zip(vh, vt, va)))
            g_vh, g_vt, g_va, my_pos = [], [], [], []       # my_pos: (global_idx, local_valid_k)
            for r, pl in enumerate(gathered):
                for k, (h, t, a) in enumerate(pl):
                    if r == rk:
                        my_pos.append((len(g_vh), k))
                    g_vh.append(h); g_vt.append(t); g_va.append(a)
            self._cot_global_nads = len(set(g_va))          # GLOBAL distinct ads (for the dynamics log)
            if self._cot_global_nads < 2:
                return out                                  # global pool single-ad -> caller uses head_reward
            if self._cot_step == 0:
                logger.info(f'[cot-grpo] pairwise+ALLGATHER: {len(g_vh)} rollouts / '
                            f'{self._cot_global_nads} distinct ads across {ws} ranks, tau={self._cot_tau}')
            rp = r_pair_soft_batch(g_vh, g_vt, g_va, t_lo=self._cot_t_lo, t_hi=self._cot_t_hi, tau=self._cot_tau)
            for gidx, k in my_pos:
                out[valid[k]] = rp[gidx]
            return out

        # --- local-only (single GPU / gather disabled / gather failed) ---
        if len(valid) < 2:
            return out
        if self._cot_step == 0:
            logger.info(f'[cot-grpo] pairwise (local): {len(valid)} rollouts, '
                        f'{len(set(va))} distinct ads, tau={self._cot_tau}')
        try:
            rp = r_pair_soft_batch(vh, vt, va, t_lo=self._cot_t_lo, t_hi=self._cot_t_hi, tau=self._cot_tau)
            for k, i in enumerate(valid):
                out[i] = rp[k]
        except Exception as e:  # noqa: BLE001 -- never crash the step; fall back to head_reward
            if self._cot_step < 3:
                logger.warning(f'[cot-grpo] pairwise reward failed '
                               f'({type(e).__name__}: {str(e)[:120]}); falling back to r_rank')
        return out

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

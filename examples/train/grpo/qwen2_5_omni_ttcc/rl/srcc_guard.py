"""In-loop held-out cross-ad SRCC guard for the head-PG RL run.

WHY THIS EXISTS
  The documented #1 failure mode is OVER-OPTIMIZATION: train reward climbs while
  held-out val cross-ad SRCC falls (RL_INVESTIGATION_LOG.md). But every full-FT RL
  config ships `eval_strategy: 'no'`, so the harm is INVISIBLE during training and
  only discovered by evaluating saved ckpts offline. This callback makes it LIVE.

WHAT IT DOES (every `guard_steps`, during training)
  1. Runs the LEAK-FREE bypass eval (assistant -> '<cot></cot>', deterministic z=mu_z)
     on a DISJOINT held-out shard -- the SAME forward as srcc_eval.py (single source
     of truth for the metric).
  2. Logs the FULL metric set to wandb (cross-ad SRCC + auc + per-t + saturation/drift
     detectors) so you SEE train-reward-up / val-SRCC-down in one view.
  3. Tracks the running best -> best-ckpt selection. OBSERVE-ONLY: it does NOT
     auto-stop training (project rule: all experiments end MANUALLY). It only
     LOGS a rollback-candidate warning when held-out SRCC has been below best for
     `patience` consecutive guard evals; the human decides when to stop.

DESIGN NOTES
  - Reuses the existing observability pipeline by logging via `wandb.log` (rank-0 only),
    the same sink swift's custom_metrics route to. No new infra; mirrors register.py.
  - The eval forward runs on ALL ranks (DeepSpeed gathers ZeRO-3 params for the fwd,
    same as swift's own evaluate); only rank-0 logs/decides. Redundant-but-parallel ->
    wall-clock ~ one pass, not world_size passes.
  - MUST be smoke-tested on the real 2-GPU topology before a production run
    (the 'test real topology early' lesson: a near-fully-frozen / ZeRO-3 fwd in a
    callback is exactly the class of thing that hangs on multi-rank if untested).
"""
from __future__ import annotations
import json
import numpy as np
import torch
from transformers import TrainerCallback


def _spearman(x, y):
    """Spearman rho via Pearson on average ranks. Identical to srcc_eval.spearman."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    def rank(a):
        order = a.argsort(); r = np.empty(len(a), float); r[order] = np.arange(len(a))
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt); starts = csum - cnt
        return ((starts + csum - 1) / 2.0)[inv]
    rx, ry = rank(x), rank(y); rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


class HeldoutSRCCGuard(TrainerCallback):
    """Run the leak-free held-out cross-ad SRCC eval in-loop + log + best/early-stop."""

    def __init__(self, holdout_jsonl, template, guard_steps=25, t_lo=1, t_hi=30,
                 ref_mu_npz=None, patience=3, margin=0.0, per_ad_timeout=120,
                 log_per_t=(1, 5, 10, 15, 20, 25, 30)):
        self.rows = []
        with open(holdout_jsonl) as f:
            for line in f:
                r = json.loads(line)
                if (r.get('R') or r.get('R_true')):
                    self.rows.append(r)
        self.template = template
        self.guard_steps = int(guard_steps)
        self.t_lo, self.t_hi = t_lo, t_hi
        self.patience, self.margin = patience, float(margin)
        self.per_ad_timeout = per_ad_timeout
        self.log_per_t = log_per_t
        # frozen-SFT reference mu_z per ad_id, for the REAL backbone opt-distance.
        # dict {ad_id: mu_z_vector}; produced once offline (see build_ref_mu()).
        self.ref_mu = None
        if ref_mu_npz:
            z = np.load(ref_mu_npz, allow_pickle=True)
            self.ref_mu = {str(a): np.asarray(z['mu'][i], float)
                           for i, a in enumerate(z['ad_ids'])}
        self.best = -1.0
        self.best_step = -1
        self.bad = 0

    # ----- the eval pass (mirrors srcc_eval.main's per-ad loop) -----
    def _bypass_eval(self, model):
        from swift.template.template_inputs import TemplateInputs
        from swift.template.base import MaxLengthError
        preds, trues, Ts, ids, muz_abs = [], [], [], [], []
        for r in self.rows:
            R_true = np.array(r.get('R') or r.get('R_true'), dtype=np.float64)
            T = len(R_true) - 1
            if T < 5:
                continue
            row = dict(r); row['messages'] = list(r['messages'])
            row['messages'][-1] = {'role': 'assistant', 'content': '<cot></cot>'}
            row['audios'] = []
            try:
                enc = self.template.encode(TemplateInputs.from_dict(row))
                batch = self.template.data_collator([enc])
                batch = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                with torch.no_grad():
                    out = model(**batch)
                rp = getattr(out, 'r_pred', None)
                if rp is None:
                    rp = model._retention_h_holder.r_pred
                R_pred = np.concatenate([[1.0], rp[0].float().cpu().numpy()])
            except (MaxLengthError, Exception):  # noqa
                continue
            preds.append(R_pred); trues.append(R_true); Ts.append(T)
            ids.append(str(r.get('ad_id', '')))
        return preds, trues, Ts, ids

    def _metrics(self, preds, trues, Ts, ids):
        m = {}
        t_range = list(range(self.t_lo, self.t_hi + 1))
        # per-second cross-ad Spearman (the eval statistic)
        per_t = {}
        for t in t_range:
            pv, tv = [], []
            for P, Tr, T in zip(preds, trues, Ts):
                if T >= t and t < len(P):
                    pv.append(P[t]); tv.append(Tr[t])
            rho = _spearman(pv, tv)
            if not np.isnan(rho):
                per_t[t] = rho
        m['eval_cross_ad_srcc'] = float(np.mean(list(per_t.values()))) if per_t else float('nan')
        for t in self.log_per_t:
            if t in per_t:
                m[f'eval_srcc_t{t}'] = per_t[t]
        # auc_spearman: single integrated-retention cross-ad Spearman (clean CI, 1-D)
        auc_p, auc_t = [], []
        for P, Tr, T in zip(preds, trues, Ts):
            hi = min(self.t_hi, T)
            if hi >= self.t_lo:
                auc_p.append(float(np.mean(P[self.t_lo:hi + 1])))
                auc_t.append(float(np.mean(Tr[self.t_lo:hi + 1])))
        m['eval_auc_spearman'] = _spearman(auc_p, auc_t) if len(auc_p) >= 3 else float('nan')
        m['eval_n_ads'] = len(preds)
        # saturation / collapse detectors on the policy-mean curve
        if preds:
            tails = [P[min(self.t_hi, len(P) - 1)] for P in preds]
            m['eval_curve_tail'] = float(np.mean(tails))
            m['eval_curve_tail_std'] = float(np.std(tails))  # outcome diversity proxy
        # real backbone opt-distance vs frozen SFT (sqrt(KL)), if reference cached
        if self.ref_mu is not None and preds:
            # reconstruct mu_z from R_pred: lam=-Δlog R; mu_z=softplus^{-1}(lam)
            dists = []
            for P, adid in zip(preds, ids):
                ref = self.ref_mu.get(adid)
                if ref is None:
                    continue
                lr = np.log(np.clip(P, 1e-8, 1.0))
                lam = np.clip(-(lr[1:] - lr[:-1]), 1e-7, None)
                muz = np.log(np.expm1(lam))
                n = min(len(muz), len(ref))
                dists.append(float(np.linalg.norm(muz[:n] - ref[:n])))
            if dists:
                m['eval_opt_distance'] = float(np.mean(dists))
        return m

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or step % self.guard_steps != 0:
            return control
        model = kwargs.get('model')
        if model is None:
            return control
        was_training = model.training
        model.eval()
        try:
            preds, trues, Ts, ids = self._bypass_eval(model)
            metrics = self._metrics(preds, trues, Ts, ids)
        finally:
            if was_training:
                model.train()
        srcc = metrics.get('eval_cross_ad_srcc', float('nan'))
        # best-ckpt + early-stop bookkeeping (computed on all ranks identically)
        is_best = (not np.isnan(srcc)) and (srcc > self.best + self.margin)
        if is_best:
            self.best, self.best_step, self.bad = srcc, step, 0
        elif not np.isnan(srcc):
            self.bad += 1
        metrics['eval_best_srcc'] = self.best
        metrics['eval_best_step'] = self.best_step
        metrics['eval_steps_since_best'] = step - self.best_step
        metrics['eval_is_best'] = float(is_best)
        # log (rank-0 only) to the live wandb run -- same sink as custom_metrics
        if args.local_rank in (-1, 0):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({**metrics, 'guard_step': step}, step=step)
            except Exception:
                pass
            print(f'[guard] step={step} cross_ad_srcc={srcc:.4f} auc={metrics.get("eval_auc_spearman", float("nan")):.4f} '
                  f'best={self.best:.4f}@{self.best_step} since_best={metrics["eval_steps_since_best"]} '
                  f'n={metrics["eval_n_ads"]} tail={metrics.get("eval_curve_tail", float("nan")):.3f}', flush=True)
        # OBSERVE-ONLY rollback CANDIDATE warning (we never auto-stop; experiments end manually).
        if self.bad >= self.patience and args.local_rank in (-1, 0):
            print(f'[guard] ROLLBACK CANDIDATE: held-out SRCC below best for {self.bad} guard-evals; '
                  f'best={self.best:.4f}@checkpoint-{self.best_step}. (NOT stopping — manual decision.)', flush=True)
        return control

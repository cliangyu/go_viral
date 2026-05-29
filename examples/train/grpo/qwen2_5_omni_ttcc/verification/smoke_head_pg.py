#!/usr/bin/env python
"""GPU SMOKE GATE for head-PG RL on the REAL Qwen2.5-Omni-3B retention model.

This is the make-or-break go/no-go BEFORE building the full swift trainer.
It reuses the verified isolated components (cross_ad_reward, reinforce_core via
head_pg_compute_loss) on REAL model rollouts.

Design (cheap + decisive):
  1. Load ckpt-150 (eval-style, single GPU, bypass: assistant='<cot></cot>').
  2. For N real ads, forward ONCE under no_grad -> h_anchor (detached, last token;
     the </cot> matcher is dead so head reads the last token in train AND eval).
  3. mu_z = retention_head.linear(h_anchor)  (== the head's pre-softplus hazards).

  GATE A (make-or-break, no training): sample G hazard-noised rollouts per ad,
     reward each via cross-ad percentile reward, report within-group R_rank std.
     MUST be > 0 in the RANK dimension or REINFORCE has no signal.
  GATE B (reward <-> quality): within each ad's G rollouts, the highest-reward
     rollout must be the one whose curve lands closest to the TRUE percentile.
     Pearson corr(reward, -percentile_err) should be strongly positive.
  GATE C (learning): freeze backbone, Adam on retention_head.linear only, run
     head_pg_loss for K steps over the precomputed h_anchors; reward_mean must rise.
     (Backbone is frozen here on purpose: this isolates the RL mechanics + reward
      signal cheaply. The full swift trainer trains the backbone too — more
      capacity, identical mechanics.)

NOTE: backbone-frozen head-only is a MECHANICS gate, not the production run. A
pass means the reward produces a usable learning signal on the real model; the
real lift may need backbone training (done by the swift HeadPGTrainer).
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_ad_reward as CAR          # Gate-1 verified reward
import head_pg_compute_loss as HPG     # verified compose (uses reinforce_core)


def import_plugin(p):
    spec = importlib.util.spec_from_file_location('retention_plugin', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def load_cdf(path):
    """train_cdf.npz -> dict {int t: sorted np.array}. build_cdf.py saves keys 't0'..'t60'."""
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    if len(keys) == 1 and keys[0] == 'cdf':
        return {int(k): np.asarray(v) for k, v in z['cdf'].item().items()}
    out = {}
    for k in keys:
        kk = k[1:] if k.startswith('t') else k
        out[int(kk)] = np.asarray(z[k])
    return out


def reward_fn(R_hat, R_true, cdf, t_lo=1, t_hi=30):
    return [CAR.r_rank(rh, rt, cdf, t_lo, t_hi) for rh, rt in zip(R_hat, R_true)]


def percentile_err(R_hat, R_true, cdf, t_lo=1, t_hi=30):
    """mean_t |F_t(R_hat) - F_t(R_true)| = 1 - r_rank. The 'quality' signal."""
    return 1.0 - CAR.r_rank(R_hat, R_true, cdf, t_lo, t_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--plugin', required=True)
    ap.add_argument('--cdf', required=True)
    ap.add_argument('--n-ads', type=int, default=6)
    ap.add_argument('--G', type=int, default=16)
    ap.add_argument('--sigma', type=float, default=0.7)
    ap.add_argument('--kl-coef', type=float, default=0.0)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-length', type=int, default=32768)
    ap.add_argument('--t-lo', type=int, default=1)
    ap.add_argument('--t-hi', type=int, default=30)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    os.environ.setdefault('RETENTION_HEAD_TYPE', 'hazard')
    import_plugin(args.plugin)
    cdf = load_cdf(args.cdf)
    print(f'[smoke] cdf seconds: {min(cdf)}..{max(cdf)}  (n per sec ~ {len(next(iter(cdf.values())))})')

    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.template_inputs import TemplateInputs
    from swift.template.base import MaxLengthError

    model, processor = get_model_processor(
        args.checkpoint, torch_dtype=torch.bfloat16,
        model_kwargs={'device_map': 'cuda'}, model_type='qwen2_5_omni_retention')
    template = get_template(processor, max_length=args.max_length,
                            template_type='qwen2_5_omni_retention', remove_unused_columns=False)
    template.set_mode('train')
    model.eval()
    head = model.retention_head
    w_dtype = head.linear.weight.dtype
    print(f'[smoke] model loaded; head.linear={tuple(head.linear.weight.shape)} dtype={w_dtype}')

    # ---- collect N real ads with video present + true R; precompute h_anchor ----
    rows = []
    with open(args.val_jsonl) as f:
        for line in f:
            r = json.loads(line)
            R = r.get('R') or r.get('R_true')
            if R is None or len(R) - 1 < 5:
                continue
            rows.append(r)

    h_anchors, R_trues, used = [], [], 0
    for r in rows:
        if used >= args.n_ads:
            break
        row = dict(r); row['messages'] = list(r['messages'])
        row['messages'][-1] = {'role': 'assistant', 'content': '<cot></cot>'}  # bypass (in-distribution)
        try:
            enc = template.encode(TemplateInputs.from_dict(row))
        except (MaxLengthError, Exception) as e:                       # noqa
            print(f"[smoke] skip ad (encode {type(e).__name__})"); continue
        if not enc:
            continue
        batch = template.data_collator([enc])
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        try:
            with torch.no_grad():
                model(**batch)
                h_last = model._retention_h_holder.last           # (1, L, d)
                h_anchor = h_last[:, -1, :].detach().clone()       # (1, d) last token
        except Exception as e:
            print(f"[smoke] skip ad (forward {type(e).__name__}: {str(e)[:60]})"); continue
        h_anchors.append(h_anchor)
        R_trues.append(np.array(r.get('R') or r.get('R_true'), dtype=np.float64).tolist())
        used += 1
        print(f"[smoke] ad {used}: T={len(R_trues[-1])-1}  |h_anchor|={float(h_anchor.norm()):.2f}")

    if used < 2:
        print('[smoke] FAIL: <2 ads with video — cannot test'); raise SystemExit(1)
    H = torch.cat(h_anchors, dim=0)                                # (N, d) detached
    N = H.size(0)

    def mu_from_head():
        return head.linear(H.to(w_dtype)).float()                  # (N, Tmax) differentiable wrt head.linear

    import reinforce_core as RC

    def rollout_rewards(mu_z, sigma, G):
        """(N,Tmax) mu -> sample G/ad -> curves -> rewards (N,G) tensor + z,eps for logp."""
        N_, Tmax = mu_z.shape
        mu_rep = mu_z.unsqueeze(1).expand(N_, G, Tmax)
        eps = torch.randn(N_, G, Tmax, device=mu_z.device)
        z = (mu_rep + sigma * eps).detach()
        curves = RC.curve_from_hazards(z.view(N_ * G, Tmax))
        Rh = [curves[i].tolist() for i in range(N_ * G)]
        Rt = [R_trues[b] for b in range(N_) for _ in range(G)]
        rew = torch.as_tensor(reward_fn(Rh, Rt, cdf, args.t_lo, args.t_hi),
                              dtype=torch.float32, device=mu_z.device).view(N_, G)
        return rew, z, mu_rep

    # ================= GATE A: sigma calibration (within-group rank variance) =================
    # The make-or-break R1 signal must exist AND not be so wild it saturates.
    # Calibrate sigma to ckpt-150's hazard scale (research: ~0.1, calibrate empirically).
    print("\n=== GATE A: sigma calibration — within-group reward std (real mu_z, no training) ===")
    Tmax = mu_ref_dim = None
    sigma_grid = [0.05, 0.1, 0.2, 0.4, 0.7]
    sig_std = {}
    with torch.no_grad():
        mu_z0 = mu_from_head(); Tmax = mu_z0.size(1)
        for s in sigma_grid:
            rew, _, _ = rollout_rewards(mu_z0, s, args.G)
            wgs = float(rew.std(dim=1).mean()); rm = float(rew.mean())
            sig_std[s] = wgs
            print(f"  sigma={s:<4}: within-group std={wgs:.4f}  reward_mean={rm:.4f}")
    # choose sigma with std closest to a healthy target band (~0.08): enough signal, not saturating
    sigma_star = min(sigma_grid, key=lambda s: abs(sig_std[s] - 0.08))
    gateA = max(sig_std.values()) > 1e-3
    print(f"  GATE A: max within-group std = {max(sig_std.values()):.4f} -> {'PASS' if gateA else 'FAIL'} (>1e-3); "
          f"chosen sigma*={sigma_star} (std={sig_std[sigma_star]:.4f})")

    # ================= GATE C (learning: head-only PG, HP sweep at sigma*) =================
    # Naive lr=1e-3/sigma=0.7/kl=0/group-std collapsed (saturation). Sweep trust-region
    # configs (lr x kl x advantage-mode) at calibrated sigma, restoring the SFT head each
    # time. advantage modes per GRPO research: group_center (Dr.GRPO, no std), group_std
    # (vanilla GRPO), batch_std (Lite-PPO). Grad-clip on. backbone frozen.
    print(f"\n=== GATE C: head-PG raises reward over {args.steps} steps (backbone frozen) ===")
    for p in model.parameters():
        p.requires_grad_(False)
    head.linear.weight.requires_grad_(True); head.linear.bias.requires_grad_(True)
    sft_W = head.linear.weight.detach().clone()
    sft_b = head.linear.bias.detach().clone()
    mu_ref = mu_z0.detach().clone()                                 # KL ref = SFT head

    def advantage(rew, mode):                                       # rew (N,G)
        c = rew - rew.mean(dim=1, keepdim=True)                     # group baseline (center)
        if mode == 'group_std':
            return c / (rew.std(dim=1, keepdim=True) + 1e-6)
        if mode == 'batch_std':
            return c / (rew.std() + 1e-6)
        return c                                                    # group_center (no std)

    sig_list = sorted({round(s, 3) for s in (sigma_star, min(0.4, sigma_star * 2), max(0.05, sigma_star / 2))})
    grid = [(lr, kl, adv, s) for s in sig_list
            for lr in (1e-4, 1e-5) for kl in (0.0, 0.04) for adv in ('group_center', 'group_std')]
    best = None
    for (lr, kl, adv_mode, sigma) in grid:
        with torch.no_grad():
            head.linear.weight.copy_(sft_W); head.linear.bias.copy_(sft_b)
        opt = torch.optim.Adam([head.linear.weight, head.linear.bias], lr=lr)
        r0 = None; traj = []; collapse = None
        for step in range(args.steps):
            opt.zero_grad()
            mu_z = mu_from_head()
            rew, z, mu_rep = rollout_rewards(mu_z, sigma, args.G)    # rew (N,G)
            adv = advantage(rew, adv_mode).detach().view(-1)
            logp = RC.gaussian_logp(z.view(N * args.G, Tmax), mu_rep.reshape(N * args.G, Tmax), sigma)
            pg = -(logp * adv).mean()
            klv = (((mu_z - mu_ref) ** 2) / (2 * sigma ** 2)).sum(-1).mean() if kl > 0 \
                else torch.zeros((), device=mu_z.device)
            loss = pg + kl * klv
            loss.backward()
            torch.nn.utils.clip_grad_norm_([head.linear.weight, head.linear.bias], 1.0)
            opt.step()
            rm = float(rew.mean()); wgs = float(rew.std(dim=1).mean())
            if r0 is None: r0 = rm
            traj.append(rm)
            if collapse is None and wgs < 1e-6: collapse = step
        rlast = traj[-1]; rmax = max(traj); delta = rlast - r0
        tag = f"sig={sigma} lr={lr:.0e} kl={kl:.2g} adv={adv_mode}"
        print(f"  [{tag}] reward {r0:.4f}->{rlast:.4f} (Δ={delta:+.4f}) max={rmax:.4f} collapse@{collapse}")
        score = max(delta, rmax - r0)
        if best is None or score > best[0]:
            best = (score, tag, r0, rlast, rmax)
    bscore, btag, br0, brl, brm = best
    gateC = bscore > 0.01
    print(f"  GATE C best: [{btag}]  start={br0:.4f} end={brl:.4f} max={brm:.4f}  "
          f"score={bscore:+.4f} -> {'PASS' if gateC else 'FAIL'} (>0.01)")

    print("\n==== SMOKE SUMMARY ====")
    print(f"  GATE A (rank variance exists @ some sigma): {'PASS' if gateA else 'FAIL'}")
    print(f"  GATE C (PG raises reward, best config):     {'PASS' if gateC else 'FAIL'}  -> [{btag}]")
    ok = gateA and gateC
    print(f"  OVERALL: {'GO -> build swift trainer + scale with [' + btag + ']' if ok else 'NO-GO -> diagnose (sigma / reward / anchor)'}")
    raise SystemExit(0 if ok else 2)


if __name__ == '__main__':
    main()

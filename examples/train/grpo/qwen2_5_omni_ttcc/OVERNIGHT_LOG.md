# CoT-GRPO overnight log (2026-05-31 night → 2026-06-01 morning)

Autonomous babysitting + HP tuning while Leon sleeps (~8h). Goals: GPUs never idle, results
always saved+logged, tune meaningfully (evidence-based, per COT_GRPO_RUN_PLAN.md). Append each cycle.

## Safeguards in place
- **GPUs busy:** 16-GPU 2-node run live; ~15-min heartbeat detects a crash/idle → relaunch (resume from latest ckpt).
- **Results saved:** save_steps=25 (local) + box-side S3 backup loop every 15 min (`ops/ckpt_backup_loop.sh` → s3://vio-juicefs-us-east-1/cotgrl/) — survives a laptop-session drop.
- **Logged:** tensorboard + logging.jsonl + completions.jsonl per run; this file per decision.

## Plan
- **Phase 1 (baseline):** temp 0.4, scale_rewards=group, warm-start ckpt-1600. Watch within-group spread (reward_std/curve_std) + reward trend + fail_frac/kl.
- **Contingency (evidence-gated, COT_GRPO_RUN_PLAN §2):** thin diversity (reward_std<0.03 sustained) → temp 0.4→0.8; noisy advantages at low spread → scale_rewards=none; KL runaway → beta↑; instability → lr↓.
- **Held-out cross-ad SRCC** (Goodhart guard) on best checkpoints at phase transitions + at the end.

## Cycle log
- **19:35Z — Phase 1 launched** (tensorboard-only, crude backup). Ran clean to step ~6.
- **19:50Z — W&B INCIDENT + recovery.** Relaunched to add W&B + the proven `rl_ckpt_backup.sh`. W&B crashed all ranks: `PermissionError` writing to `<cwd>/wandb` (repo root, not ssm-user-writable). Root-caused, killed the orphaned node-1 (was burning GPU), fixed `WANDB_DIR=<OUT>` (committed 400d53ef), relaunched both nodes (port 29516).
- **19:52Z — Phase 1 RE-LAUNCHED, healthy.** run dir `v2-20260531-114830`. **W&B: https://wandb.ai/liangyuch/ttcc/runs/ktff132d**. Step 1: reward 1.27, fail_frac 0, kl 1.86, 16 GPU @100%. Proven per-node ckpt backup (both nodes) + analyze_dynamics.py + 15-min heartbeat all live. temp 0.4 baseline; watching within-group spread (reward_std ~0.02-0.05, thin → temp 0.8 candidate).
- **12:09Z — cycle (step 12).** HEALTHY. reward 1.15, within-grp reward_std 0.074 / head_std 0.175 (diversity workable → HOLD temp 0.4), kl 1.86 flat, grad_norm 1.13. WATCH: fail_frac=0.10 (10% head forwards floored — long CoTs hitting max_length on re-encode; investigate if it climbs >0.3). No ckpts yet (first @25). SRCC eval deferred to a transition (no free GPU during the run).
- **12:25Z — cycle (step 24).** HEALTHY, reward UP 1.15→1.17 (policy learning), reward_std 0.072 (diversity OK), fail_frac 0.0 (10% earlier was transient), kl 1.78, grad_norm 1.13. ckpt-25 imminent. PLAN: first SRCC-eval transition at ~step 50-75 (stop→eval ckpt-25/50 held-out cross-ad SRCC for the Goodhart read→resume). Hold temp 0.4.
- **12:40Z — cycle (step 36).** reward UP 1.19 (climbing steadily), reward_std 0.048 (borderline), fail_frac 0.0. **kl UP 1.78→2.11** (policy drifting — WATCH; if >3 climbing → raise beta). ckpt-25 saved + **S3 auto-upload confirmed (43 obj)**. PLAN: SRCC transition at ckpt-50 (eval 25+50 trend). Reward+kl both up = Goodhart/drift watch; SRCC will tell if the reward climb is real ranking gain.
- **12:56Z — cycle (step 48).** HEALTHY. kl STABILIZED (1.78→2.11→1.84, noisy not runaway — no beta needed). reward noisy-up 1.13-1.19, reward_std 0.072 (diversity OK), fail_frac 0.05. ckpt-50 imminent. Revised plan: ONE SRCC transition at ~step 75-100 (eval ckpt-25/50/75 trend) to keep GPUs busy vs stopping every 25. Hold temp 0.4.
- **13:12Z — cycle (step 60).** HEALTHY. reward ~1.14 (noisy-up), kl 1.95 (noisy-stable), reward_std 0.061, fail_frac 0. ckpt-25 + ckpt-50 saved. TRIGGER: SRCC transition when ckpt-75 lands (eval 25/50/75 trend vs SFT-1600 bypass baseline; if flat→Goodhart/ceiling, if up→real). Hold temp 0.4.
- **13:28Z — cycle (step 74).** HEALTHY (reward 1.16 up, kl 2.1 noisy-stable, reward_std 0.041). ckpt-75 imminent. SRCC EVAL PREPPED: srcc_eval.py (--checkpoint <adapter> --val-jsonl /opt/dlami/nvme/v8_eval/data/{ttcc_val_full|val_200_no_cot}.jsonl --plugin register.py --attn-impl flash_attn); RL ckpts continue the SFT adapter so base+RL-adapter=full policy; eval ckpt-1600(baseline)/25/50/75 parallel across GPUs. TRANSITION next cycle: stop→parallel SRCC→ (Phase 2 temp 0.8 if diversity-bound, else resume Phase 1) →resume.

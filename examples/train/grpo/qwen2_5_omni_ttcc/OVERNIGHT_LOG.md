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
- **19:35Z — Phase 1 launched.** 2-node (i-06cb MASTER + i-0f1d), ckpt-1600, GA=2/eff-32, G=8/512, temp 0.4. Step 1: reward 1.27, reward_std 0.018 (thin — watching), fail_frac 0, kl 1.86, no OOM. Backup loop + heartbeat armed.

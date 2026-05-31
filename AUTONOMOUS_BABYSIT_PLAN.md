# Autonomous babysitting plan — V8 alpha=0.1 SFT (Leon asleep, 2026-05-30 ~09:30 UTC)

Run: 2-node 16xH100, W&B liangyuch/ttcc/9czlolyy, output v1-20260530-091428.
The 2 changes vs V8: cleaned CoT (de-audio+de-gain) + RETENTION_COT_ALPHA 1e-3->0.1. Logging fix live.

## Status at handoff
- Training healthy: token_acc 0.50 (step 2), cot_alpha=0.1 confirmed, loss_curve+loss_cot BOTH logging
  (the v18 blind spot is fixed). grad_norm 4019 = expected random-head shock, clipped to 1.0.
- Throughput SLOW: ~270 s/step, GPU 16-25% util (2-node ZeRO-3 comm-bound). Cliff (step ~108) ~8h out.

## The decisive test
token_acc must HOLD (>=~0.45) past the warmup cliff (~step 90-130) where v19 collapsed 0.5->0.004.

## Decision tree (autonomous)
1. token_acc holds past ~step 130 + a ckpt saved -> SUCCESS. Let it run to ~step 225 for a usable ckpt; stop manually.
2. token_acc collapses -> alpha=0.1 insufficient -> ESCALATE:
   a. First: RETENTION_COT_ALPHA=0.2 (one-number, fast). Consider SINGLE-NODE 8-GPU (2-node is comm-bound -> ~5x faster iteration).
   b. If 0.2 also collapses: launch the reserve LoRA config (structural LM protection, gate-1 verified).
3. Crash/process-died -> read rank0 log Root Cause, fix, relaunch via launch_sft_alpha_clean_2node.sh (env fixes baked in).

## Monitoring
- Box-persistent babysitter: /opt/dlami/nvme/health/v8_alpha_health.txt (+ alerts.log) on i-0f1d, updates 60s.
- ScheduleWakeup every ~25 min: poll W&B + box health, act per tree, stay silent if healthy.

## Throughput note
Full 10 epochs (2800 steps) = infeasible AND unnecessary; the decisive signal is by ~step 130-225.
If escalation is needed, switch to single-node 8-GPU for ~5x faster iteration.

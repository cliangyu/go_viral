# RL 2×8 Production-Launch Readiness — AUDIT (2026-05-29)

**VERDICT: NO-GO.** 8 verified blockers beyond GPU availability. All fixable now (no GPUs needed) except the final 2-GPU smoke + the launch itself. Source: 7-agent adversarial audit (`rl-production-readiness-audit` workflow) + my own 2-GPU empirical tests on the eval box.

## What IS ready
Algorithm (math gates + smoke GO), trainer (`compute_loss`), **DDP+static_graph verified on 2 GPUs** (grad-accum 1 & 4, eval-loadable ckpt), node1 data+ckpt, venv/CUDA/flash_attn, val data present on cluster, EFA NICs present, network/disk substrate.

## BLOCKERS (ranked; all fixable while the SFT runs)
1. **node2 is completely unprovisioned** — no `rl/` code, `rl.sh`, config, `ttcc_train_bypass.jsonl`, `train_cdf.npz`; and node2's ckpt-150 has only the deepspeed shard dir (no consolidated safetensors). A 2-node torchrun runs the entry on BOTH nodes → node2's 8 ranks crash at startup. FIX: rsync code+config+data+full ckpt-150 to node2 at identical paths.
2. **Cluster trainer is STALE** (md5 b5d9958b, line 129 double-calls `head.linear`, no `_z_cap`/static_graph) — never run under real NCCL DDP. FIX: push the fixed local trainer (with `_z_cap` hook + `_set_static_graph`) to BOTH nodes.
3. **MASTER_PORT 29500 collides** with the live SFT's c10d rendezvous (SFT pid 631503 LISTEN *:29500) — a 2nd torchrun on 29500 corrupts/hangs both. FIX: `MASTER_PORT=29501` on both nodes.
4. **No EFA/NCCL env** in `rl.sh`/`_common.sh` — cross-node NCCL stalls / falls back to TCP. The working SFT `launch_training_2node.sh` sets the EFA block (FI_PROVIDER=efa, LD_LIBRARY_PATH efa+ofi-nccl, FI_EFA_USE_DEVICE_RDMA=1, FI_EFA_FORK_SAFE=1, unset NCCL_NET, NCCL_TIMEOUT=3600, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600). FIX: mirror that block in the RL launcher.
5. **output_dir is root:root** but wandb key + watcher run as ssm-user — no single uid works. FIX: `chown -R ssm-user` the output dir, remove stale root smoke dirs, launch everything as ssm-user.
6. **No backup watcher running** + NVMe ephemeral + CB expires 2026-06-03 → checkpoints lost. FIX: start `watch_and_upload_ckpts.py` (as ssm-user, parent dir) before launch.
7. **Attention-kernel mismatch invalidates the science** — RL=FA2, eval_ibs→sdpa (default), ckpt-150 trained FA3, none persisted. h_anchor drifts across kernels → RL-vs-SFT SRCC delta is invalid. FIX: pin `--attn_impl flash_attn` for BOTH RL-train AND eval, and RE-MEASURE the ckpt-150 0.43 baseline under that kernel.
8. **KL=0.04 too weak + no in-loop SRCC** — KL→150 by step 10 (RMS hazard shift 2.24×σ); over ~280 steps can destroy the SRCC=0.43 baseline, and with `eval_strategy=no` a "reward↑ but SRCC flat/down" Goodhart failure is invisible. FIX: raise initial kl (≥0.4) or lower head lr (~3e-5) + hpg_kl auto-stop; wire per-checkpoint cross-ad SRCC on the eval box (val data present), stop if SRCC ≤ 0.43 early.

## Key RISK
- **8× effective LR**: HF skips the `loss/=grad_accum` divide (labels present + model_accepts_loss_kwargs=True), and `compute_loss` ignores `num_items_in_batch` → 8 micro-backwards sum un-normalized. `max_grad_norm=1.0` masks it only while grad_norm>1. FIX: divide loss by grad_accum (or set model_accepts_loss_kwargs=False), OR keep lr=1e-4 only if the lr-smoke (which ran GA=8) already baked in the 8×. Watch grad_norm.
- Others: frozen-feature linear-head SRCC ceiling (may need unfreeze); per-group-std advantage amplifies dead groups; reward CPU-sync cost under 16-rank DDP; wandb key not exported; anchor lexical fragility (add `assert anchor_idx==L-1`); unfreeze-run latent DDP hang on video-only rows (needs find_unused for that run); 280 steps is the low end of the SRCC-movement window.

## PREFLIGHT CHECKLIST (do in order; 1–11 are GPU-free)
1. Sync fixed `head_pg_trainer.py` (static_graph version) to BOTH nodes; verify md5.
2. Provision node2: code+config+data (verify train_cdf md5) + FULL ckpt-150 dir.
3. `chown -R ssm-user` output dir (both nodes); remove stale root smoke dirs; launch as ssm-user.
4. Add EFA/NCCL env to the RL launcher (mirror `launch_training_2node.sh`); NCCL_DEBUG=INFO run 1.
5. `MASTER_PORT=29501` on both nodes.
6. `source wandb_env.sh` on both nodes; set WANDB_NAME.
7. Resolve effective-LR (normalize by grad_accum or confirm the 8× was baked in); log effective LR.
8. Apply KL/trust-region fix (raise kl or lower lr; hpg_kl auto-stop) BEFORE the full 280 steps.
9. Stand up per-checkpoint cross-ad SRCC eval on the eval box, attn_impl pinned, vs re-measured 0.43 baseline.
10. Start the ckpt-upload watcher (ssm-user, parent dir).
11. 2-GPU NCCL torchrun smoke on the eval box with the SYNCED trainer (≥3 grad-accum windows) — confirm no reducer error + grad sync.
12. Wait for the SFT to release all 16 GPUs; then launch (do NOT co-locate).

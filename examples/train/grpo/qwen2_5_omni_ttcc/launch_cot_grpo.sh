#!/usr/bin/env bash
# Repo-canonical PRODUCTION launcher for CoT-GRPO. Uses the config defaults
# (G=8, max_completion=512, beta=0.001, lr=5e-6, LoRA warm-start from the SFT-with-CoT
# adapter). Observability: completions.jsonl (CoT + head reward + advantage per rollout),
# cot_curve_std (R1), cot_head_fail_frac, step-0 warm-start cross-ad std -> tensorboard/wandb.
#
# Experiments END MANUALLY (no max_steps cap by default) -- stop with the PID, never auto-stop.
#
# Single-node (default, e.g. the idle 8-GPU box):
#   nohup bash launch_cot_grpo.sh >/dev/null 2>&1 &
# Multi-node (2x8) -- run on EACH node with its own NODE_RANK (needs the env shims on both):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> bash launch_cot_grpo.sh   # node 0
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0-ip> bash launch_cot_grpo.sh   # node 1
#
# Env overrides: GPUS (default all visible), GA (gradient_accumulation_steps, default 2),
#   OUT, LOG, MASTER_PORT (default 29513), ADAPTERS (warm-start ckpt; default = config).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${GPUS:=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)}"
: "${GPUS:=0,1,2,3,4,5,6,7}"
: "${GA:=2}"
: "${OUT:=/opt/dlami/nvme/ssm-out/rl_cot_grpo_v1}"
: "${LOG:=${OUT}.log}"
: "${MASTER_PORT:=29513}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=localhost}"
NPROC="$(echo "${GPUS}" | tr ',' '\n' | grep -c .)"

mkdir -p "${OUT}"
# W&B (entity/project from _common.sh: liangyuch/ttcc). Only the main process logs; a per-run name.
export WANDB_NAME="${WANDB_NAME:-$(basename "${OUT}")}"
echo "[cot-grpo] node ${NODE_RANK}/${NNODES} GPUS=${GPUS} nproc=${NPROC} ga=${GA} wandb=${WANDB_NAME} -> ${LOG}"

EXTRA=()
[[ -n "${ADAPTERS:-}" ]] && EXTRA+=(--adapters "${ADAPTERS}")

RL_ENTRY=rl/train_cot_grpo.py \
NPROC_PER_NODE="${NPROC}" CUDA_VISIBLE_DEVICES="${GPUS}" \
NNODES="${NNODES}" NODE_RANK="${NODE_RANK}" MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" \
  bash "${HERE}/rl.sh" "${HERE}/configs/rl_cot_grpo.yaml" \
    --gradient_accumulation_steps "${GA}" \
    --save_steps 25 --save_total_limit 40 --logging_steps 1 \
    --report_to tensorboard wandb \
    --output_dir "${OUT}" "${EXTRA[@]}" \
  > "${LOG}" 2>&1
echo "[cot-grpo] exited -> ${LOG}"

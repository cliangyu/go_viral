#!/usr/bin/env bash
# Repo-canonical SMOKE launcher for CoT-GRPO. Tiny run to verify the integration
# (adapter load -> CoT generation -> head reward -> r_pred attach -> step-0 health
# checks: no STEP-0 ABORT, warm-start cross-ad curve std, within-group curve std)
# BEFORE the 2-node production run. Run as ssm-user on an idle GPU box.
#
# Usage:   bash smoke_cot_grpo.sh
#   detached: nohup bash smoke_cot_grpo.sh >/dev/null 2>&1 &   (real log -> $LOG)
# Env overrides: GPUS (default 0,1), STEPS (2), NGEN (4), MAXLEN (256),
#                OUT (.../cot_grpo_smoke), LOG ($OUT.log), MASTER_PORT (29512).
#
# Why 2 GPUs by default: exercises the SAME torchrun + DeepSpeed-zero2 + PEFT wrapping
# as production, so the smoke actually tests r_pred attachment under that wrapping
# (a 1-GPU no-deepspeed run would not). See the preflight audit + COT_GRPO_IMPLEMENTATION.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${GPUS:=0,1}"
: "${STEPS:=2}"
: "${NGEN:=4}"
: "${MAXLEN:=256}"
: "${OUT:=/opt/dlami/nvme/ssm-out/cot_grpo_smoke}"
: "${LOG:=${OUT}.log}"
: "${MASTER_PORT:=29512}"
NPROC="$(echo "${GPUS}" | tr ',' '\n' | grep -c .)"

mkdir -p "${OUT}"
echo "[smoke] GPUS=${GPUS} nproc=${NPROC} steps=${STEPS} G=${NGEN} maxlen=${MAXLEN} -> log ${LOG}"

RL_ENTRY=rl/train_cot_grpo.py \
NPROC_PER_NODE="${NPROC}" CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
  bash "${HERE}/rl.sh" "${HERE}/configs/rl_cot_grpo.yaml" \
    --num_generations "${NGEN}" --max_completion_length "${MAXLEN}" --max_steps "${STEPS}" \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
    --save_steps 1000 --logging_steps 1 --report_to tensorboard \
    --output_dir "${OUT}" \
  > "${LOG}" 2>&1
echo "[smoke] done -> ${LOG}"

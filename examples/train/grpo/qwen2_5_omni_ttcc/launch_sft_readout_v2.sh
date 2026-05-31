#!/usr/bin/env bash
# Readout-ablation re-SFT (B0/B1/B2). IDENTICAL to the V8 CoT-SFT recipe
# (sft_retention_hazard_lora_v8_with_cot.yaml) EXCEPT the readout:
#   RETENTION_READOUT = last_token (B0) | mean (B1) | attn_pool (B2, default).
# attn_pool adds retention_pool to modules_to_save (it has no base weight -> trained fully).
# 1-node 8-GPU LoRA + ZeRO-3 + gradient_checkpointing. Invoke from the REPO ROOT, nohup'd:
#   cd <repo> && RETENTION_READOUT=attn_pool nohup bash examples/train/grpo/qwen2_5_omni_ttcc/launch_sft_readout_v2.sh >/dev/null 2>&1 &
# Env overrides: GPUS, MASTER_PORT, OUT, LOG, WANDB_NAME.
set -euo pipefail
RO="${RETENTION_READOUT:-attn_pool}"
export RETENTION_READOUT="$RO"
MTS=(--modules_to_save retention_head)
[[ "$RO" == attn_pool ]] && MTS=(--modules_to_save retention_head retention_pool)
: "${GPUS:=0,1,2,3,4,5,6,7}"
NPROC="$(echo "${GPUS}" | tr ',' '\n' | grep -c .)"
OUT="${OUT:-/opt/dlami/nvme/ssm-out/sft_readout_v2_${RO}}"
LOG="${LOG:-${OUT}.log}"
export NNODES="${NNODES:-1}" NODE_RANK="${NODE_RANK:-0}" MASTER_ADDR="${MASTER_ADDR:-localhost}" MASTER_PORT="${MASTER_PORT:-29521}"
export NPROC_PER_NODE="${NPROC}" CUDA_VISIBLE_DEVICES="${GPUS}"
export WANDB_NAME="${WANDB_NAME:-sft_readout_${RO}}"
export WANDB_DIR="${WANDB_DIR:-${OUT}}"
mkdir -p "$OUT"
echo "[sft-readout] readout=${RO} GPUS=${GPUS} nproc=${NPROC} OUT=${OUT} MTS=[${MTS[*]}] wandb=${WANDB_NAME} -> ${LOG}"
bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
  examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_lora_v8_with_cot.yaml \
  "${MTS[@]}" \
  --output_dir "${OUT}" \
  > "${LOG}" 2>&1
echo "[sft-readout] exited -> ${LOG}"

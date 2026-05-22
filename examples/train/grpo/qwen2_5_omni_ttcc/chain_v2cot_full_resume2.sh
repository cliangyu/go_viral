#!/usr/bin/env bash
# Resume from Stage 3 (GRPO) after the CUDA OOM crash on the first GRPO attempt.
# Fix: vllm_gpu_memory_utilization 0.35 -> 0.20 (frees ~14 GB per GPU
# for the full-FT model + optimizer state).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/home/ssm-user/work}"
LOG="${WORK}/work-out/chain_v2cot_full.log"

# shellcheck disable=SC1091
source "${HERE}/_chain_lib.sh"

SFT_COT_CKPT="${WORK}/work-out/ttcc_sft_v2cot_full/v0-20260521-081339/checkpoint-450"
[[ -d "${SFT_COT_CKPT}" ]] || fail "SFT-cot ckpt missing: ${SFT_COT_CKPT}"

log "=== RESUME2 chain (Stage 3 onwards). SFT-cot ckpt: ${SFT_COT_CKPT} ==="

# Clean up the failed Stage 3 attempt directory (only logs + empty ckpt dir).
rm -rf /opt/dlami/nvme/ssm-out/ttcc_grpo_v2cot_full
mkdir -p /opt/dlami/nvme/ssm-out

# ------------------------------------------------------------
log "Stage 3/4: GRPO full-FT (vllm_gpu_mem_util=0.20)"
GRPO_LOG="/opt/dlami/nvme/ssm-out/ttcc_grpo_v2cot_full/grpo.log"
nohup env SFT_CKPT="${SFT_COT_CKPT}" bash "${HERE}/grpo_v2cot_full.sh" >/dev/null 2>&1 &
log "Stage 3 launched (pid=$!)"
wait_for_done "${GRPO_LOG}" "GRPO" 86400 || fail "GRPO failed"
GRPO_CKPT=$(pick_final_ckpt "/opt/dlami/nvme/ssm-out/ttcc_grpo_v2cot_full")
[[ -n "${GRPO_CKPT}" ]] || fail "no GRPO checkpoint found"
log "GRPO final ckpt: ${GRPO_CKPT}"
log "Stage 3 eval"
bash "${HERE}/infer_v2cot_full.sh" \
    "${GRPO_CKPT}" "grpo_v2cot_full" \
    "${WORK}/work-out/preds_grpo_v2cot_full.parquet" 2>&1 | tee -a "${LOG}" \
    || log "WARN: GRPO eval errored, continuing"

# ------------------------------------------------------------
log "Stage 4/4: RLOO full-FT"
RLOO_LOG="/opt/dlami/nvme/ssm-out/ttcc_rloo_v2cot_full/rloo.log"
nohup env SFT_CKPT="${SFT_COT_CKPT}" bash "${HERE}/rloo_v2cot_full.sh" >/dev/null 2>&1 &
log "Stage 4 launched (pid=$!)"
wait_for_done "${RLOO_LOG}" "RLOO" 86400 || fail "RLOO failed"
RLOO_CKPT=$(pick_final_ckpt "/opt/dlami/nvme/ssm-out/ttcc_rloo_v2cot_full")
[[ -n "${RLOO_CKPT}" ]] || fail "no RLOO checkpoint found"
log "RLOO final ckpt: ${RLOO_CKPT}"
log "Stage 4 eval"
bash "${HERE}/infer_v2cot_full.sh" \
    "${RLOO_CKPT}" "rloo_v2cot_full" \
    "${WORK}/work-out/preds_rloo_v2cot_full.parquet" 2>&1 | tee -a "${LOG}" \
    || log "WARN: RLOO eval errored, continuing"

log "ALL 4 EXPERIMENTS DONE. preds under ${WORK}/work-out/preds_*_v2cot_full.parquet"

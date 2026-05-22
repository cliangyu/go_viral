#!/usr/bin/env bash
# Resume the v2 full-FT 4-experiment chain after the SFT-nocot disk-full crash.
# SFT-cot full-FT already finished (final ckpt-450 on /home). New training
# outputs go to /opt/dlami/nvme/ssm-out/ to avoid /home fill-up.
#
# Chain (skip the first SFT, it's done):
#   1) eval SFT-cot ckpt-450
#   2) SFT-nocot -> eval
#   3) GRPO from SFT-cot ckpt-450 -> eval
#   4) RLOO from SFT-cot ckpt-450 -> eval
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/home/ssm-user/work}"
LOG="${WORK}/work-out/chain_v2cot_full.log"
mkdir -p "$(dirname "${LOG}")" /opt/dlami/nvme/ssm-out

# shellcheck disable=SC1091
source "${HERE}/_chain_lib.sh"

# Hard-code the already-existing SFT-cot ckpt path (was run on /home/ before
# we switched the rest of the chain over to nvme).
SFT_COT_CKPT="${WORK}/work-out/ttcc_sft_v2cot_full/v0-20260521-081339/checkpoint-450"
[[ -d "${SFT_COT_CKPT}" ]] || fail "SFT-cot ckpt missing: ${SFT_COT_CKPT}"

log "=== RESUME chain. SFT-cot ckpt: ${SFT_COT_CKPT} ==="

# ------------------------------------------------------------
log "Stage 1/4 eval: SFT-cot ckpt-450"
bash "${HERE}/infer_v2cot_full.sh" \
    "${SFT_COT_CKPT}" "sft_v2cot_full" \
    "${WORK}/work-out/preds_sft_v2cot_full.parquet" 2>&1 | tee -a "${LOG}" \
    || log "WARN: SFT-cot eval errored, continuing"

# ------------------------------------------------------------
log "Stage 2/4: SFT-nocot full-FT (output -> nvme)"
SFT_NOCOT_LOG="/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full/sft.log"
nohup bash "${HERE}/sft_nocot_v2cot_full.sh" >/dev/null 2>&1 &
log "Stage 2 launched (pid=$!)"
wait_for_done "${SFT_NOCOT_LOG}" "SFT-nocot" 86400 || fail "SFT-nocot failed"
SFT_NOCOT_CKPT=$(pick_final_ckpt "/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full")
[[ -n "${SFT_NOCOT_CKPT}" ]] || fail "no SFT-nocot checkpoint found"
log "SFT-nocot final ckpt: ${SFT_NOCOT_CKPT}"
log "Stage 2 eval"
bash "${HERE}/infer_v2cot_full.sh" \
    "${SFT_NOCOT_CKPT}" "sft_v2cot_nocot_full" \
    "${WORK}/work-out/preds_sft_v2cot_nocot_full.parquet" 2>&1 | tee -a "${LOG}" \
    || log "WARN: SFT-nocot eval errored, continuing"

# ------------------------------------------------------------
log "Stage 3/4: GRPO full-FT (from SFT-cot ckpt; output -> nvme)"
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
log "Stage 4/4: RLOO full-FT (from SFT-cot ckpt; output -> nvme)"
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

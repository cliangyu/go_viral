#!/usr/bin/env bash
# Orchestrator for the v2 full-FT round (4 experiments).
# Must run as ssm-user. SFT-cot is assumed already running (or will be
# skipped if its log already shows train_runtime). Chain:
#   1) wait for SFT-cot DONE      -> eval
#   2) SFT-nocot                  -> eval
#   3) GRPO from SFT-cot ckpt     -> eval
#   4) RLOO from SFT-cot ckpt     -> eval
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/home/ssm-user/work}"
LOG="${WORK}/work-out/chain_v2cot_full.log"
mkdir -p "$(dirname "${LOG}")"
: > "${LOG}"

# shellcheck disable=SC1091
source "${HERE}/_chain_lib.sh"

# ------------------------------------------------------------
log "Stage 1/4: wait for SFT-cot full-FT"
SFT_COT_LOG="${WORK}/work-out/ttcc_sft_v2cot_full/sft.log"
wait_for_done "${SFT_COT_LOG}" "SFT-cot" 86400 || fail "SFT-cot failed"
SFT_COT_CKPT=$(pick_final_ckpt "${WORK}/work-out/ttcc_sft_v2cot_full")
[[ -n "${SFT_COT_CKPT}" ]] || fail "no SFT-cot checkpoint found"
log "SFT-cot final ckpt: ${SFT_COT_CKPT}"
log "Stage 1 eval"
bash "${HERE}/infer_v2cot_full.sh" \
    "${SFT_COT_CKPT}" "sft_v2cot_full" \
    "${WORK}/work-out/preds_sft_v2cot_full.parquet" 2>&1 | tee -a "${LOG}" \
    || log "WARN: SFT-cot eval errored, continuing"

# ------------------------------------------------------------
log "Stage 2/4: SFT-nocot full-FT"
SFT_NOCOT_LOG="${WORK}/work-out/ttcc_sft_v2cot_nocot_full/sft.log"
nohup bash "${HERE}/sft_nocot_v2cot_full.sh" >/dev/null 2>&1 &
log "Stage 2 launched (pid=$!)"
wait_for_done "${SFT_NOCOT_LOG}" "SFT-nocot" 86400 || fail "SFT-nocot failed"
SFT_NOCOT_CKPT=$(pick_final_ckpt "${WORK}/work-out/ttcc_sft_v2cot_nocot_full")
[[ -n "${SFT_NOCOT_CKPT}" ]] || fail "no SFT-nocot checkpoint found"
log "SFT-nocot final ckpt: ${SFT_NOCOT_CKPT}"
log "Stage 2 eval"
bash "${HERE}/infer_v2cot_full.sh" \
    "${SFT_NOCOT_CKPT}" "sft_v2cot_nocot_full" \
    "${WORK}/work-out/preds_sft_v2cot_nocot_full.parquet" 2>&1 | tee -a "${LOG}" \
    || log "WARN: SFT-nocot eval errored, continuing"

# ------------------------------------------------------------
log "Stage 3/4: GRPO full-FT (from SFT-cot ckpt)"
GRPO_LOG="${WORK}/work-out/ttcc_grpo_v2cot_full/grpo.log"
nohup env SFT_CKPT="${SFT_COT_CKPT}" bash "${HERE}/grpo_v2cot_full.sh" >/dev/null 2>&1 &
log "Stage 3 launched (pid=$!)"
wait_for_done "${GRPO_LOG}" "GRPO" 86400 || fail "GRPO failed"
GRPO_CKPT=$(pick_final_ckpt "${WORK}/work-out/ttcc_grpo_v2cot_full")
[[ -n "${GRPO_CKPT}" ]] || fail "no GRPO checkpoint found"
log "GRPO final ckpt: ${GRPO_CKPT}"
log "Stage 3 eval"
bash "${HERE}/infer_v2cot_full.sh" \
    "${GRPO_CKPT}" "grpo_v2cot_full" \
    "${WORK}/work-out/preds_grpo_v2cot_full.parquet" 2>&1 | tee -a "${LOG}" \
    || log "WARN: GRPO eval errored, continuing"

# ------------------------------------------------------------
log "Stage 4/4: RLOO full-FT (from SFT-cot ckpt)"
RLOO_LOG="${WORK}/work-out/ttcc_rloo_v2cot_full/rloo.log"
nohup env SFT_CKPT="${SFT_COT_CKPT}" bash "${HERE}/rloo_v2cot_full.sh" >/dev/null 2>&1 &
log "Stage 4 launched (pid=$!)"
wait_for_done "${RLOO_LOG}" "RLOO" 86400 || fail "RLOO failed"
RLOO_CKPT=$(pick_final_ckpt "${WORK}/work-out/ttcc_rloo_v2cot_full")
[[ -n "${RLOO_CKPT}" ]] || fail "no RLOO checkpoint found"
log "RLOO final ckpt: ${RLOO_CKPT}"
log "Stage 4 eval"
bash "${HERE}/infer_v2cot_full.sh" \
    "${RLOO_CKPT}" "rloo_v2cot_full" \
    "${WORK}/work-out/preds_rloo_v2cot_full.parquet" 2>&1 | tee -a "${LOG}" \
    || log "WARN: RLOO eval errored, continuing"

log "ALL 4 EXPERIMENTS DONE. preds under ${WORK}/work-out/preds_*_v2cot_full.parquet"

# Shared helpers for the chain_v2cot_full*.sh orchestrators.
#
# Provides four functions that drive the multi-stage SFT/GRPO/RLOO chain:
#   log <msg>                       append to ${LOG} (tee to stdout)
#   fail <msg>                      log + exit 1
#   wait_for_done <logpath> <name>  block until log contains "train_runtime"
#                                   (success) or any failure pattern (error)
#   pick_final_ckpt <out_dir>       return the highest-numbered checkpoint-N
#                                   under <out_dir>/v*/  (training run dirs)
#
# The orchestrator must set ${LOG} before sourcing this file.

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"
}

fail() {
    log "FATAL: $*"
    exit 1
}

# wait_for_done logpath phase [timeout_seconds]
#
# Polls a training log every 60 s. Returns 0 when a successful
# completion marker ("train_runtime") appears; returns 1 on any of
# the failure patterns we've seen across this project (CUDA OOM,
# Python tracebacks, deepspeed enforce-fail, torch child crash).
wait_for_done() {
    local logpath="$1" phase="$2" timeout="${3:-86400}"
    local elapsed=0
    while true; do
        if [[ -f "${logpath}" ]] && grep -qE "train_runtime" "${logpath}" 2>/dev/null; then
            log "${phase}: DONE"
            return 0
        fi
        if [[ -f "${logpath}" ]] && grep -qE "Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError|NotImplementedError|enforce fail|ChildFailedError" "${logpath}" 2>/dev/null; then
            log "${phase}: ERROR in log -- tail:"
            tail -20 "${logpath}" | tee -a "${LOG}"
            return 1
        fi
        sleep 60
        elapsed=$((elapsed + 60))
        if [[ ${elapsed} -ge ${timeout} ]]; then
            log "${phase}: TIMEOUT after ${timeout}s"
            return 1
        fi
    done
}

# pick_final_ckpt <out_dir>
#
# ms-swift writes checkpoints under <out_dir>/v<INDEX>-<DATETIME>/checkpoint-<N>/.
# Return the path with the largest N across all run directories.
pick_final_ckpt() {
    local dir="$1"
    ls -d "${dir}"/v*/checkpoint-* 2>/dev/null \
      | awk -F'checkpoint-' '{print $2, $0}' | sort -n | awk '{print $2}' | tail -1
}

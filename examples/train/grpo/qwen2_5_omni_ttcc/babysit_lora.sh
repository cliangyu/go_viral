#!/usr/bin/env bash
# babysit_lora.sh — box-persistent monitor for the single-node LoRA+ZeRO-3 SFT.
# Parses the real swift step line; tracks token_acc / step-progress / throughput;
# writes health every 60s; appends ALERTs for the failure classes we've eaten:
#   PROCESS_DIED, OOM (NVLS or logits), TOKEN_ACC_COLLAPSE, STEP_STALE.
# No self-match: pgrep targets 'swift/cli/sft.py' which this script never contains.
LOG="${1:-/opt/dlami/nvme/logs/sft_lora_1node.log}"
HEALTH=/opt/dlami/nvme/health/v8_lora_health.txt
ALERTS=/opt/dlami/nvme/health/v8_lora_alerts.log
mkdir -p /opt/dlami/nvme/health
last_step=-1; last_change=$(date +%s); oom_seen=0
parse() { grep -oE "'$1': [0-9.]+" <<<"$2" | grep -oE "[0-9.]+$" | tail -1; }
while true; do
  now=$(date +%s)
  alive=$(pgrep -f "swift/cli/sft.py" | wc -l)
  line=$(grep -oE "\{'loss':[^}]*'train_speed\(s/it\)': [0-9.]+\}" "$LOG" 2>/dev/null | tail -1)
  step=$(grep -oE "'global_step/max_steps': '[0-9]+" <<<"$line" | grep -oE "[0-9]+$")
  tacc=$(parse token_acc "$line"); loss=$(parse loss "$line")
  lcot=$(parse loss_cot "$line"); lcur=$(parse loss_curve "$line")
  mem=$(parse 'memory\(GiB\)' "$line"); spd=$(grep -oE "'train_speed\(s/it\)': [0-9.]+" <<<"$line" | grep -oE "[0-9.]+$")
  [ -n "$step" ] && [ "$step" != "$last_step" ] && { last_step=$step; last_change=$now; }
  stale=$((now-last_change))
  printf '%s alive=%s step=%s token_acc=%s loss=%s loss_cot=%s loss_curve=%s mem=%sGiB s/it=%s stale=%ss\n' \
    "$(date -u +%H:%M:%S)" "$alive" "${step:-?}" "${tacc:-?}" "${loss:-?}" "${lcot:-?}" "${lcur:-?}" "${mem:-?}" "${spd:-?}" "$stale" > "$HEALTH"
  # ALERTS
  [ "$alive" -eq 0 ] && [ "$last_step" -ge 0 ] && echo "$(date -u) ALERT PROCESS_DIED at step=$step" >> "$ALERTS"
  if grep -qE "out of memory|Cuda failure|OutOfMemoryError" "$LOG" 2>/dev/null && [ "$oom_seen" -eq 0 ]; then
    oom_seen=1; echo "$(date -u) ALERT OOM_DETECTED in $LOG" >> "$ALERTS"; fi
  [ -n "$tacc" ] && [ -n "$step" ] && [ "$step" -ge 20 ] && awk "BEGIN{exit !($tacc<0.30)}" && \
    echo "$(date -u) ALERT TOKEN_ACC_COLLAPSE token_acc=$tacc step=$step" >> "$ALERTS"
  [ "$stale" -gt 1800 ] && [ "$alive" -gt 0 ] && echo "$(date -u) ALERT STEP_STALE ${stale}s step=$step" >> "$ALERTS"
  sleep 60
done

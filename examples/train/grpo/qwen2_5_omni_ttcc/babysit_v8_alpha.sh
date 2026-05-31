#!/usr/bin/env bash
# babysit_v8_alpha.sh — box-persistent monitor for the alpha 2-node SFT.
# Parses the REAL swift step line (smoke-tested against it), tracks token_acc /
# step-progress / throughput, writes health every 60s, appends ALERTs.
# No self-match: pgrep targets 'swift/cli/sft.py' which this script never contains.
LOG="${1:-/opt/dlami/nvme/logs/sft_alpha_clean_2node_rank0.log}"
HEALTH=/opt/dlami/nvme/health/v8_alpha_health.txt
ALERTS=/opt/dlami/nvme/health/v8_alpha_alerts.log
mkdir -p /opt/dlami/nvme/health
last_step=-1; last_change=$(date +%s)
parse() { grep -oE "'$1': [0-9.]+" <<<"$2" | grep -oE "[0-9.]+$" | tail -1; }
while true; do
  now=$(date +%s)
  alive=$(pgrep -f "swift/cli/sft.py" | wc -l)
  line=$(grep -oE "\{'loss':[^}]*'train_speed\(s/it\)': [0-9.]+\}" "$LOG" 2>/dev/null | tail -1)
  step=$(grep -oE "'global_step/max_steps': '[0-9]+" <<<"$line" | grep -oE "[0-9]+$")
  tacc=$(parse token_acc "$line"); loss=$(parse loss "$line")
  lcot=$(parse loss_cot "$line"); spd=$(grep -oE "'train_speed\(s/it\)': [0-9.]+" <<<"$line" | grep -oE "[0-9.]+$")
  [ -n "$step" ] && [ "$step" != "$last_step" ] && { last_step=$step; last_change=$now; }
  stale=$((now-last_change))
  printf '%s alive=%s step=%s token_acc=%s loss=%s loss_cot=%s s/it=%s stale=%ss\n' \
    "$(date -u +%H:%M:%S)" "$alive" "${step:-?}" "${tacc:-?}" "${loss:-?}" "${lcot:-?}" "${spd:-?}" "$stale" > "$HEALTH"
  # ALERTS (the things we ate losses on)
  [ "$alive" -eq 0 ] && [ "$last_step" -ge 0 ] && echo "$(date -u) ALERT PROCESS_DIED at step=$step" >> "$ALERTS"
  [ -n "$tacc" ] && [ -n "$step" ] && [ "$step" -ge 20 ] && awk "BEGIN{exit !($tacc<0.30)}" && \
    echo "$(date -u) ALERT TOKEN_ACC_COLLAPSE token_acc=$tacc step=$step" >> "$ALERTS"
  [ "$stale" -gt 1800 ] && [ "$alive" -gt 0 ] && echo "$(date -u) ALERT STEP_STALE ${stale}s step=$step" >> "$ALERTS"
  sleep 60
done

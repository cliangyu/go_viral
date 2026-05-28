#!/usr/bin/env bash
# health_writer.sh — on-box training-health monitor.
#
# Runs detached (setsid + nohup) and writes a JSON status file every 30s by
# parsing the swift distributed training log. Captures step, loss, grad_norm,
# memory, train_speed, learning_rate, proc count, and alert state.
#
# Alerts fire when:
#   - swift_sft_proc_count == 0  (training process died)
#   - stale_step_sec > 600       (step has not advanced in 10+ minutes)
#   - log contains word-bounded RuntimeError / nan_loss / CUDA error /
#     ChildFailedError
#
# WHY THIS EXISTS:
#   - Earlier ad-hoc bash babysitter had a regex self-match bug (grep'd its
#     own command echo, false-triggered on first iteration, hit `break`,
#     died silently). Result: 9h 20m monitor blackout. See
#     INCIDENT_2026-05-28_TRAINING_CRASH_AND_MONITOR_GAP.md.
#   - This version greps the log FILE directly (not SSM stdout), uses
#     word-bounded regexes, never `break`s on alert, and is meant to run
#     on the box (survives Claude session exits, SSH disconnects, etc.).
#
# DEPLOY:
#   scp this to /opt/dlami/nvme/health_writer.sh on both training nodes.
#   chmod +x /opt/dlami/nvme/health_writer.sh
#   setsid nohup bash /opt/dlami/nvme/health_writer.sh \
#       > /opt/dlami/nvme/logs/health_writer.log 2>&1 < /dev/null &
#   disown
#
# READ:
#   cat /opt/dlami/nvme/health/v8_status.json
#   tail /opt/dlami/nvme/health/v8_alerts.log

HEALTH=/opt/dlami/nvme/health
LOG=/opt/dlami/nvme/logs/v8_distributed.log
STATUS=$HEALTH/v8_status.json
ALERTS=$HEALTH/v8_alerts.log

mkdir -p "$HEALTH"
echo "[$(date -u +%FT%TZ)] writer starting, pid=$$, log=$LOG" >> "$ALERTS"

PREV_STEP=""
PREV_STEP_TS=$(date +%s)
STALE_THRESHOLD=600  # 10 min without step advance = ALERT

while :; do
  ts=$(date -u +%FT%TZ)
  now=$(date +%s)

  # Parse last training step line. Grep FILE directly (never the SSM stdout
  # of a previous remote command — that was the babysitter v1 bug).
  last=$(grep -aE "'loss'.*'grad_norm'" "$LOG" 2>/dev/null | tail -1)
  step=$(echo "$last" | grep -oE "global_step/max_steps': '[0-9]+/" | grep -oE "[0-9]+")
  loss=$(echo "$last" | grep -oE "'loss': [0-9.eE+-]+" | grep -oE "[0-9.eE+-]+$")
  gn=$(echo "$last" | grep -oE "'grad_norm': [0-9.eE+-]+" | grep -oE "[0-9.eE+-]+$")
  mem=$(echo "$last" | grep -oE "'memory\(GiB\)': [0-9.]+" | grep -oE "[0-9.]+$")
  speed=$(echo "$last" | grep -oE "'train_speed\(s/it\)': [0-9.]+" | grep -oE "[0-9.]+$")
  lr=$(echo "$last" | grep -oE "'learning_rate': [0-9.eE+-]+" | grep -oE "[0-9.eE+-]+$")

  # Process count. pgrep -fc returns "0\n" when zero matches; sanitize.
  proc=$(pgrep -fc 'swift sft' 2>/dev/null || echo 0)
  proc=${proc//[!0-9]/}
  proc=${proc:-0}

  # Stale-progress detection: if step number has not advanced in
  # STALE_THRESHOLD seconds, we have a likely hang.
  stale_sec=0
  if [ -n "$step" ]; then
    if [ "$step" != "$PREV_STEP" ]; then
      PREV_STEP=$step
      PREV_STEP_TS=$now
    fi
    stale_sec=$((now - PREV_STEP_TS))
  fi

  alert="none"
  if [ "$proc" = "0" ] && [ -s "$LOG" ]; then
    alert="PROCESS_DIED"
    echo "[$ts] ALERT: proc=0, training dead, last step=$step" >> "$ALERTS"
  elif [ "$stale_sec" -gt "$STALE_THRESHOLD" ] && [ -n "$step" ]; then
    alert="STEP_STALE_${stale_sec}s"
    echo "[$ts] ALERT: step $step has not advanced for ${stale_sec}s (hang?)" >> "$ALERTS"
  fi

  # Word-bounded failure signals. Grep the log FILE only — never the
  # command's own echoed stdout.
  if grep -aqE "\b(RuntimeError|nan_loss|CUDA error|ChildFailedError)\b" "$LOG" 2>/dev/null; then
    alert="${alert}+FAIL_SIGNAL"
  fi

  cat > "$STATUS.tmp" <<JSON
{
  "ts_utc": "$ts",
  "log_path": "$LOG",
  "step": ${step:-null},
  "loss": ${loss:-null},
  "grad_norm": ${gn:-null},
  "memory_gb": ${mem:-null},
  "train_speed_s_per_it": ${speed:-null},
  "learning_rate": ${lr:-null},
  "swift_sft_proc_count": $proc,
  "alert": "$alert",
  "stale_step_sec": $stale_sec,
  "writer_pid": $$
}
JSON
  mv "$STATUS.tmp" "$STATUS"
  sleep 30
done

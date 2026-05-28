#!/usr/bin/env bash
# post_incident_setup.sh — deploy the post-2026-05-28 defenses on a training box.
#
# WHAT IT DOES (in order, idempotent):
#   1. Writes /opt/dlami/nvme/wandb_env.sh from ~/.netrc so swift can pick up
#      WANDB_API_KEY automatically. Avoids the wandb auth failure that crashed
#      attempt #2 of the 2026-05-28 relaunch.
#
#   2. Writes /opt/dlami/nvme/RUNNING_<owner>_<runid>.lock marker so any
#      cleanup automation (like zane-ai-agent's gpu-clean.sh) can detect an
#      active training run and bail. Format is: owner, runid, hostname,
#      wandb URL, contact info.
#
#   3. Optionally disables and locks /opt/dlami/nvme/gpu-clean.sh:
#        - Original backed up to /opt/dlami/nvme/gpu-clean.sh.original
#        - Replaced with a refusal banner that prints current locks
#        - chattr +i so subsequent SendCommand can't overwrite it
#      Only fires if --lock-gpu-clean is passed. Idempotent if already locked.
#
#   4. Installs and starts the on-box health_writer.sh under setsid + nohup.
#      Writes /opt/dlami/nvme/health/v8_status.json every 30s.
#
#   5. Verifies all of the above by reading state back.
#
# WHY EACH PIECE:
#   See INCIDENT_2026-05-28_TRAINING_CRASH_AND_MONITOR_GAP.md and
#   V8_LESSONS_LEARNED.md for the events that drove each defense.
#
# USAGE:
#   bash post_incident_setup.sh \
#     --owner leon \
#     --runid v8-$(date +%Y%m%d-%H%M%S) \
#     --wandb-url https://wandb.ai/liangyuch/ttcc \
#     --contact leon@example.com \
#     [--lock-gpu-clean]   # only on shared boxes where cleanup automation runs
#
# REVERSE (undo):
#   - Remove marker:        rm /opt/dlami/nvme/RUNNING_*.lock
#   - Unlock gpu-clean:     sudo chattr -i /opt/dlami/nvme/gpu-clean.sh
#                           sudo cp /opt/dlami/nvme/gpu-clean.sh.original \
#                                   /opt/dlami/nvme/gpu-clean.sh
#   - Stop monitor:         pkill -f health_writer.sh

set -uo pipefail

OWNER=""
RUNID=""
WANDB_URL=""
CONTACT=""
LOCK_GPU_CLEAN=false
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

while [[ $# -gt 0 ]]; do
  case $1 in
    --owner)           OWNER=$2; shift 2 ;;
    --runid)           RUNID=$2; shift 2 ;;
    --wandb-url)       WANDB_URL=$2; shift 2 ;;
    --contact)         CONTACT=$2; shift 2 ;;
    --lock-gpu-clean)  LOCK_GPU_CLEAN=true; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "$OWNER" || -z "$RUNID" || -z "$CONTACT" ]]; then
  echo "USAGE: $0 --owner NAME --runid ID --contact EMAIL [--wandb-url URL] [--lock-gpu-clean]"
  exit 2
fi

NVME=/opt/dlami/nvme
LOG_DIR=$NVME/logs
HEALTH_DIR=$NVME/health
LOCK_FILE=$NVME/RUNNING_${OWNER}_${RUNID}.lock

mkdir -p "$LOG_DIR" "$HEALTH_DIR"

# ---------------------------------------------------------------------
# 1. wandb_env.sh
# ---------------------------------------------------------------------
echo "[1/5] Setting up wandb_env.sh from ~/.netrc..."
if [[ -f $NVME/wandb_env.sh ]] && grep -q "WANDB_API_KEY=" "$NVME/wandb_env.sh"; then
  echo "  wandb_env.sh already present"
else
  WANDB_KEY=$(sudo grep -A2 "machine api.wandb.ai" /home/ssm-user/.netrc 2>/dev/null \
               | grep "password" | awk '{print $2}')
  if [[ -z "$WANDB_KEY" ]]; then
    echo "  WARNING: no wandb key in ~/.netrc; skipping wandb_env.sh"
  else
    cat > "$NVME/wandb_env.sh" <<EOF
export WANDB_API_KEY=$WANDB_KEY
EOF
    sudo chown ssm-user:ssm-user "$NVME/wandb_env.sh"
    chmod 600 "$NVME/wandb_env.sh"
    echo "  wandb_env.sh written (key length: ${#WANDB_KEY})"
  fi
fi

# ---------------------------------------------------------------------
# 2. Marker file
# ---------------------------------------------------------------------
echo "[2/5] Writing marker file $LOCK_FILE..."
cat > "$LOCK_FILE" <<EOF
owner: $OWNER
runid: $RUNID
launched_utc: $(date -u +%FT%TZ)
hostname: $(hostname)
contact: $CONTACT
wandb_url: ${WANDB_URL:-(not set)}
purpose: V8 SFT-Hazard+CoT retention head training
note: DO NOT pkill swift sft / torch.distributed.run while this file exists.
note2: If you need GPUs, contact $CONTACT first.
EOF
echo "  marker written:"
sed 's/^/    /' "$LOCK_FILE"

# ---------------------------------------------------------------------
# 3. gpu-clean.sh defense (optional)
# ---------------------------------------------------------------------
if $LOCK_GPU_CLEAN; then
  echo "[3/5] Locking down /opt/dlami/nvme/gpu-clean.sh..."
  TARGET=$NVME/gpu-clean.sh
  if [[ -f "$TARGET" ]] && lsattr "$TARGET" 2>/dev/null | awk '{print $1}' | grep -q i; then
    echo "  already locked (immutable flag set)"
  else
    # Backup original if not already
    if [[ -f "$TARGET" && ! -f "$NVME/gpu-clean.sh.original" ]]; then
      sudo cp "$TARGET" "$NVME/gpu-clean.sh.original"
      echo "  backed up original"
    fi
    # Remove immutable in case partial state
    sudo chattr -i "$TARGET" 2>/dev/null || true
    # Write refusal banner
    sudo tee "$TARGET" >/dev/null <<'REFUSE'
#!/bin/bash
# DISABLED by post_incident_setup.sh. Original at /opt/dlami/nvme/gpu-clean.sh.original.
# See INCIDENT_2026-05-28_TRAINING_CRASH_AND_MONITOR_GAP.md.
echo "================================================================"
echo "  gpu-clean.sh is DISABLED. See file header for why."
echo ""
echo "  Active training markers (do NOT kill these):"
for f in /opt/dlami/nvme/RUNNING_*.lock; do
  [ -e "$f" ] || continue
  echo "  --- $f ---"
  sed 's/^/    /' "$f"
done
echo "================================================================"
exit 1
REFUSE
    sudo chmod 755 "$TARGET"
    sudo chattr +i "$TARGET"
    echo "  gpu-clean.sh locked (chattr +i)"
  fi
else
  echo "[3/5] Skipping gpu-clean.sh lock (no --lock-gpu-clean flag)"
fi

# ---------------------------------------------------------------------
# 4. health_writer.sh
# ---------------------------------------------------------------------
echo "[4/5] Deploying health_writer.sh..."
# Copy from the repo's ops/ dir to /opt/dlami/nvme/
if [[ -f "$SCRIPT_DIR/health_writer.sh" ]]; then
  cp "$SCRIPT_DIR/health_writer.sh" "$NVME/health_writer.sh"
  chmod +x "$NVME/health_writer.sh"
fi
# Kill any leftover instances
pkill -f health_writer.sh 2>/dev/null || true
sleep 1
# Start fresh
setsid nohup bash "$NVME/health_writer.sh" \
  > "$LOG_DIR/health_writer.log" 2>&1 < /dev/null &
HPID=$!
disown $HPID || true
sleep 2
if pgrep -fc health_writer.sh > /dev/null; then
  echo "  health_writer started: pid=$HPID"
else
  echo "  WARNING: health_writer failed to start; check $LOG_DIR/health_writer.log"
fi

# ---------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------
echo "[5/5] Verification:"
echo "  marker:           $(ls -la "$LOCK_FILE" 2>&1 | awk '{print $NF}')"
echo "  wandb_env.sh:     $([ -f $NVME/wandb_env.sh ] && echo present || echo MISSING)"
echo "  gpu-clean.sh:     $(ls -la $NVME/gpu-clean.sh 2>&1 | awk '{print $1, $NF}')"
echo "                    $(lsattr $NVME/gpu-clean.sh 2>&1 | awk '{print $1}')"
echo "  health_writer:    $(pgrep -af health_writer.sh | head -1)"
echo "  log path:         $LOG_DIR/v8_distributed.log"
echo ""
echo "Done. Read status:  cat $HEALTH_DIR/v8_status.json"
echo "Read alerts:        tail $HEALTH_DIR/v8_alerts.log"

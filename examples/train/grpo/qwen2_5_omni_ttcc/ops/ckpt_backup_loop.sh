#!/usr/bin/env bash
# Box-side disaster-recovery backup: sync the run output (checkpoints + logging.jsonl +
# completions.jsonl + tensorboard) to S3 every BACKUP_INTERVAL sec. Runs INDEPENDENTLY of
# the laptop/monitoring session (nohup on the box) so results survive even if monitoring drops.
# Usage: nohup bash ops/ckpt_backup_loop.sh <OUT_DIR> >> <OUT_DIR>.backup.log 2>&1 &
set -uo pipefail
OUT="${1:-/opt/dlami/nvme/ssm-out/rl_cot_grpo_2node_v1}"
DST="${DST:-s3://vio-juicefs-us-east-1/cotgrl/$(basename "$OUT")/}"
INT="${BACKUP_INTERVAL:-900}"
echo "[backup] $OUT -> $DST every ${INT}s"
while true; do
  echo "[backup $(date -u +%H:%M:%SZ)] sync ..."
  aws s3 sync "$OUT" "$DST" --region us-east-1 --only-show-errors 2>&1 | tail -2
  echo "[backup $(date -u +%H:%M:%SZ)] done; sleeping ${INT}s"
  sleep "$INT"
done

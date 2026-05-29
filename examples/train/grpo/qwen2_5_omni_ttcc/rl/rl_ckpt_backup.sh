#!/usr/bin/env bash
# Back up each completed RL checkpoint (incl deepspeed global_step optim shards) to S3, for
# node-loss / CB-end resumability. Under 2-node ZeRO-3 the optimizer state is SPLIT: node0
# holds ranks 0-7 shards + the consolidated safetensors; node1 holds ranks 8-15 shards. So
# run this on BOTH nodes (each uploads its own local shards under a per-node prefix); a full
# resume re-merges them. The consolidated safetensors (node0) also serve the SRCC guard +
# best-checkpoint selection + a weights-only "soft resume".
#
# Usage (per node, as ssm-user): setsid bash rl_ckpt_backup.sh <output_dir> <s3://prefix> >log 2>&1 &
set -u
OUT="${1:?output_dir}"; S3="${2:?s3 prefix}"; NODE="$(hostname -I | awk '{print $1}')"
echo "[backup] node=$NODE watching $OUT -> $S3/$NODE/ (poll 60s)"
while true; do
  for ck in "$OUT"/v*/checkpoint-*/; do
    [ -d "$ck" ] || continue
    gs="$(ls -d "$ck"global_step* 2>/dev/null | head -1)"; [ -n "$gs" ] || continue   # deepspeed wrote optim
    [ -f "$ck/.s3done" ] && continue
    # only back up once stable (>90s since the global_step dir was last modified)
    if [ $(( $(date +%s) - $(stat -c %Y "$gs") )) -lt 90 ]; then continue; fi
    run="$(basename "$(dirname "$ck")")"; name="$(basename "$ck")"
    if aws s3 sync "$ck" "$S3/$NODE/$run/$name/" --region us-east-1 --only-show-errors; then
      touch "$ck/.s3done"; echo "$(date '+%F %T') [backup] $run/$name -> $S3/$NODE/"
    else
      echo "$(date '+%F %T') [backup] FAILED $run/$name (will retry next poll)"
    fi
  done
  sleep 60
done

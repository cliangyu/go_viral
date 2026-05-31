#!/usr/bin/env bash
# run_lora_eval.sh <checkpoint_dir> [label] [val_jsonl]
# Out-of-loop leak-free retention eval for a LoRA checkpoint, on the idle box (i-06cb).
# Cross-ad SRCC (the target metric) via srcc_eval.py's CoT-bypass forward + raw npz dump
# (-> retention_metrics.py for IBS, no GPU). attn pinned to flash_attn (h_anchor stability).
# Baselines to beat: SRCC 0.514 (val_present) / 0.4393 (val_full); IBS lower=better.
set -uo pipefail
source /etc/profile.d/leon-env.sh
PY=/opt/dlami/nvme/work/swift_venv/bin/python      # has flash_attn 2.8.3 (.venv-eval does NOT)
CKPT="${1:?usage: run_lora_eval.sh <checkpoint_dir> [label] [val_jsonl]}"
LABEL="${2:-lora}"
VAL="${3:-/opt/dlami/nvme/v8_eval/data/val_200_no_cot.jsonl}"
REPO=/opt/dlami/nvme/go_viral
REG="$REPO/examples/custom/qwen2_5_omni_retention/register.py"
OUT=/opt/dlami/nvme/v8_eval/lora; mkdir -p "$OUT" "$(dirname "$VAL")"
# pull val data from S3 if missing (bucket is us-east-1)
[ -f "$VAL" ] || aws s3 cp "s3://vio-juicefs-us-east-1/ttcc_sync/v8_eval/$(basename "$VAL")" "$VAL" --region us-east-1 --only-show-errors
echo "[eval] ckpt=$CKPT val=$VAL ($(wc -l < "$VAL" 2>/dev/null) rows) label=$LABEL"
cd "$REPO/examples/train/grpo/qwen2_5_omni_ttcc/rl"
$PY srcc_eval.py --checkpoint "$CKPT" --val-jsonl "$VAL" --plugin "$REG" \
  --attn-impl flash_attn --head-type hazard --max-length 32768 \
  --output "$OUT/srcc_${LABEL}.json" --dump-npz "$OUT/dump_${LABEL}.npz" 2>&1 | tail -50
echo "[eval] === IBS-vs-baseline (needs an SFT baseline npz; SRCC above is the headline) ==="
BASE_NPZ="${BASELINE_NPZ:-/opt/dlami/nvme/v8_eval/lora/dump_baseline.npz}"
if [ -f "$BASE_NPZ" ]; then
  $PY retention_metrics.py --baseline "$BASE_NPZ" --candidate "$OUT/dump_${LABEL}.npz" --label "$LABEL" 2>&1 | tail -30
else
  echo "[eval] no baseline npz at $BASE_NPZ -> SRCC (printed above) is the headline vs 0.514/0.4393; set BASELINE_NPZ=<sft_ckpt_dump.npz> to add the paired IBS/SRCC bootstrap."
fi
echo "[eval] DONE label=$LABEL -> $OUT/srcc_${LABEL}.json (dump: $OUT/dump_${LABEL}.npz)"

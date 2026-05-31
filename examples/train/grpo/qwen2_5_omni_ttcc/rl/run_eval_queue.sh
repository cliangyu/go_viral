#!/usr/bin/env bash
# Production cross-ad SRCC eval QUEUE runner. One instance PER GPU; processes entries serially so the
# GPU never idles. Entry format: "channel:ckptname[:limit]"
#   channel = bypass  -> rl/srcc_eval.py        (empty <cot></cot>, 1 fwd/ad, fast)
#           = reasoned -> verification/generate_eval.py (GENERATE cot then head, slow)
#   ckptname = subdir under $CK (e.g. sft-1600, v3-100, v4-100)
#   limit (optional) = --limit N for a fast partial read; omit/0 = full val (n=1358)
# attn PINNED to sdpa (Blackwell has no flash_attn); ALL ckpts incl. SFT baseline measured on the SAME
# kernel so only RELATIVE deltas are read (cross-kernel h_anchor drift -> never compare to flash_attn runs).
# Canonical video env is baked into BOTH eval scripts (setdefault) so features match SFT training.
#
# Usage (box): setsid bash run_eval_queue.sh 0 reasoned:sft-1600:300 reasoned:sft-1600 bypass:sft-1600 ... &
set -uo pipefail
GPU="$1"; shift
REPO=/opt/dlami/nvme/go_viral
TT=$REPO/examples/train/grpo/qwen2_5_omni_ttcc
REG=$REPO/examples/custom/qwen2_5_omni_retention/register.py
VAL=/opt/dlami/nvme/v8_eval/data/val_full_no_cot.jsonl
BASE=/home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B
CK=/opt/dlami/nvme/cotgrl_eval/ckpts
OUT=/opt/dlami/nvme/cotgrl_eval/results
mkdir -p "$OUT"
export PYTHONPATH=$REPO
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1          # progress prints stream to the log (else block-buffered -> looks hung)
MAXNEW="${MAXNEW:-384}"            # CoTs cap ~320 tok in training (completions/max_length 320); 384 is plenty + faster than 600
PY=/opt/dlami/nvme/eval_venv/bin/python
echo "[$(date -u +%H:%M:%SZ) gpu$GPU] QUEUE START: $*"
for entry in "$@"; do
  ch=$(echo "$entry" | cut -d: -f1)
  nm=$(echo "$entry" | cut -d: -f2)
  lim=$(echo "$entry" | cut -d: -f3)
  tag="$nm"; LIMARG=()
  if [ -n "$lim" ] && [ "$lim" != 0 ]; then LIMARG=(--limit "$lim"); tag="${nm}_n${lim}"; fi
  log="$OUT/log_${ch}_${tag}.txt"; jout="$OUT/${ch}_${tag}.json"
  echo "[$(date -u +%H:%M:%SZ) gpu$GPU] START $ch:$tag"
  if [ "$ch" = bypass ]; then
    "$PY" "$TT/rl/srcc_eval.py" --checkpoint "$CK/$nm" --base "$BASE" --val-jsonl "$VAL" \
      --plugin "$REG" --attn-impl sdpa "${LIMARG[@]}" --output "$jout" > "$log" 2>&1 || echo "  (bypass $tag exited $?)"
  else
    "$PY" "$TT/verification/generate_eval.py" --checkpoint "$CK/$nm" --base "$BASE" --val-jsonl "$VAL" \
      --plugin "$REG" --attn-impl sdpa --max-new "$MAXNEW" "${LIMARG[@]}" --output "$jout" > "$log" 2>&1 || echo "  (reasoned $tag exited $?)"
  fi
  echo "[$(date -u +%H:%M:%SZ) gpu$GPU] DONE  $ch:$tag -> $(grep -hoE 'SRCC \(.*|REASONED.*=.*|BYPASS.*=.*|delta.*' "$log" 2>/dev/null | tail -3 | tr '\n' ' | ')"
done
echo "[$(date -u +%H:%M:%SZ) gpu$GPU] QUEUE COMPLETE"

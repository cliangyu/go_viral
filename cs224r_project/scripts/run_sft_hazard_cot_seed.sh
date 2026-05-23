#!/usr/bin/env bash
# Single-seed SFT-Hazard+CoT training.
# Assumes merge_cot.py has already produced *_with_cot.jsonl.
# Usage: bash cs224r_project/scripts/run_sft_hazard_cot_seed.sh 42
set -euo pipefail

SEED="${1:-42}"
ALPHA="${2:-0.1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/data/splits"
OUT="${ROOT}/runs/sft_hazard_cot/seed${SEED}"

python "${ROOT}/baselines/sft_hazard_cot.py" \
    --train_jsonl "${DATA}/train_with_cot.jsonl" \
    --val_jsonl   "${DATA}/val_with_cot.jsonl" \
    --output_dir  "${OUT}" \
    --alpha "${ALPHA}" \
    --seed "${SEED}"

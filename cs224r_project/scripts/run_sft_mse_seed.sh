#!/usr/bin/env bash
# Single-seed SFT-MSE training + test inference.
# Usage: bash cs224r_project/scripts/run_sft_mse_seed.sh 42
set -euo pipefail

SEED="${1:-42}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/data/splits"
OUT="${ROOT}/runs/sft_mse/seed${SEED}"

python "${ROOT}/baselines/sft_mse.py" \
    --train_jsonl "${DATA}/train.jsonl" \
    --val_jsonl   "${DATA}/val.jsonl" \
    --output_dir  "${OUT}" \
    --seed "${SEED}"

python "${ROOT}/eval/make_test_preds.py" \
    --checkpoint "${OUT}" \
    --test_jsonl "${DATA}/test.jsonl" \
    --out_parquet "${OUT}/test_preds.parquet"

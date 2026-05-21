#!/usr/bin/env bash
# Experiment (2): SFT *without* CoT in the assistant target.
# Identical to sft.sh except for the dataset path; uses the no-CoT
# JSONL where assistant content = `Curve: {"R": [...]}` only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/home/ssm-user/work}"
DATASET="${WORK}/data/ttcc_swift_nocot/ttcc_train_sft.jsonl" \
OUT="${WORK}/work-out/ttcc_sft_nocot" \
    exec "${HERE}/sft.sh"

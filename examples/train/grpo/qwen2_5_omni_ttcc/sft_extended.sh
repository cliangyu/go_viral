#!/usr/bin/env bash
# Experiment (3): SFT trained to saturation (3 epochs).
# Frame budget unified at 60 across train + infer (see docs/06_config_audit.md).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/home/ssm-user/work}"
EPOCHS=3 \
SAVE_STEPS=90 \
SAVE_LIMIT=4 \
OUT="${WORK}/work-out/ttcc_sft_extended" \
    exec "${HERE}/sft.sh"

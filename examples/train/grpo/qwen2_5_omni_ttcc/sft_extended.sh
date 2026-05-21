#!/usr/bin/env bash
# Experiment (3): SFT trained to saturation (3 epochs).
# Lowered to FPS_MAX_FRAMES=24 to match inference / GRPO budget so the
# resulting checkpoint is directly comparable downstream.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/home/ssm-user/work}"
EPOCHS=3 \
SFT_FPS_MAX_FRAMES=24 \
SAVE_STEPS=90 \
SAVE_LIMIT=4 \
OUT="${WORK}/work-out/ttcc_sft_extended" \
    exec "${HERE}/sft.sh"

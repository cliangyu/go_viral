#!/usr/bin/env bash
# Experiment (4): GRPO trained to saturation.
# Full single epoch (~179 steps) from an SFT-Extended checkpoint.
# Required env: SFT_CKPT — path to an SFT-Extended adapter.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/home/ssm-user/work}"
SAVE_STEPS=25 \
SAVE_LIMIT=5 \
OUT="${WORK}/work-out/ttcc_grpo_extended" \
    exec "${HERE}/grpo.sh"

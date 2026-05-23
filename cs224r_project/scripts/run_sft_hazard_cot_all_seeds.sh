#!/usr/bin/env bash
# §6 requires 3 seeds. Run all three sequentially.
# Usage: bash cs224r_project/scripts/run_sft_hazard_cot_all_seeds.sh [alpha]
set -euo pipefail

ALPHA="${1:-0.1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for SEED in 42 43 44; do
    echo "===== SFT-Hazard+CoT seed=${SEED} alpha=${ALPHA} ====="
    bash "${ROOT}/scripts/run_sft_hazard_cot_seed.sh" "${SEED}" "${ALPHA}"
done

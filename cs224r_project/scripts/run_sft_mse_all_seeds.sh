#!/usr/bin/env bash
# §5 requires 3 seeds. Run all three sequentially.
# Usage: bash cs224r_project/scripts/run_sft_mse_all_seeds.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for SEED in 42 43 44; do
    echo "===== SFT-MSE seed=${SEED} ====="
    bash "${ROOT}/scripts/run_sft_mse_seed.sh" "${SEED}"
done

#!/usr/bin/env bash
# SFT without CoT supervision. Output dir on nvme (3.2 TB) to avoid filling /home.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/home/ssm-user/work}"
SFT_DATA="${WORK}/data/ttcc_swift_v2cot_nocot/ttcc_train_sft.jsonl" \
OUT="/opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full" \
    exec "${HERE}/sft_v2cot_full.sh"

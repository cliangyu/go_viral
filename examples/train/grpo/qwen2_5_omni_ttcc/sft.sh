#!/usr/bin/env bash
# SFT dispatcher — runs any sft_*.yaml variant from configs/.
#
# Usage:
#   bash sft.sh configs/<variant>.yaml [--override key=value ...]
#
# What this does:
#   1. Sources _common.sh for env (paths, ENABLE_AUDIO_OUTPUT, FPS, etc.).
#   2. Extracts ENV from the YAML (variant-specific env vars like
#      RETENTION_HEAD_TYPE) and exports them.
#   3. Picks single-GPU plain python vs torchrun multi-GPU based on
#      NPROC_PER_NODE (single-card boxes don't need torchrun overhead).
#   4. Forwards the YAML and any extra CLI args to `swift sft`.
#
# Multi-machine: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT in env
# before calling. See scripts/launch_distributed.sh for the canonical
# multi-node wrapper.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_common.sh"

CONFIG="${1:?path to configs/<variant>.yaml required}"; shift || true
if [[ ! -f "${CONFIG}" ]]; then
    echo "config not found: ${CONFIG}" >&2
    exit 1
fi

# Pull variant-specific env vars (ENV: in the YAML) and export them.
# Requires yq (https://github.com/mikefarah/yq); we install it in the
# Dockerfile and the venv bin.
if command -v yq >/dev/null 2>&1 && yq e '.ENV // {} | keys' "${CONFIG}" >/dev/null 2>&1; then
    while IFS='=' read -r k v; do
        [[ -n "${k}" ]] && export "${k}=${v}"
    done < <(yq e '.ENV // {} | to_entries | .[] | .key + "=" + (.value | tostring)' "${CONFIG}")
fi

# Multi-machine env vars (default to single-node single-process).
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=localhost}"
: "${MASTER_PORT:=29500}"

# NPROC_PER_NODE default: auto-detect visible GPUs.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE="$(nvidia-smi -L | wc -l)"
    else
        NPROC_PER_NODE=1
    fi
fi
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE

# Echo what we're launching.
echo "[$(date '+%F %T')] sft.sh launching ${CONFIG}"
echo "  NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)}"

# ms-swift handles torchrun internally when NPROC_PER_NODE > 1 (via the
# `swift` CLI wrapper). For NPROC_PER_NODE=1 it falls back to plain python.
# We invoke `swift sft` directly; the framework picks the right path.
"${VENV}/bin/swift" sft "${CONFIG}" "$@"

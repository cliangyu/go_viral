#!/usr/bin/env bash
# RL launcher for head-PG — mirrors sft.sh, but runs rl/train_head_pg.py (NOT
# `swift sft`). 1-GPU -> plain python; multi-GPU/multi-node -> torchrun.
#
# Usage:
#   bash rl.sh configs/rl_head_pg.yaml [--override key=value ...]
# Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT before calling
#   (each of the two 8-card nodes runs this with its own NODE_RANK).
#
# external_plugins in the yaml are repo-root-relative, so we cd to the repo root.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_common.sh"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"     # examples/train/grpo/qwen2_5_omni_ttcc -> repo root
ENTRY="${HERE}/rl/train_head_pg.py"

CONFIG="${1:?path to configs/<variant>.yaml required}"; shift || true
[[ -f "${CONFIG}" ]] || { echo "config not found: ${CONFIG}" >&2; exit 1; }
CONFIG_ABS="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"

# Export ENV: block from the yaml (RETENTION_HEAD_TYPE, HPG_*, ...).
if command -v yq >/dev/null 2>&1 && yq e '.ENV // {} | keys' "${CONFIG_ABS}" >/dev/null 2>&1; then
    while IFS='=' read -r k v; do
        [[ -n "${k}" ]] && export "${k}=${v}"
    done < <(yq e '.ENV // {} | to_entries | .[] | .key + "=" + (.value | tostring)' "${CONFIG_ABS}")
fi

: "${NNODES:=1}"; : "${NODE_RANK:=0}"; : "${MASTER_ADDR:=localhost}"; : "${MASTER_PORT:=29500}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then NPROC_PER_NODE="$(nvidia-smi -L | wc -l)"; else NPROC_PER_NODE=1; fi
fi

echo "[$(date '+%F %T')] rl.sh launching ${CONFIG_ABS}"
echo "  NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} NPROC_PER_NODE=${NPROC_PER_NODE}"
cd "${REPO_ROOT}"

if [[ "${NNODES}" -eq 1 && "${NPROC_PER_NODE}" -eq 1 ]]; then
    exec "${VENV}/bin/python" "${ENTRY}" "${CONFIG_ABS}" "$@"
else
    exec "${VENV}/bin/torchrun" \
        --nnodes "${NNODES}" --node_rank "${NODE_RANK}" \
        --nproc_per_node "${NPROC_PER_NODE}" \
        --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
        "${ENTRY}" "${CONFIG_ABS}" "$@"
fi

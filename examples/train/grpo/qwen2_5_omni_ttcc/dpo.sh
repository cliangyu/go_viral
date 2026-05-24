#!/usr/bin/env bash
# DPO dispatcher — runs any dpo_*.yaml variant from configs/.
#
# Usage: bash dpo.sh configs/<variant>.yaml [--override key=value ...]
#
# Same shape as sft.sh; just calls `swift rlhf --rlhf_type dpo` underneath.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_common.sh"

CONFIG="${1:?path to configs/<variant>.yaml required}"; shift || true
[[ -f "${CONFIG}" ]] || { echo "config not found: ${CONFIG}" >&2; exit 1; }

if command -v yq >/dev/null 2>&1 && yq e '.ENV // {} | keys' "${CONFIG}" >/dev/null 2>&1; then
    while IFS='=' read -r k v; do
        [[ -n "${k}" ]] && export "${k}=${v}"
    done < <(yq e '.ENV // {} | to_entries | .[] | .key + "=" + (.value | tostring)' "${CONFIG}")
fi

: "${NNODES:=1}" "${NODE_RANK:=0}" "${MASTER_ADDR:=localhost}" "${MASTER_PORT:=29500}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    NPROC_PER_NODE="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)"
fi
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE

echo "[$(date '+%F %T')] dpo.sh launching ${CONFIG} (NPROC=${NPROC_PER_NODE})"
"${VENV}/bin/swift" rlhf "${CONFIG}" "$@"

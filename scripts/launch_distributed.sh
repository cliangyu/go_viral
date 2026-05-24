#!/usr/bin/env bash
# Multi-machine training launcher for TTCC variants.
#
# Same launcher runs on every node. Each node sets NODE_RANK to its rank
# 0..NNODES-1; node 0 is the rendezvous host whose address every other
# node points to via MASTER_ADDR.
#
# Pattern adapted from ms-swift's own multi-node example:
#   examples/train/grpo/multi_node/colocate_multi_node1.sh
#
# Usage on each node:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#       bash scripts/launch_distributed.sh \
#       examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
#       configs/sft_lm_full_no_cot.yaml
#
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#       bash scripts/launch_distributed.sh \
#       examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
#       configs/sft_lm_full_no_cot.yaml
#
# Single-node usage (NNODES=1 is the default):
#   bash scripts/launch_distributed.sh \
#       examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
#       configs/sft_lm_full_no_cot.yaml
#
# Single-card boxes (NPROC_PER_NODE=1 auto-detected from nvidia-smi):
#   no special flags needed.

set -euo pipefail

INNER="${1:?inner launcher path required, e.g. examples/.../sft.sh}"; shift || true

# Defaults assume single-node.
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=localhost}"
: "${MASTER_PORT:=29500}"

# Auto-detect NPROC_PER_NODE from nvidia-smi unless user fixed it.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE="$(nvidia-smi -L | wc -l)"
    else
        NPROC_PER_NODE=1
    fi
fi

export NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE

echo "===== launch_distributed.sh ====="
echo "  NNODES=${NNODES}  NODE_RANK=${NODE_RANK}"
echo "  MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  inner=${INNER} $*"
echo "================================="

bash "${INNER}" "$@"

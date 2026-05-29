#!/usr/bin/env bash
# 2-node EFA launcher for head-PG RL (option B: ZeRO-3 + full-FT). Mirrors the PROVEN
# launch_training_2node.sh EFA/NCCL block (the SFT ran 28h on it), but runs our RL entry
# (rl/train_head_pg.py) and uses MASTER_PORT 29501 (the SFT holds 29500).
#
# Usage (run on BOTH nodes; MASTER_ADDR = node A private IP):
#   Node A: NODE_RANK=0 MASTER_ADDR=172.31.1.226 bash launch_rl_2node.sh configs/rl_head_pg.yaml
#   Node B: NODE_RANK=1 MASTER_ADDR=172.31.1.226 bash launch_rl_2node.sh configs/rl_head_pg.yaml
# Single node (e.g. eval-box smoke): NNODES=1 NPROC_PER_NODE=2 bash launch_rl_2node.sh <cfg>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
ENTRY="${HERE}/rl/train_head_pg.py"

CONFIG="${1:?path to configs/<variant>.yaml required}"; shift || true
CONFIG_ABS="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"

# --- topology (set BEFORE sourcing _common.sh so our values win its ':=' defaults) ---
: "${NNODES:=2}"; : "${NODE_RANK:?set NODE_RANK (0 or 1)}"
: "${MASTER_ADDR:?set MASTER_ADDR = node-0 private IP}"
: "${MASTER_PORT:=29501}"          # NOT 29500 (the SFT torchrun holds 29500)
: "${NPROC_PER_NODE:=8}"

# --- INHERIT THE SFT'S PROVEN ENV (the env ckpt-225 was trained under; the RL MUST match it,
#     else it processes video differently -> inconsistent with ckpt-225 + the SRCC baseline).
#     Gives: FPS=1.0, VIDEO_MAX_TOKEN_NUM=8192, USE_AUDIO_IN_VIDEO=true, ENABLE_AUDIO_OUTPUT=False,
#     MAX_PIXELS/VIDEO_MAX_PIXELS=49152, WANDB_ENTITY=liangyuch, WANDB_PROJECT=ttcc, VENV, PYTHONPATH.
#     _common.sh uses ':=' so our topology vars above are NOT clobbered. ---
source "${HERE}/_common.sh"

# --- SFT-launcher overrides NOT in _common.sh (mirror launch_training_2node.sh EXACTLY, the
#     ground-truth env ckpt-225 was trained under). _common.sh's USE_AUDIO_IN_VIDEO default is
#     a STALE 'true'; the SFT overrode it to false in the launcher (drop-audio fix for AUDIO_OOB).
#     The RL MUST match (audio off) to be consistent with ckpt-225 AND to avoid re-triggering OOB. ---
export USE_AUDIO_IN_VIDEO=false     # ckpt-225 trained WITHOUT audio (launch_training_2node.sh:51)
export DS_BUILD_OPS=0               # don't JIT-build deepspeed CUDA ops (the SFT's approach; avoids nvcc dep)
export AWS_DEFAULT_REGION=us-east-2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- paths / venv (override per box; _common.sh already set these via ':=') ---
: "${VENV:=/opt/dlami/nvme/work/swift_venv}"
: "${TTCC_REPO:=${REPO_ROOT}}"
export PYTHONPATH="${TTCC_REPO}:${PYTHONPATH:-}"

# --- CUDA toolkit for the deepspeed import (host lacks /usr/local/cuda; use bundled cu13) ---
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME:-/none}/bin/nvcc" ]]; then
    for _c in /usr/local/cuda /opt/pytorch/lib/python*/site-packages/nvidia/cu* \
              "${VENV%/*}"/lib/python*/site-packages/nvidia/cu* /opt/*/lib/python*/site-packages/nvidia/cu*; do
        [[ -x "${_c}/bin/nvcc" ]] && { export CUDA_HOME="${_c}"; break; }
    done
fi

# --- EFA / NCCL (THE CRITICAL DIFFERENCE FROM TCP — copied from launch_training_2node.sh) ---
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH:-}
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
unset NCCL_NET || true
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
: "${NCCL_DEBUG:=INFO}"; export NCCL_DEBUG     # confirm "Selected Provider is EFA" on run 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- wandb (key lives in the ssm-user env file; required or trainer init can crash) ---
[[ -f /opt/dlami/nvme/wandb_env.sh ]] && source /opt/dlami/nvme/wandb_env.sh || true
: "${WANDB_NAME:=rl_head_pg_v1_$(echo "${MASTER_ADDR}" | tr . _)}"; export WANDB_NAME

echo "[$(date '+%F %T')] launch_rl_2node: ${CONFIG_ABS}"
echo "  NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT} NPROC=${NPROC_PER_NODE}"
echo "  CUDA_HOME=${CUDA_HOME:-(unset)}  FI_PROVIDER=${FI_PROVIDER}  VENV=${VENV}"
cd "${REPO_ROOT}"
exec "${VENV}/bin/python" -m torch.distributed.run \
    --nnodes "${NNODES}" --node_rank "${NODE_RANK}" --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
    "${ENTRY}" "${CONFIG_ABS}" "$@"

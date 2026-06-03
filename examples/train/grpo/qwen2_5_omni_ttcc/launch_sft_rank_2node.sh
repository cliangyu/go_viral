#!/usr/bin/env bash
# launch_sft_rank_2node.sh <NODE_RANK> [extra sft args...]
# Rung 3a: LoRA + attn_pool readout + cross-ad rank loss. Mirrors the PROVEN
# launch_sft_alpha_clean_2node.sh EFA/CUDA/wandb env (the ckpt-225 lineage), but
# points at the LoRA-rank config + register_rank.py. Topology is overridable so the
# single-node smoke reuses this exact path:
#   16-GPU run:  node0(i-0f1d): bash launch_sft_rank_2node.sh 0
#                node1(i-06cb): bash launch_sft_rank_2node.sh 1
#   2-GPU smoke: NNODES=1 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline \
#                bash launch_sft_rank_2node.sh 0 --max_steps 20 --save_steps 20 --eval_steps 20 \
#                  --gradient_accumulation_steps 1
set -uo pipefail
NODE_RANK_ARG="${1:?usage: launch_sft_rank_2node.sh <NODE_RANK 0|1> [extra args...]}"; shift || true

source /etc/profile.d/leon-env.sh
echo "CUDA_HOME=$CUDA_HOME nvcc=$(ls "$CUDA_HOME/bin/nvcc" 2>/dev/null)"
export WANDB_API_KEY="$(python3 -c "import netrc; a=netrc.netrc('/home/ssm-user/.netrc').authenticators('api.wandb.ai'); print(a[2] if a else '')" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] && echo "WANDB_API_KEY loaded (len ${#WANDB_API_KEY})" || echo "WARN: WANDB_API_KEY EMPTY"
cd /opt/dlami/nvme/go_viral

# topology — defaults = full 16-GPU 2-node; smoke overrides via env before the call
: "${NNODES:=2}"
: "${NPROC_PER_NODE:=8}"
: "${MASTER_ADDR:=172.31.1.226}"
: "${MASTER_PORT:=29500}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export NNODES NODE_RANK="$NODE_RANK_ARG" MASTER_ADDR MASTER_PORT NPROC_PER_NODE CUDA_VISIBLE_DEVICES
export USE_AUDIO_IN_VIDEO=false               # V8 data is audio-off (ckpt-225 lineage)
: "${WANDB_NAME:=rank_3a_node${NODE_RANK_ARG}}"; export WANDB_NAME
mkdir -p /opt/dlami/nvme/logs
LOG="/opt/dlami/nvme/logs/rank_3a_rank${NODE_RANK_ARG}_$(date +%Y%m%d_%H%M%S).log"
echo "[launch] NNODES=$NNODES NPROC=$NPROC_PER_NODE NODE_RANK=$NODE_RANK_ARG MASTER=$MASTER_ADDR:$MASTER_PORT CVD=$CUDA_VISIBLE_DEVICES -> $LOG"

CFG=examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_lora_rank_3a.yaml
bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh "$CFG" \
  --dataloader_num_workers "${DL_WORKERS:-4}" \
  --dataloader_prefetch_factor "${DL_PREFETCH:-4}" \
  "$@" \
  > "$LOG" 2>&1
echo "[launch] exited rc=$? log=$LOG"

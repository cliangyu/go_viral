#!/usr/bin/env bash
# launch_sft_alpha_clean_2node.sh <NODE_RANK>
#
# 2-node launcher for the alpha-fix V8 SFT: full-FT hazard head, CLEANED CoT,
# RETENTION_COT_ALPHA=0.1 (the only changes vs V8). Run on BOTH p5.48xlarge:
#   node-0 (i-0f1d, 172.31.1.226):  bash launch_sft_alpha_clean_2node.sh 0
#   node-1 (i-06cb, 172.31.11.191): bash launch_sft_alpha_clean_2node.sh 1
#
# Sources /etc/profile.d/leon-env.sh because SSM runs as root in a NON-login
# shell -> CUDA_HOME (deepspeed nvcc) would otherwise be unset (the 2x-hit bug).
# USE_AUDIO_IN_VIDEO=false: V8 data is audio-off; _common.sh defaults it true.
set -uo pipefail
NODE_RANK="${1:?usage: launch_sft_alpha_clean_2node.sh <NODE_RANK 0|1>}"

source /etc/profile.d/leon-env.sh
echo "CUDA_HOME=$CUDA_HOME nvcc=$(ls "$CUDA_HOME/bin/nvcc" 2>/dev/null)"
# W&B key: SSM root non-login shell can't see ssm-user's env; pull it from the netrc
# (python netrc parser = robust; no key hardcoded in this file).
export WANDB_API_KEY="$(python3 -c "import netrc; a=netrc.netrc('/home/ssm-user/.netrc').authenticators('api.wandb.ai'); print(a[2] if a else '')" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] && echo "WANDB_API_KEY loaded (len ${#WANDB_API_KEY})" || echo "WARN: WANDB_API_KEY EMPTY -> wandb.init will fail"
cd /opt/dlami/nvme/go_viral

export NNODES=2 NODE_RANK="$NODE_RANK" MASTER_ADDR=172.31.1.226 MASTER_PORT=29500 NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # MUST set: _common.sh defaults it to 0,1 -> would crash 8 procs on 2 GPUs
export USE_AUDIO_IN_VIDEO=false
export WANDB_NAME="v8_alpha_clean_2node"        # online -> liangyuch/ttcc, monitorable
mkdir -p /opt/dlami/nvme/logs

bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
  examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_cleancot_alpha.yaml \
  --dataloader_num_workers "${DL_WORKERS:-4}" \
  --dataloader_prefetch_factor "${DL_PREFETCH:-4}" \
  > "/opt/dlami/nvme/logs/sft_alpha_clean_2node_rank${NODE_RANK}.log" 2>&1

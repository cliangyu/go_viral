#!/usr/bin/env bash
# launch_sft_lora_shiftfix_1node.sh — single-node 8-GPU LoRA+ZeRO-3, on i-06cb.
# THE SHIFT-FIX A/B: identical to launch_sft_lora_1node.sh EXCEPT it runs with the
# register.py whose loss_cot now shifts labels (torch.roll(labels,-1)) — the verified
# root-cause fix for the token_acc collapse. Control = the buggy run on i-0f1d (W&B
# v8_lora_clean_1node, token_acc 0.50->0.005). If token_acc HOLDS here while loss_cot
# still falls, the missing shift was the cause and the CoT channel is finally real.
set -uo pipefail
source /etc/profile.d/leon-env.sh
echo "CUDA_HOME=$CUDA_HOME"
export WANDB_API_KEY="$(python3 -c "import netrc; a=netrc.netrc('/home/ssm-user/.netrc').authenticators('api.wandb.ai'); print(a[2] if a else '')" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] && echo "WANDB_API_KEY loaded (len ${#WANDB_API_KEY})" || echo "WARN: no WANDB_API_KEY"
cd /opt/dlami/nvme/go_viral
export NNODES=1 NODE_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29503 NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_AUDIO_IN_VIDEO=false
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_NAME="v8_lora_shiftfix_1node"
mkdir -p /opt/dlami/nvme/logs
bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
  examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_lora_v8_with_cot.yaml \
  --dataloader_num_workers "${DL_WORKERS:-16}" \
  --dataloader_prefetch_factor "${DL_PREFETCH:-6}" \
  ${MAX_STEPS_OVERRIDE:+--max_steps $MAX_STEPS_OVERRIDE} \
  > /opt/dlami/nvme/logs/sft_lora_shiftfix.log 2>&1

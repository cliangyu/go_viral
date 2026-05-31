#!/usr/bin/env bash
# launch_sft_lora_1node.sh  — single-node 8-GPU LoRA escalation.
# WHY: alpha=0.1 full-FT did NOT hold token_acc (crashed 0.50->0.286, LM degenerating
# despite CE protection). LoRA FREEZES the base LM -> token_acc structurally cannot
# degrade. Single-node (no inter-node comm); DDP (LoRA is light); no deepspeed
# (avoids the ZeRO-2 audio_tower bug on audios=[] data).
set -uo pipefail
source /etc/profile.d/leon-env.sh
echo "CUDA_HOME=$CUDA_HOME"
export WANDB_API_KEY="$(python3 -c "import netrc; a=netrc.netrc('/home/ssm-user/.netrc').authenticators('api.wandb.ai'); print(a[2] if a else '')" 2>/dev/null)"
[ -n "$WANDB_API_KEY" ] && echo "WANDB_API_KEY loaded (len ${#WANDB_API_KEY})" || echo "WARN: no WANDB_API_KEY"
cd /opt/dlami/nvme/go_viral
export NNODES=1 NODE_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29501 NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_AUDIO_IN_VIDEO=false
# OOM FIX (evidence: rank4 'Cuda failure 2 out of memory' in transport/nvls.cc:270,
# GPUs at 48-71GB on the longest video sample under single-node DDP base-replication):
#  - NVLS off: removes the NVLink-SHARP multicast buffer that OOM'd. LoRA trains only
#    30M params so the gradient all-reduce is tiny -> NVLS gives ~0 benefit here.
#  - expandable_segments: kills allocator fragmentation from variable-length video seqs.
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_NAME="v8_lora_clean_1node"
mkdir -p /opt/dlami/nvme/logs
bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
  examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_lora_v8_with_cot.yaml \
  --dataloader_num_workers "${DL_WORKERS:-16}" \
  --dataloader_prefetch_factor "${DL_PREFETCH:-6}" \
  ${MAX_STEPS_OVERRIDE:+--max_steps $MAX_STEPS_OVERRIDE} \
  > /opt/dlami/nvme/logs/sft_lora_1node.log 2>&1

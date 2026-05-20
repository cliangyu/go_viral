#!/usr/bin/env bash
# GRPO with TTCC IBS reward on Qwen2.5-Omni-3B.
# v2: vLLM colocate rollouts (~10x faster than HF), shorter max_completion,
# lower temperature to stay near SFT format, 2 generations instead of 4.
set -euo pipefail

WORK="${WORK:-/home/ssm-user/work}"
VENV="${VENV:-/opt/dlami/nvme/work/swift_venv}"
GRPO_DATA="${WORK}/data/ttcc_swift/ttcc_train_grpo.jsonl"
SFT_CKPT="${SFT_CKPT:?SFT_CKPT must be set}"
OUT="${WORK}/work-out/ttcc_grpo"

mkdir -p "${OUT}"
export PYTHONPATH="/home/ubuntu/go_viral:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_PIXELS=49152 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 FPS_MAX_FRAMES=24 FPS=1.0 VIDEO_MAX_TOKEN_NUM=4096 \
"${VENV}/bin/python" -m swift.cli.main rlhf \
    --rlhf_type grpo \
    --model /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    --adapters "${SFT_CKPT}" \
    --reward_funcs ttcc_ibs_reward ttcc_format \
    --reward_weights 1.0 0.2 \
    --external_plugins \
        "/home/ubuntu/go_viral/examples/train/grpo/plugin/ttcc_ibs_plugin.py" \
        "/home/ubuntu/go_viral/examples/train/grpo/plugin/ttcc_format_plugin.py" \
    --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
    --torch_dtype bfloat16 --gradient_checkpointing true \
    --dataset "${GRPO_DATA}" \
    --max_length 8192 \
    --max_pixels 49152 \
    --max_completion_length 384 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.05 \
    --logging_steps 2 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --output_dir "${OUT}" \
    --deepspeed zero2 \
    --dataloader_num_workers 2 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.35 \
    --num_generations 2 \
    --temperature 0.4 \
    --top_p 0.95 \
    --beta 0.04 \
    --log_completions true \
    2>&1 | tee "${OUT}/grpo.log"

#!/usr/bin/env bash
# Full FT of LLM only (ViT + aligner frozen) on v2 (causal) CoT corpus.
# Standalone -- does not source _common.sh because of full-FT-specific defaults
# (lr=1e-5, max_pixels=200704, fps_max_frames=32, no LoRA flags).
set -euo pipefail

WORK="${WORK:-/home/ssm-user/work}"
VENV="${VENV:-/opt/dlami/nvme/work/swift_venv}"
SFT_DATA="${SFT_DATA:-${WORK}/data/ttcc_swift_v2cot/ttcc_train_sft.jsonl}"
OUT="${OUT:-${WORK}/work-out/ttcc_sft_v2cot_full}"

mkdir -p "${OUT}"
export PYTHONPATH="/home/ubuntu/go_viral:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_ENTITY="${WANDB_ENTITY:-liangyuch}"
export WANDB_PROJECT="${WANDB_PROJECT:-ttcc}"
: "${WANDB_NAME:=$(basename "${OUT}")}"
export WANDB_NAME

MAX_PIXELS=200704 \
VIDEO_MAX_PIXELS=200704 \
FPS_MAX_FRAMES=32 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
"${VENV}/bin/python" -m swift.cli.main sft \
    --model /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --dataset "${SFT_DATA}" \
    --max_length 8192 \
    --truncation_strategy delete \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --logging_steps 5 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 3 \
    --output_dir "${OUT}" \
    --deepspeed zero2 \
    --dataloader_num_workers 2 \
    --report_to tensorboard wandb \
    2>&1 | tee "${OUT}/sft.log"

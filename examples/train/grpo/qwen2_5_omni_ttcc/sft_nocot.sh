#!/usr/bin/env bash
# Experiment (2): SFT without CoT in assistant target.
# Identical to sft.sh except --dataset points at the no-CoT variant
# (assistant content = JSON curve only). Same 1-epoch budget so the
# resulting IBS is directly comparable to the original SFT.
set -euo pipefail

WORK="${WORK:-/home/ssm-user/work}"
VENV="${VENV:-/opt/dlami/nvme/work/swift_venv}"
SFT_DATA="${WORK}/data/ttcc_swift_nocot/ttcc_train_sft.jsonl"
OUT="${WORK}/work-out/ttcc_sft_nocot"

mkdir -p "${OUT}"
export PYTHONPATH="/home/ubuntu/go_viral:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MAX_PIXELS=49152 \
VIDEO_MAX_PIXELS=49152 \
FPS_MAX_FRAMES=32 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
"${VENV}/bin/python" -m swift.cli.main sft \
    --model /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "${SFT_DATA}" \
    --max_length 8192 --max_pixels 49152 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --learning_rate 1e-4 --warmup_ratio 0.05 \
    --logging_steps 5 --eval_steps 50 --save_steps 50 --save_total_limit 3 \
    --output_dir "${OUT}" \
    --deepspeed zero2 --dataloader_num_workers 2 \
    2>&1 | tee "${OUT}/sft_nocot.log"

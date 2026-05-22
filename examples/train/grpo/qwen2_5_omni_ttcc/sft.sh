#!/usr/bin/env bash
# SFT seed: train Qwen2.5-Omni-3B on the CoT-distilled TTCC dataset.
#
# Defaults: 1 epoch, FPS=1.0 with FPS_MAX_FRAMES=60 (covers every T_i in
# our test set without tail-truncation; see docs/06_config_audit.md),
# cosine LR, audio_tower + visual encoder frozen so LoRA only attaches
# to the text decoder.
#
# Overridable env vars (see _common.sh for the full set):
#   DATASET     path to ms-swift JSONL
#   OUT         output directory for checkpoints + log
#   EPOCHS      number of training epochs
#   LR          peak learning rate (cosine to 0 by end of training)
#   SAVE_STEPS  save every N steps; controls --save_steps and --eval_steps
#   SAVE_LIMIT  --save_total_limit
#   FPS_MAX_FRAMES  (default 60 from _common.sh; covers T_max=60 at FPS=1)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_common.sh"

# Per-experiment knobs (env-overridable).
: "${DATASET:=${WORK}/data/ttcc_swift/ttcc_train_sft.jsonl}"
: "${OUT:=${WORK}/work-out/ttcc_sft}"
: "${EPOCHS:=1}"
: "${LR:=1e-4}"
: "${SAVE_STEPS:=50}"
: "${SAVE_LIMIT:=3}"
: "${LOGGING_STEPS:=5}"

mkdir -p "${OUT}"

MAX_PIXELS="${MAX_PIXELS}" \
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS}" \
FPS_MAX_FRAMES="${FPS_MAX_FRAMES}" \
FPS="${FPS}" \
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${VENV}/bin/python" -m swift.cli.main sft \
    --model "${MODEL}" \
    --tuner_type lora \
    --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --dataset "${DATASET}" \
    --max_length "${MAX_LENGTH}" --max_pixels "${MAX_PIXELS}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing true \
    --learning_rate "${LR}" --warmup_ratio "${WARMUP_RATIO}" \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_steps "${SAVE_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_LIMIT}" \
    --output_dir "${OUT}" \
    --deepspeed zero2 --dataloader_num_workers 2 \
    2>&1 | tee "${OUT}/sft.log"

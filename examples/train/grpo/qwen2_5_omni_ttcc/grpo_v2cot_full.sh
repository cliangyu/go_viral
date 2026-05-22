#!/usr/bin/env bash
# GRPO continuing full FT from a full-FT SFT checkpoint. ViT + aligner frozen.
# Required env: SFT_CKPT (path to full-FT SFT checkpoint dir).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_common.sh"

SFT_CKPT="${SFT_CKPT:?SFT_CKPT must be set (path to full-FT SFT checkpoint)}"
: "${GRPO_DATASET:=${WORK}/data/ttcc_swift_v2cot/ttcc_train_grpo.jsonl}"
: "${OUT:=/opt/dlami/nvme/ssm-out/ttcc_grpo_v2cot_full}"
: "${EPOCHS:=1}"
: "${LR:=5e-6}"
: "${BETA:=0.001}"
: "${NUM_GENERATIONS:=4}"
: "${TEMPERATURE:=0.4}"
: "${TOP_P:=0.95}"
: "${MAX_COMPLETION_LENGTH:=1024}"
: "${SAVE_STEPS:=50}"
: "${SAVE_LIMIT:=3}"
: "${LOGGING_STEPS:=2}"
: "${REWARD_WEIGHTS:=1.0 0.2}"
: "${VLLM_GPU_MEM_UTIL:=0.20}"

# Visual config override (full-FT round): native pixels, 60 frames cover
# all T_i in [5, 60] at FPS=1.0. See docs/06_config_audit.md.
MAX_PIXELS=200704
VIDEO_MAX_PIXELS=200704
FPS_MAX_FRAMES=60
VIDEO_MAX_TOKEN_NUM=8192

mkdir -p "${OUT}"

export WANDB_ENTITY="${WANDB_ENTITY:-liangyuch}"
export WANDB_PROJECT="${WANDB_PROJECT:-ttcc}"
: "${WANDB_NAME:=$(basename "${OUT}")}"
export WANDB_NAME

MAX_PIXELS="${MAX_PIXELS}" \
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS}" \
FPS_MAX_FRAMES="${FPS_MAX_FRAMES}" \
FPS="${FPS}" \
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${VENV}/bin/python" -m swift.cli.main rlhf \
    --rlhf_type grpo \
    --model "${SFT_CKPT}" \
    --reward_funcs ttcc_ibs_reward ttcc_format \
    --reward_weights ${REWARD_WEIGHTS} \
    --external_plugins "${IBS_PLUGIN}" "${FORMAT_PLUGIN}" \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 --gradient_checkpointing true \
    --dataset "${GRPO_DATASET}" \
    --max_length "${MAX_LENGTH}" --max_pixels "${MAX_PIXELS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" --warmup_ratio "${WARMUP_RATIO}" \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_steps "${SAVE_STEPS}" --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_LIMIT}" \
    --output_dir "${OUT}" \
    --deepspeed zero2 --dataloader_num_workers 2 \
    --use_vllm true --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEM_UTIL}" \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" --top_p "${TOP_P}" \
    --beta "${BETA}" \
    --log_completions true \
    --report_to tensorboard wandb \
    2>&1 | tee "${OUT}/grpo.log"

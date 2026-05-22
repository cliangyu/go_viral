#!/usr/bin/env bash
# GRPO with TTCC IBS reward on Qwen2.5-Omni-3B.
#
# Defaults follow docs/06_config_audit.md:
#   - beta = 0.001   (DeepSeek-R1; was 0.04 -- 16x too strong KL pull)
#   - num_generations = 4  (literature minimum; was 2)
#   - max_completion_length = 1024  (was 384, observed 87% clipped)
#   - audio_tower + visual frozen (--freeze_vit/--freeze_aligner true);
#     LoRA attaches to the text decoder only
#
# Required env: SFT_CKPT — path to the SFT adapter to warm-start from.
# Overridable env vars:
#   GRPO_DATASET, OUT, EPOCHS, LR, BETA, NUM_GENERATIONS, TEMPERATURE,
#   MAX_COMPLETION_LENGTH, SAVE_STEPS, SAVE_LIMIT, REWARD_WEIGHTS
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_common.sh"

SFT_CKPT="${SFT_CKPT:?SFT_CKPT must be set (path to SFT adapter for warm-start)}"
: "${GRPO_DATASET:=${WORK}/data/ttcc_swift/ttcc_train_grpo.jsonl}"
: "${OUT:=${WORK}/work-out/ttcc_grpo}"
: "${EPOCHS:=1}"
: "${LR:=5e-6}"
: "${BETA:=0.001}"
: "${NUM_GENERATIONS:=4}"
: "${TEMPERATURE:=0.4}"
: "${TOP_P:=0.95}"
: "${MAX_COMPLETION_LENGTH:=1024}"
: "${SAVE_STEPS:=50}"
: "${SAVE_LIMIT:=2}"
: "${LOGGING_STEPS:=2}"
: "${REWARD_WEIGHTS:=1.0 0.2}"
: "${VLLM_GPU_MEM_UTIL:=0.35}"

mkdir -p "${OUT}"

MAX_PIXELS="${MAX_PIXELS}" \
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS}" \
FPS_MAX_FRAMES="${FPS_MAX_FRAMES}" \
FPS="${FPS}" \
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${VENV}/bin/python" -m swift.cli.main rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --adapters "${SFT_CKPT}" \
    --reward_funcs ttcc_ibs_reward ttcc_format \
    --reward_weights ${REWARD_WEIGHTS} \
    --external_plugins "${IBS_PLUGIN}" "${FORMAT_PLUGIN}" \
    --tuner_type lora \
    --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" \
    --target_modules all-linear \
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
    2>&1 | tee "${OUT}/grpo.log"

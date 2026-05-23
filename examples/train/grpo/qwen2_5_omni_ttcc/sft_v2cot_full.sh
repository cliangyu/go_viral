#!/usr/bin/env bash
# Full FT of LLM only (ViT + aligner frozen) on v2 (causal) CoT corpus.
# Standalone -- does not source _common.sh because of full-FT-specific defaults
# (lr=1e-5, max_pixels=200704, fps_max_frames=60, no LoRA flags).
#
# Recipe knobs (post system sweep T1-T6, 2026-05-23 — see ttcc-rl/docs/09 + 10):
#   max_length=24576              fits dataset p99 (19,278 tok) with 27% slack
#   lazy_tokenize=true            varlen per row; no padding to max_length
#   group_by_length=false         incompatible with lazy_tokenize (missing 'lengths' column)
#   attn_impl=flash_attn          FA2.8.3 compiled for sm_120 Blackwell (FA3/4 unsupported)
#   torch_compile=true            inductor fuses logits.float() cast with CE loss → -32 GiB peak
#   per_device_batch=2, accum=4   effective batch 16; T4b verified +bs2 fits at 87 GiB peak
#   dataloader_num_workers=4      T6a sweep: U-curve inflection (w<4 underfeeds, w>4 oversubscribes)
#   persistent_workers=true       amortize spawn cost across epochs
#   prefetch_factor=4             absorb tail-latency from long-video decodes
#   deepspeed=zero3 (no offload)  T1c verified plain zero2 trades +5 GiB for ~2% — not worth
#   use_liger_kernel=false        T1b: Qwen2.5-Omni bypasses transformers' ForCausalLMLoss path
#   ENABLE_AUDIO_OUTPUT=False     disables Talker (~833 M params, ~1.5 GiB GPU)
# Net result: ~63 s/step steady-state (vs 227 s/step baseline = 3.3x speedup).
#
# Env vars (all optional):
#   SFT_DATA       train JSONL  (default: ttcc_swift_v2cot/ttcc_train_sft.jsonl)
#   VAL_DATA       val JSONL    (default: parallel ttcc_val.jsonl next to SFT_DATA;
#                                unset -> no val_dataset, no val loss tracking)
#   OUT            output dir   (default: work-out/ttcc_sft_v2cot_full)
#   EPOCHS         num_train_epochs            (default: 10)
#   LR             learning rate               (default: 1e-5)
#   SAVE_STEPS     eval+save cadence           (default: 50)
#   SAVE_LIMIT     keep this many ckpts        (default: 3)
#   LOGGING_STEPS  train-loss log cadence      (default: 5)
set -euo pipefail

WORK="${WORK:-/home/ssm-user/work}"
VENV="${VENV:-/opt/dlami/nvme/work/swift_venv}"
SFT_DATA="${SFT_DATA:-${WORK}/data/ttcc_swift_v2cot/ttcc_train_sft.jsonl}"
# Auto-derive VAL_DATA from SFT_DATA's directory (one dir per dataset variant).
_SFT_DIR="$(dirname "${SFT_DATA}")"
VAL_DATA="${VAL_DATA-${_SFT_DIR}/ttcc_val.jsonl}"
OUT="${OUT:-${WORK}/work-out/ttcc_sft_v2cot_full}"
: "${EPOCHS:=10}"
: "${LR:=1e-5}"
: "${SAVE_STEPS:=50}"
: "${SAVE_LIMIT:=3}"
: "${LOGGING_STEPS:=5}"

mkdir -p "${OUT}"
export PYTHONPATH="/home/ubuntu/go_viral:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Disable Talker (~833 M params we never use). Loads thinker-only.
export ENABLE_AUDIO_OUTPUT="False"
export WANDB_ENTITY="${WANDB_ENTITY:-liangyuch}"
export WANDB_PROJECT="${WANDB_PROJECT:-ttcc}"
: "${WANDB_NAME:=$(basename "${OUT}")}"
export WANDB_NAME

# Build val_dataset flag conditionally — pass only when the file exists,
# so calling code can disable val tracking by setting VAL_DATA="".
VAL_ARGS=()
if [[ -n "${VAL_DATA}" && -f "${VAL_DATA}" ]]; then
    VAL_ARGS=(--val_dataset "${VAL_DATA}")
    echo "[$(date '+%F %T')] val_dataset = ${VAL_DATA}" | tee -a "${OUT}/sft.log"
else
    echo "[$(date '+%F %T')] val_dataset = (none — overfit test, no val loss tracking)" | tee -a "${OUT}/sft.log"
fi

MAX_PIXELS=200704 \
VIDEO_MAX_PIXELS=200704 \
FPS_MAX_FRAMES=60 \
VIDEO_MAX_TOKEN_NUM=16384 \
OMP_NUM_THREADS=6 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
"${VENV}/bin/python" -m swift.cli.main sft \
    --model /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --dataset "${SFT_DATA}" \
    "${VAL_ARGS[@]}" \
    --max_length 24576 \
    --truncation_strategy delete \
    --lazy_tokenize true \
    --strict false \
    --dataset_num_proc 1 \
    --group_by_length false \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --torch_compile true \
    --learning_rate "${LR}" \
    --warmup_ratio 0.05 \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_steps "${SAVE_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_LIMIT}" \
    --load_best_model_at_end true \
    --metric_for_best_model loss \
    --greater_is_better false \
    --output_dir "${OUT}" \
    --deepspeed zero3 \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 4 \
    --report_to tensorboard wandb \
    2>&1 | tee -a "${OUT}/sft.log"

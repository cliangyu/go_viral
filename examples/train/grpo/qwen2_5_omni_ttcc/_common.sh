# Shared environment for TTCC SFT/GRPO/DPO/RLOO/Inference launchers.
#
# Source this from sft.sh / grpo.sh / dpo.sh / rloo.sh / infer.sh /
# chain.sh; per-experiment wrappers and per-variant YAMLs override any of
# the variables below.
#
# Hard-coded paths are the EC2 host this project was developed on. To
# port to a new machine, override WORK / VENV / MODEL via env before
# sourcing.

: "${WORK:=/home/ssm-user/work}"
: "${VENV:=/opt/dlami/nvme/work/swift_venv}"
: "${MODEL:=/home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B}"
: "${TTCC_REPO:=/home/ubuntu/go_viral}"
: "${IBS_PLUGIN:=${TTCC_REPO}/examples/train/grpo/plugin/ttcc_ibs_plugin.py}"
: "${FORMAT_PLUGIN:=${TTCC_REPO}/examples/train/grpo/plugin/ttcc_format_plugin.py}"
: "${RETENTION_PLUGIN:=${TTCC_REPO}/examples/custom/qwen2_5_omni_retention/register.py}"

# --- Shared training defaults (legacy LoRA-seed knobs; per-variant YAMLs override) ---
: "${LORA_RANK:=16}"
: "${LORA_ALPHA:=32}"
: "${MAX_LENGTH:=8192}"
: "${MAX_PIXELS:=49152}"
: "${PER_DEVICE_BS:=1}"
: "${GRAD_ACCUM:=4}"
: "${WARMUP_RATIO:=0.05}"

# --- Video / audio processing (task constants — same in all variants) ---
#
# FPS=1.0 + FPS_MAX_FRAMES=60: one frame per second covering all T_i in
# [5, 60] without truncation. See docs/06_config_audit.md.
: "${FPS:=1.0}"
: "${FPS_MAX_FRAMES:=60}"
: "${VIDEO_MAX_PIXELS:=49152}"
: "${VIDEO_MAX_TOKEN_NUM:=8192}"

# Critical: timeline-aligned audio-in-video must match between training
# and inference. Qwen2.5-Omni's documented multimodal mode is on. Default
# is True here so launchers that source this file don't silently disable it.
: "${USE_AUDIO_IN_VIDEO:=true}"
export USE_AUDIO_IN_VIDEO

# --- Multi-GPU / multi-machine defaults ---
#
# NPROC_PER_NODE default 2 matches the historic 2-card dev box; sft.sh /
# dpo.sh auto-detect from nvidia-smi when sourced on the 8-card prod box
# or single-card cloud boxes.
: "${NPROC_PER_NODE:=2}"
: "${CUDA_VISIBLE_DEVICES:=0,1}"

# Multi-machine env vars (defaults = single-node). Override at the shell
# before invoking a launcher; scripts/launch_distributed.sh is the
# canonical multi-node wrapper.
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=localhost}"
: "${MASTER_PORT:=29500}"

# --- Python / CUDA ---
export PYTHONPATH="${TTCC_REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Disable Talker (speech generation) + token2wav (vocoder) — we never use
# them. Per ms-swift FAQ, this loads only the thinker submodule, freeing
# ~833 M params (~1.5 GB at bf16) of GPU memory per device.
: "${ENABLE_AUDIO_OUTPUT:=False}"
export ENABLE_AUDIO_OUTPUT

# Export the multimodal env vars consumed by Qwen-Omni / ms-swift.
export FPS FPS_MAX_FRAMES MAX_PIXELS VIDEO_MAX_PIXELS VIDEO_MAX_TOKEN_NUM

# --- W&B defaults (per-launcher WANDB_NAME overrides) ---
export WANDB_ENTITY="${WANDB_ENTITY:-liangyuch}"
export WANDB_PROJECT="${WANDB_PROJECT:-ttcc}"

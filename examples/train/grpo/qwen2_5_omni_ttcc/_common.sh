# Shared environment for TTCC SFT/GRPO launchers.
#
# Source this from sft.sh / grpo.sh; per-experiment wrappers override
# any of the variables below via the environment before sourcing.
#
# Hard-coded paths are the EC2 host this project was developed on. To
# port to a new machine, override WORK / VENV / MODEL.

: "${WORK:=/home/ssm-user/work}"
: "${VENV:=/opt/dlami/nvme/work/swift_venv}"
: "${MODEL:=/home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B}"
: "${TTCC_REPO:=/home/ubuntu/go_viral}"
: "${IBS_PLUGIN:=${TTCC_REPO}/examples/train/grpo/plugin/ttcc_ibs_plugin.py}"
: "${FORMAT_PLUGIN:=${TTCC_REPO}/examples/train/grpo/plugin/ttcc_format_plugin.py}"

# Shared training defaults; per-script wrappers override.
: "${LORA_RANK:=16}"
: "${LORA_ALPHA:=32}"
: "${MAX_LENGTH:=8192}"
: "${MAX_PIXELS:=49152}"
: "${PER_DEVICE_BS:=1}"
: "${GRAD_ACCUM:=4}"
: "${WARMUP_RATIO:=0.05}"

# Video preprocessing (passed as env vars to the swift CLI).
: "${FPS:=1.0}"
: "${FPS_MAX_FRAMES:=24}"
: "${VIDEO_MAX_PIXELS:=49152}"
: "${VIDEO_MAX_TOKEN_NUM:=4096}"

# Multi-GPU.
: "${NPROC_PER_NODE:=2}"
: "${CUDA_VISIBLE_DEVICES:=0,1}"

export PYTHONPATH="${TTCC_REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

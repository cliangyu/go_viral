# TTCC training container.
#
# Multi-arch build (default sm_80;sm_90 covers A100 + H100 + H20; pass
# FLASH_ATTN_CUDA_ARCHS=80;89;90 for A100/L40S/4090/H100, or 80;90;120 for
# A100/H100/Blackwell, etc.). sm_90 only is the fastest single-arch build
# (~7 min on 32 parallel jobs).
#
# What this image provides:
#   - CUDA 13.0 toolchain (matches torch 2.11.0+cu130)
#   - System libs: ffmpeg, ninja-build (DeepSpeed JIT), libsndfile, libavcodec
#   - yq (https://github.com/mikefarah/yq) for YAML parsing in launchers
#   - Python 3.12 venv with the exact pip stack from validated runs
#   - flash-attn 2.8.3 built from source, archs from FLASH_ATTN_CUDA_ARCHS
#   - ms-swift installed from the cliangyu/go_viral fork (ttcc-rl branch)
#
# What this image does NOT provide (must be done on the host):
#   - NVIDIA kernel module (host-level)
#   - nvidia-fabricmanager service (only needed for H100 NVSwitch)
#   - Filesystem paths /home/ubuntu/go_viral — see host-setup.sh
#
# Build for 8-card AWS p5.48xlarge (H100 only):
#   docker build --build-arg FLASH_ATTN_CUDA_ARCHS=90 -t ttcc-sft:h100 .
#
# Build for mixed-card dev box (A100 + H100):
#   docker build --build-arg FLASH_ATTN_CUDA_ARCHS=80;90 -t ttcc-sft:mixed .
#
# Build for single-card Modal H100:
#   docker build --build-arg FLASH_ATTN_CUDA_ARCHS=90 -t ttcc-sft:modal .
#
# Run (8-card AWS, all GPUs):
#   docker run --gpus all --ipc=host \
#     -v /opt/dlami/nvme/data:/workspace/data \
#     -v /opt/dlami/nvme/ssm-out:/workspace/out \
#     -e WANDB_API_KEY=$WANDB_API_KEY \
#     ttcc-sft:h100 \
#     bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
#          examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_lm_full_no_cot.yaml
#
# Run (single-card Modal H100):
#   docker run --gpus 1 --ipc=host \
#     ttcc-sft:modal \
#     bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
#          examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_lora_with_cot.yaml

ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

# Multi-arch flash-attn (default H100 only — fastest build).
# Override at build with --build-arg FLASH_ATTN_CUDA_ARCHS=80;90;120
ARG FLASH_ATTN_CUDA_ARCHS=90

# ---------- Stable layers (rarely changes; cache early) ----------------------

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/opt/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# System packages we learned the hard way are mandatory.
# - ffmpeg: librosa fallback for mp4 audio extraction
# - ninja-build: DeepSpeed JIT compile of CPU Adam op (subprocess shell needs /usr/bin/ninja)
# - libsndfile1: audio decode
# - git: clone ms-swift fork
# - python3.12-venv: isolated environment
# - yq: YAML parsing in launcher shell scripts
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev python3-pip \
      ffmpeg \
      ninja-build \
      libsndfile1 \
      libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
      git curl ca-certificates wget \
      build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    # Install yq (Go binary, ~5 MB)
    wget -qO /usr/local/bin/yq \
      'https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64' && \
    chmod +x /usr/local/bin/yq

RUN python3.12 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip wheel setuptools

# ---------- Python stack (pinned to validated set) ---------------------------

COPY requirements.lock /tmp/requirements.lock

RUN /opt/venv/bin/pip install \
      --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 || \
    /opt/venv/bin/pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

RUN /opt/venv/bin/pip install -r /tmp/requirements.lock

# ---------- flash-attn 2.8.3 (multi-arch) ------------------------------------
# FLASH_ATTN_CUDA_ARCHS is the correct env var (TORCH_CUDA_ARCH_LIST is IGNORED
# by flash-attn 2.8.3). Default sm_90 only is ~7 min on 32 parallel jobs.
# Multi-arch (e.g. 80;89;90) takes ~25 min.
ENV FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} \
    NVCC_THREADS=2 \
    MAX_JOBS=32 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE
RUN /opt/venv/bin/pip install --no-build-isolation flash-attn==2.8.3

# ---------- ms-swift fork (volatile layer; rebuild on branch update) ---------

ARG GO_VIRAL_REPO=https://github.com/cliangyu/go_viral.git
ARG GO_VIRAL_REF=ttcc-rl
RUN git clone --branch ${GO_VIRAL_REF} --single-branch ${GO_VIRAL_REPO} /opt/go_viral && \
    /opt/venv/bin/pip install -e /opt/go_viral

WORKDIR /opt/go_viral

# ---------- Training-time defaults -------------------------------------------

ENV FORCE_QWENVL_VIDEO_READER=decord \
    ENABLE_AUDIO_OUTPUT=False \
    USE_AUDIO_IN_VIDEO=true \
    WORK=/workspace \
    HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["/bin/bash"]

# TTCC SFT training container.
#
# What this image provides:
#   - CUDA 13.0 toolchain (matches torch 2.11.0+cu130)
#   - System libs: ffmpeg, ninja-build (DeepSpeed JIT), libsndfile, libavcodec
#   - Python 3.12 venv with the exact pip stack from 2-card + 8-card validated runs
#   - flash-attn 2.8.3 built from source, sm_90 only (~7 min on a 192-vCPU box)
#   - ms-swift installed from the cliangyu/go_viral fork (ttcc-rl branch)
#
# What this image does NOT provide (must be done on the host):
#   - NVIDIA kernel module (linux-modules-nvidia-595-open) — host-level
#   - nvidia-fabricmanager service — host-level (H100 NVSwitch requires it)
#   - Filesystem paths /home/ubuntu/go_viral (symlink target) — see scripts/host-setup.sh
#
# Build (on a CUDA 13 capable host):
#   docker build -t ttcc-sft:8card --build-arg GO_VIRAL_REF=ttcc-rl .
#
# Run (on the host with NVIDIA Container Toolkit):
#   docker run --gpus all --ipc=host \
#     -v /opt/dlami/nvme/data:/workspace/data \
#     -v /opt/dlami/nvme/ssm-out:/workspace/out \
#     -e FORCE_QWENVL_VIDEO_READER=decord \
#     -e WANDB_API_KEY=$WANDB_API_KEY \
#     ttcc-sft:8card \
#     bash examples/train/grpo/qwen2_5_omni_ttcc/sft_nocot_v2cot_full.sh

ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

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
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev python3-pip \
      ffmpeg \
      ninja-build \
      libsndfile1 \
      libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
      git curl ca-certificates \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip wheel setuptools

# ---------- Python stack (pinned to validated set) ---------------------------

# Copy lockfile BEFORE source clone so stack changes don't bust source layer.
COPY requirements.lock /tmp/requirements.lock

# Install the pinned PyTorch stack first from the pytorch wheel index
# (the public index doesn't always have +cu130 builds; pytorch.org/whl is canonical).
RUN /opt/venv/bin/pip install \
      --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 || \
    /opt/venv/bin/pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# Then install the rest from lockfile.
RUN /opt/venv/bin/pip install -r /tmp/requirements.lock

# ---------- flash-attn 2.8.3 (sm_90 only) ------------------------------------
# Lesson learned: FLASH_ATTN_CUDA_ARCHS is the correct env var; TORCH_CUDA_ARCH_LIST is IGNORED.
# Building for 4 arches (default) takes 1-2 h; sm_90 only is ~7 min on 32 parallel jobs.
ENV FLASH_ATTN_CUDA_ARCHS=90 \
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
    WORK=/workspace \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Default to interactive shell; production runs override CMD.
CMD ["/bin/bash"]

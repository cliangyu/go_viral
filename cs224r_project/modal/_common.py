"""
Shared Modal definitions: Image, Volume, Mount, Secret.

All Modal apps in cs224r_project/modal/ import from here so the worker image
and the persistent volume are consistent across prep / train / eval.

Volume layout (mounted at /vol inside containers):
  /vol/hf_cache/                # HuggingFace dataset + model cache
  /vol/data/audios/{ad_id}.wav  # extracted 16kHz mono audio
  /vol/data/videos/{ad_id}.mp4  # symlinks/copies of mp4
  /vol/data/splits/*.jsonl      # train/val/test JSONL
  /vol/runs/{method}/seed{N}/   # checkpoints + test_preds.parquet

Local code is mounted at /root/cs224r_project so scripts can import the same
modules used locally.
"""

import os
from pathlib import Path

import modal


# Path on the host of the cs224r_project/ directory (one level up from here).
LOCAL_PROJECT_DIR = Path(__file__).resolve().parents[1]

# Persistent volume name; survives across runs. Pick something unique to you
# if your Modal workspace is shared.
VOLUME_NAME = "cs224r-ttcc-retention"

# Single shared volume mounted at /vol in every container.
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUME_PATH = "/vol"


def make_cpu_image() -> modal.Image:
    """Image for data prep: needs ffmpeg + datasets + decord + ttcc-eval.

    ttcc-eval is a private repo. Rather than thread GITHUB_TOKEN through
    Modal (which silently fails to expand in pip_install URLs), we bundle a
    snapshot of the repo under ``cs224r_project/third_party/ttcc-eval`` and
    pip-install it from the attached local directory. To bump the version,
    re-clone the upstream and replace that directory.
    """
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("ffmpeg", "git")
        .pip_install(
            "datasets>=2.18",
            "decord",
            "tqdm",
            "huggingface_hub",
            "pandas",
            "pyarrow",
            "soundfile",
            "scipy",
            "numpy",
        )
    )


def make_gpu_image() -> modal.Image:
    """Image for training + inference. Includes CUDA torch and Qwen-Omni deps."""
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("ffmpeg", "git")
        .pip_install(
            # Pin torch to a CUDA build compatible with H100. modern
            # transformers refuses torch < 2.6 due to torch.load CVE.
            "torch==2.6.0",
            "torchvision==0.21.0",
            "transformers>=4.52,<4.58",
            "accelerate>=0.33",
            "peft>=0.11",
            "datasets>=2.18",
            "decord",
            "soundfile",
            "qwen-omni-utils>=0.0.9",
            "scipy",
            "numpy",
            "pandas",
            "pyarrow",
            "tqdm",
            "wandb",
        )
        # Make HuggingFace cache live on the persistent volume.
        .env({
            "HF_HOME": "/vol/hf_cache",
            "TRANSFORMERS_CACHE": "/vol/hf_cache",
            "HF_DATASETS_CACHE": "/vol/hf_cache",
        })
    )


# Code mount: ships cs224r_project/ into /root/cs224r_project. The image
# also has ttcc-eval installed in editable mode from
# /root/cs224r_project/third_party/ttcc-eval, which is the source of truth
# for ground-truth preprocessing. We additionally use
# `add_local_python_source('_common')` so the entrypoint scripts'
# `from _common import ...` works on the remote side without any sys.path
# munging — Modal places _common.py at /root on PYTHONPATH.
def attach_code(image: modal.Image) -> modal.Image:
    """Layer the local cs224r_project/ directory + ttcc-eval install."""
    return (
        image
        .add_local_dir(
            str(LOCAL_PROJECT_DIR),
            remote_path="/root/cs224r_project",
            copy=True,
        )
        .run_commands(
            "pip install /root/cs224r_project/third_party/ttcc-eval"
        )
        .add_local_python_source("_common", copy=True)
    )


# Optional WANDB secret. Create with:
#   modal secret create wandb WANDB_API_KEY=<your_token>
wandb_secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])

# GitHub PAT for installing the private ttcc-eval repo. Create with:
#   modal secret create github GITHUB_TOKEN=<your_pat>
# Token needs "Contents: read" on cliangyu/ttcc-eval (+ ttcc-inference if used).
github_secret = modal.Secret.from_name("github", required_keys=["GITHUB_TOKEN"])


# HuggingFace secret is intentionally not exported. Qwen2.5-Omni-3B is
# public, so no HF auth is needed. If you later need to pull a gated model:
#   modal secret create huggingface HF_TOKEN=<your_token>
# then add it as a secret only on the specific entrypoint that needs it
# (don't make it a global import — Modal raises at any reference site if
# the secret doesn't exist).
hf_secret = None


def common_env() -> dict[str, str]:
    """Env vars set on every GPU container."""
    return {
        "PYTHONPATH": "/root",
        "TOKENIZERS_PARALLELISM": "false",
        # WANDB project + run-grouping is set inside each script.
        "WANDB_PROJECT": "cs224r-ttcc-retention",
        # Cap video sampling to fit 30-second ads on one H100. Without these
        # qwen_omni_utils emits 150k+ video patches per ad (≈30k LM tokens),
        # which blows out activations even at batch 1 with grad checkpointing.
        # 50176 px ≈ 224x224 per patch; 12 frames keeps the temporal cue.
        "VIDEO_MAX_PIXELS": "50176",
        "FPS_MAX_FRAMES": "12",
        "MAX_PIXELS": "1003520",
        # Reduce CUDA fragmentation: the OOM trace was sitting on ~13GB
        # reserved-but-unallocated.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

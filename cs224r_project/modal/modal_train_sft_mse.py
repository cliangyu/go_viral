"""
Modal entrypoint: SFT-MSE training (Stage 1, §5).

Runs on H100. Reads prepared splits from the volume, writes the checkpoint
back to /vol/runs/sft_mse/seed{N}/. WANDB logs are written if the secret
named "wandb" is configured.

Run a single seed:
  modal run cs224r_project/modal/modal_train_sft_mse.py --seed 42

Detached (recommended for real runs — survives terminal closure):
  modal run --detach cs224r_project/modal/modal_train_sft_mse.py --seed 42

Smoke verification only (loads first batch, prints shapes, exits):
  modal run cs224r_project/modal/modal_train_sft_mse.py --seed 42 --smoke-only
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import (
    VOLUME_PATH, attach_code, common_env, hf_secret,
    make_gpu_image, volume, wandb_secret,
)


app = modal.App("cs224r-sft-mse")
image = attach_code(make_gpu_image())


_secrets = [wandb_secret]
if hf_secret is not None:
    _secrets.append(hf_secret)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=_secrets,
    gpu="H100",
    timeout=24 * 3600,  # 24 h ceiling; --detach if longer.
    cpu=8.0,
    memory=64 * 1024,
)
def train(
    seed: int = 42,
    lr: float = 1e-5,
    # 30k-token MM sequences blow up activations; per-device 1 + grad-accum 16
    # gives the §5 effective batch (16) and fits within 80GB after gradient
    # checkpointing.
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    num_train_epochs: float = 10.0,
    lora_rank: int = 8,
    nframes: int = 32,
    max_pixels: int = 200704,
    smoke_only: bool = False,
    out_subdir: str = "",
) -> str:
    import os
    os.environ.update(common_env())

    sub = out_subdir or f"seed{seed}"
    out_dir = f"/vol/runs/sft_mse/{sub}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "/root/cs224r_project/baselines/sft_mse.py",
        "--train_jsonl", "/vol/data/splits/train.jsonl",
        "--val_jsonl",   "/vol/data/splits/val.jsonl",
        "--output_dir",  out_dir,
        "--seed",        str(seed),
        "--lr",          str(lr),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--num_train_epochs", str(num_train_epochs),
        "--lora_rank",   str(lora_rank),
        "--nframes",     str(nframes),
        "--max_pixels",  str(max_pixels),
    ]
    if smoke_only:
        cmd.append("--smoke_only")

    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    return out_dir


@app.local_entrypoint()
def main(
    seed: int = 42,
    lr: float = 1e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    num_train_epochs: float = 10.0,
    lora_rank: int = 8,
    nframes: int = 32,
    max_pixels: int = 200704,
    smoke_only: bool = False,
    out_subdir: str = "",
    all_seeds: bool = False,
):
    if all_seeds:
        seeds = [42, 43, 44]
        for s in seeds:
            sub = out_subdir.replace("{seed}", str(s)) if "{seed}" in out_subdir else (out_subdir or f"seed{s}")
            print(f"===== SFT-MSE seed={s} out_subdir={sub} =====")
            train.remote(
                seed=s,
                lr=lr,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_train_epochs,
                lora_rank=lora_rank,
                nframes=nframes,
                max_pixels=max_pixels,
                smoke_only=smoke_only,
                out_subdir=sub,
            )
    else:
        out = train.remote(
            seed=seed,
            lr=lr,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lora_rank=lora_rank,
            nframes=nframes,
            max_pixels=max_pixels,
            smoke_only=smoke_only,
            out_subdir=out_subdir,
        )
        print(f"checkpoint at: {out}")

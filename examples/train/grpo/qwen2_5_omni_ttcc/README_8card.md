# 8-card p5.48xlarge Setup Runbook

Production training of TTCC retention-curve SFT on a fresh p5.48xlarge
(8× H100 SXM 80 GB) requires several system-level fixes beyond the
2-card RTX PRO 6000 reference setup. Document them here so the next person
(or future-you) doesn't re-debug.

## System asymmetry vs 2-card

| | 2-card (g7e.12xlarge) | 8-card (p5.48xlarge) |
|---|---|---|
| GPU | 2× RTX PRO 6000 Blackwell (96 GB) | 8× H100 SXM (80 GB) |
| Interconnect | PCIe | NVSwitch (NVLink fabric) |
| ZeRO-3 activations | replicated per rank, 96 GB headroom | replicated per rank, 80 GB headroom |

Implication: 8-card has **less per-GPU memory** despite more total GPUs, because
ZeRO-3 shards params/optimizer/grads across ranks but NOT activations.
Long-sequence training (24576 tokens) needs `vit_gradient_checkpointing=true`
to fit on 80 GB H100; without it the run OOMs at ~79 GB.

## Driver + Fabric Manager (one-time per new instance)

The DLAMI `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)` ships
NVIDIA Open Kernel Module 595.64 but no fabricmanager — and apt only has
fabricmanager 595.71.05 (minor mismatch, FM rejects).

```bash
# Install matching kernel module from apt:
sudo apt-get install -y linux-modules-nvidia-595-open-6.17.0-1015-aws
sudo dkms uninstall nvidia/595.64 -k $(uname -r)
sudo depmod -a
sudo modprobe nvidia          # loads 595.71.05 (matches apt fabricmanager)

# Userspace libs (must match kernel module):
sudo apt-get install -y \
  nvidia-utils-595-server \
  libnvidia-compute-595-server libnvidia-decode-595-server \
  libnvidia-encode-595-server libnvidia-extra-595-server \
  libnvidia-fbc1-595-server libnvidia-gl-595-server \
  libnvidia-nscq-595 nvidia-fabricmanager-595

# Start fabric manager (H100 NVSwitch requires it):
sudo systemctl daemon-reload
sudo systemctl start nvidia-fabricmanager
sudo systemctl enable nvidia-fabricmanager

# Verify:
nvidia-smi  # should show driver 595.71.05, 8 GPUs
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==8"
```

NOTE: reboot clears `linux-modules-nvidia-595-open` and FM packages on this AMI
(unattended-upgrades or apt cleanup). Re-run the install block above after any
reboot.

## ninja (DeepSpeed CPU Adam JIT requires it)

```bash
sudo apt-get install -y ninja-build   # /usr/bin/ninja, system-wide
```

The DeepSpeed CPU Adam op compiles at training startup via torch C++ extension,
which calls ninja from a subprocess shell with the default PATH. Without
ninja in /usr/bin, DS falls back to slow distutils OR fails with "ninja not
found" depending on path.

## flash-attn 2.8.3 build

No prebuilt wheel exists for torch 2.11.0+cu130+cp312+cxx11abiTRUE. Build from
source, but limit arch to sm_90 (Hopper) — the default builds 4 arches
(80/90/100/120) for a ~1 GB fat binary that takes 1-2 hours.

```bash
export MAX_JOBS=32
export FLASH_ATTN_CUDA_ARCHS=90   # critical: not TORCH_CUDA_ARCH_LIST (ignored by flash-attn 2.8.3)
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/opt/dlami/nvme/work/swift_venv/bin:/usr/local/cuda-13.0/bin:$PATH
export FLASH_ATTENTION_FORCE_BUILD=TRUE
pip install --no-build-isolation flash-attn==2.8.3
```

sm_90 only takes ~7 min on this box (vs 1-2 h for 4 arches).

## Python stack (pinned to 2-card versions)

| package | version | notes |
|---|---|---|
| torch | 2.11.0+cu130 | matches 2-card; 2.12 ABI breaks torchvision |
| torchvision | 0.26.0 | matches torch 2.11 ABI |
| torchaudio | 2.11.0 | matches |
| flash_attn | 2.8.3 | built from source (sm_90) |
| transformers | 5.8.1 | per ms-swift requirements |
| accelerate | 1.13.0 | |
| deepspeed | 0.19.0 | |
| trl | 0.29.1 | |
| huggingface_hub | 1.15.0 | |
| qwen-omni-utils | 0.0.9 | latest available |
| decord | 0.6.0 | video reader (set `FORCE_QWENVL_VIDEO_READER=decord`) |
| peft | 0.19.1 | |

## Path symlinks (scripts have hardcoded `/home/ubuntu/go_viral`)

```bash
sudo mkdir -p /home/ubuntu
sudo chmod 755 /home/ubuntu
sudo ln -sfn /opt/dlami/nvme/go_viral /home/ubuntu/go_viral
ln -sfn /opt/dlami/nvme/data/ttcc-v0_2_0 /home/ssm-user/work/data/ttcc
mkdir -p /opt/dlami/nvme/hf-cache
ln -sfn /opt/dlami/nvme/hf-cache /home/ssm-user/work/hf-cache
```

NVMe ephemeral storage (`/opt/dlami/nvme`, ~28 TB) is wiped on instance termination
but survives reboot. Put venv, model cache, dataset, and outputs there to keep
the 72 GB root disk clean.

## SFT launch (production config)

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export FORCE_QWENVL_VIDEO_READER=decord
export WORK=/home/ssm-user/work

bash examples/train/grpo/qwen2_5_omni_ttcc/sft_nocot_v2cot_full.sh
```

The wrapper sets `SFT_DATA` + `OUT` then execs `sft_v2cot_full.sh`, which on the
8-card config runs:
- 8 GPUs, ZeRO-3 (no offload), flash_attn
- per_device_train_batch_size=1, gradient_accumulation_steps=8 → effective batch 64
- max_length=24576, lazy_tokenize, group_by_length=false
- vit_gradient_checkpointing=true (critical for memory)
- torch_compile=true (one-time ~6-min warmup at step 1, ~57s/step steady-state)

Expected memory: ~35 GB/rank (with ~45 GB headroom).
Expected throughput: ~57 s/step.

## Known failure modes + fixes

| Failure | Symptom | Fix |
|---|---|---|
| FM not started | `torch.cuda.is_available() == False` despite driver loaded | `systemctl start nvidia-fabricmanager` |
| ABI mismatch | `torchvision::nms does not exist` | downgrade torch/torchvision to 2.11/0.26 |
| ViT OOM at 24576 seq | OOM at ~79 GB during forward | `--vit_gradient_checkpointing true` |
| flash-attn build 4-arch | nvcc compile takes 1-2 hr | `FLASH_ATTN_CUDA_ARCHS=90` (NOT `TORCH_CUDA_ARCH_LIST`) |
| audio decode crash | `soundfile.LibsndfileError: Format not recognised` mid-training | run `prepare_dataset.py --validate` (ffprobe filters no-audio mp4) |
| CUDA mismatch (DS CPU Adam) | `CUDAMismatchException: 13.2 != 13.0` | `export CUDA_HOME=/usr/local/cuda-13.0` |

## Reproducibility

- Code: git branch `ttcc-rl`, see commits since `a2442de8`.
- Data: HuggingFace `liangyuch/ttcc-v0_2_0` (sha pinned by HF revision).
- Env: see Python stack table above.
- Hardware: AWS p5.48xlarge, AMI `ami-04ad8889fce3b8b67`
  (Deep Learning Base OSS Nvidia Driver Ubuntu 24.04 20260515).

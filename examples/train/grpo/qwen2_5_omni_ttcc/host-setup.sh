#!/usr/bin/env bash
# host-setup.sh — Host-level prerequisites for p5.48xlarge SFT training.
#
# These can NOT live in a Docker image because they touch the kernel module
# and host-level systemd services. Run this once per fresh instance (or after
# any reboot that wiped apt packages — the DLAMI's unattended-upgrades has
# been observed to remove linux-modules-nvidia-595-open across reboots).
#
# Idempotent: safe to re-run.
#
# Usage:
#   sudo bash scripts/host-setup.sh
#
# Verification at the end:
#   - torch.cuda.is_available() == True
#   - nvidia-smi shows 8 H100 80GB GPUs at driver 595.71.05
#   - nvidia-fabricmanager.service is active

set -euo pipefail

echo "[host-setup] $(date) — starting"

# ---- 1. NVIDIA kernel module + userspace libs ------------------------------
# AMI ships driver 595.64 (NVIDIA Open Kernel Module from DLAMI build) but apt
# only has fabricmanager 595.71.05. Mismatch → FM refuses to start.
# Fix: install the matching apt kernel module and uninstall the DKMS 595.64.

KERNEL=$(uname -r)
echo "[host-setup] kernel: ${KERNEL}"

# Install matching kernel module + fabricmanager + userspace libs
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  linux-modules-nvidia-595-open-${KERNEL} \
  nvidia-fabricmanager-595 \
  nvidia-utils-595-server \
  libnvidia-compute-595-server \
  libnvidia-decode-595-server \
  libnvidia-encode-595-server \
  libnvidia-extra-595-server \
  libnvidia-fbc1-595-server \
  libnvidia-gl-595-server \
  libnvidia-nscq-595 \
  ninja-build \
  ffmpeg

# Swap kernel module: uninstall stale DKMS 595.64, load apt's 595.71.05.
if sudo dkms status nvidia/595.64 -k "${KERNEL}" 2>&1 | grep -q "installed"; then
  echo "[host-setup] uninstalling stale DKMS nvidia/595.64..."
  sudo dkms uninstall nvidia/595.64 -k "${KERNEL}" || true
fi
sudo depmod -a "${KERNEL}"

# Load matching nvidia module (only if not already loaded with right version).
LOADED_VER=$(cat /proc/driver/nvidia/version 2>/dev/null | grep -oE '595\.[0-9]+\.[0-9]+' | head -1 || true)
if [[ "${LOADED_VER}" != "595.71.05" ]]; then
  # Need to unload current module first if anything else is using it.
  sudo systemctl stop nvidia-dcgm 2>/dev/null || true
  sudo pkill -f nvidia-persistenced 2>/dev/null || true
  sleep 2
  for m in nvidia_uvm efa_nv_peermem gdrdrv nvidia_modeset nvidia_drm nvidia; do
    sudo rmmod "${m}" 2>/dev/null || true
  done
  sudo modprobe nvidia
fi

# ---- 2. fabricmanager service ----------------------------------------------
# H100 SXM requires fabricmanager for NVSwitch routing; without it
# torch.cuda.is_available() returns False with error 802.

sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-fabricmanager

# ---- 3. Filesystem layout --------------------------------------------------
# Scripts hardcode /home/ubuntu/go_viral. Make sure that path resolves.

if [[ -d /opt/dlami/nvme ]]; then
  # Symlink go_viral if a clone exists at the canonical NVMe location.
  if [[ -d /opt/dlami/nvme/go_viral ]] && [[ ! -e /home/ubuntu/go_viral ]]; then
    sudo mkdir -p /home/ubuntu
    sudo chmod 755 /home/ubuntu
    sudo ln -sfn /opt/dlami/nvme/go_viral /home/ubuntu/go_viral
  fi
  # /scratch -> NVMe (sft_v2cot_full.sh default VENV path)
  if [[ ! -e /scratch ]]; then
    sudo ln -sfn /opt/dlami/nvme /scratch
  fi
fi

# ---- 4. Verification --------------------------------------------------------

echo
echo "[host-setup] === verification ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
echo
echo "[host-setup] fabricmanager: $(systemctl is-active nvidia-fabricmanager)"
echo
if command -v python3 >/dev/null; then
  python3 - <<'PY'
import torch
ok = torch.cuda.is_available()
n = torch.cuda.device_count() if ok else 0
print(f"[host-setup] torch.cuda.is_available() = {ok}, device_count = {n}")
assert ok and n == 8, "GPU stack not ready"
PY
fi

echo "[host-setup] done — $(date)"

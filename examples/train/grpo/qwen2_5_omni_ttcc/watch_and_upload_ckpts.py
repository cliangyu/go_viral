"""watch_and_upload_ckpts.py — Push training checkpoints to HuggingFace as they're written.

Why this exists:
  Training runs on ephemeral NVMe (/opt/dlami/nvme/ssm-out/...). When the
  CB ends or the instance is terminated, NVMe is wiped — checkpoints die with
  the box. This script watches the output_dir and uploads each new
  `checkpoint-<step>` directory to a HF model repo, providing a versioned,
  transferable backup.

Behavior:
  - Polls output_dir every --poll-sec seconds.
  - For each new checkpoint-* directory that contains a 'model.safetensors'
    or *.bin shard AND a config.json, uploads via `hf upload`.
  - Each upload becomes a HF commit. Repo grows monotonically; HF stores
    full history.
  - Only uploads checkpoints that ms-swift has finished writing
    (detected by presence of 'training_args.bin' or 'trainer_state.json').
  - Idempotent: re-running won't re-upload.

Usage (on the training box, in background):
  setsid /opt/venv/bin/python scripts/watch_and_upload_ckpts.py \
    --output-dir /opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full \
    --hf-repo liangyuch/ttcc-sft-qwen25omni-3b \
    --poll-sec 60 \
    >/opt/dlami/nvme/logs/ckpt_upload.log 2>&1 &
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SENTINEL = ".uploaded_to_hf"  # marker file we drop after successful upload


def is_complete_ckpt(d: Path) -> bool:
    """Heuristic: ms-swift writes trainer_state.json last; if present, ckpt is done."""
    return (d / "trainer_state.json").exists() and (
        (d / "model.safetensors").exists()
        or any(d.glob("model-*-of-*.safetensors"))
        or any(d.glob("pytorch_model*.bin"))
    )


def upload_one(ckpt_dir: Path, repo: str) -> bool:
    """Upload one checkpoint dir to HF, mounted under path <run_dir>/<ckpt_name>."""
    run_name = ckpt_dir.parent.name  # e.g. "v19-20260523-181720"
    ckpt_name = ckpt_dir.name        # e.g. "checkpoint-200"
    remote_path = f"{run_name}/{ckpt_name}"
    cmd = [
        "/opt/dlami/nvme/work/swift_venv/bin/hf", "upload", repo, str(ckpt_dir), remote_path,
        "--commit-message", f"sft ckpt: {run_name}/{ckpt_name}",
        "--repo-type", "model",
    ]
    print(f"[{time.strftime('%H:%M:%S')}] uploading {ckpt_dir} -> {repo}:{remote_path}",
          flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED rc={r.returncode}", flush=True)
        print(f"  stderr: {r.stderr[:500]}", flush=True)
        return False
    # Drop sentinel file so we don't re-upload.
    (ckpt_dir / SENTINEL).write_text(
        json.dumps({"uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "repo": repo, "remote_path": remote_path}))
    print("  OK", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Parent of the run dir (which contains checkpoint-* dirs).")
    ap.add_argument("--hf-repo", required=True, help="e.g. liangyuch/ttcc-sft-qwen25omni-3b")
    ap.add_argument("--poll-sec", type=int, default=60)
    args = ap.parse_args()

    if not args.output_dir.exists():
        print(f"output_dir not found: {args.output_dir}", file=sys.stderr)
        return 1
    print(f"watching {args.output_dir} -> {args.hf_repo} (poll {args.poll_sec}s)",
          flush=True)

    seen: set[Path] = set()
    while True:
        # ms-swift creates v<N>-<DATE>/checkpoint-<step> nested dirs.
        for ckpt in args.output_dir.glob("v*/checkpoint-*"):
            if not ckpt.is_dir() or ckpt in seen:
                continue
            if (ckpt / SENTINEL).exists():
                seen.add(ckpt)
                continue
            if not is_complete_ckpt(ckpt):
                continue
            if upload_one(ckpt, args.hf_repo):
                seen.add(ckpt)
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())

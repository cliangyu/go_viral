"""Modal entrypoint: run failure-mode diagnostics on v2 submissions."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal
from _common import VOLUME_PATH, attach_code, common_env, make_cpu_image, volume

app = modal.App("cs224r-diagnose")
image = attach_code(make_cpu_image())


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=300,
              cpu=2.0, memory=4 * 1024)
def diagnose():
    import os
    os.environ.update(common_env())
    volume.reload()
    cmd = [
        "python", "/root/cs224r_project/eval/diagnose_failure_mode.py",
        "--gt", "/vol/data/splits/test.jsonl",
        "--preds",
        "/vol/runs/B1_mean_train_curve/submission.parquet:B1",
        "/vol/runs/sft_mse/seed43_v2/submission.parquet:MSE-v2-43",
        "/vol/runs/sft_hazard_cot/seed43_a0.1_v2/submission.parquet:CoT-v2-43",
    ]
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main():
    diagnose.remote()

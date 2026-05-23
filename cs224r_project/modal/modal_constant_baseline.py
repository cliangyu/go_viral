"""
Modal entrypoint: compute the content-blind constant-curve baseline +
ttcc-eval evaluate. CPU only (no Qwen forward), so quick.

Run:
  modal run cs224r_project/modal/modal_constant_baseline.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import VOLUME_PATH, attach_code, common_env, make_cpu_image, volume


app = modal.App("cs224r-constant-baseline")
image = attach_code(make_cpu_image())


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=900,
    cpu=2.0,
    memory=8 * 1024,
)
def run() -> dict:
    import os
    os.environ.update(common_env())
    volume.reload()

    sub_parquet = "/vol/runs/constant_mean/submission.parquet"
    report_json = "/vol/reports/constant_mean.json"

    Path(sub_parquet).parent.mkdir(parents=True, exist_ok=True)
    Path(report_json).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "/root/cs224r_project/baselines/constant_curve_baseline.py",
        "--train_jsonl", "/vol/data/splits/train.jsonl",
        "--test_jsonl",  "/vol/data/splits/test.jsonl",
        "--out_submission", sub_parquet,
        "--method", "constant_mean",
        "--seed", "0",
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    cmd = [
        "ttcc-eval", "evaluate",
        "--predictions", sub_parquet,
        "--report",      report_json,
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()

    with open(report_json) as f:
        payload = json.load(f)
    return {"report_path": report_json, "raw": payload}


@app.local_entrypoint()
def main():
    result = run.remote()
    print(json.dumps(result, indent=2, default=str))

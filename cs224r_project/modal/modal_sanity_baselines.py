"""
Modal entrypoint: build B0/B1/B2 sanity baseline submissions on the volume.

  modal run cs224r_project/modal/modal_sanity_baselines.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import VOLUME_PATH, attach_code, common_env, make_cpu_image, volume


app = modal.App("cs224r-sanity-baselines")
image = attach_code(make_cpu_image())


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
    cpu=2.0,
    memory=8 * 1024,
)
def run() -> dict:
    import os
    os.environ.update(common_env())
    volume.reload()

    cmd = [
        "python", "/root/cs224r_project/baselines/sanity_baselines.py",
        "--train_jsonl", "/vol/data/splits/train.jsonl",
        "--test_jsonl",  "/vol/data/splits/test.jsonl",
        "--out_dir",     "/vol/runs",
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    return {"out_dir": "/vol/runs/{B0_constant_05,B1_mean_train_curve,B2_uniform_decay}"}


@app.local_entrypoint()
def main():
    result = run.remote()
    print(json.dumps(result, indent=2, default=str))

"""
Modal entrypoint: data preparation.

Runs `cs224r_project/data/prep_ttcc.py` on a CPU container. Output (HF cache,
extracted audio, mp4 symlinks, JSONL splits) lands on the persistent volume
under /vol/data/.

Run:
  modal run cs224r_project/modal/modal_prep.py
  modal run cs224r_project/modal/modal_prep.py --max-duration 60
"""

import subprocess
import sys
from pathlib import Path

# Ensure _common.py is importable regardless of the CWD modal run uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import (
    VOLUME_PATH, attach_code, common_env, make_cpu_image, volume,
)


app = modal.App("cs224r-ttcc-prep")
image = attach_code(make_cpu_image())


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,  # 1h is plenty; bump for the 50K dump.
    cpu=4.0,
    memory=16 * 1024,
)
def prep(
    hf_dataset: str = "liangyuch/ttcc-v0_1_0",
    n_shards: int = 17,
    train_max_duration: int = 60,
) -> dict:
    import os
    os.environ.update(common_env())

    cmd = [
        "python", "/root/cs224r_project/data/prep_ttcc.py",
        "--hf_dataset", hf_dataset,
        "--n_shards", str(n_shards),
        "--out_dir", "/vol/data",
        "--train_max_duration", str(train_max_duration),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    volume.commit()

    import json
    manifest_path = "/vol/data/splits/split_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"splits: {manifest['counts']}")
    return manifest["counts"]


@app.local_entrypoint()
def main(
    hf_dataset: str = "liangyuch/ttcc-v0_1_0",
    n_shards: int = 17,
    train_max_duration: int = 60,
):
    result = prep.remote(
        hf_dataset=hf_dataset,
        n_shards=n_shards,
        train_max_duration=train_max_duration,
    )
    print(f"prep complete: {result}")

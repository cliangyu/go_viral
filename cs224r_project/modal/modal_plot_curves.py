"""
Modal entrypoint: diagnostic A — sample 5 ads, print R_hat curves.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import VOLUME_PATH, attach_code, common_env, make_cpu_image, volume


app = modal.App("cs224r-plot-curves")
image = attach_code(make_cpu_image())


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
    cpu=2.0,
    memory=4 * 1024,
)
def plot(seed: int = 0, n: int = 5) -> str:
    import os
    os.environ.update(common_env())
    volume.reload()
    cmd = [
        "python", "/root/cs224r_project/eval/plot_sample_curves.py",
        "--gt", "/vol/data/splits/test.jsonl",
        "--b1", "/vol/runs/B1_mean_train_curve/submission.parquet",
        "--sft_cot", "/vol/runs/sft_hazard_cot/seed42_a0.1_strict/submission.parquet",
        "--sft_mse", "/vol/runs/sft_mse/seed42_strict/submission.parquet",
        "--out_dir", "/vol/runs/diagnostics",
        "--n", str(n),
        "--seed", str(seed),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    return "/vol/runs/diagnostics/sample_ad_summary.tsv"


@app.local_entrypoint()
def main(seed: int = 0, n: int = 5):
    print(plot.remote(seed=seed, n=n))

"""Modal entrypoint: PNG diagnostic plots."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import VOLUME_PATH, attach_code, common_env, make_cpu_image, volume


app = modal.App("cs224r-plot-png")
# matplotlib not in the base CPU image; add it here.
image = attach_code(make_cpu_image().pip_install("matplotlib>=3.7"))


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
    cpu=2.0,
    memory=4 * 1024,
)
def plot(seed: int = 0, n: int = 8, strategy: str = "by_T",
         sft_cot_seed: int = 43, out_subdir: str = "diagnostics_v2") -> str:
    import os
    os.environ.update(common_env())
    volume.reload()
    cmd = [
        "python", "/root/cs224r_project/eval/plot_curves_png.py",
        "--gt", "/vol/data/splits/test.jsonl",
        "--b1", "/vol/runs/B1_mean_train_curve/submission.parquet",
        "--sft_cot", f"/vol/runs/sft_hazard_cot/seed{sft_cot_seed}_a0.1_v2/submission.parquet",
        "--sft_mse", f"/vol/runs/sft_mse/seed{sft_cot_seed}_v2/submission.parquet",
        "--out_dir", f"/vol/runs/{out_subdir}",
        "--n", str(n),
        "--seed", str(seed),
        "--strategy", strategy.replace("-", "_"),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    return f"/vol/runs/{out_subdir}/sample_curves_grid.png"


@app.local_entrypoint()
def main(
    seed: int = 0,
    n: int = 8,
    strategy: str = "by_T",
    sft_cot_seed: int = 43,
    out_subdir: str = "diagnostics_v2",
):
    p = plot.remote(seed=seed, n=n, strategy=strategy,
                    sft_cot_seed=sft_cot_seed, out_subdir=out_subdir)
    print(f"grid at {p}")

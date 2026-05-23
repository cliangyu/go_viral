"""
Modal entrypoint: paired BCa comparison between two methods via
`ttcc-eval compare`. Use this once both methods have written
submission.parquet files to the volume.

Default compares SFT-MSE seed 42 vs SFT-Hazard+CoT seed 42 (alpha=0.1).

Run:
  modal run cs224r_project/modal/modal_compare.py \\
      --baseline /vol/runs/sft_mse/seed42/submission.parquet \\
      --candidate /vol/runs/sft_hazard_cot/seed42_a0.1/submission.parquet \\
      --report /vol/reports/sft_hazard_cot_vs_sft_mse_seed42.json
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import VOLUME_PATH, attach_code, common_env, make_cpu_image, volume


app = modal.App("cs224r-ttcc-compare")
image = attach_code(make_cpu_image())


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=900,
    cpu=2.0,
    memory=8 * 1024,
)
def compare(baseline: str, candidate: str, report: str) -> dict:
    import os
    os.environ.update(common_env())
    volume.reload()

    Path(report).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ttcc-eval", "compare",
        "--baseline", baseline,
        "--candidate", candidate,
        "--report", report,
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()

    with open(report) as f:
        payload = json.load(f)
    return {"report_path": report, "raw": payload}


@app.local_entrypoint()
def main(
    baseline: str = "/vol/runs/sft_mse/seed42/submission.parquet",
    candidate: str = "/vol/runs/sft_hazard_cot/seed42_a0.1/submission.parquet",
    report: str = "/vol/reports/sft_hazard_cot_vs_sft_mse_seed42.json",
):
    result = compare.remote(baseline=baseline, candidate=candidate, report=report)
    print(json.dumps(result, indent=2, default=str))

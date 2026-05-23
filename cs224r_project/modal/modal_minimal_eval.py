"""
Modal entrypoint: run ttcc-eval's canonical minimal_eval.py protocol.

Headline = 6 numbers per method + paired BCa ΔIBS vs B1 (constant-mean baseline).

Run (default sweep — both stages, all strict seeds, vs B1):
  modal run cs224r_project/modal/modal_minimal_eval.py

Custom (one method only):
  modal run cs224r_project/modal/modal_minimal_eval.py \\
      --preds-spec "/vol/runs/constant_mean/submission.parquet:B1,/vol/runs/sft_mse/seed42_strict/submission.parquet:SFT-MSE-42" \\
      --ref B1
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import VOLUME_PATH, attach_code, common_env, make_cpu_image, volume


app = modal.App("cs224r-minimal-eval")
image = attach_code(make_cpu_image())


# Default set of submissions to compare. Customize via --preds-spec.
# B1 is the reference baseline per docs/05 + docs/07; B0 + B2 included as
# additional sanity checkpoints.
DEFAULT_SPEC = ",".join([
    "/vol/runs/B0_constant_05/submission.parquet:B0",
    "/vol/runs/B1_mean_train_curve/submission.parquet:B1",
    "/vol/runs/B2_uniform_decay/submission.parquet:B2",
    "/vol/runs/sft_mse/seed42_strict/submission.parquet:SFT-MSE-42",
    "/vol/runs/sft_mse/seed43_strict/submission.parquet:SFT-MSE-43",
    "/vol/runs/sft_mse/seed44_strict/submission.parquet:SFT-MSE-44",
    "/vol/runs/sft_hazard_cot/seed42_a0.1_strict/submission.parquet:SFT-CoT-42",
    "/vol/runs/sft_hazard_cot/seed43_a0.1_strict/submission.parquet:SFT-CoT-43",
    "/vol/runs/sft_hazard_cot/seed44_a0.1_strict/submission.parquet:SFT-CoT-44",
])


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=1800,
    cpu=4.0,
    memory=16 * 1024,
)
def eval_minimal(preds_spec: str, ref: str, with_windows: bool = True) -> dict:
    import os
    os.environ.update(common_env())
    volume.reload()

    # 1) Build GT JSONL in minimal_eval format (T, R_true).
    gt_path = "/vol/data/splits/test_minimal_eval.jsonl"
    Path(gt_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "/root/cs224r_project/eval/build_minimal_eval_gt.py",
        "--in_jsonl",  "/vol/data/splits/test.jsonl",
        "--out_jsonl", gt_path,
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 2) Run minimal_eval.py with all submissions.
    preds_args = preds_spec.split(",")
    # Pre-flight: verify every parquet exists.
    for spec in preds_args:
        p, m = spec.rsplit(":", 1)
        if not Path(p).exists():
            print(f"  MISSING: {p} ({m})")
    cmd = [
        "python", "/root/cs224r_project/third_party/ttcc-eval/scripts/minimal_eval.py",
        "--preds", *preds_args,
        "--gt",    gt_path,
        "--ref",   ref,
    ]
    if with_windows:
        cmd.append("--with-windows")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    return {"preds_spec": preds_spec, "ref": ref}


@app.local_entrypoint()
def main(
    preds_spec: str = "",
    ref: str = "B1",
    with_windows: bool = True,
):
    spec = preds_spec or DEFAULT_SPEC
    result = eval_minimal.remote(preds_spec=spec, ref=ref, with_windows=with_windows)
    print(json.dumps(result, indent=2, default=str))

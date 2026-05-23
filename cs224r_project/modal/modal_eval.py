"""
Modal entrypoint: produce predictions + run ttcc-eval evaluate.

Pipeline per checkpoint:
  1. make_test_preds.py  → /vol/runs/.../test_preds.parquet (diagnostic, has R_true)
  2. write_submission.py → /vol/runs/.../submission.parquet (eval-contract format)
  3. ttcc-eval evaluate  → /vol/reports/{method}_seed{N}.json

Run:
  modal run cs224r_project/modal/modal_eval.py \\
      --checkpoint /vol/runs/sft_mse/seed42 \\
      --method sft_mse --seed 42

Sweep all 3 SFT-MSE seeds in parallel:
  modal run cs224r_project/modal/modal_eval.py --all-seeds-sft-mse
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import (
    VOLUME_PATH, attach_code, common_env, hf_secret,
    make_gpu_image, volume,
)


app = modal.App("cs224r-sft-eval")
image = attach_code(make_gpu_image())

_secrets = []
if hf_secret is not None:
    _secrets.append(hf_secret)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=_secrets,
    gpu="H100",
    timeout=4 * 3600,
    cpu=4.0,
    memory=32 * 1024,
)
def make_preds_and_eval(
    checkpoint: str,
    method: str,
    seed: int,
    test_jsonl: str = "/vol/data/splits/test.jsonl",
    batch_size: int = 2,
    skip_inference_if_cached: bool = True,
) -> dict:
    import os
    os.environ.update(common_env())

    # Pick up checkpoints just written by another container (training job).
    # Without reload(), Modal volumes serve the snapshot they had at container
    # start, which races with concurrent writers.
    volume.reload()

    run_dir = Path(checkpoint)
    diag_parquet = run_dir / "test_preds.parquet"
    sub_parquet = run_dir / "submission.parquet"
    report_dir = Path("/vol/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / f"{method}_seed{seed}.json"

    # 1) Predict. Route by method: hazard methods use hazard-head inference.
    # Skip Qwen forward pass if cached predictions already exist (useful for
    # re-eval after a ttcc-eval upgrade — no need to redo expensive inference).
    if skip_inference_if_cached and diag_parquet.exists():
        print(f"[skip] reusing cached predictions at {diag_parquet}")
    else:
        if "hazard" in method or "grpo" in method:
            infer_script = "/root/cs224r_project/eval/make_test_preds_hazard.py"
        else:
            infer_script = "/root/cs224r_project/eval/make_test_preds.py"
        cmd = [
            "python", infer_script,
            "--checkpoint", str(run_dir),
            "--test_jsonl", test_jsonl,
            "--out_parquet", str(diag_parquet),
            "--batch_size", str(batch_size),
        ]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # 2) Write eval-contract submission parquet.
    cmd = [
        "python", "/root/cs224r_project/eval/write_submission.py",
        "--diagnostic_parquet", str(diag_parquet),
        "--out_parquet",        str(sub_parquet),
        "--method", method,
        "--seed",   str(seed),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 3) Run ttcc-eval evaluate.
    cmd = [
        "ttcc-eval", "evaluate",
        "--predictions", str(sub_parquet),
        "--report",      str(report_json),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    volume.commit()

    # Echo the headline metrics back to the caller.
    if report_json.exists():
        with open(report_json) as f:
            report = json.load(f)
        # Pull the three primary metrics if present. ttcc-eval's report
        # schema may evolve — we surface the raw JSON if our guess misses.
        summary = {"report_path": str(report_json), "raw": report}
        for key in ("rho_hook", "rho_comp", "rho_shape_mean", "mae_hook", "mae_comp"):
            if key in report:
                summary[key] = report[key]
        return summary
    return {"report_path": str(report_json), "warning": "report not written"}


@app.local_entrypoint()
def main(
    checkpoint: str = "",
    method: str = "",
    seed: int = -1,
    test_jsonl: str = "",
    batch_size: int = 2,
    all_seeds_sft_mse: bool = False,
    all_seeds_sft_hazard_cot: bool = False,
    alpha: float = 0.1,
):
    if all_seeds_sft_mse:
        tj = test_jsonl or "/vol/data/splits/test.jsonl"
        configs = [
            (f"/vol/runs/sft_mse/seed{s}", "sft_mse", s, tj, batch_size)
            for s in [42, 43, 44]
        ]
        for result in make_preds_and_eval.starmap(configs):
            print(json.dumps(result, indent=2, default=str))
        return

    if all_seeds_sft_hazard_cot:
        tj = test_jsonl or "/vol/data/splits/test_with_cot.jsonl"
        configs = [
            (f"/vol/runs/sft_hazard_cot/seed{s}_a{alpha}",
             "sft_hazard_cot", s, tj, batch_size)
            for s in [42, 43, 44]
        ]
        for result in make_preds_and_eval.starmap(configs):
            print(json.dumps(result, indent=2, default=str))
        return

    if not (checkpoint and method and seed >= 0):
        raise SystemExit(
            "either pass --all-seeds-sft-mse / --all-seeds-sft-hazard-cot or "
            "all of --checkpoint, --method, --seed"
        )
    # Default test_jsonl: hazard methods need the CoT-tagged version.
    tj = test_jsonl or (
        "/vol/data/splits/test_with_cot.jsonl"
        if "hazard" in method or "grpo" in method
        else "/vol/data/splits/test.jsonl"
    )
    result = make_preds_and_eval.remote(
        checkpoint=checkpoint, method=method, seed=seed,
        test_jsonl=tj, batch_size=batch_size,
    )
    print(json.dumps(result, indent=2, default=str))

"""
Modal entrypoint: 3-rollout CoT generation inference (Fix 1 / §9).

Each Stage 2 strict checkpoint gets a new submission written under
`gen3_strict` subdir.

Run:
  modal run --detach cs224r_project/modal/modal_hazard_generate.py --all-seeds
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import (
    VOLUME_PATH, attach_code, common_env, hf_secret, make_gpu_image, volume,
)


app = modal.App("cs224r-hazard-generate")
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
def generate_eval(
    seed: int = 42,
    rollouts: int = 3,
    temp: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 200,
) -> dict:
    import os
    os.environ.update(common_env())
    volume.reload()

    ckpt = f"/vol/runs/sft_hazard_cot/seed{seed}_a0.1_strict"
    out_parquet = f"/vol/runs/sft_hazard_cot/seed{seed}_a0.1_strict_gen3/test_preds.parquet"
    sub_parquet = f"/vol/runs/sft_hazard_cot/seed{seed}_a0.1_strict_gen3/submission.parquet"
    report_dir = "/vol/reports"

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # 1) Generate test_preds.parquet with K-rollout CoT generation.
    cmd = [
        "python", "/root/cs224r_project/eval/make_test_preds_hazard_generate.py",
        "--checkpoint", ckpt,
        "--test_jsonl", "/vol/data/splits/test.jsonl",
        "--out_parquet", out_parquet,
        "--rollouts", str(rollouts),
        "--temp", str(temp),
        "--top_p", str(top_p),
        "--max_new_tokens", str(max_new_tokens),
        "--seed", str(seed),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 2) Submission parquet (ttcc-eval contract format).
    cmd = [
        "python", "/root/cs224r_project/eval/write_submission.py",
        "--diagnostic_parquet", out_parquet,
        "--out_parquet",        sub_parquet,
        "--method", "sft_hazard_cot_strict_gen3",
        "--seed",   str(seed),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    return {"submission": sub_parquet}


@app.local_entrypoint()
def main(
    seed: int = 42,
    rollouts: int = 3,
    temp: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 200,
    all_seeds: bool = False,
):
    if all_seeds:
        for s in [42, 43, 44]:
            print(f"===== generate seed={s} =====")
            generate_eval.remote(
                seed=s, rollouts=rollouts, temp=temp, top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
    else:
        result = generate_eval.remote(
            seed=seed, rollouts=rollouts, temp=temp, top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        print(json.dumps(result, indent=2, default=str))

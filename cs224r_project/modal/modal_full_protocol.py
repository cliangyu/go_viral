"""
Modal entrypoint: run the full ttcc-eval protocol from /tmp/ttcc-eval-readme bundle.

Scripts executed (in order):
  0. ttcc-eval verify         (regression test: GT-as-prediction)
  1. minimal_eval.py          (headline 6 numbers + paired ΔIBS vs B1)
  2. full_eval.py             (BSS + Murphy REL/RES/UNC + slope, sanity checks)
  3. conditional_eval.py      (novelty Q2Q3 subset — content-awareness signal)
  4. segment_eval.py          (hook + completion MSE + AUC + paired BCa)
  5. rmst_eval.py             (RMST MAE in seconds + C-index)
  6. weighted_eval.py         (4 weighting schemes including excess-skill D)
  7. inflection_analysis.py   (drop-localization, qualitative)

All outputs land at /vol/reports/protocol/ on the volume.

Run:
  modal run cs224r_project/modal/modal_full_protocol.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import VOLUME_PATH, attach_code, common_env, make_cpu_image, volume


app = modal.App("cs224r-full-protocol")
image = attach_code(make_cpu_image().pip_install("matplotlib>=3.7"))


GT = "/vol/data/splits/test_minimal_eval.jsonl"
TRAIN_GT = "/vol/data/splits/train_minimal_eval.jsonl"

# Submissions ordered: B1 first (it's the reference for most paired tests).
SUBMISSIONS = [
    ("B1", "/vol/runs/B1_mean_train_curve/submission.parquet"),
    ("B0", "/vol/runs/B0_constant_05/submission.parquet"),
    ("B2", "/vol/runs/B2_uniform_decay/submission.parquet"),
    # v2 = Liangyu-aligned hparams: 32 frames, 200k px, 10 epochs.
    ("SFT-MSE-v2-42",  "/vol/runs/sft_mse/seed42_v2/submission.parquet"),
    ("SFT-MSE-v2-43",  "/vol/runs/sft_mse/seed43_v2/submission.parquet"),
    ("SFT-MSE-v2-44",  "/vol/runs/sft_mse/seed44_v2/submission.parquet"),
    ("SFT-CoT-v2-42",  "/vol/runs/sft_hazard_cot/seed42_a0.1_v2/submission.parquet"),
    ("SFT-CoT-v2-43",  "/vol/runs/sft_hazard_cot/seed43_a0.1_v2/submission.parquet"),
    ("SFT-CoT-v2-44",  "/vol/runs/sft_hazard_cot/seed44_a0.1_v2/submission.parquet"),
]


def _preds_args() -> list[str]:
    return [f"{path}:{name}" for name, path in SUBMISSIONS]


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=2400,
    cpu=4.0,
    memory=16 * 1024,
)
def run_protocol() -> dict:
    import os
    os.environ.update(common_env())
    volume.reload()

    proto_dir = Path("/vol/reports/protocol_v2")
    proto_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0) build train + test GTs in minimal_eval format ----
    for in_jsonl, out_jsonl in [
        ("/vol/data/splits/test.jsonl",  GT),
        ("/vol/data/splits/train.jsonl", TRAIN_GT),
    ]:
        cmd = [
            "python", "/root/cs224r_project/eval/build_minimal_eval_gt.py",
            "--in_jsonl", in_jsonl, "--out_jsonl", out_jsonl,
        ]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    scripts_dir = "/root/cs224r_project/third_party/ttcc-eval/scripts"
    b1_path = next(p for n, p in SUBMISSIONS if n == "B1")
    preds_args = _preds_args()

    def run_and_save(name: str, cmd: list[str]) -> Path:
        out = proto_dir / f"{name}.txt"
        print("\n" + "=" * 80)
        print(f"[{name}]  $ {' '.join(cmd)}")
        print("=" * 80)
        proc = subprocess.run(
            cmd, check=False, capture_output=True, text=True,
        )
        out.write_text(
            f"$ {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- returncode ---\n{proc.returncode}\n"
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"!! {name} returned {proc.returncode}; stderr:")
            print(proc.stderr)
        return out

    results: dict[str, str] = {}

    # 0) Identity verify
    results["00_verify"] = str(run_and_save(
        "00_verify",
        ["ttcc-eval", "verify"],
    ))

    # 1) minimal_eval
    results["01_minimal"] = str(run_and_save(
        "01_minimal",
        ["python", f"{scripts_dir}/minimal_eval.py",
         "--preds", *preds_args, "--gt", GT, "--ref", "B1",
         "--with-windows"],
    ))

    # 2) full_eval (BSS + Murphy)
    results["02_full"] = str(run_and_save(
        "02_full",
        ["python", f"{scripts_dir}/full_eval.py",
         "--preds", *preds_args, "--b1-preds", b1_path, "--gt", GT],
    ))

    # 3) conditional_eval on Q2Q3 (middle 50% novelty)
    results["03_conditional_Q2Q3"] = str(run_and_save(
        "03_conditional_Q2Q3",
        ["python", f"{scripts_dir}/conditional_eval.py",
         "--preds", *preds_args, "--b1-preds", b1_path, "--gt", GT,
         "--subset", "Q2Q3", "--ref", "B1"],
    ))
    # also Q1 and Q4 for context
    for sub in ("Q1", "Q4", "all"):
        results[f"03_conditional_{sub}"] = str(run_and_save(
            f"03_conditional_{sub}",
            ["python", f"{scripts_dir}/conditional_eval.py",
             "--preds", *preds_args, "--b1-preds", b1_path, "--gt", GT,
             "--subset", sub, "--ref", "B1"],
        ))

    # 4) segment_eval
    results["04_segment"] = str(run_and_save(
        "04_segment",
        ["python", f"{scripts_dir}/segment_eval.py",
         "--preds", *preds_args, "--b1-preds", b1_path, "--gt", GT,
         "--train-gt", TRAIN_GT],
    ))

    # 5) rmst_eval
    results["05_rmst"] = str(run_and_save(
        "05_rmst",
        ["python", f"{scripts_dir}/rmst_eval.py",
         "--preds", *preds_args, "--b1-preds", b1_path, "--gt", GT],
    ))

    # 6) weighted_eval
    results["06_weighted"] = str(run_and_save(
        "06_weighted",
        ["python", f"{scripts_dir}/weighted_eval.py",
         "--preds", *preds_args, "--b1-preds", b1_path, "--gt", GT],
    ))

    # 7) inflection_analysis (writes a PNG)
    results["07_inflection"] = str(run_and_save(
        "07_inflection",
        ["python", f"{scripts_dir}/inflection_analysis.py",
         "--preds", *preds_args, "--gt", GT,
         "--out", str(proto_dir / "inflection.png")],
    ))

    volume.commit()
    return results


@app.local_entrypoint()
def main():
    result = run_protocol.remote()
    print(json.dumps(result, indent=2, default=str))

"""
Modal entrypoint: download Liangyu's CoT distillation jsonl from HF and
inject into the existing splits to produce {train,val,test}_with_cot.jsonl
on the persistent volume.

Pulls from:  https://huggingface.co/datasets/liangyuch/ttcc-cot

Run:
  modal run cs224r_project/modal/modal_merge_cot.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import VOLUME_PATH, attach_code, common_env, make_cpu_image, volume


app = modal.App("cs224r-ttcc-merge-cot")
image = attach_code(make_cpu_image())


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
    cpu=2.0,
    memory=4 * 1024,
)
def merge(
    cot_repo: str = "liangyuch/ttcc-cot",
    cot_filename: str = "cot_v2_causal_instruct.jsonl",
) -> dict:
    import os
    import json
    os.environ.update(common_env())

    # Make sure we see the latest splits the prep job committed.
    volume.reload()

    from huggingface_hub import hf_hub_download
    print(f"Downloading {cot_filename} from {cot_repo}")
    cot_local = hf_hub_download(
        cot_repo, cot_filename,
        repo_type="dataset",
        cache_dir="/vol/hf_cache",
    )
    # Stash a copy under /vol/data/cot/ so it's discoverable later.
    cot_dir = Path("/vol/data/cot")
    cot_dir.mkdir(parents=True, exist_ok=True)
    cot_dst = cot_dir / cot_filename
    if not cot_dst.exists():
        from shutil import copy2
        copy2(cot_local, cot_dst)

    cmd = [
        "python", "/root/cs224r_project/data/merge_cot.py",
        "--cot_manifest", str(cot_dst),
        "--splits_dir",   "/vol/data/splits",
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()

    # Count rows in each output for a sanity echo.
    out = {}
    splits_dir = Path("/vol/data/splits")
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}_with_cot.jsonl"
        if not path.exists():
            out[split] = None
            continue
        n_total = 0
        n_with_cot = 0
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                n_total += 1
                cot = row.get("_meta", {}).get("cot", "")
                if cot:
                    n_with_cot += 1
        out[split] = {"n_total": n_total, "n_with_cot": n_with_cot}
    print(f"merge result: {out}")
    return out


@app.local_entrypoint()
def main(
    cot_repo: str = "liangyuch/ttcc-cot",
    cot_filename: str = "cot_v2_causal_instruct.jsonl",
):
    result = merge.remote(cot_repo=cot_repo, cot_filename=cot_filename)
    print(f"merge complete: {result}")

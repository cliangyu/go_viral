"""Modal entrypoint to run scripts/inspect_freeze.py on a GPU container."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

from _common import VOLUME_PATH, attach_code, common_env, make_gpu_image, volume


app = modal.App("cs224r-inspect-freeze")
image = attach_code(make_gpu_image())


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    gpu="H100",
    timeout=900,
    cpu=2.0,
    memory=16 * 1024,
)
def inspect():
    import os
    os.environ.update(common_env())
    subprocess.run(
        ["python", "/root/cs224r_project/scripts/inspect_freeze.py"],
        check=True,
    )


@app.local_entrypoint()
def main():
    inspect.remote()

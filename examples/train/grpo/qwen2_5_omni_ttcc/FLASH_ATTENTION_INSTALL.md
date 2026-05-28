# Flash Attention Install on p5.48xlarge (DLAMI Ubuntu 24.04, torch 2.12+cu130)

This is the playbook for installing flash-attention on the V8 training cluster.
Lessons learned 2026-05-28; this stack is bleeding-edge (torch 2.12 nightly +
CUDA 13.0) and most prebuilt wheels do NOT cover it.

## TL;DR — copy-paste recipe

```bash
VENV=/opt/dlami/nvme/work/swift_venv
SP=$VENV/lib/python3.12/site-packages

# 1. Pin huggingface_hub to a transformers-compatible version
$VENV/bin/pip install 'huggingface_hub>=0.34,<1.0'

# 2. Download varunneal/flash-attention-3 HF kernel (only known source
#    with the full Python API on cu130+torch2120)
mkdir -p /opt/dlami/nvme/work/fa3_kernel
cd /opt/dlami/nvme/work/fa3_kernel
$VENV/bin/huggingface-cli download varunneal/flash-attention-3 \
    --include 'build/torch212-cxx11-cu130-x86_64-linux/**' \
    --local-dir .

# 3. Install into site-packages as `flash_attn_3` + add shim for
#    `flash_attn_interface` (the runtime import path)
KSRC=/opt/dlami/nvme/work/fa3_kernel/build/torch212-cxx11-cu130-x86_64-linux/flash_attention_3
cp -r $KSRC $SP/flash_attn_3
echo 'from flash_attn_3.flash_attn_interface import *' > $SP/flash_attn_interface.py

# 3b. CRITICAL: cp -r does NOT register pip metadata. transformers'
#     _is_package_available() calls importlib.metadata.version() which
#     returns PackageNotFoundError → returns False → FA3 is_available
#     returns False even though imports work. Fake the dist-info:
mkdir -p $SP/flash_attn_3-3.0.0.dist-info
printf 'Metadata-Version: 2.1\nName: flash_attn_3\nVersion: 3.0.0\n' \
    > $SP/flash_attn_3-3.0.0.dist-info/METADATA

# 4. Verify
$VENV/bin/python -c '
import importlib.util
assert importlib.util.find_spec("flash_attn_3") is not None
from flash_attn_interface import flash_attn_func
from flash_attn_3 import flash_attn_func as f2
print("FA3 ready")
'

# 5. Use it: yaml sets attn_impl: flash_attention_3 (NOT flash_attn or flash_attn_3)
```

## What we tried that DID NOT work — and why

### 1. `pip install flash-attn==2.8.3 --no-build-isolation` (no MAX_JOBS)
- Result: **Killed** at compile time (OOM kill, exit -9)
- Cause: ninja runs 192 parallel nvcc on this 192-vCPU box; each nvcc peak
  ~30 GB during CUTLASS/cute::tuple template instantiation; despite 2 TB
  total RAM, swap=0 + transient high-watermark = OS SIGKILLs the process

### 2. `MAX_JOBS=4 pip install flash-attn==2.8.3`
- Result: **Killed**
- Cause: 4 parallel nvcc still hits the same per-process peak. The bottleneck
  is per-process memory, not aggregate.

### 3. `MAX_JOBS=64 NVCC_THREADS=8 TORCH_CUDA_ARCH_LIST=9.0` (Zuocan's attempt)
- Result: **Killed** at 27/72 .o files
- Cause: `TORCH_CUDA_ARCH_LIST` is **IGNORED by flash-attn's root setup.py**.
  The `-gencode` flags get filtered but **the .cu files all compile anyway**.
  First file produced was `flash_bwd_hdim128_bf16_causal_sm80.o` — proof
  that sm_80 builds happened despite the env var.

### 4. Build from `flash-attention/hopper/` subdir with disable flags
```bash
FLASH_ATTENTION_DISABLE_SM80=TRUE
FLASH_ATTENTION_DISABLE_FP16=TRUE
MAX_JOBS=4 NVCC_THREADS=4
pip install -e .
```
- Result: **Killed** at `flash_bwd_hdim192_bf16_softcapall_sm90.cu` (nvcc exit -9)
- Cause: Even the necessary sm_90 bf16 softcap kernels hit single-file peak
  > 30 GB; OS kills the nvcc process. Disable flags reduce kernel **count**
  but not per-kernel **memory**.

### 5. `windreamer/flash-attention3-wheels` cu130+torch2120 wheel
- Result: Installs without error but **stub** — only ships `_C` extension,
  not Python API
- Verify: `dir(flash_attn_3) == []`, `from flash_attn_interface import flash_attn_func`
  → ModuleNotFoundError
- Cause: upstream commit `e2743ab` of `flash-attention/hopper/` is missing
  `__init__.py`; `find_packages()` returns empty → only `_C` ships

### 6. `pip install flash-attn-3 --index-url https://download.pytorch.org/whl/flash-attn-3/`
- Result: "No matching distribution found"
- Cause: PyTorch's official FA3 wheel index doesn't actually host this name.

## What works

### `varunneal/flash-attention-3` on HuggingFace

Maintainer forked `flash-attention/hopper/` and **added the missing
`__init__.py`**, packaged via HF kernels-hub layout. Their build at
`torch212-cxx11-cu130-x86_64-linux/flash_attention_3/` contains:

- `__init__.py` (`from .flash_attn_interface import *`)
- `flash_attn_interface.py` (41 KB — the real Tri Dao Python API)
- `flash_attn_config.py` (946 B)
- `_C.abi3.so` (125 MB compiled extension, cp39-abi3 → works on cp312)

The cp39-abi3 stable ABI tag lets the same .so load on Python 3.12.

### transformers integration

transformers v4.56.2 supports FA3 via:
```python
model = AutoModel.from_pretrained(..., attn_implementation="flash_attention_3")
```

The dispatch path (`modeling_flash_attention_utils.py:78-92`):
```python
is_fa3 = is_flash_attn_3_available()   # checks find_spec("flash_attn_3")
if implementation == "flash_attention_3" or (implementation is None and is_fa3):
    from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
```

So we need TWO things importable:
1. `flash_attn_3` as a package (for `is_flash_attn_3_available` gate)
2. `flash_attn_interface` at top level (for the runtime forward)

Our install achieves both: `cp -r .../flash_attention_3 site-packages/flash_attn_3`
(rename → satisfies the gate) + shim file `flash_attn_interface.py`
(re-exports → satisfies the runtime import).

## What about FA2?

For H100 (SM_90), **use FA3, not FA2**. FA3 is the Hopper-native version
with better kernel ergonomics for the architecture. FA2 has no prebuilt
wheel for cu130+torch2120 AND building from source on this box keeps OOMing.
Even if you built it, FA2 is slower than FA3 on H100 by a measurable margin.

## Open issues to be aware of

1. **`huggingface_hub` 1.x breaks transformers 4.56**: `pip install` of newer
   `huggingface_hub` (>=1.0) shows a dependency conflict and changes the CLI
   from `huggingface-cli` to `hf`. Pin to `>=0.34,<1.0`.

2. **`accelerate` 1.13 vs Zuocan's runbook 0.34**: ms-swift requires
   `accelerate >= 1.1.0` (for `data_seed` arg). Zuocan's runbook documented
   0.34 from before this requirement. We use 1.13. No known incompatibility,
   but worth noting if a regression appears.

3. **FA3 + Qwen2.5-Omni vision/audio path**: `modeling_qwen2_5_omni.py` has
   literal `== "flash_attention_2"` checks at lines 748, 930 that gate the
   varlen path for vision/audio subblocks. Under FA3 those branches fall
   back to padded calls. **Perf regression for densely-packed vision tokens
   vs FA2**. For Omni training of language head, this is a minor cost.

4. **transformers issue #41179** documents the `find_spec('flash_attn_3')`
   gate bug. Our shim sidesteps it.

## References

- [varunneal/flash-attention-3 on HuggingFace](https://huggingface.co/varunneal/flash-attention-3)
- [transformers issue #41179](https://github.com/huggingface/transformers/issues/41179)
- [Dao-AILab flash-attention setup.py](https://github.com/Dao-AILab/flash-attention/blob/main/setup.py)
- [windreamer wheels](https://windreamer.github.io/flash-attention3-wheels/) — known stub for cu130
- [PyTorch FA3 wheel index (does not include cu130)](https://download.pytorch.org/whl/flash-attn-3/)

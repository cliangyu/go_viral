#!/usr/bin/env bash
# Make a box's swift_venv CoT-GRPO-ready (run once per node BEFORE a run that uses the
# GRPO path). Idempotent. The SFT path never needed these; the GRPO path does.
#   1) msgspec  — swift GRPO import dep, absent in the SFT-era venv.
#   2) peft EmbeddingParallel shim — peft 0.19.1 vs transformers 4.56.2 skew (see ops/peft_embeddingparallel_shim.py).
# Does NOT disturb a running SFT (it already imported its modules).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/opt/dlami/nvme/work/swift_venv}"

echo "[shims] venv=${VENV}"
"${VENV}/bin/pip" install -q msgspec 2>&1 | tail -1 || true
"${VENV}/bin/python" "${HERE}/peft_embeddingparallel_shim.py"
echo "[shims] verify:"
"${VENV}/bin/python" -c "import msgspec; from peft.utils.save_and_load import _maybe_shard_state_dict_for_tp; from swift.rlhf_trainers import GRPOTrainer; print('  msgspec + peft + GRPOTrainer import OK (no vllm)')"

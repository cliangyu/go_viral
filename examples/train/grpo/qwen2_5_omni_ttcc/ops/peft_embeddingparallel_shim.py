#!/usr/bin/env python
"""Compatibility shim: peft 0.19.1's _maybe_shard_state_dict_for_tp hard-imports
EmbeddingParallel from transformers.integrations.tensor_parallel BEFORE its non-TP
early-skip, but transformers 4.56.2 (pinned for our Qwen2.5-Omni-retention model)
lacks it -> ImportError on every adapter LOAD (--adapters warm-start). The TP path
is unused here, so make that one symbol optional. Idempotent; keeps a .bak_cotgrl backup.

Run once per box venv before any CoT-GRPO run that warm-starts from an adapter:
  <venv>/bin/python ops/peft_embeddingparallel_shim.py
Proper fix is aligning peft/transformers versions; this is a scoped, reversible shim.
"""
import shutil
import sys

F = '/opt/dlami/nvme/work/swift_venv/lib/python3.12/site-packages/peft/utils/save_and_load.py'
OLD = ("    from transformers.integrations.tensor_parallel import (\n"
       "        ALL_PARALLEL_STYLES,\n        ColwiseParallel,\n        EmbeddingParallel,\n"
       "        RowwiseParallel,\n    )\n")
NEW = ("    from transformers.integrations.tensor_parallel import (\n"
       "        ALL_PARALLEL_STYLES,\n        ColwiseParallel,\n        RowwiseParallel,\n    )\n"
       "    try:  # cotgrl shim: transformers 4.56.2 lacks EmbeddingParallel; TP path unused here\n"
       "        from transformers.integrations.tensor_parallel import EmbeddingParallel\n"
       "    except ImportError:\n        EmbeddingParallel = None\n")


def main():
    src = open(F).read()
    if 'cotgrl shim' in src:
        print('peft already patched'); return 0
    if OLD not in src:
        print('IMPORT BLOCK NOT FOUND VERBATIM — aborting, no change'); return 1
    shutil.copy(F, F + '.bak_cotgrl')
    open(F, 'w').write(src.replace(OLD, NEW, 1))
    print(f'peft patched + backup {F}.bak_cotgrl')
    return 0


if __name__ == '__main__':
    sys.exit(main())

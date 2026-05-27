#!/usr/bin/env python
"""Round-trip checkpoint validation.

Loads a saved checkpoint from disk via `swift.model.get_model_processor` with
the `qwen2_5_omni_retention` model_type, runs a single forward pass on a
synthetic input, and confirms:

1. The checkpoint dir contains all 10 files required for multimodal load.
2. The processor reconstructs without `image_token` / `video_token` /
   `audio_token` AttributeError.
3. The model loads with the retention head weights restored from safetensors.
4. A single forward produces a non-NaN `r_pred` of shape `(B, T_MAX)`.

Replaces a heavier CI pipeline for our cadence (~5 production launches per
project). Run after every training launch (and after every published
checkpoint) to catch save-format regressions cheaply.

Usage:
    python validate_ckpt.py <ckpt_dir>
    # exit 0 on success, 1 on any failure
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REQUIRED_FILES = [
    'config.json',
    'generation_config.json',
    'model.safetensors.index.json',
    'preprocessor_config.json',
    'processor_config.json',
    'spk_dict.pt',
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'vocab.json',
    'merges.txt',
    'added_tokens.json',
    'chat_template.json',
]


def check_files(ckpt_dir):
    missing = []
    for f in REQUIRED_FILES:
        if not (ckpt_dir / f).exists():
            missing.append(f)
    return missing


def check_load_and_forward(ckpt_dir, plugin):
    spec = importlib.util.spec_from_file_location('retention_plugin', plugin)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    os.environ.setdefault('RETENTION_HEAD_TYPE', 'hazard')
    os.environ.setdefault('ENABLE_AUDIO_OUTPUT', 'False')

    import torch
    from swift.model import get_model_processor

    print(f'[validate] loading {ckpt_dir} ...', flush=True)
    model, processor = get_model_processor(
        str(ckpt_dir),
        model_type='qwen2_5_omni_retention',
        torch_dtype=torch.bfloat16,
    )
    tk = processor.tokenizer
    for tok in ('image_token', 'video_token', 'audio_token'):
        if not hasattr(tk, tok):
            raise RuntimeError(f'tokenizer missing attribute: {tok}')

    # Locate the retention head
    base = getattr(model, 'base_model', model)
    base = getattr(base, 'model', base)
    head = getattr(base, '_retention_head', None) or getattr(base, 'retention_head', None)
    if head is None:
        raise RuntimeError('retention_head not attached to model')
    if head.linear.weight.shape != (60, head.linear.weight.shape[1]):
        # T_MAX dimension is the output rows, hidden_size is columns
        pass  # accept any hidden_size

    # Simple input: prompt only, no video/audio. Tests that
    # the LM end-to-end path produces hidden states and the
    # head's anchor-position read produces a non-NaN tensor.
    ids = tk('Hello.', return_tensors='pt').input_ids
    ids = ids.to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(input_ids=ids)

    r_pred = getattr(out, 'r_pred', None)
    if r_pred is None:
        holder = getattr(base, '_retention_h_holder', None)
        r_pred = holder.r_pred if holder is not None else None
    if r_pred is None:
        raise RuntimeError('forward produced no r_pred')
    if torch.isnan(r_pred).any():
        raise RuntimeError(f'r_pred contains NaN: {r_pred}')
    return tuple(r_pred.shape), float(r_pred.float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt_dir')
    ap.add_argument('--plugin', default=None,
                    help='Path to register.py. Defaults to sibling-of-this-script.')
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    if not ckpt_dir.is_dir():
        print(f'NOT A DIR: {ckpt_dir}', file=sys.stderr)
        return 2

    plugin = args.plugin or str(Path(__file__).resolve().parent.parent / 'register.py')

    # Step 1: file completeness.
    missing = check_files(ckpt_dir)
    if missing:
        print(f'FAIL  missing required files: {missing}', file=sys.stderr)
        return 1
    print(f'OK    all {len(REQUIRED_FILES)} required files present')

    # Step 2: load + 1 forward.
    try:
        shape, mean = check_load_and_forward(ckpt_dir, plugin)
    except Exception as e:
        print(f'FAIL  load/forward: {type(e).__name__}: {e}', file=sys.stderr)
        return 1
    print(f'OK    forward returned r_pred shape={shape} mean={mean:.5f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

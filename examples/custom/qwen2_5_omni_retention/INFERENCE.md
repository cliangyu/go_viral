# Inference Guide — Retention-Curve Predictor (Qwen2.5-Omni-3B + hazard head)

> **⚠ IMPORTANT (2026-05-26): V7 checkpoints under `v2-20260524-103930/checkpoint-*`
> were trained with a label-in-input leak. Three independent audits (randomization
> probe, video-swap, constant-predictor sanity) confirm the V7 backbone does NOT
> use the video signal — predictions are essentially constant per ad once the
> leaked ground-truth curve is removed from the input. Do NOT use V7 checkpoints
> for new inference. They are kept on HF for reproducibility / audit only.
>
> A leak-free V8 (with CoT distillation, assistant span = `<cot>...</cot>` only)
> is in preparation. This guide will be updated when V8 checkpoints land.
>
> See `INCIDENT_2026-05-26_EVAL_LEAK.md` in
> [cliangyu/ttcc-eval](https://github.com/cliangyu/ttcc-eval) for the full
> three-part incident review and the affected metrics.

This guide shows how to load a trained TTCC retention checkpoint and predict the per-second retention curve `R(t)` for a TikTok ad video. Single forward pass per ad; works on a single 24 GB+ GPU.

## What the model does

Input: an MP4 ad video (with embedded audio) + a short text prompt describing the task.

Output: a non-increasing retention curve `R(t)` for t = 0..60 seconds, where `R(t)` is the fraction of viewers still watching at second `t`. `R(0) = 1` by convention.

The retention curve comes from a small structural head on top of the Qwen2.5-Omni-3B multimodal language model. The head reads a single hidden state, emits 60 non-negative hazards `λ(1..60)` via softplus, and recovers `R(t) = exp(-Σ_{s≤t} λ(s))`. This parameterization guarantees the curve is monotone non-increasing — you never have to clamp the output post-hoc.

## Where to get the checkpoint

**HuggingFace:** [`liangyuch/ttcc-sft-qwen25omni-3b`](https://huggingface.co/liangyuch/ttcc-sft-qwen25omni-3b)

The repo contains many intermediate saves under the path `v2-20260524-103930/checkpoint-{step}/`. The current production-quality checkpoint is `checkpoint-3675` — the last saved step of the full SFT run. Reported mean Integrated Brier Score (IBS) on a 3,375-ad held-out test split is **0.000092**, equivalent to ~0.85% absolute error per second of retention curve.

| Checkpoint | Step | When to use |
|---|---|---|
| `v2-20260524-103930/checkpoint-3675` | 3675 | Recommended for inference. Latest saved checkpoint with full multimodal weights + retention head. |
| Earlier checkpoints (e.g. `checkpoint-2175`) | 2175 etc. | For ablation: how quality scales with training. Otherwise use the latest. |

## Setup

```bash
# 1) Get this repo on the inference branch (contains the model registration plugin).
git clone -b ttcc-rl https://github.com/cliangyu/go_viral.git
cd go_viral

# 2) Python environment.
python -m venv venv
source venv/bin/activate
pip install -e .

# 3) Download the base Qwen2.5-Omni-3B (we need its tokenizer / processor files;
#    see "Known issue" below).
huggingface-cli download Qwen/Qwen2.5-Omni-3B \
    --local-dir ~/work/hf-cache/Qwen2.5-Omni-3B

# 4) Download the trained checkpoint. Skip the DeepSpeed shards to save ~30 GB;
#    they're only needed if you want to resume training.
huggingface-cli download liangyuch/ttcc-sft-qwen25omni-3b \
    --include 'v2-20260524-103930/checkpoint-3675/*' \
    --exclude '*/global_step*/*' \
    --local-dir ~/work/ttcc-ckpt

# 5) IMPORTANT: overlay the multimodal tokenizer/processor files from the base
#    model. See "Known issue" below for why this is needed.
CKPT=~/work/ttcc-ckpt/v2-20260524-103930/checkpoint-3675
BASE=~/work/hf-cache/Qwen2.5-Omni-3B
for F in added_tokens.json merges.txt special_tokens_map.json vocab.json \
         chat_template.json tokenizer_config.json; do
    cp "$BASE/$F" "$CKPT/$F"
done
```

## Minimal inference script (single ad → curve)

```python
"""Predict the per-second retention curve for one TikTok ad."""
import os
import numpy as np
import torch
import importlib.util

# 1) Load the plugin. This registers the custom model_type 'qwen2_5_omni_retention',
#    the retention template, and the loss. It must be loaded before model construction.
PLUGIN = 'examples/custom/qwen2_5_omni_retention/register.py'
spec = importlib.util.spec_from_file_location('retention_plugin', PLUGIN)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

os.environ['RETENTION_HEAD_TYPE'] = 'hazard'      # use the softplus -> cumsum -> exp head
os.environ['ENABLE_AUDIO_OUTPUT'] = 'False'       # skip Talker (speech gen), save ~1.5 GB

# 2) Load the checkpoint with the custom model_type.
from swift.model import get_model_processor

CKPT = os.path.expanduser(
    '~/work/ttcc-ckpt/v2-20260524-103930/checkpoint-3675'
)
model, processor = get_model_processor(
    CKPT,
    model_type='qwen2_5_omni_retention',
    torch_dtype=torch.bfloat16,
)
model = model.to('cuda:0').eval()

# 3) Build one inference row. Schema matches the training JSONL.
ad_seconds = 15
row = {
    'messages': [
        {
            'role': 'system',
            'content': (
                'You are an expert in short-form video advertising. You forecast '
                'second-by-second audience retention curves. R(t) is the fraction '
                'of viewers still watching at second t, with R(0) = 1 by convention.'
            ),
        },
        {
            'role': 'user',
            'content': f'This ad is {ad_seconds} seconds long. '
                       f'Estimate the per-second retention curve.',
        },
        {'role': 'assistant', 'content': ''},   # empty: retention head reads h[last_token]
    ],
    # Same path for video and audio: Qwen-Omni reads audio embedded in the MP4.
    'videos': ['/path/to/your/ad.mp4'],
    'audios': ['/path/to/your/ad.mp4'],
    'T': ad_seconds,
}

# 4) Encode + collate + forward.
from swift.template import get_template
from swift.template.template_inputs import TemplateInputs

template = get_template(
    processor,
    template_type='qwen2_5_omni_retention',
    truncation_strategy='right',
    max_length=32768,
)
enc = template.encode(TemplateInputs.from_dict(row))
batch = template.data_collator([enc])
batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

with torch.no_grad():
    out = model(**batch)

# 5) Read the prediction. The head emits R(1..60); prepend R(0)=1 and trim to ad length.
r_pred = out.r_pred[0].float().cpu().numpy()              # shape (60,)
R = np.concatenate([[1.0], r_pred])[: row['T'] + 1]       # shape (T+1,)

print(f'Predicted retention curve for {row["T"]}s ad:')
for t, v in enumerate(R):
    print(f'  t={t:2d}s   R={v:.4f}')
```

Example output:
```
t= 0s   R=1.0000
t= 1s   R=0.8423
t= 2s   R=0.7156
...
t=15s   R=0.2891
```

## Batched inference

Build multiple rows and collate them together:

```python
encs = [template.encode(TemplateInputs.from_dict(r)) for r in rows]
batch = template.data_collator(encs)
batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
with torch.no_grad():
    out = model(**batch)
# out.r_pred has shape (B, 60)
```

Memory: ~13 GB per ad at batch 1, ~3 GB more per added ad (rough estimate, dominated by visual encoder activations). On a 24 GB GPU you can comfortably fit batches up to 3-4.

## Architecture in one diagram

```
Input: video frames + embedded audio + system+user text
   │
   ▼
Qwen2.5-Omni encoder (vision + audio + aligner, frozen during this SFT)
   │
   ▼
LM decoder (full fine-tuned; reads multimodal embeddings + text)
   │
   ▼
hidden state h at the last input token  ──► RetentionHead (nn.Linear(2048 → 60))
                                                  │
                                                  ▼
                                          softplus → 60 non-negative hazards λ(1..60)
                                                  │
                                                  ▼
                                          cumsum + exp(-·) → R(t) for t=1..60
                                                  │
                                                  ▼
                                          (always monotone non-increasing)
```

The retention head is small (~123K params) but trained jointly with the multimodal encoder + LM. The full model is `Qwen2.5-Omni-3B` + this head; ~3B parameters total.

## Known issue: tokenizer overlay is mandatory

The published checkpoints save only a partial tokenizer (the LM tokenizer, not the full multimodal processor). Loading the checkpoint without the overlay produces:

```
AttributeError: Qwen2TokenizerFast has no attribute image_token
```

Cause: this is a roundtrip-save bug in how multimodal processors are serialized by the training framework (HuggingFace `ProcessorMixin.save_pretrained` documents that it "calls feature_extractor.save_pretrained and tokenizer.save_pretrained" — but the tokenizer attached to the Qwen2.5-Omni processor at training time is a stripped LM-only variant, so the multimodal placeholder tokens `<|IMAGE|>`, `<|VIDEO|>`, `<|AUDIO|>` don't survive the save).

Fix: copy these 6 files from the base `Qwen2.5-Omni-3B/` into the checkpoint directory before loading:

- `added_tokens.json`
- `merges.txt`
- `special_tokens_map.json`
- `vocab.json`
- `chat_template.json`
- `tokenizer_config.json`

After overlay, `from_pretrained` succeeds.

## Things to watch out for

1. **Audio path = video path.** Qwen2.5-Omni reads audio from the MP4 (`USE_AUDIO_IN_VIDEO=true`). Don't pass a separate WAV.

2. **`max_length=32768`.** Very long ads + heavy multimodal content can exceed this; the template raises `MaxLengthError`. Catch it and skip the ad.

3. **`R(0)=1` is a convention.** The head emits R(1..60); always prepend 1.0 to get the full curve, then slice to T+1 entries for a T-second ad.

4. **`out.r_pred` is the prediction, not the model's text output.** The system+user prompt may ask the model to "write a JSON curve" — that text is decorative during SFT, not what you want. Use the structured head output.

5. **DeepSpeed shards are optional for inference.** The `global_step*/` directory is only needed to resume training. Save the bandwidth.

6. **`spk_dict.pt` is mandatory.** This is the speaker dictionary for Qwen-Omni; the loader expects it. If missing, copy from the base model.

## Compute reference

| Operation | 1× L40S (48 GB) | 1× A100 (40 GB) | 1× RTX 4090 (24 GB) |
|---|---|---|---|
| Load model | ~30 s | ~25 s | ~30 s |
| Single 15s ad inference | ~3 s | ~2 s | ~4 s |
| Batch 4 ads | ~5 s | ~4 s | OOM possible at long ads |
| 5000-ad eval (sequential, bs=1) | ~3-4 hr | ~3 hr | ~4 hr |

Bottleneck is the visual encoder over video frames, not the retention head.

## Files referenced

- Plugin: `examples/custom/qwen2_5_omni_retention/register.py`
- Reference eval harness: `examples/custom/qwen2_5_omni_retention/eval_ibs.py`
- Training-side README (deeper architecture notes): `examples/custom/qwen2_5_omni_retention/README.md`

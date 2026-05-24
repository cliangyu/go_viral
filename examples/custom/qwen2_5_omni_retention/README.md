# qwen2_5_omni_retention — retention-curve head plugin for ms-swift

A custom model registration that wraps `Qwen2.5-Omni-3B` (or `7B`) with a
small per-second retention-curve head. Two head architectures are
selectable at runtime; both produce R(t) of length 60 for the TTCC task.

## Variants

| `head_type` | Transform | Loss | Prior | Milestone ref |
|---|---|---|---|---|
| `hazard` | `softplus(Linear(h)) → λ(t); R = exp(-cumsum(λ))` | log-hazard MSE | monotone non-increasing by construction | §6 SFT-Hazard+CoT |
| `sigmoid` | `sigmoid(Linear(h)) per second` | masked MSE | bounded `[0, 1]` only | §5 SFT-MSE |

`hazard` is the survival-analysis formulation. `λ(t) ≥ 0` for all `t` so
`R(t)` cannot increase over time — a true structural property of retention
curves. References: DeepHit (Lee et al., AAAI 2018), SurvTRACE (Wang & Sun,
CHIL 2022), `pycox`, `lifelines`.

`sigmoid` is the simplest bounded baseline: each second is independent,
so the model can predict `R(5)=0.5, R(6)=0.7` (illegal but unpunished by
the loss). Inference-time post-processing is needed to clamp monotonicity.

## Anchor position

The head reads a single hidden state per row. The anchor is the last
`</cot>` token if present (with-CoT variants), else the last input token
(without-CoT variants). No code change between the two — `_locate_anchor_positions`
handles both cases.

## Loading

```bash
swift sft <variant>.yaml \
  --external_plugins examples/custom/qwen2_5_omni_retention/register.py \
  --model_type qwen2_5_omni_retention \
  --loss_type retention_loss
```

The `--external_plugins` flag imports `register.py`; the registrations
fire as import side-effects and `qwen2_5_omni_retention` becomes a valid
`--model_type` value.

## Configuration

| Env var / flag | Default | Meaning |
|---|---|---|
| `RETENTION_HEAD_TYPE` | `hazard` | Head architecture: `hazard` or `sigmoid`. |
| `RETENTION_COT_ALPHA` | `0.0` | Weight on the LM CoT cross-entropy term (only when labels are present, i.e. with-CoT variants). Try `0.05`, `0.1`, `0.2`. |
| `ENABLE_AUDIO_OUTPUT` | `false` | Whether to load Talker. Default off (saves ~833 M params). |

## Data contract

The training/eval JSONL must carry an `R` field per row: a list of floats
of length `T_i + 1`, with `R[0] == 1.0` by convention. Rows without `R`
(e.g. inference) skip the retention target — the template makes the
`r_true` / `r_mask` tensors optional.

## Tuning

`--tuner_type {full, lora}` works unchanged. The retention head itself is
a single `nn.Linear` in fp32; it participates in the optimizer regardless
of tuner type, because PEFT's `target_modules=all-linear` finds it.

For full FT on long-sequence Qwen-Omni: pair with `--deepspeed zero3
--vit_gradient_checkpointing true --gradient_checkpointing true`.

For LoRA: pair with `--deepspeed zero2 --lora_rank 16 --lora_alpha 32`.

## Architecture provenance

This plugin is the ms-swift-native port of the hazard-head architecture
originally prototyped against raw HF Trainer in
`wanjia/main:cs224r_project/baselines/retention_vlm.py`. That parallel
codebase has been retired; this plugin is the single source of truth.

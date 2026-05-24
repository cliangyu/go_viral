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


## Loss function

Both heads (hazard, sigmoid) train with **plain masked MSE on R(t)**:

```
L = mean over (t, ad) of (R_pred(t) - R_true(t))² * duration_mask
```

This matches the eval metric (IBS) exactly. The monotonicity prior (for
hazard) lives in the architecture (`softplus → cumsum → exp`), not the
loss — switching the loss does not weaken the prior.

We considered the survival-analysis canonical choice, **log-hazard MSE**
(wanjia milestone §3):

```
L_LH = mean over (t, ad) of (log λ_pred(t) - log λ_true(t))² * mask
```

but rejected it because:

1. **Flat-tail explosion**: retention curves are typically flat in their
   last ~30s (loyal viewers stay). At those positions `λ_true ≈ 0`,
   clamped to `ε = 1e-6`, giving `log(ε) ≈ -13.8`. A reasonable model
   prediction `log(λ_pred) ≈ -3` produces `(-3 - (-13.8))² ≈ 117` per
   position. This noise dominates the loss and inflates `grad_norm`
   by ~10× (observed 25K vs 2.4K under plain MSE in V6 vs V7 8-GPU
   prod runs).
2. **Not the survival-lit standard anyway**: DeepHit, MTLR, Nnet-survival,
   PC-Hazard, SurvTRACE all use Bernoulli/PMF NLL on hazards, not
   log-hazard MSE. The "spec" loss was idiosyncratic.
3. **Train-on-eval-metric is correct** when the metric is strictly proper
   (Brier 1950) and there's no censoring (our case). Sonabend et al.
   ECML-PKDD 2024 confirmed empirically that matching loss to metric is
   competitive-to-better than survival-style proxies.

`RETENTION_COT_ALPHA > 0` adds an LM cross-entropy term on the assistant
turn for with-CoT variants. Default 0 (no auxiliary loss).

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

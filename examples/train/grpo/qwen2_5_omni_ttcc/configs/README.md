# Variant configs

One YAML per training variant. Pattern follows LLaMA-Factory and axolotl:
self-contained YAML, no inheritance, CLI overrides for one-off changes.

## Naming convention

`<stage>_<emission>[_<head>]_<tuner>_<cot>.yaml`

- `stage` — `sft`, `dpo`, `grpo`, `rloo`, `kto`, etc.
- `emission` — `lm` (the model writes the curve as text) or `retention`
  (custom head produces R(t)).
- `head` — `hazard`, `sigmoid`. Present only for `emission=retention`.
- `tuner` — `full` or `lora`. (qlora etc. can be added later.)
- `cot` — `with_cot` or `no_cot`. Indicates whether the dataset row has
  `<cot>...</cot>` reasoning between prompt and answer.

Examples:

- `sft_lm_full_no_cot.yaml` — SFT, language head, full FT, no CoT.
- `sft_retention_hazard_lora_with_cot.yaml` — SFT, hazard head, LoRA, CoT.
- `dpo_lm_full_no_cot.yaml` — DPO on top of an SFT checkpoint.

## Round-1 variants

| File | Variant | Question it answers |
|---|---|---|
| `sft_lm_full_no_cot.yaml` | V1 | baseline (currently running) |
| `sft_lm_full_with_cot.yaml` | V2 | does CoT help the LM head? |
| `sft_retention_hazard_lora_with_cot.yaml` | V3 | hazard head + CoT (wanjia §6) |
| `sft_retention_sigmoid_lora_no_cot.yaml` | V4 | sigmoid head (wanjia §5) |
| `dpo_lm_full_no_cot.yaml` | V5 | does DPO add to V1? |

## Running a variant

```bash
bash sft.sh configs/sft_lm_full_no_cot.yaml
bash sft.sh configs/sft_retention_hazard_lora_with_cot.yaml
bash dpo.sh configs/dpo_lm_full_no_cot.yaml
```

Override one knob without forking the YAML:

```bash
bash sft.sh configs/sft_lm_full_no_cot.yaml --learning_rate 5e-6 --num_train_epochs 5
```

`sft.sh` passes everything after the YAML path through to `swift sft`.

## Discipline

1. **A variable lives in the LOWEST layer where it is shared.** Since we
   use one-file-per-variant (no family inheritance), most knobs duplicate
   across variants. That is acceptable: it keeps each file self-readable.
   Reference: LLaMA-Factory `examples/train_full/qwen3_full_sft.yaml`,
   axolotl `examples/llama-3/{lora-1b.yml, fft-8b.yaml}`.
2. **One-off overrides go on the CLI**, not into a new YAML.
3. **Run output dirs are `<variant>_<date>_<short_hash>`**, never `v1` `v2`
   `v3`. The wrapper resolves `<date>` and `<short_hash>` automatically.
4. **Variant names are semantic** — read top-to-bottom, you know what it
   is. Anti-patterns to avoid: `v2cot`, `extended`, `nocot_v2cot_full`,
   numeric versions in filenames.


## Loss

All retention variants (hazard, sigmoid) train with plain masked MSE on R(t). See `examples/custom/qwen2_5_omni_retention/README.md` for the rationale.

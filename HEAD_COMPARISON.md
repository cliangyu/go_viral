# Retention-Curve Head Comparison

Two parallel training paths live in this repo for head-to-head A/B
evaluation of retention-curve prediction on TTCC.

## TL;DR

| | **Path A: Language head** | **Path B: Hazard head** |
|---|---|---|
| Output | Curve as a string of tokens | 60-dim hazard vector -> curve |
| Loss | Token cross-entropy | masked log-hazard MSE (+ alpha * CoT CE) |
| Tuning | Full fine-tune | LoRA |
| Framework | ms-swift | HF Trainer (direct) |
| Compute | AWS p5.48xlarge (8x H100) | Modal (H100 on-demand) |
| Owner | Liangyu | Wanjia |
| Launcher | examples/train/grpo/qwen2_5_omni_ttcc/sft_v2cot_full.sh | cs224r_project/scripts/run_sft_hazard_cot_seed.sh |
| Code root | examples/train/grpo/qwen2_5_omni_ttcc/ | cs224r_project/ |

Both paths share: base model (Qwen2.5-Omni-3B Thinker), dataset
(liangyuch/ttcc-v0_2_0), and final eval (ttcc-eval).

## A. Language-head path (Liangyu)

**Architecture.** Vanilla Qwen2.5-Omni-3B language head; retention curve is
emitted as a numeric sequence in the assistant turn (e.g. "0.92 0.85 ...").
No new modules.

**Training.** Full fine-tune via ms-swift (--tuner_type full). DeepSpeed
ZeRO-3, flash-attn 2.8.3, vit_gradient_checkpointing=true,
torch_compile=true. 8x H100 80GB SXM, effective batch 64, lr 1e-5, 10
epochs over 39,468 train rows.

**Why this design.** Lets the GRPO-RL stage that follows operate on the
same token interface as the eval prompt: no head surgery between SFT and
RL. Curve numerics are part of the language distribution.

**Reproduction.** See examples/train/grpo/qwen2_5_omni_ttcc/README_8card.md.

## B. Hazard-head path (Wanjia)

**Architecture.** RetentionVLM = Qwen2.5-Omni-3B Thinker trunk + softplus
hazard head reading the hidden state at the last `</cot>` token. Output
is a 60-dim hazard vector; the retention curve is recovered as
R(t) = exp(- sum_{s<=t} lambda(s)). Hazard head is fp32 (cumsum/exp
precision).

**Training.** LoRA on Thinker; HF Trainer directly. Joint loss
L = L_hazard + alpha * L_cot where L_hazard is masked log-hazard MSE and
L_cot is next-token CE on the distilled CoT span only. Sweep
alpha in {0.05, 0.1, 0.2}. Modal H100, 1 epoch per seed in {42, 43, 44}.

**Why this design.** Survival-analysis prior: hazards are non-negative,
retention is monotone non-increasing by construction (exp of cumsum of a
non-negative quantity). Matches the structure of the TTCC eval metric.

**Reproduction.** See cs224r_project/README.md.

## A/B methodology

Both paths produce a test_preds.parquet consumable by ttcc-eval
(https://github.com/cliangyu/ttcc-eval). Comparison is on:

1. Overall MAE / RMSE on the held-out test split.
2. Conditional metrics by T_i quartile (Q1 short, Q4 long) - captures
   whether either head specializes to a duration regime.
3. Inflection-point alignment - where the predicted curve's
   second-derivative-zero crosses vs ground truth.
4. RMST - restricted mean survival time, the integrated curve.

Identical eval pipeline runs against both paths. The winner of the SFT
A/B determines the initialization for the GRPO-RL stage.

## Repo layout

```
go_viral/
  examples/train/grpo/qwen2_5_omni_ttcc/  # PATH A - language head
    sft_v2cot_full.sh                     #   training launcher
    prepare_dataset.py                    #   data prep (--no-cot, --validate)
    Dockerfile + host-setup.sh            #   p5.48xlarge bring-up
    watch_and_upload_ckpts.py             #   HF ckpt sync
    README_8card.md                       #   AWS runbook

  cs224r_project/                         # PATH B - hazard head
    baselines/
      retention_vlm.py                    #   trunk + softplus head module
      sft_hazard_cot.py                   #   hazard + CoT joint training
      sft_mse.py                          #   MSE baseline (per-second sigmoid)
    modal/                                #   Modal launchers
    eval/                                 #   per-method test_preds.parquet
    scripts/                              #   shell wrappers per-seed

  swift/                                  # ms-swift core - shared, unchanged
```

The two paths are independent at the file level. Changing one does not
affect the other.

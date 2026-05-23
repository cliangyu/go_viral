# CS224R — SFT baselines for retention-curve prediction

Owner: Wanjia. Companion to the milestone report + experiment plan (Qwen2.5-Omni-3B + ms-swift).

## Layout
```
cs224r_project/
├── data/
│   ├── prep_ttcc.py             # HF dataset → JSONL + audio extraction + split
│   ├── raw/                     # HF cache for liangyuch/ttcc-v0_1_0
│   ├── videos/                  # symlinks to mp4 files, named {ad_id}.mp4
│   ├── audios/                  # ffmpeg-extracted 16kHz mono wav
│   └── splits/                  # train.jsonl / val.jsonl / test.jsonl + manifest
├── baselines/
│   ├── sft_mse.py               # §5: per-second sigmoid head, LoRA, MSE
│   ├── retention_vlm.py         # §2: trunk + softplus hazard head wrapper
│   └── sft_hazard_cot.py        # §6: joint hazard MSE + CoT cross-entropy
├── eval/
│   └── make_test_preds.py       # §9 test_preds.parquet (SFT-MSE for now)
├── scripts/
│   ├── run_sft_mse_seed.sh           # Stage 1 wrapper (single seed)
│   ├── run_sft_mse_all_seeds.sh      # Stage 1 wrapper (3 seeds)
│   ├── run_sft_hazard_cot_seed.sh    # Stage 2 wrapper
│   └── run_sft_hazard_cot_all_seeds.sh
└── modal/
    ├── _common.py                    # shared Image + Volume + Mount
    ├── modal_prep.py                 # CPU container: data prep
    ├── modal_train_sft_mse.py        # H100 container: Stage 1
    ├── modal_train_sft_hazard_cot.py # H100 container: Stage 2
    └── modal_eval.py                 # H100 container: test_preds.parquet
└── runs/                        # checkpoints + test_preds.parquet per (method, seed)
```

## Stage 1: SFT-MSE on the 100-ad smoke set

### One-time data prep
```bash
python cs224r_project/data/prep_ttcc.py \
    --hf_dataset liangyuch/ttcc-v0_1_0 \
    --out_dir cs224r_project/data \
    --max_duration 30
```
Expected output: `data/splits/{train,val,test}.jsonl` and `split_manifest.json`. On the
100-ad smoke set this gives roughly 80/10/10 after the `T_i ∈ [5,30]` filter.

### Training (single seed)
```bash
bash cs224r_project/scripts/run_sft_mse_seed.sh 42
```
This runs LoRA-tuned Qwen2.5-Omni-3B for 1 epoch with the per-second sigmoid head and
then writes `runs/sft_mse/seed42/test_preds.parquet`.

To do the full sweep:
```bash
for s in 42 43 44; do bash cs224r_project/scripts/run_sft_mse_seed.sh $s; done
```

## Stage 2: SFT-Hazard+CoT (the RL initialization)

Waits on Liangyu's CoT distillation. Once it lands as a JSONL with one
`{ad_id, cot}` per line (or a JSON dict `{ad_id: cot}`):

```bash
python cs224r_project/data/merge_cot.py \
    --cot_manifest cs224r_project/data/cot/ttcc_train_cot.jsonl \
    --splits_dir   cs224r_project/data/splits
# produces splits/{train,val,test}_with_cot.jsonl
# (val/test get empty CoT — distillation runs train-only per the milestone)

bash cs224r_project/scripts/run_sft_hazard_cot_seed.sh 42 0.1
```

`alpha=0.1` is the joint-loss weight on CoT cross-entropy (§6); candidate
sweep `{0.05, 0.1, 0.2}` if CoT under-trains on val.

Output: LoRA adapter + `hazard_head.pt` per seed, ready for the GRPO stage.

## Running on Modal (H100)

All experiments target Modal per CS224R's compute guide. Setup once:

```bash
pip install modal
modal setup                  # browser auth with Stanford email
modal secret create wandb WANDB_API_KEY=<your_token>
# Optional — only needed for gated HF models:
# modal secret create huggingface HF_TOKEN=<your_token>
```

### ttcc-eval bundling

The private ttcc-eval repo is **bundled** into `third_party/ttcc-eval/`
(no Modal secret needed; we copied the source in directly). To bump:

```bash
cd /tmp && git clone https://github.com/cliangyu/ttcc-eval.git
rm -rf /Users/zwj/ms-swift/cs224r_project/third_party/ttcc-eval
cp -r /tmp/ttcc-eval /Users/zwj/ms-swift/cs224r_project/third_party/ttcc-eval
rm -rf /Users/zwj/ms-swift/cs224r_project/third_party/ttcc-eval/.git
```

The Modal image build does `pip install /root/cs224r_project/third_party/ttcc-eval`.

Volume `cs224r-ttcc-retention` is auto-created on first use and persists:
- `/vol/hf_cache/`   — HuggingFace dataset + model cache
- `/vol/data/`       — extracted audio, mp4 symlinks, JSONL splits
- `/vol/runs/{method}/seed{N}/` — checkpoints and test_preds.parquet

### Stage 0: data prep (CPU container)

`prep_ttcc.py` delegates ground-truth preprocessing to ttcc-eval (peak-normalize
0–100 → 0–1, monotone smoothing, T_i computation, drop rules). Splits come from
the dataset's `split` column (train/val/test), **not** a custom shuffle. The
training-set filter `--train-max-duration` controls only which long ads are
dropped from training; val/test keep full clean T_i ∈ [5, 60] so our submissions
align with eval ground truth row-for-row.

```bash
modal run cs224r_project/modal/modal_prep.py                          # train T_i <= 60
modal run cs224r_project/modal/modal_prep.py --train-max-duration 30  # strict audio cap
```

### Stage 1: SFT-MSE training (H100)

Smoke test first — loads one batch on the real model and exits:
```bash
modal run cs224r_project/modal/modal_train_sft_mse.py --seed 42 --smoke-only
```

Real run (use `--detach` so your terminal can close):
```bash
modal run --detach cs224r_project/modal/modal_train_sft_mse.py --seed 42
# or all 3 seeds:
modal run --detach cs224r_project/modal/modal_train_sft_mse.py --all-seeds
```

### Stage 1 inference + ttcc-eval evaluation

`modal_eval.py` runs three steps per checkpoint:
1. `make_test_preds.py` writes diagnostic parquet (includes R_true).
2. `write_submission.py` writes a contract-compliant parquet (ad_id, R_hat, method, seed).
3. `ttcc-eval evaluate` writes `/vol/reports/{method}_seed{N}.json` with
   ρ_hook / ρ_comp / ρ̄_shape and 95% BCa bootstrap CIs.

```bash
# Single checkpoint:
modal run cs224r_project/modal/modal_eval.py \
    --checkpoint /vol/runs/sft_mse/seed42 \
    --method sft_mse --seed 42

# Sweep all three SFT-MSE seeds in parallel:
modal run cs224r_project/modal/modal_eval.py --all-seeds-sft-mse
```

Reports land on the volume at `/vol/reports/`. Pull them locally with:

```bash
modal volume get cs224r-ttcc-retention /reports/ ./reports/
```

### Stage 2: SFT-Hazard+CoT (after CoT distillation lands)

```bash
# After merge_cot.py has populated splits/train_with_cot.jsonl on the volume:
modal run --detach cs224r_project/modal/modal_train_sft_hazard_cot.py \
    --seed 42 --alpha 0.1
```

### Debugging on Modal

```bash
# Open a shell in the same image as the training function:
modal shell cs224r_project/modal/modal_train_sft_mse.py::train

# Watch a detached run:
modal app logs cs224r-sft-mse

# Live profile a stuck container: visit modal.com dashboard → Containers tab.
```

To run *locally* (no Modal) the scripts under `scripts/` still work — the
Modal wrappers just shell out to the same `python ...` commands.

## Status
- [x] Data prep script
- [x] SFT-MSE trainer + LoRA wrapper
- [x] Test inference → parquet
- [ ] Smoke run on 100 ads via `modal run … --smoke-only` (next: actually execute)
- [x] RetentionVLM wrapper (§2)
- [x] SFT-Hazard+CoT trainer (Stage 2; **waits on CoT distillation** to actually run)
- [x] Modal entrypoints for prep / train / eval with WANDB hook
- [x] ttcc-eval integration: dataset split column, peak-norm via preprocess, submission contract, automatic evaluate call
- [ ] make_test_preds for SFT-Hazard+CoT (Stage 2 inference)
- [ ] Multi-seed R_hat averaging in write_submission (§9 — 3 rollouts per ad for CoT methods)
- [ ] Paired BCa via `ttcc-eval compare` (SFT-MSE vs SFT-Hazard+CoT vs GRPO)

## Open issues / known caveats
- **100-ad smoke set only**: numbers will not be meaningful; this stage validates the
  pipeline. Real baselines wait on a larger TTCC dump.
- **CoT field is empty** in JSONL (`"<cot></cot>"`). SFT-MSE doesn't use it. SFT-Hazard+CoT
  will need Liangyu's distillation to fill in `_meta.cot`.
- **Audio-cap policy**: filtering to `T_i ≤ 30` per the experiment plan §3 recommended
  policy. Toggle with `--max_duration 60` if you want to keep the longer ads.
- **ms-swift integration is deferred** for SFT-MSE. This stage uses plain
  `transformers.Trainer` because the per-second sigmoid head is custom. ms-swift is
  reserved for SFT-Hazard+CoT (Stage 2) where the omni template carries audio/video
  tokenization.

## Tested on
- Single A100 / H100 80 GB, bf16, LoRA rank 8.
- `ffmpeg` available on PATH for `prep_ttcc.py`.
- `pip install datasets transformers peft scipy pandas pyarrow tqdm`
- `pip install 'ttcc-eval @ git+https://github.com/cliangyu/ttcc-eval.git@main'`
  (private repo — needs git+SSH or a PAT in the URL).

# 8-card SFT Work Log — Session 2026-05-23

This file is a hand-off note so a context-compressed agent (or next-day me) can resume
the work without losing state. Authoritative location on training box:
`/opt/dlami/nvme/go_viral/examples/train/grpo/qwen2_5_omni_ttcc/8CARD_WORK_LOG.md`

## TL;DR — where things stand at handoff

| Item | State |
|---|---|
| SFT training | **RUNNING** on i-00712a40af67ab9c5 (us-east-2), step ~250/6190, loss ~0.47 |
| W&B run | https://wandb.ai/liangyuch/ttcc/runs/m89eiy4r |
| Ckpt watcher | **RUNNING** (PID 1009087), uploading to `liangyuch/ttcc-sft-qwen25omni-3b` |
| My commits | Pushed to **`cliangyu/go_viral:8card-runbook`** (6 commits, see below) |
| Conflict | PR-ready against `ttcc-rl`; conflict on `sft_v2cot_full.sh` (Leon's `664f297e bs=2→1` overlaps) |
| CB window | Ends **2026-05-26 11:30 UTC** (~65 hr from session end) |

## What I committed (branch: `cliangyu/go_viral:8card-runbook`)

```
0efd6132 fix(watcher): use abs path to hf CLI (subprocess PATH does not have venv bin)
11e98299 infra(8card): Dockerfile + host-setup.sh + requirements.lock + ckpt watcher
4708e87c docs(8card): runbook for p5.48xlarge — driver/FM/ninja/flash-attn/path fixes
fd14da63 config(launchers): production H2 config + scale to 8 GPUs
6d5c58e1 config(deepspeed): enable overlap_comm in zero3.json
8cd7a7f4 refactor(prep): unify prepare_dataset.py with --no-cot + --validate flags
─── origin/ttcc-rl @ 664f297e (Leon's latest, NOT yet merged) ─────────────────
```

PR URL: https://github.com/cliangyu/go_viral/pull/new/8card-runbook

## Key files touched

All under `examples/train/grpo/qwen2_5_omni_ttcc/`:

| File | Status | Purpose |
|---|---|---|
| `prepare_dataset.py` | rewritten | Unified: --no-cot flag, --validate (ffprobe), prep_stats.json |
| `prepare_dataset_nocot.py` | **deleted** | Was duplicate of prepare_dataset.py |
| `sft_v2cot_full.sh` | modified | 8-GPU, H2 config flags, flash_attn, vit_grad_checkpointing |
| `infer_v2cot_full.sh` | modified | CUDA_VISIBLE_DEVICES=0..7 |
| `Dockerfile` | new | nvidia/cuda:13.0 + ffmpeg + ninja + pip stack + flash-attn 2.8.3 sm_90 |
| `host-setup.sh` | new | NVIDIA driver+FM+modules install (host-level, can't Dockerize) |
| `requirements.lock` | new | pip freeze of swift_venv (176 packages) |
| `watch_and_upload_ckpts.py` | new | watches output_dir, uploads ckpts to HF |
| `README_8card.md` | new | Runbook for p5.48xlarge with all painful setup |

`swift/config/zero3.json` — modified to set `overlap_comm: true` (was false; DS official recommends true).

## Production SFT config (currently running)

In `sft_v2cot_full.sh` after my edits:

```
NPROC_PER_NODE=8  CUDA_VISIBLE_DEVICES=0..7
OMP_NUM_THREADS=6
MAX_PIXELS=200704  VIDEO_MAX_PIXELS=200704  FPS_MAX_FRAMES=60  VIDEO_MAX_TOKEN_NUM=16384
ENABLE_AUDIO_OUTPUT=False
FORCE_QWENVL_VIDEO_READER=decord  (env)
CUDA_HOME=/usr/local/cuda-13.0  (env)

--tuner_type full
--freeze_vit true  --freeze_aligner true
--torch_dtype bfloat16
--attn_impl flash_attn
--dataset /home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_train_sft.jsonl
--max_length 24576
--truncation_strategy delete
--lazy_tokenize true
--strict false
--dataset_num_proc 1
--group_by_length false
--num_train_epochs 10
--per_device_train_batch_size 1
--gradient_accumulation_steps 8   (→ effective batch 64)
--gradient_checkpointing true
--vit_gradient_checkpointing true   ← critical for 80GB H100 memory
--torch_compile true
--learning_rate 1e-5
--warmup_ratio 0.05
--logging_steps 5
--save_steps 50  --save_total_limit 3
--output_dir /opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full
--deepspeed zero3
--dataloader_num_workers 4
--dataloader_persistent_workers true
--dataloader_prefetch_factor 4
--report_to tensorboard wandb
--resume_from_checkpoint /opt/dlami/nvme/ssm-out/ttcc_sft_v2cot_nocot_full/v19-20260523-181720/checkpoint-200
```

## Data state

| Dataset | Path / repo | Rows |
|---|---|---|
| Raw | hf://datasets/liangyuch/ttcc-v0_2_0 | 61,789 |
| Filtered train SFT | `/home/ssm-user/work/data/ttcc_swift_v2cot_nocot/ttcc_train_sft.jsonl` | 39,468 (was 39,617, dropped 148 no-audio + 1 corrupt) |
| Filter logic | T∈[5,60] strict reject + audio stream required | |
| Prep backup files | `.bak1` (39,617 ads), `.bak2` (39,616 ads) in same dir | |
| No-audio list | `/opt/dlami/nvme/logs/no_audio_mp4s.txt` (148 paths) | |

## Pending work (in priority order)

1. **MERGE wanjia's fork** ← Leon's new ask (see "Active task" below)
2. **Resolve PR conflict** with `ttcc-rl` HEAD (`664f297e bs=2→1`)
3. **`run_meta.json` emit** at training start (designed in launcher patch, didn't apply due to SSM EOF)
4. **Test Dockerfile** — never actually `docker build`-ed; might have bugs
5. **prep_manifest.jsonl upload** to a separate HF dataset repo `liangyuch/ttcc-prep-v0_2_0`
6. **Atomic writes** in prepare_dataset.py (write-then-rename pattern)
7. **Pre-flight checks** at training start (git clean, wandb auth, lockfile drift)

## Active task — merge wanjia's fork

**Wanjia's fork:** https://github.com/wanjiaZhao1203/ms-swift.git
**Leon's fork:**   https://github.com/cliangyu/go_viral.git

Wanjia added a NEW HEAD ("horizontal head" / "horse head" — possibly a numeric retention-curve head separate from the language head). Leon's branch trains purely language-based (predicting the curve as a string of tokens).

Steps to do:
1. Add wanjia's fork as a remote on the training box
2. Fetch all branches; identify which contains the head addition
3. Diff vs my `8card-runbook`: figure out scope of changes (new model classes, new launcher script, new loss?)
4. Plan merge that preserves BOTH paths:
   - **Language head** (current path): existing `sft_v2cot_full.sh` flow
   - **Numeric head** (wanjia's): probably needs separate launcher `sft_v2cot_full_numhead.sh` + maybe new tuner_type or model class registration
5. Config differentiation: use either separate launchers OR a `--head-type {language,numeric}` flag in a unified launcher. Recommend separate launchers (clearer head-to-head comparison artifacts).
6. Commit per concern; push to my branch.

**Critical:** Keep both paths runnable. Goal is head-to-head A/B, NOT replace.

## Useful commands for handoff

```bash
# SSH-via-SSM into training box:
TARGET=i-00712a40af67ab9c5 REGION=us-east-2 AWS_PROFILE=gpu-box \
  bash /Users/marvl/Documents/stanford/cs224r/projects/ttcc-inference/scripts/aws/ssm_run.sh '<cmd>'

# Check SFT alive:
ps -ef | grep sft.py | grep -v grep | wc -l   # expect 41

# Latest loss:
grep -E "loss.*epoch.*global_step" /opt/dlami/nvme/logs/sft.log | tail -3

# Watcher state:
ps -ef | grep watch_and_upload | grep -v grep
tail /opt/dlami/nvme/logs/ckpt_upload.log

# Branch state:
cd /opt/dlami/nvme/go_viral && git log --oneline -10 && git status --short
```

## Lessons learned this session (for SE practice)

1. **Don't reach for tools without searching first** — wasted 30 min on soundfile (wrong tool for mp4) before searching. Per Leon: "search first, conclude second."
2. **File versioning (`_v2`, `_nocot`) is anti-pattern** — use single source + flags + git.
3. **`audios: []` is canonical Qwen2.5-Omni pattern** for video-only samples (per HF docs).
4. **Every artifact needs a versioned + transferable home**: code→git, data→HF, ckpts→HF model repo, metrics→wandb, env→Dockerfile+lockfile.
5. **Ephemeral NVMe**: anything only on the box dies with the box. Push everything externally.
6. **Atomic commits**: one commit per concern with `feat:/fix:/refactor:/config:/docs:` prefix + rationale.
7. **`vit_gradient_checkpointing=true` is the magic flag** for fitting 24576-seq SFT on 80GB H100 (saves ~45 GB).
8. **`FLASH_ATTN_CUDA_ARCHS=90`** is the flash-attn 2.8.3 build env var (not `TORCH_CUDA_ARCH_LIST`, which is ignored).
9. **DLAMI + apt drift**: `linux-modules-nvidia-595-open` gets removed on reboot via unattended-upgrades; run host-setup.sh after every reboot.

## CB time accounting

```
CB end:        2026-05-26 11:30 UTC
Session end:   ~2026-05-23 22:00 UTC (estimate)
Remaining:     ~65 hours
SFT step time: ~57 s/step (steady)
Total steps:   6190 (10 epochs × 619 steps/epoch)
ETA at 57 s/step: 97 hours  ← EXCEEDS CB window
```

→ Will only finish ~7 epochs in CB window. Either accept partial, or reduce epochs.
Leon previously preferred letting it run and use whatever epochs land.

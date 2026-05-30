# TTCC Scripts Map + V8 Provenance + Canonical Eval Protocol

**Why this doc exists:** we burned a lot of time confusing two different SFT lineages
(v18 single-node vs V8 2-node) and mis-reading one script's config as V8's. This is the
ground truth. Verified-by-experiment (Tier-1) and verified-by-doc are marked.

---

## 1. The single most important fact: two SFT lineages, do not confuse them

| | **v18 lineage** (earlier, superseded) | **V8 lineage** (ckpt-225, what we eval/RL on) |
|---|---|---|
| launcher | `sft_v2cot_full.sh` (standalone, **8-card single-node**) | `launch_training_2node.sh` (**2-node EFA**, ran 28h) |
| sources `_common.sh`? | **NO** (standalone, full-FT-specific defaults) | **YES** |
| **FPS** | **2.0** (unset → swift default) | **1.0** (`_common.sh ${FPS:=1.0}`) ✅ Tier-1 experiment |
| dataset | `ttcc_swift_v2cot/ttcc_train_sft.jsonl` | `ttcc_v8/ttcc_train_with_cot.jsonl` (args.json) |
| date | 2026-05-23 (T1-T6 sweep) | 2026-05-26+ → run dir `v19-20260528`, `checkpoint-225` |
| HF repo | `liangyuch/ttcc-sft-qwen25omni-3b` (STALE, 3.5 TB) | `liangyuch/ttcc-sft-qwen25omni-3b-v8` (9.4 GB, clean) |
| config yaml | (LoRA/full sweep variants) | `configs/sft_retention_hazard_full_with_cot.yaml` |

**The trap:** `sft_v2cot_full.sh` *looks* like the V8 SFT but is the v18 single-node run.
Because it is standalone (doesn't source `_common.sh`) its FPS defaults to 2.0, max_pixels
to 200704, vmt to 16384 — all different from V8. Reading it led to a wrong eval config.

**V8 (ckpt-225) provenance chain (verified):**
`launch_training_2node.sh` → sources `_common.sh` (FPS=1.0, audio-handling) → `swift sft`
with `configs/sft_retention_hazard_full_with_cot.yaml` on `ttcc_v8/ttcc_train_with_cot.jsonl`
→ run dir `.../sft_retention_hazard_full_with_cot/v19-20260528-180726/checkpoint-225`.
`launch_training_2node.sh` is NOT in this checkout (it lived on the now-torn-down
p5.48xlarge cluster); its config is documented in `launch_rl_2node.sh:24-35`.

---

## 2. Script map (one line each)

### Shared / launchers
- `_common.sh` — **shared env**, sourced by SFT/RL/DPO/RLOO launchers. Sets the V8 media
  constants: `FPS=1.0`, `FPS_MAX_FRAMES=60`, `VIDEO_MAX_TOKEN_NUM=8192`,
  `MAX_PIXELS=VIDEO_MAX_PIXELS=49152`, `USE_AUDIO_IN_VIDEO` default. **(`sft_v2cot_full.sh`
  is the one launcher that does NOT source it.)**
- `_chain_lib.sh` — helpers for the multi-stage `chain_*` orchestrators.
- `sft.sh` — SFT dispatcher (runs any `configs/sft_*.yaml`).
- `grpo.sh` / `rloo.sh` / `dpo.sh` — GRPO / RLOO / DPO dispatchers.
- `rl.sh` — head-PG RL launcher (mirrors `sft.sh` but runs `rl/train_head_pg.py`, NOT `swift sft`).
- `launch_rl_2node.sh` — 2-node EFA launcher for head-PG RL (mirrors `launch_training_2node.sh`).
- `infer.sh` / `infer_v2cot_full.sh` — inference / test-split eval launchers.
- `host-setup.sh` — p5.48xlarge host prerequisites.

### The v2-full-FT "chain" round (the 4-experiment sweep, v18 era)
- `chain_v2cot_full.sh` (+ `_resume`, `_resume2`) — orchestrator for the 4-experiment round.
- `sft_v2cot_full.sh` — **Exp 1: full-FT SFT WITH CoT (v18 single-node, FPS=2.0). NOT V8.**
- `sft_nocot.sh` / `sft_nocot_v2cot_full.sh` — Exp 2: SFT without CoT supervision.
- `sft_extended.sh` — Exp 3: SFT to saturation (3 epochs).
- `grpo_extended.sh` — Exp 4: GRPO to saturation.
- `grpo_v2cot_full.sh` / `rloo_v2cot_full.sh` — GRPO/RLOO continuing full-FT from an SFT ckpt.

### Head-PG RL (the current work)
- `rl/train_head_pg.py` — RL entry point (head-PG REINFORCE). **RL, not SFT.**
- `rl/head_pg_trainer.py` — the RL trainer: reward / advantage / KL. Reward currently
  `crossad` concordance vs a per-GPU buffer (the red-team flagged this as mis-specified).
- `rl/build_bypass_dataset.py` — builds the RL bypass train set (assistant span → `<cot></cot>`).
- `rl/build_train_cdf.py` — per-second train-population CDF for the cross-ad reward.
- `rl/srcc_eval.py` — **THE eval / ruler** (cross-ad SRCC, leak-free bypass). See §3.
- `rl/paired_bootstrap.py` — paired-bootstrap CI for SRCC(candidate) − SRCC(baseline).

### Plugin + data tools (`examples/custom/qwen2_5_omni_retention/`)
- `register.py` — the retention-head plugin: hazard head `R=exp(-cumsum(softplus(z)))`,
  reads the hidden at the last `</cot>` token (else last token).
- `eval_ibs.py` — IBS (integrated Brier score) eval.
- `make_v8_from_hf.py` — rebuild the leak-free V8 train JSONL from the 2 PUBLIC HF repos
  (the Wanjia handoff). **CAVEAT: currently emits the SHORT prompt + `audios=[]`; must be
  re-checked against V8's actual long-prompt + audio-off training format.**
- `build_v8_from_hf.py` / `build_v8_train_jsonl.py` / `extract_videos_from_hf.py` — earlier
  data-build variants (superseded by `make_v8_from_hf.py`).
- `prepare_dataset.py` — ffprobe-based data prep; sets `audios:[mp4]` only if an audio
  stream exists (now moot — V8 is audio-off).
- `watch_and_upload_ckpts.py` — watches an output dir and uploads checkpoints to HF.

---

## 3. CANONICAL EVAL PROTOCOL (the ruler) — verified to reproduce 0.5142

`rl/srcc_eval.py` on `ckpt-225` MUST reproduce the baseline. **Verified by experiment:**

```
FPS=1.0, audios=[], native 720p, max_length=32768, short prompt, bypass cot
  → ckpt-225 / val_present  =  SRCC 0.5142,  n_ads=158,  skipped=10   ✅ == canonical
(FPS=2.0 was the bug: → 0.5628, n=116, skipped=52 — DISCARD all fps=2.0 numbers)
```

**Per-knob status (what binds and what doesn't):**
| knob | value | binds? | evidence |
|---|---|---|---|
| **FPS** | **1.0** | **YES — the only binding media knob** | Tier-1: fps=1.0 reproduces 0.5142+n=158+skip=10; fps=2.0 gives 0.5628 |
| audios | `[]` (off) | yes (modality) | AUDIO_OOB: tower PE cap 1500 < 4517 frames for >15s ads → audio dropped |
| resolution | native 720×1280 | — (not downscaled) | Tier-1: ffprobe == grid_thw |
| max_length | 32768 | yes (drops ~6% >55s videos at fps=1.0) | `ckpt-225/args.json` |
| VIDEO_MAX_TOKEN_NUM | 8192 (_common) / 16384 (srcc_eval) | **NO** — native < both ceilings | both give 0.5142 |
| MAX_PIXELS / VIDEO_MAX_PIXELS | 49152 (_common) / 200704 (srcc_eval) | **NO** — native < both | both give 0.5142 |
| prompt | short vs long | within noise (0.563 vs 0.538) | Tier-1; keep short |
| cot | empty `<cot></cot>` bypass | held fixed (Leon: leave it) | — |

So the **only** media knob that must be pinned is **FPS=1.0** (now `setdefault`-ed in
`srcc_eval.py`). The pixel/token caps don't bind at native 720p, so the wrong values from
`sft_v2cot_full.sh` happened to give the right answer once FPS was corrected.

### val_full baseline settings (the scaled ruler — TO RUN)
Same protocol as above, on `val_full_no_cot.jsonl` (n=1358):
`python rl/srcc_eval.py --checkpoint ckpt-225 --val-jsonl data/val_full_no_cot.jsonl
  --plugin .../register.py --attn-impl flash_attn --video-max-tokens 16384 --dump-npz <out>`
with `FPS=1.0` (now the default). Expected drop ≈ 6% (the >55s videos) → n ≈ 1280.
This npz is the baseline every RL checkpoint is paired-bootstrapped against.

---

## 4. The no-audio standard (this stage's task)
`audios=[]` everywhere for train + eval (V8 is already audio-off). Residual cleanup:
strip `audios=[mp4]` from any training data still carrying it; optionally clean "watch and
**listen**" / "video and **audio**" wording from the prompt for internal consistency.

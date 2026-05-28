# V8 SFT Integrity Audit (2026-05-28)

**Run:** `v14-20260528-061946`
**Wandb:** https://wandb.ai/liangyuch/ttcc/runs/fxryqedt
**Code:** `go_viral` HEAD `9c9c478c`, branch `ttcc-rl`, git status clean
**Audit performed at step 8/2800 (~0.3% in, 31 min elapsed)**

## Why this audit exists

V7 had a label-in-input leak: ground-truth `R(t)` lived inside the assistant
span, and the retention head read the hidden state at the last `}` of the answer
JSON — it trivially echoed the answer. Three independent audits
(randomization probe, video-swap, constant-predictor sanity) confirmed V7's
backbone never learned video-retention reasoning. See
`INCIDENT_2026-05-26_EVAL_LEAK.md` in the `ttcc-eval` repo.

Before V8 can be trusted, we must verify the V7 failure class is closed AND
that no other common-ML failure mode is present. This document records the
evidence for each check.

## Verdict

**PASS.** All 16 integrity checks below are clean. Pipeline is well-formed.
The remaining unknowns (does the model actually use video signal?) are
unobservable until step ~75 (first checkpoint); the **randomization probe is
staged** and will fire at that point.

---

## A. Train-data leak scan (V7 failure class)

**Check:** Train assistant span must not contain R values, decimal lists, or
exact retention-curve numbers.

**Evidence:** Full corpus scan over all 35,793 rows (`/tmp/full_scan.py`):

| Pattern | Hits |
|---|---|
| Literal `"R":` key in assistant | 0 / 35,793 |
| Decimal list pattern `[0.xx, ...]` | 0 / 35,793 |
| Exact `R[k]` value as substring (3 decimal places) | 0 / 35,793 |
| Content AFTER `</cot>` close tag | 0 / 35,793 |
| Missing `</cot>` close tag | 0 / 35,793 |
| CoT float matching any `R[k]` (tolerance ±0.01) | 0 / 35,793 |

**Sample inspection (5 rows):** all assistant spans are well-formed
`<cot>...</cot>` blocks containing qualitative reasoning (per-second visual /
audio / impact descriptions + an overall "Shape:" summary). One false-positive
hit on the "retention + number" regex (`"4.6 million yen reward reveal"`) was
currency, not a retention value.

**PASS.**

## B. Val-data leak scan

**Check:** Val rows must have no leak surface (empty assistant content; no R
substring anywhere).

**Evidence:** Scan over all 200 val rows (`val_200_no_cot.jsonl`):

| Property | Value |
|---|---|
| Non-empty assistant content | 0 / 200 |
| `<cot>` present in assistant | 0 / 200 |
| Any `R[k]` as substring in assistant | 0 / 200 |
| `audios: []` (audio-tower bypass) | 200 / 200 |

Sample: `val[0].messages[-1].content == ''`, `val[0].R[:5] = [1.0, 0.144, 0.052, 0.034, 0.027]`. Ground truth lives only in the `R` field of the JSONL row, never in any text span sent to the model.

**PASS.**

## C. `</cot>` anchor position safety

**Check:** The retention head reads `h[</cot>]` (or the last input token as
fallback). At that position, the model's input MUST NOT contain any R-related
information — otherwise the head could trivially copy.

**Evidence:**
- 100 / 100 sampled train rows have a `</cot>` close tag → head anchors at
  `h[</cot>]` (per `_locate_anchor_positions` in `register.py:94`).
- 0 / 100 rows have any content after `</cot>` → the anchor is the LAST
  token of the assistant span; nothing follows it that could pollute upstream
  hidden state.
- For val rows (no `<cot>`), the fallback path uses `h[last_input_token]`, and
  the assistant content is empty → the anchor is the model's state after the
  user prompt finished. This creates a train/val anchor mismatch (post-CoT vs
  post-prompt) but is monitoring-only; downstream IBS eval generates the CoT.

**PASS.**

## D. Model init source (no V7-ckpt contamination)

**Check:** V8 must init from base Qwen2.5-Omni-3B, not from any V7 checkpoint
(V7 weights carry the leak-path corruption).

**Evidence:**
- `model_dir = /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B`
- Files: `model-{00001,00002,00003}-of-00003.safetensors` (4.99 + 5.00 + 1.98 GB = **11.97 GB**, matches HF release of Qwen2.5-Omni-3B)
- `config.model_type = "qwen2_5_omni"` (BASE, not `qwen2_5_omni_retention`)
- `config.architectures = ["Qwen2_5OmniModel"]` (BASE class)
- `model.safetensors.index.json`: 2,544 tensors, **0 retention/head_type keys**
- Sentinel tensor counts (sanity): `thinker.lm_head` 1, `embed_tokens` 1,
  `audio_tower` 489, `visual` 518, `talker` 293 → matches the public Omni-3B
  architecture exactly.

**PASS.**

## E. Loss math wiring

**Check:** Hazard formula, masked MSE, LM CE term, and `RETENTION_COT_ALPHA`
must be wired exactly as the yaml comment specifies.

**Evidence (`register.py:600-695`):**

| Item | Code | Status |
|---|---|---|
| Hazard formula `R(t) = exp(-cumsum(softplus(Linear(h))))` | `RetentionHead.forward:155-158` | ✓ |
| Head bias init `-3.0` (gives `R(60) ≈ 0.05` at init, avoids log-hazard explosion) | `RetentionHead.__init__:146` | ✓ |
| Masked MSE on `R_pred` vs `R_true` | `_masked_mse(r_pred, r_true, r_mask)` line 656 | ✓ |
| LM cross-entropy on assistant tokens (gated by `alpha > 0` AND `labels is not None`) | line 660-666 | ✓ |
| `alpha = float(get_env_args('RETENTION_COT_ALPHA', str, '0.0'))` reads env | line 658 | ✓ |
| Yaml sets `RETENTION_COT_ALPHA: '1e-3'` | yaml line 53 | ✓ |
| Total loss `= loss_curve + alpha * loss_cot` (when CoT present) | line 667 | ✓ |
| Total loss `= loss_curve` only (when CoT absent — val path) | line 669 | ✓ |

**PASS.**

## F. Retention head gradient flow

**Check:** Head must receive gradients (not silently frozen or detached).

**Evidence:**
- `model.retention_head = head` attached as nn.Module submodule (`register.py:347`)
- No `head.requires_grad_(False)` call anywhere
- `head.to(device)` called after attach (`register.py:368`)
- Head is **not** in the freeze list (only `thinker.audio_tower`, `thinker.visual`, `thinker.audio_tower.proj`, `thinker.visual.merger`, `talker`, `token2wav` are frozen — head is omitted, so it trains)
- Live `grad_norm` per step ≈ 4000 over steps 1–8 → the head's MSE gradient
  (initial Kaiming weight on `Linear(3584 → 60)` against true R curves with
  values down to 0.014) is the dominant gradient contributor. Confirms head
  is in the optimizer.

**PASS.**

## G. CoT contains no ground-truth numerals

**Check:** The Gemini-distilled CoT must not include R values (would let the
model trivially memorize).

**Evidence:** Full-corpus scan over all 35,793 train rows for any decimal
inside `<cot>...</cot>` matching any `R[k]` within ±0.01:

- **0 / 35,793 CoT spans contain a number matching any R[k] value.**

CoT spans contain only qualitative descriptors ("strong hook", "gradual decay")
and second timestamps ("0-3s", "4-10s"). The only numbers are second-timestamps,
which are bounded by `T` (not in the [0,1] range that would match R).

**PASS.**

## H. Randomization probe staged

**Check:** A probe that perturbs video input (zero / swap with another ad) and
measures whether the prediction changes. V7's failing values were < 0.1 for both
`zero_dist/between_ad_dist` and `swap_dist/between_ad_dist`. V8 should produce
ratios ≫ 0.5 if it actually conditions on video.

**Evidence:** Script exists at
`examples/custom/qwen2_5_omni_retention/tools/randomization_probe.py`. It runs
on a saved checkpoint dir. First V8 ckpt lands at step 75 (~5.5 h from launch).

**Plan:** Fire at step 75:
```bash
python randomization_probe.py \
    --ckpt /opt/dlami/nvme/ssm-out/sft_retention_hazard_full_with_cot/v14-20260528-061946/checkpoint-75 \
    --val-jsonl /home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl \
    --n-ads 20
```
Hard-fail criterion: either ratio < 0.3 → halt training and investigate.

**STAGED.**

## J. Train/val ad_id overlap

**Check:** A val ad appearing in train would mean memorization gets credit as
generalization.

**Evidence:** `train_ids ∩ val_ids = ∅` (0 shared ad_ids across 35,793 train +
200 val).

**PASS.**

## K. R-label sanity

**Check:** R values must be a valid retention curve: monotone non-increasing,
in `[0,1]`, with `R[0] = 1.0` and `len(R) = T + 1`.

**Evidence:** Full corpus scan over all 35,793 rows:

| Violation | Count |
|---|---|
| `R[0] != 1.0` | 0 |
| Any `R[k] ∉ [0, 1]` | 0 |
| R not monotone non-increasing | 0 |
| `len(R) != T + 1` | 0 |

Distribution sanity:
- `T`: min 5s, median 24s, max 60s, mean 26.5s (TTCC short-form ads ✓)
- `R[1]` (post-hook retention): median **0.37** — most ads lose ~63% of viewers
  in the first second (consistent with TikTok scroll behavior)
- `R[T]` (final retention): median **0.023** — typical ad keeps ~2% of viewers
  to the end

**PASS.**

## L. Frozen modules

**Check:** Vision tower, audio tower, talker, token2wav must be frozen (matches
milestone spec; only LM body + retention head train).

**Evidence (from launch log):**
```
freeze_parameters: ['thinker.audio_tower', 'thinker.visual',
                    'thinker.audio_tower.proj', 'thinker.visual.merger',
                    'talker', 'token2wav']
freeze_vit=True, freeze_aligner=True
```

The retention head is **not** in the freeze list → trains. Visual+audio towers
still produce embeddings (forward pass), but their weights don't update — this
is standard multimodal SFT practice.

**PASS.**

## M. NaN/Inf scan

**Check:** No NaN or Inf in loss or grad_norm at any logged step.

**Evidence:** 8 logged steps so far:
- `loss`: range 29.01 – 35.51 (warmup floor, no NaN/Inf)
- `grad_norm`: range 3686.78 – 4355.27 (huge but stable, clipped by `max_grad_norm: 1.0`)
- `learning_rate`: ramping 6e-8 → 4.8e-7 (warmup over 84 steps to 5e-6)
- `token_acc`: stable 0.554 – 0.564 (LM body not drifting, α=1e-3 is small)

**PASS.**

## N. FA3 actually active (launch argument)

**Check:** `attn_impl: flash_attention_3` in yaml must propagate to the actual
launch arg (V7's near-miss was a silent SDPA override in a launcher script).

**Evidence:**
- Launch command (from log): `--attn_impl flash_attention_3`
- 0 SDPA mentions in log (would warn if fallback occurred)
- 0 FA2-specific mentions
- TMRoPE-FA3 monkey-patch in `register.py` is active (verified separately)

**PASS.**

## O. Reproducibility metadata

**Check:** Run is reproducible: commit pinned, seed set, package versions
captured.

**Evidence:**
- `go_viral` HEAD: `9c9c478c42313838475d679a992fa5349e00b9fd`
- branch: `ttcc-rl`
- `git status`: clean
- `seed=42` (set by ms-swift default; appears in log)
- Pinned versions:
  - `transformers==4.56.2` (TMRoPE `_is_packed_sequence` patch targets this)
  - `torch==2.12.0`
  - `flash-attn==2.8.3` (FA2, installed but not used; coexists with FA3 .so)
  - `deepspeed==0.15.2`
  - `accelerate==1.13.0`

**PASS.**

## P. TMRoPE `_is_packed_sequence` patch active

**Check:** Qwen2.5-Omni's 3D position_ids (shape `[3, batch, seq]` for time/H/W)
break transformers v4.56.2's `_is_packed_sequence` which assumes 2D shape and
falsely returns True → wrong-sized `cu_seqlens` → FA3's strict 1D check raises
`RuntimeError`. PR #44911 fixes this upstream; we monkey-patch locally.

**Evidence:** Patch present in `register.py` (lines 1-30 ish). It wraps the
upstream `_is_packed_sequence` and returns False whenever `position_ids.dim() > 2`.

**PASS.**

## Q. Video file existence

**Check:** Referenced `.mp4` paths must resolve to readable files.

**Evidence:** 50-row random sample:
- 0 / 50 missing or truncated
- First video: `/home/ssm-user/work/data/videos/7631598650221608968.mp4`, 7.4 MB,
  `ffprobe duration = 45.14s` ← matches `T = 45` in the jsonl row

**PASS.**

## R. FA3 actually loaded at runtime

**Check:** Beyond the launch argument, the `flash_attn_3` C extension must
actually be loaded into the rank processes' memory.

**Evidence:** `/proc/<rank-pid>/maps` on all 8 rank workers:

```
pid=398821: /opt/dlami/nvme/work/swift_venv/lib/python3.12/site-packages/flash_attn_3/_C.abi3.so
pid=398822: /opt/dlami/nvme/work/swift_venv/lib/python3.12/site-packages/flash_attn_3/_C.abi3.so
pid=398823: /opt/dlami/nvme/work/swift_venv/lib/python3.12/site-packages/flash_attn_3/_C.abi3.so
(all 8 ranks: identical)
```

(FA2's `.so` is also mapped — it's installed in the venv — but transformers
dispatches to FA3 based on the launch arg, not the import order.)

**PASS.**

---

## Open monitoring items (not failures, signals to watch)

1. **Loss turnover (step 30–80).** Warmup ends at step 84 (`warmup_ratio: 0.03 × 2800`). Loss should transition from "oscillating in [29, 36]" to "monotone decreasing" around then. If it doesn't, the effective LR is too small for the gradient to overcome clipping → may need to re-examine clip threshold.

2. **First eval (step 50, ~3.5 h from launch).** Val MSE will be high due to the train/val anchor mismatch (train anchors at `h[</cot>]`, val falls back to `h[last_input]`). Watch for explosion, not the absolute value.

3. **First ckpt + randomization probe (step 75, ~5.5 h from launch).** Verify:
   - `safetensors` contains `retention_head.linear.{weight,bias}` keys
   - Probe ratios `zero_dist / between_ad_dist` and `swap_dist / between_ad_dist` both ≫ 0.5
   - If either ratio < 0.3: **halt training** and investigate.

4. **Memory plateau.** Currently 53 GB / 80 GB on head node. If it climbs past 70 GB across 100+ steps, we have a leak.

5. **Step time.** Currently 230–246 s/step. At this rate the CB (~133 h remaining) gets us ~6.95 epochs, matching V7's natural plateau. Acceptable.

6. **CapBlock expiration: 2026-06-03 19:30 BJ.** Training will be cut off mechanically; save_total_limit=10 means we'll have ckpts from step ~1200 onward.

---

## What remains genuinely unobservable

- Whether the trained model has learned to **interpret** video content (vs. learn shape priors that happen to fit the corpus). The randomization probe at step 75 is the strongest test we have for this.
- Whether the audio drop (`audios: []` + `USE_AUDIO_IN_VIDEO: false`) costs us measurable IBS. We can only know by training a V9 with audio reintroduced (via PE interpolation or windowed chunking) and comparing.

## Related incidents

- `INCIDENT_2026-05-26_EVAL_LEAK.md` (ttcc-eval) — V7 R(t) leak in assistant span; reason this audit exists.
- `INCIDENT_2026-05-28_SDPA_OVERRIDE.md` — Silent SDPA override in a launcher script defeated the FA3 install; fixed via launcher edit and AUDIT-N now guards against recurrence.
- `INCIDENT_2026-05-28_AUDIO_OOB.md` — Audio tower positional embedding overflow (4517 frames > 1500 capacity) for ads > 15 s; fixed by dropping audio for V8.

## Reproduction

Re-run any of the above checks:
```bash
# Train/val leak + anchor + init + label sanity + frozen modules + NaN scan + FA3 + repro metadata
TARGET=i-0f1d5118e482bceeb REGION=us-east-2 AWS_PROFILE=gpu-box \
    ssm_run.sh "python3 /tmp/full_scan.py && python3 /tmp/audit_jkl.py"

# Video existence + FA3 .so runtime check
TARGET=i-0f1d5118e482bceeb REGION=us-east-2 AWS_PROFILE=gpu-box \
    ssm_run.sh "python3 /tmp/audit_qr.py && python3 /tmp/audit_r2.py"
```

The audit scripts themselves live in `/tmp/` on the head node; they're
ephemeral, but the queries in this document are reproducible from scratch.

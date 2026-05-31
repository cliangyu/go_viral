# Batched reasoned-retention eval — plan, correctness gates, smoke

**Goal:** cut the long-pole reasoned eval (`generate_eval.py`, ~12–20 s/ad at
batch=1, ~7.5 h for full n=1358) by ~5–10× by batching the autoregressive CoT
generation. Generation is the bottleneck (~300 sequential decode steps over a
multi-thousand-token multimodal context, done one ad at a time). Batching
amortizes the per-decode-step cost across K ads.

**Deliverables (all NEW; the running `generate_eval.py` is untouched):**
- `verification/generate_eval_batched.py` — batched path + per-ad fallback.
- `verification/smoke_batched_equiv.py` — n=4 both-ways equivalence guard.
- this doc.

Status: drafted + syntax-checked locally (`py_compile` clean). **No GPU job
run** (both eval GPUs busy). Smoke below is for Leon to run.

---

## 1. Batching approach (what is batched, what is not)

| Step | Cost | Batched? | Why |
|---|---|---|---|
| 1. encode prompt (per ad, video decode + token expansion) | ~3 s/ad | No (inherently per-sample) | multimodal encode is per-row; collation only stacks |
| 2. **autoregressive CoT generation** | **dominant** | **YES, K ads/call** | the long pole; one `generate()` over K ads amortizes per-step cost |
| 3. head readout (re-encode prompt+CoT, 1 forward, read `r_pred`) | ~0.1 s/ad | **No — kept per-ad** | a single cheap forward; the head reads ONE token's hidden at last `</cot>` and is right-pad-fragile (see §4) — per-ad is bit-identical to `generate_eval.head_curve()` and to `srcc_eval.py`'s bypass forward |

So the speedup is entirely in step 2. Step 3 stays exactly as `generate_eval.py`
does it — this keeps the reasoned/bypass numbers directly comparable to the
documented baseline and to the bypass eval.

**Collation is the processor/template's own `data_collator`, not hand-rolled
padding** (the task said to prefer this). `tmpl_gen.set_mode('transformers')`
then `tmpl_gen.data_collator(encs)`:
- LEFT-pads `input_ids` / `attention_mask` (swift `base.py:1817`:
  `padding_side = self.padding_side if self.is_training else 'left'`),
- `torch.concat`s `pixel_values_videos` and row-concats `video_grid_thw`
  (`base.py:_data_collator_mm_data`) — ragged per-ad video token counts collate
  correctly because the model indexes videos by `grid_thw`, not by batch row,
- in inference mode it does **NOT** precompute `position_ids`
  (`qwen.py:933` `if not self.is_training: return {}`).

Generated-token slicing: with LEFT padding, every row's prompt ends at column
`L-1` (where `L = padded prompt width`), so the new tokens for **every** row are
exactly `gen[i, L:]`. (Right-padding would need per-row offsets; left-pad makes
the slice uniform — that's the reason decoder generation uses left-pad.)

---

## 2. The exact left-pad / position_ids handling (the #1 risk)

We pass `input_ids` + `attention_mask` + stacked video tensors, and **NO
`position_ids`** (explicitly `batch.pop('position_ids', None)` as a belt-and-
suspenders). Qwen2.5-Omni's `thinker.forward` then computes mrope itself
(`modeling_qwen2_5_omni.py`, branch `if attention_mask is not None and
position_ids is None:`):

```
# prefill (first forward):
delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)        # = # left-pad tokens per row
position_ids, rope_deltas = self.get_rope_index(input_ids, ..., attention_mask, ...)
rope_deltas = rope_deltas - delta0                            # <-- LEFT-PAD CORRECTION
self.rope_deltas = rope_deltas
# decode steps reuse: position_ids = arange(seq) + (past_len + rope_deltas)
```

Two things make left-pad correct here:
1. `get_rope_index` masks padded positions per row before assigning mrope coords
   (`input_ids = input_ids[attention_mask[i]]`, `modeling…:301`), so the
   temporal/H/W rope coordinates of the real tokens are unaffected by left-pad.
2. the `rope_deltas - delta0` term offsets the decode-step position arithmetic by
   the per-row left-pad width, so continued generation stays aligned per row.

⇒ **We MUST pass `attention_mask` and MUST NOT pass `position_ids`.** We do both.
This is also exactly what the batch=1 path already relies on (it passes no
position_ids either) — batching changes only K.

### Residual VERSION risk (the thing the smoke exists to catch)
The `rope_deltas - delta0` left-pad correction must be present in the
transformers build on the eval box. **Verified present in transformers 5.x**
(cached wheel inspected) and it has been in HF since the Qwen2.5-Omni left-pad
fix (~4.51+). `register.py`'s header comment mentions "transformers v4.56.x";
4.56 has the fix. **But this is environment-dependent — do not assume.** If the
box ran an older build without it, left-padded multimodal decode positions drift
and batched CoTs diverge from batch=1. **The equivalence smoke (§3) is the
guard.** Until it passes on the actual eval box + ckpt, batched numbers are not
trustworthy.

---

## 3. Equivalence smoke (n=4, both ways) — the correctness gate

One process, one model load, same 4 ads through batch=1 and batch=K. Asserts:
- **(A) CoTs identical per ad** — greedy is deterministic; if left-pad/mrope is
  wrong the batched decode diverges and this fails (the primary signal).
- **(B) reasoned `r_pred` curves match** within `rtol` (head is per-ad in both →
  should be bit-identical up to kernel numerics).
- **(C) cross-ad reasoned SRCC matches** within `rtol`.

Exit 0 = equivalent; exit 1 = divergence (prints the first char where a CoT
diverges, for debugging). Command (run on the eval box, on a **spare** GPU — the
script is tiny; **do not** touch the two busy eval GPUs):

```bash
cd <go_viral repo root>
REG=examples/custom/qwen2_5_omni_retention/register.py
CK=/opt/dlami/nvme/cotgrl_eval/ckpts/<ckpt-name>     # e.g. the SFT baseline used for reasoned eval
BASE=Qwen/Qwen2.5-Omni-3B                            # or the adapter's true base
VAL=/opt/dlami/nvme/v8_eval/data/val_full_no_cot.jsonl
PY=/opt/dlami/nvme/eval_venv/bin/python

CUDA_VISIBLE_DEVICES=<spare_gpu> "$PY" \
  examples/train/grpo/qwen2_5_omni_ttcc/verification/smoke_batched_equiv.py \
  --checkpoint "$CK" --base "$BASE" --val-jsonl "$VAL" --plugin "$REG" \
  --attn-impl sdpa --max-new 600 --n 4 --batch-size 4
```

Expected on PASS:
```
ad0: CoT_match=True  r_pred_maxabsdiff=~1e-4 (<= 0.002? True)
...
cross-ad SRCC: single=0.xxxx batched=0.xxxx (|diff|<= 0.002? True)
SMOKE PASS
```

If **CoT_match=False**: left-pad/posids is wrong for this transformers build —
STOP, do not use the batched path; report the build, and either upgrade
transformers to a build with the `rope_deltas - delta0` fix or fall back to
`--batch-size 1`. If only `r_pred` differs slightly (> rtol but CoT matches):
loosen `--rtol` cautiously — single-token readout is documented kernel-sensitive
on sdpa (see `research/EVAL_SPEED_AND_KERNEL.md`); a few e-3 is plausible, a
large diff is not.

> Note: the smoke also exercises an independent diff harness — the full batched
> script can be run with `--dump-cots out.jsonl` for a larger A/B if desired, but
> n=4 in-process is the cheap definitive check.

---

## 4. Why the head readout is NOT batched (deliberate)

The head reads a single token's hidden state at the **last `</cot>`**
(`register.py:_locate_anchor_positions`), falling back to the last input token
(`L-1`) when a row has no `</cot>`. In a **right-padded** batch the `L-1`
fallback lands on a PAD token — `head_pg_trainer.py:19-20` documents exactly this
("bs>1 right-pads → h_anchor at L-1 becomes a PAD"). The `srcc_eval.py` bypass
and the current `generate_eval.py` head read are both bs=1 for this reason.
Keeping step 3 per-ad makes the batched script's r_pred **bit-comparable** to the
existing baseline with zero anchor risk, at negligible cost (it's a single cheap
forward, not the long pole). A batched head readout is possible later (locate
anchors per row + gather) but is out of scope: correctness first.

---

## 5. Recommended K and the risk list

**Recommended `--batch-size 8` to start.** Drop to **4** if OOM on long videos,
raise toward **16** if VRAM is free. Generation VRAM scales ~ `K × (video tokens
+ prompt + max_new + KV cache)`; long videos hit ~36k video tokens each at the
canonical `VIDEO_MAX_TOKEN_NUM=16384` cap, so K×that is the dominant term. The
script's batched-generate has a **per-ad fallback on any batch OOM/error**, so a
single pathological long video degrades that batch to batch=1 rather than
crashing the run.

### Risks (flagged explicitly)
1. **Left-pad + mrope position_ids (P0).** Correct only if the transformers
   build has the `rope_deltas - delta0` left-pad fix (verified in 5.x / 4.51+;
   register.py header says 4.56.x). **The n=4 smoke is the guard.** This is the
   single most likely way the batched path is silently wrong.
2. **Ragged video token counts across the batch.** Handled — `pixel_values_videos`
   concat + per-row `video_grid_thw`; the model indexes by grid_thw. Low risk but
   the smoke covers it (the 4 ads have different video lengths).
3. **OOM at large K + long videos.** Mitigation: start K=8, per-ad fallback on
   batch failure, scale K with free VRAM. Watch `nvidia-smi` on the first full
   batch.
4. **Attention-mask leakage from padded positions.** Handled — the collator emits
   a per-row `attention_mask` (1 for real tokens, 0 for left-pad); the model's
   causal mask + this 2D mask zero out pad attention. The smoke's CoT-match
   assertion would fail if pad positions leaked into the real tokens' attention.
5. **Kernel sensitivity (sdpa on the Blackwell eval box).** The single-token head
   read is documented kernel-sensitive (bypass SRCC ~0.51 on flash_attn vs ~0.07
   on sdpa, `EVAL_SPEED_AND_KERNEL.md`). Batching does **not** change this — the
   head read is identical per-ad — but it means absolute numbers from this script
   inherit the same sdpa caveat; only relative (RL vs SFT, both sdpa) comparisons
   are valid, same as the existing eval.
6. **`stop_words` / EOS handling.** The batched `generate()` uses the same
   `do_sample=False, num_beams=1, max_new` as batch=1 and the same `_clean_cot`
   wrap; rows that hit EOS early are padded by HF and sliced at `gen[i, L:]`, and
   `skip_special_tokens=True` drops the trailing pad/eos on decode — same as the
   batch=1 path.

---

## 6. Expected speedup

Per `EVAL_SPEED_AND_KERNEL.md`: batched generation (8–16 ads) ≈ **5–10×** on the
generation step ⇒ full-n reasoned ~7.5 h → **~1 h** at K=8 on the current
sdpa Blackwell box (encode + head readout are not batched, so the realized
end-to-end speedup is a bit under the pure-generation factor; generation
dominates, so it stays in the 5–10× band). flash_attn + batch would be minutes,
but the eval box has no flash_attn build (sdpa only).

**Acceptance order:** (1) run the n=4 smoke → must print `SMOKE PASS`; (2) only
then run `generate_eval_batched.py` with `--batch-size 8` on a spare GPU and
confirm its reasoned/bypass SRCC matches a short `generate_eval.py --limit 32`
run within ~rtol; (3) scale to full n.

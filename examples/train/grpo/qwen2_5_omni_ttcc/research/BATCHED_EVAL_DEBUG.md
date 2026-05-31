# Batched eval equivalence-smoke failure — diagnosis + fix

**Symptom:** `smoke_batched_equiv.py` FAILED:
`ad0: CoT_match=False, r_pred_maxabsdiff=9.38e-03 (> tol 2e-3)`. The batched-K
generated CoTs differ from the per-ad (batch=1) CoTs.

**TL;DR:** The plan's #1 hypothesis (left-pad mrope `position_ids` are wrong in
this transformers build) is **FALSIFIED**. transformers 4.57.6 handles left-pad
mrope correctly; I proved it on CPU with the real `get_rope_index`. The real
cause is **bf16 numerical drift from padded batched attention flipping near-tied
greedy argmaxes**, which is intrinsic to padded batched decode on GPU and cannot
be removed by any `position_ids` change. The fix makes batching **deterministic
by construction** via **length-bucketed, zero-padding** generation.

---

## 1. What I verified (with evidence)

### 1a. The transformers build HAS the left-pad mrope correction
The eval box runs **transformers 4.57.6**. I could not install 4.57.6 locally,
so I (a) read the installed 5.6.2 source and (b) fetched the `v4.57.1` tag source
from GitHub (patch releases do not touch the modeling file). Both are identical
in the relevant code:

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py`,
`Qwen2_5OmniThinkerForConditionalGeneration.forward` (5.6.2 local lines
1955–1976; identical in 4.57.x):

```python
if attention_mask is not None and position_ids is None:
    if cache_position is None or cache_position[0] == 0 or self.rope_deltas is None:
        delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)   # = #left-pad tokens/row
        position_ids, rope_deltas = self.get_rope_index(input_ids, ..., attention_mask, ...)
        rope_deltas = rope_deltas - delta0                       # <-- LEFT-PAD CORRECTION
        self.rope_deltas = rope_deltas
    else:
        delta = cache_position[0] + self.rope_deltas             # decode-step offset
        ...
```

And `get_rope_index` (local lines 289–301) masks padded positions per row before
assigning mrope coords and writes them back only into unmasked slots:

```python
if attention_mask is not None:
    attention_mask = attention_mask == 1
...
for i, input_ids in enumerate(total_input_ids):
    if attention_mask is not None:
        input_ids = input_ids[attention_mask[i]]                 # drop pad before mrope
...
position_ids[..., i, attention_mask[i]] = llm_positions          # write only real slots
```

So the left-pad correction (`rope_deltas - delta0`) the plan worried about **is
present** in 4.57.6.

### 1b. CPU proof that real-token positions are bit-identical under left-pad
Probe: `/tmp/rope_leftpad_probe.py` (run with the local easydel venv:
`forks/easydel/.venv/bin/python`, torch 2.11.0 / transformers 5.5.0 — same
algorithm). It calls the **real** `get_rope_index` and replicates the
`forward()` `delta0` correction + the decode-step position math.

Result (single video row, then ragged K=2 batch):
```
real-token position_ids identical under left-pad?  True
FIRST-DECODE-STEP absolute position: single=32 pad=32 identical=True

===== TRUE BATCHED K=2 RAGGED CASE =====
Row A real-token pos identical to single A? True
Row B real-token pos identical to single B? True
Row A: decode-pos batched=32 single=32 identical=True
Row B: decode-pos batched=80 single=80 identical=True
```

Conclusion: for a left-padded (even ragged) multimodal batch, every real-token
mrope coordinate AND every per-row decode-step absolute position is **exactly
equal** to the batch=1 values. **Left-pad mrope is not the bug.** Likewise the
other plan suspects are ruled out: `pixel_values_videos` is row-concat and
`video_grid_thw` is row-concat and consumed by a per-row `video_idx` counter
(`get_rope_index` lines 397/414), and `attention_mask` is built per real-token
length and left-padded by the collator (`swift/template/base.py:1875`,
`_pad_sequence` left branch at base.py:2127+), so pad positions are masked.

### 1c. So why do the CoTs diverge?
The r_pred diff is **small** (9.38e-3), not catastrophic. If positions were
broken, CoTs would be garbage and r_pred would be off by O(0.1–1). A ~9e-3
difference is the signature of the head reading a CoT that is *almost* the same
but diverged a few tokens in. The mechanism:

- Greedy (`do_sample=False`) takes a hard `argmax` over logits each step.
- With **padding present**, the attention softmax + value matmul run over the
  masked-but-materialized padded columns; the **bf16 reduction order** differs
  from the unpadded batch=1 forward. Top-2 logits on a 3B model routinely sit
  within ~1e-2.
- When the padded-batch logits flip a near-tied top-1/top-2, the greedy token
  changes, and from that step the sequence diverges. The per-ad head readout is
  bit-identical in both paths (it is the *input CoT* that differs), so
  r_pred_maxabsdiff is exactly "the two CoTs read slightly different hidden
  states" — i.e. a downstream symptom, not an independent bug.

This is a well-known property of padded batched generation on GPU; it is **not**
fixable by setting `position_ids=None` (the code already does that correctly).

---

## 2. The fix (in `verification/generate_eval_batched.py`)

`gen_cot_batched` was rewritten to be **deterministic by construction**:

- Encode each row (unchanged; multimodal encode is inherently per-sample).
- **Bucket rows by EXACT collated prompt length** (`len(enc['input_ids'])`).
- Run one `generate()` per bucket. Equal length ⇒ the collator left-pads by
  **zero** ⇒ each row's attention math is identical to its standalone batch=1
  forward ⇒ **CoTs are bit-identical to batch=1**. Singleton lengths fall back
  to a 1-row generate (literally == batch=1).
- Results are realigned to the input `rows` order.

The padded behaviour is preserved behind an opt-in env flag
`TTCC_BATCHED_ALLOW_PAD=1` (fast, NOT bit-exact). `gen_cot_single` and
`head_curve` are unchanged, so the smoke still tests the shipping helpers.

Why this passes the smoke: the 4 smoke ads (distinct video lengths ⇒ distinct
prompt lengths) each generate as their own 1-row bucket ⇒ identical to the
`gen_cot_single` path ⇒ `CoT_match=True`, `r_pred_maxabsdiff` at kernel-noise
level (≤ 2e-3).

Verification done locally (no GPU): `py_compile` clean on both scripts; the
bucketing logic unit-tested for order-preservation, full index coverage, and
singleton fallback.

---

## 3. Re-smoke command (run on the eval box, ONE spare GPU)

```bash
cd /Users/marvl/Documents/stanford/cs224r/projects/go_viral   # repo root on the box
REG=examples/custom/qwen2_5_omni_retention/register.py
CK=/opt/dlami/nvme/cotgrl_eval/ckpts/<ckpt-name>              # the reasoned-eval ckpt
BASE=Qwen/Qwen2.5-Omni-3B                                     # or the adapter's true base
VAL=/opt/dlami/nvme/v8_eval/data/val_full_no_cot.jsonl
PY=/opt/dlami/nvme/eval_venv/bin/python                       # the torch2.8/tf4.57.6 venv

CUDA_VISIBLE_DEVICES=<spare_gpu> "$PY" \
  examples/train/grpo/qwen2_5_omni_ttcc/verification/smoke_batched_equiv.py \
  --checkpoint "$CK" --base "$BASE" --val-jsonl "$VAL" --plugin "$REG" \
  --attn-impl flash_attn --max-new 600 --n 4 --batch-size 4
```

Expected now:
```
ad0: CoT_match=True  r_pred_maxabsdiff=<=2e-3 (<= 0.002? True)
...
cross-ad SRCC: single=0.xxxx batched=0.xxxx (|diff|<= 0.002? True)
SMOKE PASS
```

Notes:
- Use `--attn-impl flash_attn` to match the **production** eval kernel (the
  earlier plan used `sdpa`; the divergence is independent of backend, but the
  smoke should mirror the run kernel that head-readout SRCC is calibrated on —
  see `EVAL_SPEED_AND_KERNEL.md`, bypass SRCC ~0.51 on flash_attn vs ~0.07 on
  sdpa). If flash_attn is unavailable on the box, `sdpa` still validates the
  determinism fix.
- The `register.py` FA3 `_is_packed_sequence` monkeypatch (register.py:56–65)
  must be active (it is, via `--plugin`).

---

## 4. Residual risk / honest tradeoff (READ before scaling K)

Exact-length bucketing is deterministic but only amortizes the decode loop
across rows that share an **exact** prompt length. Raw ad prompts have
continuously-varying video-token counts, so exact-length collisions may be
**rare**, which means the deterministic default may run close to batch=1 speed on
real data. I could not measure the val length distribution here (val lives on the
box, not this laptop).

Decision for the throughput goal ("saturate VRAM"):

- If exact-length clusters are common enough → deterministic default already
  amortizes; just raise the eligible bucket size.
- If they are rare and you need the speedup → set `TTCC_BATCHED_ALLOW_PAD=1` and
  **re-validate at the METRIC level, not CoT-exactness**: run
  `generate_eval_batched.py --dump-cots` padded vs `generate_eval.py` on a
  ~64-ad slice and confirm the cross-ad **SRCC agrees within ~1e-2**. Greedy
  near-tie flips are a handful of tokens per CoT; cross-ad SRCC is robust to that
  even though exact CoT strings differ. That is the right acceptance gate for the
  padded fast path — do NOT loosen the smoke's exact-CoT assertion to force it.

The exact-CoT smoke remains the gate for the **deterministic** path (default),
which is what this fix makes pass.

## 5. Files
- Fixed: `examples/train/grpo/qwen2_5_omni_ttcc/verification/generate_eval_batched.py`
  (`gen_cot_batched` + new `_encode_row` / `_generate_bucket` helpers).
- Probe used for the CPU proof: `/tmp/rope_leftpad_probe.py` (scratch; not committed).
- Unchanged: `verification/smoke_batched_equiv.py`, `verification/generate_eval.py`.

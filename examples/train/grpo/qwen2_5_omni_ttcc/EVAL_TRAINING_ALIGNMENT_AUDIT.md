# V8 Eval ↔ Training Alignment Audit (2026-05-29)

Authoritative source for training: `ckpt-225/args.json` →
`dataset = ['/home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl']` (39,375 rows),
`model_type=qwen2_5_omni_retention`, `template=qwen2_5_omni_retention`,
`loss_type=retention_loss`, `truncation_strategy=delete`, `max_length=32768` (SFT *script*
said `--max_length 24576`; saved arg is 32768 — minor discrepancy to resolve).

Eval = `srcc_eval.py` on `val_full_no_cot.jsonl` (n=1358) / `val_present.jsonl` (n=168).

## Side-by-side (verified from the actual files)

| dimension | **V8 TRAINING** (ttcc_train_with_cot) | **EVAL** (val_*_no_cot + srcc_eval) | aligned? |
|---|---|---|---|
| **audio** | `audios=[mp4]` — AUDIO **ON**; system/user say "Watch and **listen**" | `audios=[]` — AUDIO **OFF** | ❌ **NO** |
| **user prompt** | 580c: "Watch and listen… write analysis on three labeled lines… `Content:`/`Drops:`… finish with JSON curve" | 68c: "Estimate the per-second retention curve." | ❌ **NO** |
| **system prompt** | 297c: "…R(0)=1 by **definition**. R(t) is monotone non-increasing. Use the vi…" | 196c: "…R(0)=1 by **convention**." | ❌ NO |
| **assistant / CoT** | `<cot>` with REAL visual+audio reasoning ("0s \| visual:… \| audio: voiceover…"), then JSON | `''` → srcc_eval overwrites to empty `<cot></cot>` | ❌ NO (bypass) |
| **head readout** | reads hidden at `</cot>` anchor — conditioned on full reasoning | reads `</cot>` anchor — conditioned on NOTHING | ❌ NO (consequence of CoT) |
| **video resolution** | VMT=16384 + native frames → grid **92×52** (~1196 tok/frame) | was VMT=256 → **42×24** (OOD); **FIXED → 16384** | ✅ after fix |
| **video token cap** | MAX_PIXELS=200704 (non-binding; native<cap), FPS_MAX=60 | same env now pinned | ✅ after fix |
| **attn_impl** | flash_attn | flash_attn (pinned) | ✅ |
| **metric target** | retention_loss on R(0..T) | cross-ad Spearman of R(t) | ✅ (intended) |

## The four real misalignments (in severity order)

1. **AUDIO OFF at eval, ON in training.** The model was explicitly trained to "watch and
   listen"; its CoT reasoning cites audio cues ("voiceover 'Hey' + synth beat"). The eval
   feeds `audios=[]`. The model is being asked to predict retention with a modality it was
   trained to depend on **removed**. This is the single biggest distribution shift.
2. **User prompt is a different task.** Training: a structured analysis task (Content/Drops/
   three labeled lines → JSON). Eval: a bare "estimate the curve." Different instruction =
   different conditioning of the whole forward.
3. **CoT bypass.** Training put real reasoning before `</cot>`; eval puts nothing. The head
   reads a hidden conditioned on rich reasoning at train time, on an empty span at eval. The
   "leak-free" rationale is real (teacher CoT could leak the curve), but the cost is the head
   reads an out-of-distribution representation.
4. **System prompt wording** ("definition" vs "convention", extra monotonicity instruction).
   Smallest, but still a non-identical conditioning string.

## Implication

**The 0.514 "baseline/ceiling" was measured in a heavily out-of-distribution condition**
(no audio + minimal prompt + empty CoT). It is a faithful number *for that bypass protocol*,
but it is **not** the model's in-distribution ability, and the "ceiling" may be largely a
measurement artifact rather than a capability limit. This reframes the RL problem: before
asking RL to push past 0.514, confirm whether 0.514 is even the right number — an
**audio-on, prompt-matched** eval may move the baseline materially (cheap to test: rebuild a
small val slice WITH `audios=[mp4]` + the training prompt, re-score ckpt-225).

## Open items to resolve
- `max_length` 24576 (script) vs 32768 (saved arg) — which did ckpt-225 actually use for drops?
- Is `val_full_no_cot` the V8 held-out split (no ad_id/brand leak vs the 39k train)?
- Does `make_v8_from_hf.py` (the Wanjia handoff) reproduce the LONG-prompt + audio-ON format,
  or the short/no-audio format? If the latter, the handoff trains a different model than V8.

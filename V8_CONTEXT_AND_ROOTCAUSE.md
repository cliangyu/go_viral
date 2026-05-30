# V8 — Context + Root Cause (evidence-verified, 2026-05-30)

Consolidated understanding of the V8 pipeline and the **verified** root cause of why we are
stuck at ~0.44–0.515 cross-ad SRCC with a decorative CoT. Corrects one confident-but-wrong
claim from the strategy workflow's `DECISION_COT_VS_RL_20260530.md` (the "dead anchor is FATAL").

---

## Part A — What V8 is (the pipeline)

- **Task:** predict a per-second retention curve `R(t)`, t=0..60, for a TikTok ad from video.
  **Audio OFF** (AUDIO_OOB decode failures). Video fps=1.0, VMT=16384, max_length 32768.
- **Model:** Qwen2.5-Omni-3B (thinker) + a small **hazard retention head** —
  `z = Linear(h_anchor)`; `R(t) = exp(−cumsum(softplus(z)))`. Head reads ONE hidden state.
- **Data (`ttcc_v8`):** ~39k train ads. Two variants of the assistant message:
  - *with-cot* (`ttcc_train_with_cot.jsonl`): assistant = `<cot>` + per-second reasoning
    (Gemini-distilled: "0s | Car image… | gain: strong hook…") + `</cot>`. **The message ENDS
    at `</cot>`** — there is NO curve JSON after it (verified: `</cot>` at char 1483/1489).
  - *bypass* (`ttcc_train_bypass.jsonl`): assistant = `<cot></cot>` (empty).
- **Loss** (`register.py:724-735`): `L = masked_MSE(R_pred,R_true) + alpha·CE(cot_tokens)`,
  **alpha = RETENTION_COT_ALPHA = 1e-3** (config:53).
- **SFT:** full fine-tune (freeze_vit/aligner), 2-node, fps=1.0 → `v19/checkpoint-225` = baseline.
- **Eval:** LEAK-FREE BYPASS — assistant overwritten to `<cot></cot>`, single forward, head reads
  the last token; cross-ad SRCC over t∈[1,30] + IBS. **baseline: val_full 0.4393 SRCC / 0.00672 IBS
  (n=1332); val_present/CV ≈ 0.514 (n=158).**
- **RL tried:** head-PG REINFORCE + dense `rank_sft`, both with the full backbone trainable. Both
  NULL on SRCC, both WORSE on IBS (see RL_STRATEGY_WORKTHROUGH §3-4).
- **head_oracle (`verification/head_oracle.py`, PROVEN prior result):** best linear ridge readout of
  the frozen h_anchor features → CV-SRCC ≈ 0.514 ≈ baseline ⇒ **the linear readout is saturated.**

---

## Part B — The root cause (verified causal chain)

**Symptoms:** (1) RL can't beat SFT on SRCC; (2) CoT is decorative (bypass ΔIBS≈0); (3) token_acc
collapsed 0.557→0.004; (4) everything sits at ~0.44/0.515.

### What is NOT the root cause (corrected)
- **The "dead `</cot>` anchor" is a RED HERRING.** Yes, `_locate_anchor_positions` falls back to
  `L-1` when the 3-token `</cot>` window misses. **But the assistant message ends at `</cot>`**, so
  `L-1` (≈ the trailing `</cot>` / `<|im_end|>` token) has **causally attended to the entire CoT**.
  Whether the matcher fires or falls back, it reads the *same* end-of-CoT position. **CoT content IS
  routed to the head's input.** The workflow's "Break 1: CoT never reaches the head — FATAL" is
  wrong on this data. (Verified by the assistant-message structure.)
- **token_acc collapse is SECOND-ORDER.** Real (alpha=1e-3 ⇒ CoT-CE is <1% of the gradient ⇒ the
  curve-MSE reshapes the shared backbone and the LM head collapses — RL_STRATEGY §9). But it only
  blocks the model from *generating* its own CoT; it does NOT block a *teacher-forced* CoT from
  reaching the head. So it is not why CoT is decorative.

### What IS the root cause (the real chain)
1. **The readout is saturated.** A best-case linear readout of the frozen bypass features already
   reaches the baseline (head_oracle CV-SRCC ≈ 0.514). The SFT head is not under-extracting — there
   is no linear headroom left in the features as they stand. RL/LTR on the head therefore can only
   re-weight a feature that is already at its linear ceiling ⇒ ≤noise gain (confirmed: rank_sft, the
   dense exact-gradient upper bound, found +0.016, CI includes 0).
2. **CoT adds no exogenous information.** The CoT is `f(video)` (Gemini described the same clip the
   backbone already encodes). By the data-processing inequality it cannot raise the information
   ceiling — at best it *re-expresses* video features.
3. **The eval structurally bypasses CoT**, so the 0.44/0.515 baseline is measured with an EMPTY CoT;
   and the prior bypass-vs-generate ΔIBS≈0 says that even a present CoT doesn't move the prediction.

**So:** the system is at a **feature/representation ceiling**, not an optimizer or a routing bug.
RL re-weights a saturated feature; CoT can't add information; the head already reads the end-of-CoT
state. token_acc collapse and the dead anchor are real artifacts but neither is the binding cause.

### The ONE thing this does NOT yet foreclose (the live question)
Serializing the model's reasoning into the context and re-reading it can, in principle, make latent
video information **more linearly accessible** (a representation/test-time-compute benefit, not new
information). That is the only un-refuted way CoT could help — and it is **directly testable**.

---

## Part C — The decisive probe (cheap, no training, ~1–2 GPU-hr on the idle box)

Extend `feature_dump.py` + `head_oracle.py` on frozen ckpt-225, K-fold CV on val, compare readouts:
- **L0** — linear ridge on EMPTY-CoT bypass `h_anchor` (reproduce ≈0.514).
- **N0** — 2–3 layer MLP probe on the SAME bypass features (linear-vs-nonlinear headroom).
- **G0** — linear ridge on `h_anchor` under **teacher-forced GROUND-TRUTH (Gemini) CoT** in the
  assistant span, single forward (NO `model.generate` — the generate→head path is broken, TMRoPE).

**This is the clean root-cause discriminator.** It separates "can the model produce CoT" (irrelevant
— teacher-forced) from "does in-context CoT move the readout" (the only open question).

**GO / NO-GO:**
- **N0 and G0 both ≈ 0.514 (CI overlaps)** ⇒ readout + CoT are both dead ⇒ **ship SFT + write the
  negative result. No retrain.** (This is the modal outcome predicted by the saturated linear oracle.)
- **G0 beats 0.514 CI-clean, IBS not worse** ⇒ in-context CoT *does* carry extractable signal ⇒
  greenlight ONE time-boxed Path-1 retrain (fix the anchor properly + GradNorm/phased alpha so
  token_acc survives AND curve-MSE holds), then re-measure.

---

## Part D — Recommendation

1. **Lock the fallback now:** SFT bypass baseline (0.4393 / 0.00672, beats climatology) IS a
   shippable CS224R deliverable; the RL story (head-PG REINFORCE + the dense-gradient upper-bound
   null + level-invariance calibration rot + the head_oracle ceiling) is a rigorous negative result.
2. **Run the probe tonight on the idle H100** — it gates whether any CoT path is worth GPU.
3. **Do NOT** start the hybrid (CoT-RL) before the deadline: it needs the anchor re-architected, the
   broken generate→head path repaired, a new trainer, and a warm-start — infeasible in 4 days. Park
   as future work, gated on a GO from the probe.
4. **Stop the idle box** when not running the probe; **kill the un-guarded `reinforce` run** (it's
   over-optimizing a saturated, calibration-rotting objective).

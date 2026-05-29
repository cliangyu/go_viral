# V8-Next: Big Picture (First Principles Synthesis)

**Date**: 2026-05-28
**Author**: Leon + Claude
**Purpose**: Step back from the math, the wandb traces, the reward formulas, and lay out what we're actually trying to build, why, and what the structural choices imply.
**Method**: First principles. Strip away inherited framing from the proposal/milestone. Start from the customer decision and trace back.

---

## 1. The actual problem (in customer terms)

Vispie wants to answer this question for an agency or creator:

> "I have N creative variants of an ad. Which K should I run?"

This is a **ranking** decision. Not a calibration decision. The customer doesn't care if our prediction of "this ad's retention curve is 0.42 at second 10" is exactly right. They care if our ordering of the N creatives matches their actual run-time performance.

The decision is also **per-time-horizon**: agencies optimize for different view-completion targets (3s, 6s, 15s, 30s) depending on platform and KPI. So the underlying primitive is *"at fixed t, which ads retain more?"* — cross-ad ranking per second.

---

## 2. What's needed mechanistically to solve it

Predicting "at second 10, ad A retains better than ad B" requires the model to:

1. **Perceive video content of each ad** — what's on screen, who's there, what's happening, what's the motion energy
2. **Perceive audio content** — voice quality, music energy, beat, sound design
3. **Reason about retention drivers** — hook quality at t=0, pacing, payoff timing, content-audience fit, production polish, reveal/twist structure
4. **Calibrate relative magnitude** — even if A has a "stronger hook" than B in our perception, by how much does that translate to retention difference?

Steps 1–2 are perception. Step 3 is *high-level semantic reasoning over multimodal input*. Step 4 is *cross-ad comparison capability*, which emerges from exposure to many ads and learning what's relatively predictive.

---

## 3. Why a VLM — first principles

Steps 1–2 (perception) could be done by a vision tower + audio encoder. Why use a Video Language Model?

Because step 3 (reasoning about retention drivers) needs a **conceptual vocabulary** that:
- Knows what a "hook" is and recognizes its features
- Knows the difference between "fast cuts" and "slow buildup" pacing
- Knows what kinds of content payoffs work for different audiences
- Has priors about engagement patterns from training data

This vocabulary lives inside a pretrained LM's parameter space — the body's hidden representations encode it. A VLM can route video+audio percepts INTO this conceptual vocabulary because of its vision-language alignment training.

**The key insight**: we don't need to OUTPUT language. But we need the body to INTERNALLY route through language-grade conceptual representations. The act of being able to generate text about a video forces the body to project video → semantic concepts. If we destroy that routing capability, we lose access to the conceptual vocabulary, even if we never wanted text output.

This is why language matters even though our final output is R(t): **language is the carrier of the reasoning we depend on**.

---

## 4. What V8 has actually built

| Component | Built? | Healthy? |
|---|---|---|
| Vision perception | yes (Qwen2.5-Omni vision tower) | likely yes |
| Audio perception | yes (Qwen2.5-Omni audio tower) | likely yes |
| Conceptual reasoning vocabulary (LM body) | inherited from pretraining | **destroyed by step 152** (token_acc 0.036) |
| Cross-ad calibration | trained via per-ad MSE | weak — see §6 |
| Retention head (h_last → R(t)) | yes | yes, head MSE eval = 0.0161 |

Two of the five are broken. The diagnosis: **per-ad MSE loss bias**.

---

## 5. The fundamental misalignment

We are training on a **per-ad calibration** loss (`_masked_mse(R̂, R)`) and a **CoT-distillation** auxiliary loss (`α · CE(CoT)`), to optimize for a **cross-ad ranking** decision.

These three are not the same objective. They are related but not aligned:

| Objective | Loss term | Optimizes |
|---|---|---|
| Per-ad calibration | `_masked_mse` | Predict THIS ad's curve correctly |
| CoT distillation | `α · CE(CoT)` | Mimic Gemini's reasoning text |
| Cross-ad ranking (customer) | NONE | Rank N ads at fixed t correctly |

The customer-facing objective doesn't appear in our loss anywhere. We are optimizing two proxies and hoping they generalize. They don't, structurally:

- Per-ad MSE regresses toward the mean curve when ads are diverse → it sacrifices cross-ad discrimination to be "close enough" to everyone
- CoT distillation forces text mimicry but doesn't connect to retention outcome — it's a free-floating language anchor

The V8 trajectory we saw in wandb (eval/loss 0.027 → 0.016, token_acc 0.50 → 0.036 between step 101 and 152) is **the model trading reasoning capacity for per-ad memorization** — improving the wrong objective at the cost of the right one.

---

## 6. The cross-ad signal lives in language reasoning

This is the move that ties everything together:

Cross-ad discrimination requires the model to perceive that *"ad A has a stronger hook than ad B"* — a comparative semantic judgment. This judgment lives in the conceptual vocabulary (§3). A vision-only encoder can extract features ("there's a face at t=0, motion is fast") but cannot compare them semantically across ads without a reasoning layer.

So:
- **Cross-ad ranking** is enabled by reasoning
- **Reasoning** is carried by language-grade representations
- **Language-grade representations** are what V8 SFT is destroying

This is why V8 SFT can improve per-ad IBS while damaging the actual capability we need.

---

## 7. The corrective: RL on the right objective

We need a training signal that:
- Directly rewards cross-ad ranking (not per-ad calibration)
- Operates through the language pathway (so it preserves the reasoning carrier)
- Optionally has a per-ad calibration anchor for variance reduction

Cross-ad per-second SRCC on parsed text completions does all three:
- Reward = mean over t of SRCC across ads in the batch → **cross-ad ranking objective**
- Reward computed from text completion → **language pathway must stay alive to score non-zero**
- Optional composite with `α · (1−IBS)` → calibration anchor

This is the design we converged on. But there are pre-requisites we hadn't articulated:

1. The RL initialization checkpoint must have language alive (token_acc > 0.4)
2. The RL training must include a KL anchor to a language-alive reference (prevents collapse during RL)
3. We need empirical evidence that the early ckpt (with language) ranks ads better than the late ckpt (with destroyed language)

---

## 8. What we have empirically validated

| Claim | Evidence | Status |
|---|---|---|
| Within-ad Spearman is degenerate | machine-precision across 200 val ads × 200 random preds | ✓ verified |
| Two strict-monotone-decreasing sequences (even crossing) → ρ_S = 1 | 1000 random crossing pairs, all rho=1 | ✓ verified |
| `_masked_mse` = IBS metric per ad | bit-identical at float64 | ✓ verified |
| GRPO `1−IBS` reward is detached Python float, head W not updated | source code direct | ✓ verified |
| Cross-ad rank gradient ⊥ SFT MSE gradient | cos sim = −0.0116 | ✓ verified |
| Cross-ad signal in val_200: t ∈ [1, 30] discriminative | gap F empirical | ✓ verified |
| Gemini CoT data is retention-relevant (not Scenario B) | hand-inspected 5 examples | ✓ verified |
| V8 SFT destroys language: token_acc 0.50→0.036 step 101→152 | wandb run b5rhw2te | ✓ verified |
| V8 SFT inferred CoT contribution = 97.7% of train loss but model can't predict CoT | wandb loss split analysis | ✓ verified |

## 9. What we have NOT yet validated (open empirical questions)

| Open question | Test | Status |
|---|---|---|
| Can ckpt-75 generate parseable JSON? | model.generate() on 10 val ads, check parse rate | RUNNING (in background) |
| What is V8 ckpt-75's per-ad IBS vs B1 baseline on val_200? | eval_ibs.py --limit 200 | RUNNING (in background) |
| Does V8 step 152 beat step 50 on per-ad IBS but lose on cross-ad SRCC? | run both ckpts, compute both metrics | PENDING |
| Does inference-time generated CoT actually help head h_last or hurt? | compare teacher-forced vs generated CoT IBS | PENDING |
| Does a pure vision+MLP baseline (no LM body) match V8 IBS? | needs B0 baseline training | DEFERRED (out of scope this week) |

The first two are running right now on the 2-card box. Will close the next two before launching E3 training.

---

## 10. The reframed proposal

Putting it all together:

### What we ship

| Experiment | Purpose | Status |
|---|---|---|
| E1 = V8 SFT (already running) | The "current approach" baseline | training (step ~167 of cosine decay) |
| E2 = V7-style GRPO with `1−IBS` reward | "Redundant RL" baseline | code exists, can re-run |
| E3 = GRPO with composite cross-ad SRCC reward | The contribution — direct cross-ad optimization | TO BUILD |

### Required mechanisms in E3 (gleaned from this synthesis)

1. **Reward** (from PROPOSAL.md): `r = α·(1−IBS) + β·S_LOO + γ·format − η·KL(π||π_ref)`
   - α=0.2, β=0.7, γ=0.1, η=0.02
2. **Initialization**: V8 step 50–75 ckpt (token_acc > 0.4), NOT the final V8 ckpt
3. **KL reference**: V8 step 50 or base Qwen2.5-Omni
4. **Per-rollout reward**: leave-one-out form (Form 2a in PROPOSAL.md) — no trainer changes needed

### What the report claims

> SFT with per-ad MSE loss structurally underfits the cross-ad ranking decision that drives customer value. Late-stage SFT trades language-grade reasoning capacity (token_acc 0.50→0.036 in 50 steps) for per-ad calibration improvement — improving the wrong objective. We propose RL with a cross-ad SRCC reward, anchored to a language-alive checkpoint via KL, as the structurally correct training signal. We validate the math (cross-ad rank gradient ⊥ SFT MSE gradient, cos sim = −0.0116) and the empirical opening (V8 destroys language while improving IBS), and provide a reference implementation. [Empirical E3 results follow if training completes by 6/3.]

This is a defensible CS224R submission. The math, the diagnosis of V8's failure mode, and the reference implementation are the contribution. The running RL is the demonstration.

---

## 11. The risks if our reframing is wrong

| If we're wrong about... | Then... | Mitigation |
|---|---|---|
| Customer decision being cross-ad ranking | We're optimizing the wrong metric for Vispie | Verify with Vispie product team |
| Language being the reasoning carrier | KL anchor and ckpt selection are unnecessary | Run B0 baseline (vision-only) — but slow |
| CoT data being useful | We're propping up dead weight | Hand-inspection says it's good; could be wrong at scale |
| Cross-ad SRCC reward producing positive RL signal | E3 fails to beat E1 | Math contribution stands even if empirical fails |

---

## 12. What we do NEXT (concrete, in order)

1. **Wait for ckpt-75 eval to finish** (running): get per-ad IBS vs B1.
2. **Run ckpt-75 generation test**: 10 val ads, check parse rate of generated text. **This is the gate for E3 — if ckpt-75 can't generate parseable JSON, RL doesn't work from this ckpt.**
3. **If gate passes, queue up step-50 ckpt download** and run the same two tests on it.
4. **Hand the data to Leon for the empirical decision**:
   - Best RL init ckpt
   - Whether E3 design (composite reward) is approved
5. **Implement `ttcc_composite_reward_plugin.py`** (~120 LOC + unit tests)
6. **Launch E3 training** on 16×H100 once V8 SFT finishes

The entire 4-day plan in PROPOSAL.md is unchanged in structure; this synthesis just makes the *reasons* explicit and adds the KL/ckpt-selection mechanisms.

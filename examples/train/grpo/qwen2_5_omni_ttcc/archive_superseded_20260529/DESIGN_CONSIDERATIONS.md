# V8-Next Design Considerations

**Date**: 2026-05-28 (working notes; Leon + Claude conversation)
**Status**: Living document — captures the conceptual moves leading to the final reward design
**Companion docs**: [PROPOSAL.md](PROPOSAL.md), [VERIFICATION_WALKTHROUGH.md](VERIFICATION_WALKTHROUGH.md), [BUSINESS_RATIONALE.md](BUSINESS_RATIONALE.md)

This doc captures the design considerations that aren't obvious from the math verification or the business case alone — they emerged from first-principles questions Leon raised about why we're using a VLM at all.

---

## Consideration 1: Are we actually using Video and Language?

### Leon's framing

> 我们用的是 Video Language Model。如果我们这其中连 Video 都没有了，或者说我们整个过程中连 Language 都没有了，我们是否有用 Video Language Model 的必要呢？

Translated: if our pipeline doesn't actually USE video or language, why pay for a VLM?

### Three training modes and what each activates

| Mode | Video used? | Language used? | Head used? | VLM justified? |
|---|---|---|---|---|
| **Pure head-MSE SFT** (α=0) | ✓ | ✗ (catastrophically destroyed) | ✓ | No — vision tower + MLP would do |
| **head-MSE + α·CoT CE** (V8, α=1e-3) | ✓ | partial (and destroyed by step 152) | ✓ | depends on α; current α insufficient |
| **GRPO with parsed-text reward** | ✓ | ✓ (must generate text) | ✗ (frozen by absence) | yes, but head wasted |
| **Ideal** | ✓ | ✓ | ✓ | ✓ |

### Direct evidence from V8 wandb (run `liangyuch/ttcc/b5rhw2te`)

| step | epoch | eval/loss (head MSE) | eval/token_acc |
|---|---|---|---|
| 50 | 0.18 | 0.0316 | **0.5000** |
| 101 | 0.36 | 0.0269 | **0.5000** |
| 152 | 0.54 | **0.0161** | **0.0361** |

**Phase transition**: between step 101 and step 152, eval/token_acc went from 0.50 → 0.036. The model lost ~93% of its language modeling ability in 50 steps while improving head MSE by ~40%.

**Loss-split inference** at step 167:
- `train/loss = 0.7003` (total)
- `eval/loss = 0.0161` (head MSE proxy — eval-time has no CoT CE)
- Inferred `α·CoT_CE ≈ 0.6842` (97.7% of train loss)
- If α=1e-3, then CoT CE per ad ≈ 684 nats ≈ 3.4 nats/token (essentially random)

The CoT CE term DOMINATES train loss but the model has STOPPED actually predicting CoT well. It's burning 97% of training compute on a signal it can no longer act on.

### Conclusion for Consideration 1

V8 is not currently a healthy VLM. The Language pathway is destroyed. We have to choose:

- (a) Accept VLM destruction → drop CoT and use vision tower + MLP (simpler model)
- (b) Preserve VLM → mechanisms to stop language collapse (KL anchor, higher α, earlier ckpt)
- (c) Hybrid — fine for E1 baseline, but RL initialization must come from a language-alive ckpt

We choose (b)+(c): the RL story requires language to be alive at inference time (parsed-text reward needs parseable text). Therefore RL must start from a ckpt before the phase transition (step 75 or earlier).

---

## Consideration 2: Reasoning ≠ Language, but Language carries reasoning

### Leon's framing

> 我们的 Video Language Model 对 Retention 应该是有用的。也就是说，要用到 Video Perception 和 Language Reasoning 的能力，才能够把这个 Retention 给做好。这个 Reasoning 不一定非要以 Language 的形式呈现，但我们假设 Language 里面包含了一些对 Retention 有用的信号。

### Mechanistic interpretation

Reasoning happens in the body's hidden representations. Language is one CHANNEL that exposes these representations, but the representations themselves are also accessed by the retention head (via h_last).

Why language matters even if we don't OUTPUT pretty text:
- The pretrained VLM's vocabulary encodes retention-relevant concepts: hook, pacing, payoff, transition, content-type, production-quality, audio-appeal
- These concepts live in the body's parameter space because the body was trained on language data
- When the body is shaped purely by retention MSE, the parameter space drifts toward "retention encoding" and AWAY from "concept representation"
- The drift destroys the concepts even though we never asked the model to output them

This explains why eval/token_acc → 0.036 matters even if our final output is just R(t):
- Loss of token_acc = loss of concept-grade representations in the body
- The head reads h_last from a body whose internal concepts have collapsed
- h_last still has SOME signal (eval/loss = 0.0161 is good locally), but it has lost generalization power because the concept hierarchy is gone

### Verifiable prediction (untested as of writing)

If language=reasoning carrier hypothesis is correct:
- V8 step 152 (token_acc=0.036, eval/loss=0.0161) has memorized per-ad shapes via head W
- V8 step 50 (token_acc=0.50, eval/loss=0.0316) has retained reasoning
- On **per-ad IBS** (in-distribution metric): step 152 wins
- On **cross-ad SRCC** (requires comparing across ads = reasoning): step 50 should be COMPARABLE OR BETTER than step 152

This is a falsifiable test. We run it.

---

## Consideration 3: Is the CoT data itself useful?

### Leon's framing

> 也有可能是我们当前 CoT training data 就有问题，对 retention 也没有什么帮助

### Three scenarios (named for tracking)

| Scenario | CoT data quality | Training preserves language? | What to do |
|---|---|---|---|
| **A** | useful | NO (V8 destroys it) | KL anchor + start RL from early ckpt |
| **B** | NOT useful (shallow / generic) | doesn't matter | drop CoT, simpler architecture |
| **C** | partially useful | partially destroyed | both fixes |

### Empirical inspection of `ttcc_train_with_cot.jsonl` (35,793 examples)

Sampled 5 CoTs (indices 0, 100, 1000, 10000, 30000). Format:

```
<cot>0s | visual: ... | audio: ... | impact: gain/hold/lose: <reasoning>
1-2s | ... | ...
...
Shape: ...</cot>
```

**Per-second timeline** with explicit `gain`/`hold`/`lose` annotations — DIRECTLY retention-relevant. Reasoning specifies:
- Hook quality at t=0
- Visual content per second
- Audio content per second
- Transition costs ("transition to app demo increases drop-off")
- Reward delivery moments ("4.6 million yen reveal — reward delivery")
- Closing patterns ("standard CTA drop-off")

**Verdict**: the CoT data is **high quality** and **retention-relevant**. Gemini is producing the right shape of reasoning. Not Scenario B.

### Caveat — discrimination calibration

Looking at R[1] (retention at second 1) across the 5 examples:
- Ex 0 (Alphard ad, "strong hook"): R[1] = 0.23
- Ex 100 (RSA marketing, "strong hook"): R[1] = 0.22
- Ex 1000 (slot machine, "strong hook"): R[1] = 0.20
- Ex 10000 (Spanish lifestyle, "strong hook"): R[1] = **0.61**
- Ex 30000 (Thai medical, "strong hook"): R[1] = 0.31

All CoTs say "strong hook" but R[1] varies 3x. The CoT correctly identifies retention DRIVERS but doesn't perfectly calibrate retention MAGNITUDE. This is fine — the CoT is meant to be a reasoning scaffold, not a direct predictor.

### Conclusion for Consideration 3

We are in Scenario A. The CoT data is good. The training is destroying the carrier.

---

## Consideration 4: ckpt selection for RL initialization

### What the data say

| ckpt | eval/loss (IBS) | eval/token_acc | RL feasibility |
|---|---|---|---|
| step 50 | 0.0316 | 0.50 | ✓ healthy language, RL likely works |
| step 75 | ~0.030 (interpolated) | ~0.48 | ✓ same |
| step 101 | 0.0269 | 0.50 | ✓ same |
| step 152 | 0.0161 | **0.036** | ✗ language destroyed, RL on parsed-text reward will hit format-cliff catastrophically |
| step ~165 (current latest) | ~0.016 | ~0.005 | ✗ worse |

The IBS improvement from step 101 → 152 (0.0269 → 0.0161) is bought with language destruction.

### Three-way trade-off

We need to choose the RL initialization ckpt by:
1. **Local IBS skill** (higher is better — but post-RL IBS matters more than pre-RL IBS)
2. **Language alive** (token_acc > 0.4 — required for RL format compliance)
3. **Reasoning capacity** (proxied by token_acc; required for cross-ad SRCC reward to teach anything new)

step 75 (token_acc ~0.48, eval/loss ~0.030) is the sweet spot. step 152 is overfit.

### Open question (testable)

Does the IBS improvement at step 152 come from:
- (a) genuine generalization improvement (then step 152 wins everywhere including SRCC)
- (b) per-ad memorization (then step 152 wins on IBS but LOSES on cross-ad SRCC)

This is the verification we're running.

---

## Consideration 5: What the RL reward design must achieve

Synthesizing Considerations 1-4, the RL reward design must:

1. **Operate on parsed text** (because that's where the language pathway lives; this preserves the reasoning carrier through training)
2. **Reward cross-ad discrimination** (because SFT structurally underfits this and it's the business-relevant signal)
3. **Anchor to a language-alive reference** (KL term to base model or to V8 step ~50 — prevents drift back into language collapse)
4. **Start from a language-alive ckpt** (step 50–75 range)
5. **Carry a small calibration anchor** (1−IBS term in composite reward — for variance reduction, not for signal)
6. **Carry a format reward** (prevents parse-cliff)

The composite reward proposed in [PROPOSAL.md](PROPOSAL.md) §5.1 covers (1), (2), (5), (6). This doc adds (3) and (4) — KL anchor and ckpt selection.

### Updated composite reward (revision)

$$r_i = \alpha \cdot (1 - \text{IBS}_i) + \beta \cdot S_i^{\text{LOO}} + \gamma \cdot \text{format}_i - \eta \cdot \text{KL}(\pi_i || \pi_{\text{ref}})$$

with **α = 0.2, β = 0.7, γ = 0.1, η = 0.02** as starting weights, and π_ref = V8 step ~50 (language-alive).

The KL term is added in this revision (Consideration 2 implies it's load-bearing for preserving the reasoning carrier).

---

## Considerations 6-N: TBD

Open considerations that may yet enter the design:

- **6: Does CoT help or hurt at inference time?** During eval, the model auto-generates CoT. If the generated CoT is degenerate (Gemini-mimicry but content-wrong), h_last may be worse than no-CoT. Testable.
- **7: Differentiable RL through the head?** If we use head(h_last) as the predicted curve instead of parsed text, we get differentiable gradient. But this loses the language pathway. Same trade-off as before.
- **8: Wanjia's parallel path.** Wanjia is doing hazard-head SFT separately on Modal. Does her ckpt differ from V8 in language preservation? If yes, comparison adds rigor.

---

## Decision log

| # | Decision | Reasoning | Date |
|---|---|---|---|
| D1 | Use parsed-text reward (not differentiable head reward) for E3 | Preserves language pathway; aligns with VLM justification | 2026-05-28 |
| D2 | Start E3 from V8 step ~75, not the final V8 ckpt | Token_acc must be > 0.4 for RL format compliance | 2026-05-28 |
| D3 | Add KL anchor (η=0.02) to V8 step ~50 reference | Prevents language drift back into collapse | 2026-05-28 |
| D4 | Composite reward (not pure SRCC) | Stability/variance reduction via dense IBS anchor | 2026-05-28 |
| D5 | Use cross-ad SRCC over t ∈ [1, 30] (verified discriminative range) | Gap F empirical | 2026-05-28 |

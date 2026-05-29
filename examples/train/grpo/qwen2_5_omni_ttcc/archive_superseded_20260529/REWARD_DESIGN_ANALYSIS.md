# Reward Design — Rigorous Math Analysis

**Author:** Claude (Sonnet 4.6) for Leon
**Date:** 2026-05-28
**Purpose:** Verify which claims about reward design (Spearman, 1−IBS, MSE) are objectively true, which are false, and what that implies for what we should actually do.

This document examines five claims that have been made (by me, by the proposal, or by the milestone). For each, we state it precisely, prove or disprove it, and state the implications.

---

## Setup

We have $N$ ads indexed by $i$. For ad $i$:

- True retention curve: $R_i(t)$ for $t = 0, 1, \ldots, T_i$, with $R_i(0) = 1$ and $R_i(t) \in [0, 1]$
- Predicted retention curve: $\hat R_i(t)$
- Hazard parameterization (V8): $\hat R_i(t) = \exp\!\big(-\sum_{s \le t} \mathrm{softplus}(z_i(s))\big)$ where $z_i(s)$ is a learned function of the model's hidden state at $\langle\text{/cot}\rangle$
- $T_i \in [5, 60]$, with median $\approx 24$ in our train distribution

**Key structural fact**: under the hazard parameterization, $\hat R_i(t+1) \le \hat R_i(t)$ for all $t$. Predicted curves are monotone non-increasing **by construction**.

**Empirical fact**: ground-truth $R_i(t)$ is *approximately* monotone non-increasing. It has occasional small bumps from finite-sample noise (R is the empirical fraction from a finite audience). Ties (plateaus) are common.

---

## Claim A — Within-ad Spearman on hazard-parameterized predictions

### A.1 Statement (precise)

Let $\rho_S(\hat R_i, R_i)$ denote the Spearman rank correlation between the two length-$(T_i + 1)$ vectors $\hat R_i$ and $R_i$. Spearman is computed as the Pearson correlation of the rank vectors (using mid-rank for ties).

**Claim (mine, earlier):** Under hazard parameterization, $\rho_S(\hat R_i, R_i) \equiv 1$ identically.

**Claim (milestone):** $B_1$ (the train-mean baseline) achieves $\bar\rho_\text{shape} = +0.95$.

These cannot both be true.

### A.2 Proof — my claim is FALSE in general; the milestone claim is consistent with a more nuanced truth

**The truth is intermediate**: if both $\hat R_i$ and $R_i$ are *strictly* monotone non-increasing (no ties), then yes, $\rho_S = 1$ exactly. But ties break this.

**Counterexample showing $\rho_S < 1$ is possible**:

$$R = [1.0, 0.5, 0.5, 0.2], \qquad \hat R = [0.8, 0.6, 0.4, 0.3]$$

Both are monotone non-increasing. Compute ranks (largest gets highest rank, ties get mid-rank):

- Ranks of $R$: $[4, 2.5, 2.5, 1]$ — the 0.5 tie averages ranks 2 and 3
- Ranks of $\hat R$: $[4, 3, 2, 1]$ — no ties

Pearson on these rank vectors:

$$\bar r_R = \frac{4 + 2.5 + 2.5 + 1}{4} = 2.5, \qquad \bar r_{\hat R} = 2.5$$

$$\text{cov} = \frac{1}{4}\sum (r_{R,k} - 2.5)(r_{\hat R,k} - 2.5) = \frac{1.5 \cdot 1.5 + 0 + 0 + (-1.5)(-1.5)}{4} = \frac{4.5}{4} = 1.125$$

$$\text{var}(r_R) = \frac{1.5^2 + 0 + 0 + 1.5^2}{4} = 1.125, \qquad \text{var}(r_{\hat R}) = \frac{1.5^2 + 0.5^2 + 0.5^2 + 1.5^2}{4} = 1.25$$

$$\rho_S = \frac{1.125}{\sqrt{1.125 \cdot 1.25}} \approx 0.949$$

So **ties in the truth drag Spearman down from 1**.

This is consistent with the milestone's empirical observation: B1 scores 0.95, not 1.0, because ground-truth $R_i$ has ties.

### A.3 The real claim that holds rigorously

**Refined claim (CORRECT):** Under hazard parameterization, $\rho_S(\hat R_i, R_i)$ depends only on the tie pattern of the ground truth $R_i$. It is **independent of the predicted values $\hat R_i$** as long as $\hat R_i$ is strictly monotone non-increasing (no predicted ties).

**Why**: when $\hat R_i$ is strictly monotone, its rank vector is fixed at $[T_i + 1, T_i, \ldots, 1]$ regardless of the specific predicted values. So $\rho_S$ depends only on the rank vector of $R_i$.

**Implication**: within-ad Spearman is **uninformative about the predicted curve** when the prediction structure is monotone. The reward gives the same value for any monotone $\hat R_i$ — even one that's catastrophically wrong (e.g., $\hat R_i(t) = 1 - 0.001 t$ when the true is sharply decaying).

This is a real degeneracy. Not "identically 1," but **identically the same value** across all monotone predictions for a given ad. That value happens to depend on the ground truth's tie pattern.

### A.4 Verdict on Claim A

| Claimant | Claim | Status |
|---|---|---|
| Me (earlier) | $\rho_S \equiv 1$ for monotone predictions | **WRONG** |
| Milestone | B1 scores $\bar\rho_\text{shape} = 0.95$ | **CONSISTENT** with refined claim |
| Both | Within-ad Spearman cannot distinguish good vs. bad monotone predictors | **CORRECT** but for a more subtle reason than I gave |

---

## Claim B — `1 − IBS` reward is gradient-equivalent to SFT MSE

### B.1 Statement (precise)

Define:

$$\text{IBS}_i = \frac{1}{T_i + 1}\sum_{t=0}^{T_i} \big(\hat R_i(t) - R_i(t)\big)^2$$

The SFT loss in our register.py is (after unmasking and rescaling):

$$\mathcal{L}_{\text{SFT}}^{(i)} = \sum_{t} m_i(t) \cdot \big(\hat R_i(t) - R_i(t)\big)^2$$

where $m_i(t) \in \{0, 1\}$ is a mask indicating which time steps to include. For ads where $m_i(t) = 1$ for $t \in [0, T_i]$, this is $(T_i + 1) \cdot \text{IBS}_i$.

The proposed RL reward is $r_i = 1 - \text{IBS}_i$.

**Claim (mine, earlier):** $\partial r_i / \partial \theta = -\partial \mathcal{L}_{\text{SFT}}^{(i)} / \partial \theta$ up to scale, so the RL gradient direction matches SFT gradient direction.

### B.2 Proof — TRUE for deterministic head, FALSE for token-policy RL

The subtle question is *what gets sampled* in the RL setup.

**Case B.2.a — Deterministic head, no CoT sampling (degenerate setup)**

If the head is deterministic given the input and no CoT tokens are sampled, then:

$$\frac{\partial r_i}{\partial \theta} = -\frac{\partial \text{IBS}_i}{\partial \theta} = -\frac{1}{T_i + 1}\sum_t 2(\hat R_i(t) - R_i(t)) \cdot \frac{\partial \hat R_i(t)}{\partial \theta}$$

$$\frac{\partial \mathcal{L}_{\text{SFT}}^{(i)}}{\partial \theta} = \sum_t 2(\hat R_i(t) - R_i(t)) \cdot \frac{\partial \hat R_i(t)}{\partial \theta}$$

These are equal up to the constant factor $-1/(T_i + 1)$.

**Verdict for Case B.2.a:** RL gradient = SFT gradient (rescaled). RL with this reward is mathematically equivalent to SFT with a learning rate rescaling. **RL adds nothing.**

**Case B.2.b — CoT token sampling, with deterministic head conditional on hidden state**

Now the policy is over CoT tokens $\tau$. The reward depends on which CoT $\tau$ is sampled (because different CoTs lead to different hidden states $h[\langle\text{/cot}\rangle | \tau]$, which lead to different $\hat R_i$).

The policy-gradient estimator (REINFORCE, GRPO, etc.) is:

$$\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\!\big[\nabla_\theta \log \pi_\theta(\tau) \cdot A(\tau)\big]$$

where $A(\tau) = r(\tau) - b$ is the advantage (some baseline $b$).

Now there are **two distinct gradient paths**:

**(B.2.b.i) — Gradient through the LM body via token log-probs**

The $\nabla_\theta \log \pi_\theta(\tau)$ term flows through every token-emission decision. For each CoT token chosen, we get a gradient that says: "increase the log-prob of this token if its sampled trajectory had higher-than-baseline reward, decrease otherwise."

**This gradient is NOT equal to SFT MSE gradient.** SFT MSE has no per-token signal — it has a single hidden-state error signal flowing into the head. RL via CoT sampling re-allocates credit to each token-emission decision.

This is the gradient signal the team is actually banking on.

**(B.2.b.ii) — Gradient through the head via R̂**

For each rollout, we also get a "head-side" gradient: how to adjust the head parameters given the hidden state and the reward. For a fixed sampled trajectory $\tau$ with reward $r(\tau)$:

$$\frac{\partial r(\tau)}{\partial W_{\text{head}}} = -\frac{1}{T_i + 1} \sum_t 2(\hat R_i^\tau(t) - R_i(t)) \cdot \frac{\partial \hat R_i^\tau(t)}{\partial W_{\text{head}}}$$

This is the per-rollout SFT MSE gradient. RL with K rollouts gives an advantage-weighted version of this. **For the head, RL converges in expectation to the same gradient direction as SFT.**

### B.3 What this means concretely

- **Gradient on the retention head**: RL with `1 − IBS` reward is essentially noisy SFT. No new signal.
- **Gradient on the LM body (via CoT tokens)**: RL with `1 − IBS` reward provides a real policy gradient over which tokens to emit. The signal IS new vs. teacher-forced SFT — RL allows exploration of CoT space to find token sequences that result in better predictions.

But there's a catch: **the signal on the LM body is `r(τ) − baseline`, where `r(τ) = 1 − IBS(τ)`**. So the policy gradient is "increase probability of tokens that lead to lower IBS." That IS the same training signal as "increase probability of tokens whose generated continuations have low MSE." In expectation, this trains the LM to generate CoTs that *correlate with low-MSE predictions*.

**The deep question**: is "minimize MSE" a different optimization target when applied via a CoT policy gradient vs. when applied via teacher-forced SFT?

**Answer**: yes, slightly — RL with policy gradient explores CoT space (off-policy from the Gemini-distilled CoT), while SFT enforces matching the distilled CoT. Different things can happen:

- SFT trains the LM to *imitate* the Gemini CoT → distribution-match the demonstrations
- RL trains the LM to *generate any CoT that lowers MSE* → optimize the downstream objective directly

These can diverge if the LM finds an unexpected token sequence that yields low MSE. But the **scalar signal driving both is the same** (MSE → IBS), and the search space the LM can explore is constrained by the KL anchor.

### B.4 Verdict on Claim B

| My claim | Status |
|---|---|
| RL with `1 − IBS` is gradient-equivalent to SFT for the head | **TRUE** (in expectation) |
| RL with `1 − IBS` is gradient-equivalent to SFT for the LM body | **FALSE** — they are different training signals, though both ultimately minimize MSE |
| RL with `1 − IBS` cannot improve over SFT in expectation | **OVERSTATED** — it can find different LM token sequences that lower MSE, which SFT cannot do |
| RL with `1 − IBS` is unlikely to significantly improve over SFT for a well-distilled CoT | **PROBABLY TRUE** — high noise, marginal gain |

So I was **wrong to claim flat gradient equivalence**. The right statement is: "the scalar signal is the same (MSE), so RL with this reward cannot optimize a meaningfully different objective from SFT. RL may find slightly different solutions due to exploration, but the optimum is the same."

This is still a strong reason to want a different reward. But it's not the slam-dunk "RL ≡ SFT" claim I was making.

---

## Claim C — IBS is a "strictly proper" scoring rule

### C.1 Statement (precise)

A scoring rule $S(\hat p, p)$ is *strictly proper* if, for any true distribution $p$, the unique minimizer over $\hat p$ is $\hat p = p$ — that is:

$$\arg\min_{\hat p} \mathbb{E}_{X \sim p}\!\big[S(\hat p, X)\big] = p$$

The Brier score $S(\hat p, x) = (\hat p - x)^2$ is strictly proper for binary outcomes. The IBS we use is the time-averaged Brier score:

$$\text{IBS}(\hat R, R) = \frac{1}{T+1}\sum_t (\hat R(t) - R(t))^2$$

If we treat $R(t)$ as a probability and $\hat R(t)$ as an estimate, IBS at each $t$ is the Brier score for the binary outcome "viewer still watching at second $t$."

### C.2 Proof — TRUE in standard formulation, NUANCED in our setting

**Standard textbook result**: Brier score is strictly proper. ✓

**In our actual setting**: $R(t)$ is not a single binary outcome — it's the empirical fraction over a finite audience. We're estimating the population retention probability. IBS minimization is "predict the true population retention" — also a strictly proper objective.

**Verdict for Claim C:** Correct. IBS is strictly proper and cannot be gamed by predicting any specific *form* of curve (e.g., constant, exponential).

### C.3 Why "strictly proper" doesn't save IBS from the RL problem

Strict propriety is about whether the optimum is unique. It doesn't mean IBS is the right loss for **learning the optimum**.

- IBS is **minimized at $\hat R = R$**, the true curve.
- SFT MSE is **minimized at $\hat R = R$**, the same true curve.
- Therefore SFT and `RL with 1−IBS` have the **same optimum**.

If we already have SFT pushing toward this optimum, "switch to RL with 1−IBS as reward" doesn't change the destination. It might change the path (more exploration), but not the goal.

For RL to be useful, the reward needs to push toward a **different optimum** that captures something MSE doesn't — for example, calibration, ranking, or downstream utility.

---

## Claim D — B1 baseline scores $\bar\rho_\text{shape} \approx 0.95$

### D.1 Statement (precise)

$B_1(t) = \frac{1}{N_{\text{train}}}\sum_{i=1}^{N_{\text{train}}} R_i(t)$ is the train-mean curve. This is itself monotone non-increasing (mean of monotone functions, on the support where each ad has data).

**Milestone claim**: $\bar\rho_\text{shape}(B_1) = \frac{1}{N_\text{test}}\sum_i \rho_S(B_1, R_i) \approx 0.95$.

### D.2 Verification — CONSISTENT with the rigorous math

From Claim A.3: when both $B_1$ and $R_i$ are monotone non-increasing, $\rho_S(B_1, R_i)$ depends only on the tie patterns. If $B_1$ has no ties (likely, since it's a mean over many ads) and $R_i$ has typical retention-curve tie patterns (some plateaus), the Spearman should land somewhere in the 0.9–0.99 range.

0.95 is plausible for typical TTCC curves with moderate tie density.

**Verdict for Claim D:** Empirically consistent with the rigorous math derivation.

---

## Claim E — A monotone predictor "trivially" beats Spearman

### E.1 Statement (precise)

**Milestone**: "any monotone-decreasing curve trivially yields $\bar\rho_S \approx +1.0$ ... so this metric cannot distinguish content-aware from content-blind predictors."

### E.2 Verification — TRUE in our regime, but for the reason in Claim A.3, not the reason the milestone implies

From Claim A.3 (refined): within-ad Spearman is **identically the same value** for all monotone predictors, regardless of how good they are at predicting the curve's *magnitude*. That value is determined entirely by ground-truth tie patterns.

So a content-aware monotone predictor and a content-blind monotone predictor get the **same Spearman score** for each ad. They cannot be distinguished by this metric.

**Verdict for Claim E:** TRUE. The metric is information-free for our prediction structure.

---

## Synthesis of which claims hold

| # | Claim | Source | Status |
|---|---|---|---|
| A1 | Within-ad Spearman ≡ 1 for monotone predictions | Me (earlier) | **FALSE** — depends on ground-truth tie pattern |
| A2 | Within-ad Spearman is information-free across monotone predictors | Refined from A1 + milestone | **TRUE** |
| B1 | RL with `1−IBS` reward is gradient-equivalent to SFT for the head | Me | **TRUE in expectation** |
| B2 | RL with `1−IBS` reward is gradient-equivalent to SFT for the LM body | Me | **FALSE** — policy gradient over tokens is a different signal |
| B3 | RL with `1−IBS` reward has the same OPTIMUM as SFT | Implicit | **TRUE** (same strictly-proper objective) |
| B4 | RL with `1−IBS` reward is "useless" | Me (earlier, sloppy) | **OVERSTATED** — it can find different LM token sequences, but cannot optimize a different objective |
| C | IBS is strictly proper | Milestone | **TRUE** |
| C.3 | "Strictly proper" doesn't justify IBS as an RL reward | Me | **TRUE** — strict propriety only guarantees the optimum is unique, not that RL adds value over SFT |
| D | $B_1$ scores $\bar\rho_\text{shape} \approx 0.95$ | Milestone | **CONSISTENT** with math |
| E | Monotone predictors can't be ranked by within-ad Spearman | Milestone | **TRUE** |

## What this means for what we should do

Given the rigorous math:

1. **Within-ad Spearman is dead.** Both my analysis and the milestone agree on this. ✓
2. **`1 − IBS` is not as broken as I claimed**, but it is fundamentally **the same OPTIMUM as SFT**. So if SFT is already approaching that optimum (V8 val_loss 0.027 suggests yes), RL with this reward will not improve the optimum — it will just take a noisier path to it.
3. **For RL to materially help, the reward must define a different OPTIMUM than SFT.** That requires the reward to be a different objective, not a scaled version of MSE.

### Concrete reward functions that satisfy "different optimum"

A reward function $r(\hat R, R)$ has a different optimum than SFT MSE if and only if:

$$\arg\max_{\hat R} \mathbb{E}_R[r(\hat R, R)] \neq \arg\min_{\hat R} \mathbb{E}_R[(\hat R - R)^2]$$

That is: the function that maximizes expected reward is not the same as the function that minimizes expected MSE.

Examples that satisfy this:

**E1 — Cross-ad rank reward at fixed second:**

$$r_\text{hook}(\hat R_i, R_i; \text{batch}) = \rho_S\!\big(\{\hat R_j(3)\}_{j \in \text{batch}}, \{R_j(3)\}_{j \in \text{batch}}\big)$$

Optimum is "predict $\hat R_i(3)$ such that the cross-ad ranking is preserved." This is NOT the same as predicting $\hat R_i(3) = R_i(3)$ for every $i$ — only the *ranking* matters, not the absolute values. A model that adds a constant offset to every prediction still gets reward 1, while incurring positive MSE.

**E2 — Calibration reward (ECE):**

For predicted values in a bin $[p, p + \delta]$, the calibrated reward is high when the empirical mean of $R_i(t)$ for predictions in that bin equals the bin center. SFT MSE doesn't require this.

**E3 — Top-K ad selection:**

$r = \text{mean true } R \text{ of top-K ads selected by } \hat R$. Maximizes ranking quality at the operational decision point. Not equal to MSE.

### What I now propose

Given the rigorous math:

1. **Use cross-ad Spearman at $t = 3$ (hook) and $t = T_i$ (completion) as the RL reward.** This is what the proposal originally said, and it satisfies the "different optimum" criterion that 1−IBS fails.

2. **Keep `1 − IBS` as a small auxiliary reward term** (weight 0.1–0.2) to prevent the policy from learning ranking-correct-but-magnitude-broken predictions. This is the "anchor" role.

3. **Drop within-ad Spearman entirely.** It is information-free.

Concrete reward:

$$r_i = \alpha \cdot \big(\rho_\text{hook}^\text{batch} + \rho_\text{comp}^\text{batch}\big)/2 \;+\; (1 - \alpha)(1 - \text{IBS}_i)$$

with $\alpha \in [0.7, 0.9]$. The first term contributes to the cross-ad ranking goal; the second term anchors absolute predictions to ground truth.

This **does** define a different optimum than SFT, because the cross-ad rank term is not MSE-equivalent.

## Open questions I haven't resolved

1. **Per-rollout vs. batch-level reward in GRPO**: cross-ad rank requires a batch of ads, which is awkward when GRPO samples K rollouts per prompt. Need to define carefully: is "batch" = set of distinct ads in the GRPO batch? Or = set of K rollouts for one ad? The former is correct semantically but requires a redesign of the GRPO loop.

2. **Whether RL on a token policy can effectively learn the cross-ad ranking signal**: the reward signal is at the batch level, but the policy gradient flows through individual tokens. Credit assignment from a single scalar batch reward to individual CoT tokens may have very high variance.

3. **Whether the eval_ibs.py implementation matches the milestone's text**: I have not verified whether the head is actually frozen during RL (as App C claims) or trainable (as the policy gradient interpretation would require).

## Action items

1. **Verify Claim B numerically**: implement both reward functions in a tiny synthetic setup, train for a few steps, compare gradient norms and direction. Confirm that `1−IBS` gradient really does match SFT MSE gradient in expectation.

2. **Verify Claim A numerically**: compute $\bar\rho_\text{shape}(B_1)$ on the actual val_200 set and confirm it's ~0.95.

3. **Implement the cross-ad rank reward** as a standalone Python function with unit tests. ~80 LOC.

4. **Test whether GRPO trainer supports batch-coupled rewards** or requires per-rollout rewards.

These are Day 1 PM tasks.

---

## Summary in one paragraph

I made two math errors in earlier discussion: (1) I claimed within-ad Spearman is identically 1 for monotone predictions — it's actually identically *the same value* across all monotone predictions (often around 0.95 due to ground-truth ties), which is the same conclusion but for a different reason. (2) I claimed `1−IBS` reward is "the same as SFT MSE" — this is true for the head's gradient but not for the LM body's policy gradient. However, the **optimum** is the same as SFT, so RL with `1−IBS` reward cannot find a fundamentally different solution. The milestone correctly identified Spearman as broken but landed on a reward whose optimum is identical to SFT's. The right reward is **cross-ad** Spearman at specific seconds (the proposal's original design), with a small `1−IBS` anchor. This **does** have a different optimum than SFT and is therefore the only reward family among those discussed that can give RL a non-trivial role.

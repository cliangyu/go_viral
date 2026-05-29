# Cross-ad SRCC: Business Rationale for Normalized Retention

**Date**: 2026-05-28
**Author**: Claude (for Leon / Vispie)
**Purpose**: Explain *why* cross-ad per-second SRCC is the right reward for retention-curve prediction in the Vispie product context, why it remains valid (and is arguably stronger) with peak-normalized curves, and where its semantic limits lie.
**Companion docs**:
- [FINAL_PROPOSAL.md](FINAL_PROPOSAL.md) — 4-day execution plan
- [VERIFICATION_WALKTHROUGH.md](VERIFICATION_WALKTHROUGH.md) — math verification
- [verification/verify_gap_f.out](verification/verify_gap_f.out) — empirical evidence of per-second cross-ad signal

---

## TL;DR

**Cross-ad per-second SRCC** rewards correctly ranking ads against each other at each second of playback. This matches the business decision Vispie's customers actually face — "which creatives should I run?" — and works *better* on peak-normalized curves than on absolute reach, because normalization isolates the creative-quality signal from the budget/distribution signal.

Within-ad metrics (IBS, within-ad Spearman) answer a different question: "is one ad's predicted curve right?" Useful for model calibration analysis; not the reward signal a customer's decision depends on.

---

## The two retention axes

Retention prediction has two independent axes:

| Axis | What's compared | Question answered | Business use |
|---|---|---|---|
| **Within-ad (time axis)** | R̂(t₁), R̂(t₂), ..., R̂(t_T) for ONE ad | "Is the shape of THIS ad's predicted curve right?" | Per-ad calibration analysis |
| **Cross-ad (ad axis)** | R̂_A(t), R̂_B(t), R̂_C(t) at FIXED t | "Of these ads, which one retains best at t?" | Creative selection / ranking |

Most of our prior work conflated these. Within-ad Spearman is empirically and mathematically degenerate under any monotone-non-increasing predictor (see VERIFICATION_WALKTHROUGH.md Rounds 1–3). Cross-ad Spearman doesn't share that degeneracy and maps directly to the decision Vispie's customers make.

---

## What cross-ad per-second SRCC means in customer decisions

**Spearman rank correlation between predicted and true retention at second t, computed across N ads in a batch, averaged over t.**

Concretely: for each second t, sort the N ads in two ways — by predicted R̂(t) and by true R(t). Compute Spearman correlation of the two orderings. Average over t.

### Decision map

| Customer scenario | Decision | Right metric |
|---|---|---|
| Agency picks top-K creatives from 100 candidates | "Which 10 will retain best at the 15s view-completion mark?" | Cross-ad SRCC at t=15 |
| Real-time bidding for an ad slot | "At target view-duration T, who will hold most viewers?" | Cross-ad SRCC at t=T |
| Creative A/B test pre-launch | "Is variant B's retention shape better than A?" | Cross-ad SRCC (paired) |
| Quarterly creative-portfolio audit | "Across our 500 ads, which 50 underperform?" | Cross-ad SRCC, averaged over all t |
| Single-ad debugging | "Is my predicted curve for ad X wiggling correctly over time?" | Within-ad metric (IBS) — not the reward |

The first four are the actual product use cases. Cross-ad SRCC is aligned with all of them.

### Why averaging over t makes business sense

Different customers care about different view horizons:
- TikTok creator economy: 3s view rate, 6s completion
- Brand campaigns: 15s+ view-completion
- Long-form storytelling: 30s+ retention

Averaging over t in the discriminative range trains the model to be useful across all these horizons simultaneously. Optional t-weighting (covered in §"Refinements") lets us emphasize particular horizons per customer segment.

---

## Why peak-normalization makes the metric *stronger*, not weaker

### What peak-normalization does

Our R values are normalized so R(0) = 1.0 for every ad. This collapses the **absolute reach** dimension (peak views, impressions, audience size) and isolates the **shape** of retention.

### The rank-flip example

Peak normalization can flip the ranking of two ads vs. unnormalized data:

| Ad | Peak views | Views at t=10 | Unnormalized rank | R(10) normalized | Normalized rank |
|---|---|---|---|---|---|
| A | 1,000,000 | 500,000 | 1st (more absolute viewers) | 0.50 | 2nd |
| B | 100,000 | 80,000 | 2nd | 0.80 | **1st** |

**This is a feature, not a bug** for the creative-quality prediction problem.

**Why**: A's larger absolute number at t=10 is almost entirely a function of campaign budget and distribution muscle (1M peak views), not creative quality. B's 80% retention through t=10 vs. A's 50% means **B's creative is holding attention better** — that's the signal a creative selection algorithm should optimize for. Absolute reach is a downstream multiplier the customer applies separately based on their budget.

### The semantic decomposition

Total view-seconds = (peak reach) × (retention shape integral)

| Component | Owned by | Predictable from creative content? |
|---|---|---|
| Peak reach | Campaign budget, distribution algorithm, audience targeting | No — depends on platform/budget |
| Retention shape | Creative quality (hook, pacing, content) | **Yes — this is what the model should predict** |

Peak-normalization makes the model focus on the predictable part and ignore the confound. This is the same logic as why VCR (view-completion rate) is normalized — you don't compare a $1M campaign's absolute VCR to a $10K campaign's absolute VCR; you compare their VCR rates.

---

## Why rank-based (Spearman) and not value-based (Pearson, MSE)

### The rank-based vs. value-based tradeoff

| Property | Rank-based (Spearman) | Value-based (Pearson / MSE) |
|---|---|---|
| Robustness to systematic miscalibration | ✓ Model can be globally too high or too low; rank-correct still gets full reward | ✗ Penalized |
| Top-K selection alignment | ✓ Top-K is a rank decision | Indirect |
| Confidence info retained | ✗ [0.51, 0.49] = [0.99, 0.01] same Spearman | ✓ Distinguished |
| Outlier robustness | ✓ | ✗ |

### When rank-based wins (our case)

Vispie's primary customer workflow:
1. Submit 100 creative variants
2. Get predicted retention scores
3. **Pick top K to actually run**

Step 3 is a discrete rank decision. You cannot "half-pick" the 11th-ranked ad to express less confidence. The marginal value of confidence above what rank already captures is zero in this workflow.

### When value-based would win (not our case)

If the customer workflow were:
1. Predict retention curve for one creative
2. Decide *how much budget* to allocate based on expected retention

Then value-calibration matters — a 60% predicted retention vs. 55% changes the budget allocation. But this isn't how the Vispie product is positioned; budget allocation is a separate ML model with different features (audience, time-of-day, competition).

---

## Where cross-ad SRCC has limits (and what to do)

### Limit 1: degeneracy at t=0

R(0) = 1.0 for every ad by peak-normalization. Cross-ad rank at t=0 is fully tied → SRCC undefined.

**Mitigation**: restrict the reward to t ∈ [1, T_i]. (Verified by [verification/verify_gap_f.out](verification/verify_gap_f.out): std(R) at t=0 is 0.0000 with 1 unique value; at t=1 it jumps to 0.215 with 198/200 unique.)

### Limit 2: signal weakens at long t (tail saturation)

By t=30, ~57% of ads have R_i(t) ≈ R_i(t-1) within 0.001 (they've plateaued or hit zero). By t=60, 90% of ads are saturated and only 31/200 ads are still alive.

**Mitigation options**:
- **Simple**: average SRCC uniformly over t ∈ [1, min(T_i, 30)].
- **Better**: weight per-t contributions by `n_unique(t) / N` so saturated tails contribute less.
- **Best (deferred)**: per-customer t-weighting reflecting business-relevant view horizons.

We'll ship the simple version first. The 5-LOC swap to weighted is trivial if signal is weak.

### Limit 3: small-batch noise

SRCC over N=4 ads is noise; SRCC over N=16+ is usable. Our V8 training config (16 GPUs × 1 ad × 4 rollouts = 64 rollouts per batch, equivalent to 16 effective ads) is right at the floor.

**Mitigation**: monitor per-batch SRCC variance during smoke training. If high, increase rollouts-per-ad or ads-per-batch.

### Limit 4: format cliff inherited from existing reward pipeline

The current `ttcc_ibs_reward` plugin parses completions via regex. Any parse-failure → reward = 0 (verified, see VERIFICATION_WALKTHROUGH.md B3'). The cross-ad SRCC reward will inherit this. Format compliance becomes a binary gate.

**Mitigation**: carry `ttcc_format` reward (already in V8 config) at small weight (~0.2). Monitor parse rate during training; early-stop if it drops below 60%.

### Limit 5: rank reward incentivizes ordering, not value accuracy

A model that predicts [0.51, 0.49] across two ads gets full SRCC if the true order is the same. So does a model that predicts [0.99, 0.01]. The first is barely confident; the second is highly confident. Cross-ad SRCC doesn't reward confidence.

**Mitigation if needed**: add a small auxiliary value-calibration term (e.g., 0.1 × per-ad IBS) to retain some calibration pressure. We'll ship without it first to see if pure rank-reward is enough.

---

## How this maps to the model architecture

The cross-ad SRCC reward operates on the **text completion** (per the existing GRPO reward pipeline, see B3' in VERIFICATION_WALKTHROUGH.md). The model emits a curve as `{"R": [1.0, 0.95, 0.92, ...]}` text. The reward parses, ranks across the batch's rollouts at each t, computes Spearman, averages.

The retention head W still exists in the architecture (SFT trained it). During GRPO it sits unused — same as the current `ttcc_ibs_reward`. This is *fine* for the cross-ad SRCC story: we are explicitly demonstrating that **the body learns to emit cross-ad-discriminative numeric text under this reward**. The head's role is "SFT calibration anchor"; GRPO's role is "cross-ad rank discrimination via text generation."

This is a cleaner story than the current `1−IBS` reward, where the body learns to emit value-matched text under a signal that's gradient-redundant with what SFT already did.

---

## Implementation sketch

```python
class TTCCCrossAdSRCCReward(ORM):
    name = "ttcc_cross_ad_srcc_reward"

    def __call__(self, completions, **kwargs):
        R_true_batch = kwargs.get("R_true")   # list[list[float]]
        T_batch = kwargs.get("T")             # list[int]
        N = len(completions)

        # Parse all completions -> R_hat_batch (N x T_max), with NaN for parse-fail
        R_hat_batch = [...]  # use _parse_curve from existing plugin

        # For each t in 1..max(T): compute Spearman across ads at that t
        # Skip t where < 4 ads are valid (degenerate)
        srccs_per_t = []
        for t in range(1, max(T_batch) + 1):
            valid_mask = [...]  # ads with R(t) defined AND R_hat(t) parsed
            if valid_mask.sum() < 4:
                continue
            srcc_t, _ = spearmanr(R_hat_at_t[valid_mask], R_true_at_t[valid_mask])
            if np.isnan(srcc_t):
                continue
            srccs_per_t.append(srcc_t)

        batch_srcc = np.mean(srccs_per_t)

        # Per-rollout reward via leave-one-out
        rewards = []
        for i in range(N):
            srcc_without_i = recompute_excluding_i(...)
            r_i = batch_srcc - srcc_without_i  # marginal contribution
            rewards.append(float(r_i))
        return rewards
```

This is Form 2a (per-rollout leave-one-out). Form 2b (batch-broadcast) is the fallback if leave-one-out signal is too weak.

Full implementation in the next sprint per FINAL_PROPOSAL.md timeline.

---

## What this paper claims

If we run the full E1/E2/E3 comparison:

> "We mathematically and empirically demonstrate that the current 1−IBS reward is gradient-redundant with SFT on the body and a no-op on the head. We propose cross-ad per-second SRCC as the orthogonal-signal alternative (cos sim = −0.0116 with SFT MSE gradient). We validate that peak-normalized retention curves carry strong cross-ad signal in t ∈ [1, 30] (verify_gap_f.out). [Experimental results on E3 vs. E1/E2 follow.]"

If E3 doesn't train in time:

> "[Math + measurement claims as above.] We provide a reference implementation of cross-ad SRCC reward (ttcc_cross_ad_srcc_plugin.py). Empirical validation deferred to future work due to project-window constraints."

Both forms are defensible CS224R submissions. The math is the contribution; the running RL is the demonstration.

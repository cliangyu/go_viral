# Cross-ad SRCC vs Per-ad IBS: Relationship, Tension, and Variable-Length Handling

**Date**: 2026-05-28
**Purpose**: Answer Leon's two sharp questions precisely, from first principles.
1. If you rank correctly, is cross-ad SRCC necessarily high? Is it mutually exclusive with per-ad IBS?
2. For two videos of different length, how is cross-ad SRCC computed?

---

## Part 1: The relationship between cross-ad SRCC and per-ad IBS

### Definitions (precise)

**Per-ad IBS** for ad $i$ with horizon $T_i$:
$$\text{IBS}_i = \frac{1}{T_i + 1}\sum_{t=0}^{T_i}\big(\hat R_i(t) - R_i(t)\big)^2$$
Then averaged over ads. Measures **absolute value accuracy** of each curve.

**Cross-ad SRCC** at fixed second $t$, over the set $\mathcal{A}(t) = \{i : T_i \ge t\}$ of ads alive at $t$:
$$\text{SRCC}(t) = \rho_S\big(\{\hat R_i(t)\}_{i\in\mathcal{A}(t)},\ \{R_i(t)\}_{i\in\mathcal{A}(t)}\big)$$
Then averaged over $t$ in a chosen range. Measures **relative ordering** of ads at each second.

### Claim 1: They share the same global optimum

If $\text{IBS}_i = 0$ for all $i$ (perfect value prediction), then $\hat R_i(t) = R_i(t)$ exactly, so ranking by $\hat R(t)$ equals ranking by $R(t)$ at every $t$ $\Rightarrow \text{SRCC}(t) = 1$ for all $t$.

**Perfect calibration $\Rightarrow$ perfect ranking.** The converse is false (Claim 2). So at the global optimum they agree; they are NOT mutually exclusive.

### Claim 2: SRCC = 1 does NOT imply low IBS

Counterexample: $\hat R_i(t) = 0.5 \cdot R_i(t)$ for all $i, t$. This is a monotone transform, so ranks are preserved $\Rightarrow \text{SRCC}(t) = 1$ everywhere. But
$$\text{IBS}_i = \frac{1}{T_i+1}\sum_t (0.5 R_i(t) - R_i(t))^2 = 0.25 \cdot \overline{R_i^2} > 0.$$

So you can have **perfect ranking with arbitrarily bad calibration**. SRCC is invariant to any monotone-increasing transform of the predictions; IBS is not.

**Answer to "if you rank correctly is SRCC necessarily high?"**: Yes, by definition — SRCC literally measures rank-correctness. But high SRCC does NOT mean the problem is "solved", because a rank-correct model can be badly miscalibrated in level (off by a scale/shift). High SRCC is necessary but not sufficient for good prediction.

### Claim 3: They are NOT mutually exclusive, but there IS a practical trade-off

Not mutually exclusive: the global optimum (perfect prediction) maximizes both.

But away from the optimum, in a finite-capacity model under uncertainty, there is a **real trade-off**:

- **To minimize IBS, the safe move under uncertainty is to hedge toward the population-mean curve.** Predicting closer to the mean reduces the expected squared error when you're unsure. This is classic regression-to-the-mean. But hedging compresses all predictions toward the same value $\Rightarrow$ destroys cross-ad ranking ability $\Rightarrow$ SRCC drops.
- **To maximize SRCC, you want to spread predictions apart** so the ordering is robust to noise, even at the cost of some per-ad squared error.

So in the practical (sub-optimal) regime they pull in different directions. This is exactly the regime our model lives in.

### Claim 4: The decisive empirical contrast — what the baselines score

This is the crux of why cross-ad SRCC is the right axis and within-ad metrics are not.

| Metric | B1 (predict population-mean curve for every ad) |
|---|---|
| Within-ad Spearman | **0.97** (saturated — verified, verify_math_v3.out) |
| Per-ad IBS | low-ish (mean curve is "close enough" to each ad) |
| **Cross-ad SRCC** | **≈ 0** (all ads get the SAME predicted curve $\Rightarrow$ all ranks tied $\Rightarrow$ Spearman undefined/zero) |

**The mean-baseline B1 scores ~0.97 on within-ad Spearman but ~0 on cross-ad SRCC.**

This is the entire argument in one line: cross-ad SRCC is a metric on which the trivial mean-hedging baseline FAILS. So it cannot be gamed by regression-to-the-mean. Within-ad Spearman and (to a lesser degree) per-ad IBS can be.

### Summary of Part 1

```
                IBS = 0  ──────────────►  SRCC = 1     (perfect prediction: both win)
                SRCC = 1  ──╳──►  low IBS              (rank-correct ≠ calibrated)

   Practical regime trade-off:
     minimize IBS  →  hedge to mean  →  SRCC ↓   (SFT's failure mode)
     maximize SRCC →  spread apart   →  IBS may ↑ on some ads

   Baseline contrast (the decisive fact):
     B1 mean-curve:  within-ad Spearman 0.97,  cross-ad SRCC ≈ 0
```

They are complementary, not mutually exclusive, not redundant. Per-ad IBS = "is each curve's level right". Cross-ad SRCC = "is the ordering right". The customer decision (pick top-K creatives) is the ordering. SFT optimizes the level and structurally sacrifices the ordering. That gap is the RL opening.

---

## Part 2: Variable-length videos — how to compute cross-ad SRCC

### The problem

Ads have different horizons $T_i$ (e.g., 19s vs 60s). Cross-ad SRCC at second $t$ compares $\hat R_i(t)$ across ads at the SAME $t$. But at $t=50$, a 19-second ad has no $R(50)$ — it already ended.

### The correct method: per-$t$ valid set, absolute time

At each second $t$, only include ads that are at least $t$ seconds long:
$$\mathcal{A}(t) = \{\,i : T_i \ge t\,\}$$
Compute SRCC over $\mathcal{A}(t)$ only. Then average SRCC over a chosen range of $t$.

This is what `verification/verify_gap_f.py` already implemented and measured:

| $t$ | $|\mathcal{A}(t)|$ (ads alive) | std($R(t)$) | usable? |
|---|---|---|---|
| 1 | 200 | 0.215 | strong |
| 10 | 179 | 0.145 | strong |
| 15 | 162 | 0.142 | good |
| 30 | 102 | 0.126 | ok |
| 60 | 31 | 0.125 | thin (few ads) |

### Why absolute time, NOT normalized/relative time

A tempting alternative: rescale every curve to $[0,1]$ relative time and resample to a common grid. **This is wrong for our problem**, because:

- A 19-second ad's "second 10" is at relative position $10/19 = 0.53$ (more than halfway through).
- A 60-second ad's "second 10" is at relative position $10/60 = 0.17$ (early).
- These are NOT comparable engagement moments. Comparing them cross-ad would mix "mid-ad fatigue" with "early hook".
- The customer's view-completion targets (3s, 6s, 15s) are **absolute seconds**, not fractions. The product decision is in absolute time.

So we keep absolute $t$ and accept that $|\mathcal{A}(t)|$ shrinks as $t$ grows.

### Consequences for the reward / metric

1. **Restrict to a discriminative range.** Large-$t$ SRCC is computed over few ads (31 at $t=60$) $\Rightarrow$ noisy. Restrict reward/metric to $t \in [1, 30]$ where $|\mathcal{A}(t)| \ge \sim 100$. (gap_f confirms signal is strong there.)

2. **Exclude $t = 0$.** Every ad has $R(0) = 1$ by peak-normalization $\Rightarrow$ all tied $\Rightarrow$ SRCC undefined. Start at $t = 1$.

3. **In a GRPO batch**, a short ad only contributes to early-$t$ SRCC terms; a long ad contributes across the whole range. The leave-one-out attribution handles this automatically — a short ad's marginal contribution reflects only the early-$t$ ranking it participated in. No special handling needed beyond the per-$t$ valid set.

4. **Minimum ads per $t$.** If $|\mathcal{A}(t)| < 4$, skip that $t$ (Spearman over <4 points is meaningless). Already in gap_f as the `n_valid >= 4` guard.

### Optional refinement (weighting)

If we want each $t$ to contribute according to how much signal it carries:
$$\text{SRCC}_{\text{weighted}} = \frac{\sum_t w(t)\,\text{SRCC}(t)}{\sum_t w(t)},\quad w(t) = \frac{|\mathcal{A}(t)|}{N}\cdot \text{std}_i(R_i(t))$$
This down-weights thin/saturated tails. Ship uniform-over-$[1,30]$ first; weighting is a 5-line swap.

---

## Part 3: The deeper hypothesis Leon raised (and how to test it)

> Video Language Model 里面有两个部分:(1) Video Perception (2) Language Model Reasoning。我们的假设是 Retention Value 的预测同时需要这两个信息。这样的假设有可能是错的。

This is the crux. The whole "RL on the language/reasoning pathway helps" story rests on this hypothesis. If retention is predictable from **video perception alone** (no reasoning), then the LM body is dead weight and RL on it is pointless.

### What the V8 trajectory already weakly tells us

V8 SFT improved per-ad IBS (eval/loss 0.027 → 0.016) **while destroying language** (token_acc 0.50 → 0.036) between steps 101 and 152.

Interpretation: **per-ad IBS does NOT seem to need language reasoning** — IBS kept improving after the language capacity was gone. Perception + per-ad memorization was sufficient for value accuracy.

This suggests a refined, testable hypothesis that ties everything together:

> **Per-ad value prediction (IBS) needs only perception + memorization. Cross-ad ranking (SRCC) needs comparative semantic reasoning, which needs the language pathway.**

If true, this is the cleanest possible story for the report:
- SFT improves IBS by perception+memorization, kills reasoning, and therefore CANNOT improve (and may hurt) cross-ad ranking.
- RL with a cross-ad SRCC reward, on a language-alive checkpoint, exercises the reasoning pathway and improves the ranking that SFT structurally cannot.

### The decisive test (cheap, ~1 hour)

Take two V8 checkpoints:
- **Early** (step ~50): token_acc ≈ 0.50 (language alive), eval/loss ≈ 0.032
- **Late** (step ~152): token_acc ≈ 0.036 (language dead), eval/loss ≈ 0.016

Compute on val_200 for both:
- per-ad IBS
- cross-ad SRCC at $t \in \{3, 6, 15, 30\}$

**Predicted outcome if the hypothesis is true:**

| ckpt | per-ad IBS | cross-ad SRCC |
|---|---|---|
| early (lang alive) | higher (worse) | **higher (better)** |
| late (lang dead) | lower (better) | **lower (worse)** |

i.e., **a crossing**: late wins on IBS, early wins on SRCC. That crossing would be direct evidence that (a) the two metrics are distinct, (b) language carries the ranking-relevant signal, (c) late-stage SFT is actively trading away the customer-relevant capability.

**If there's no crossing** (late wins on both, or early wins on both), the hypothesis is wrong or the signal is elsewhere, and we rethink.

This single experiment is the highest-information thing we can run. It validates or kills the central hypothesis before we invest in RL.

### If the hypothesis is wrong

If retention (including ranking) is fully predictable from perception alone:
- The VLM's language half is not pulling weight.
- RL on reasoning won't help.
- But cross-ad SRCC RL could STILL work — it would just be teaching the perception→prediction map to spread predictions for better ranking, rather than invoking reasoning. RL still has a structural opening (SFT's mean-hedging) even if reasoning isn't the mechanism.

So: the hypothesis being wrong weakens the "RL exercises reasoning" narrative but does NOT kill "RL improves cross-ad ranking". The reward design is robust to this uncertainty; only the explanatory story changes.

# V8-Next Proposal: Cross-Ad Retention Discrimination via Composite RL

**Date**: 2026-05-28
**Author**: Leon + Claude
**Status**: AUTHORITATIVE — supersedes prior milestone-anchored framing
**Deadline**: 2026-06-03 19:30 BJ

This document is self-contained. It does not depend on the research proposal or the milestone report. The reward design is justified from first principles + verified math + Vispie's actual customer decision context.

Supporting artifacts (all in this directory):
- [VERIFICATION_WALKTHROUGH.md](VERIFICATION_WALKTHROUGH.md) — math verification, Rounds 1-3
- [BUSINESS_RATIONALE.md](BUSINESS_RATIONALE.md) — customer decision context
- [verification/](verification/) — reproducible scripts + raw outputs

---

## 1. Thesis

> **SFT alone collapses toward mean-curve prediction; it loses the cross-ad discrimination signal Vispie's customers depend on. We propose a composite RL reward that dominates on the orthogonal cross-ad ranking pathway while anchoring on per-ad calibration, achieving both value accuracy AND rank discrimination in a single training pass.**

The verified math establishes that the current `1−IBS` reward is gradient-redundant with SFT MSE on the body and a no-op on the head. The cross-ad rank reward is gradient-orthogonal to SFT MSE (cos sim = −0.0116). The proposed composite uses the orthogonal signal as the dominant term and the redundant signal as a small calibration anchor. The result targets cross-ad ranking improvement without sacrificing per-ad calibration.

---

## 2. Settled facts (machine-precision verified, reproducible)

| # | Claim | Source | Confidence |
|---|---|---|---|
| F1 | `_masked_mse` = IBS metric per ad, mean-mode, bit-identical | [verify_math_v3.py B1](verification/verify_math_v3.py) | bit-identical at float64 |
| F2 | Within-ad Spearman is degenerate: any monotone predictor gives the SAME ρ_S on a given truth; std across 200 random monotone preds = 0 for all 200 val ads | [verify_math_v3.py A1+A5](verification/verify_math_v3.py) | std = 0.0 across 40K samples |
| F3 | Two strict-monotone-decreasing sequences with no ties → ρ_S = 1 regardless of value-space crossing | [verify_crossing.py](verification/verify_crossing.py) | std = 0 across 1000 crossing pairs |
| F4 | `ttcc_ibs_reward` returns `list[float]` — non-differentiable, computed from parsed text; head W is NEVER updated by GRPO | [verify_math_v3.py B3'](verification/verify_math_v3.py) | source code direct |
| F5 | Cross-ad rank gradient ⊥ SFT MSE gradient | [verify_math_v3.py B3](verification/verify_math_v3.py) | cos sim = −0.0116 |
| F6 | Cross-ad signal exists in val_200: 198/200 ads have unique R(1), discriminative band t ∈ [1, 30], tail saturation beyond | [verify_gap_f.py](verification/verify_gap_f.py) | empirical |

What these jointly imply:
- A pure value-accuracy reward (`1−IBS`) cannot teach the model anything SFT didn't already teach the body and cannot touch the head at all.
- Within-ad Spearman is dead as a reward (saturated by monotone construction).
- Cross-ad SRCC is the orthogonal signal direction.
- The cross-ad signal IS empirically present in our data.

---

## 3. Business framing (1 paragraph)

Vispie's customer workflow is creative selection: "Of these N creatives, which K should we run?" That is a rank decision. Per-ad calibration (IBS) is useful for model debugging but is **not** the decision the customer makes. The reward we optimize must align with the rank decision — cross-ad SRCC at fixed view-horizons (3s, 6s, 15s, 30s). Peak-normalized retention curves are the correct primitive here because they isolate retention SHAPE (the creative-quality signal we can predict from content) from peak REACH (a budget/distribution signal we cannot predict from content). Full argument in [BUSINESS_RATIONALE.md](BUSINESS_RATIONALE.md).

---

## 4. Why SFT alone is insufficient (the empirical opening)

This is the mechanistic argument for why RL with the right reward can beat SFT:

**SFT minimizes per-ad MSE → biased toward mean-curve prediction (regression to the mean).**

Concretely: an ad with very steep decay and an ad with very flat decay both contribute equally to SFT loss. The MSE gradient pulls predictions toward something that fits both reasonably — i.e., toward the average decay rate. The result is *under-discrimination* across ads.

For per-ad IBS, mean-reverted predictions can still score well (each ad's mean curve is "close enough" to truth). But for **cross-ad SRCC, mean-reverted predictions are catastrophic** — if every ad is predicted to be near the mean, you can't rank them.

This is the empirical opening: **SFT's exact failure mode (under-discrimination) is precisely what cross-ad SRCC reward corrects.** We are not fighting against SFT's strengths; we are filling SFT's structural gap.

---

## 5. Proposed reward design

### 5.1 Formulation

Per-rollout reward over N ads × K rollouts per ad:

$$r_i = \alpha \cdot (1 - \text{IBS}_i) + \beta \cdot S_i^{\text{LOO}} + \gamma \cdot \text{format}_i$$

with **α = 0.2, β = 0.7, γ = 0.1** as the starting weights.

Where:
- $\text{IBS}_i = \frac{1}{T_i+1} \sum_t (\hat R_i(t) - R_i(t))^2$ — per-ad value calibration (the redundant-with-SFT but stable anchor)
- $S_i^{\text{LOO}}$ — per-rollout leave-one-out cross-ad SRCC contribution (the orthogonal signal, defined below)
- $\text{format}_i \in \{0, 1\}$ — parse success indicator (operational anti-cliff)

### 5.2 Cross-ad SRCC leave-one-out contribution

For each batch of N rollouts × K samples per rollout, compute per-second cross-ad SRCC averaged over the discriminative t-range:

$$S^{\text{batch}} = \frac{1}{|T_{\text{disc}}|} \sum_{t \in T_{\text{disc}}} \rho_S(\{\hat R_j(t)\}_{j=1}^{NK}, \{R_j(t)\}_{j=1}^{NK})$$

where $T_{\text{disc}} = \{t : t \geq 1 \text{ and } n_{\text{valid}}(t) \geq 4\}$ — verified empirically discriminative range from [verify_gap_f.out](verification/verify_gap_f.out).

Per-rollout leave-one-out contribution:

$$S_i^{\text{LOO}} = NK \cdot (S^{\text{batch}} - S^{\text{batch}\setminus i})$$

The $NK$ scale factor normalizes the marginal contribution to ~unit variance per rollout. This is the standard leave-one-out attribution for batch-coupled rewards in PPO-family algorithms.

### 5.3 Why these specific weights

- **β = 0.7 (dominant SRCC)** — the orthogonal signal, the actual contribution. Must dominate to drive the body toward discrimination.
- **α = 0.2 (small IBS anchor)** — provides dense per-rollout gradient signal to reduce policy-gradient variance. Even though gradient-redundant with SFT body in principle, it acts as a calibration anchor that prevents reward-sparsity from destabilizing training. Not load-bearing for the contribution; load-bearing for training stability.
- **γ = 0.1 (format)** — small but non-zero. Prevents parse-cliff catastrophe (parse-fail → 0 reward dominates if format compliance drifts).

### 5.4 What this is NOT

This is **not** "RL on top of SFT" in the trivial sense. This is **RL with a reward that targets a signal direction (cross-ad rank) that SFT structurally underfits.** The two are complementary, not redundant.

---

## 6. Why this has high empirical odds of working

### 6.1 Mechanistic argument

- SFT structurally regresses to mean → under-discrimination across ads
- Cross-ad SRCC reward directly penalizes under-discrimination
- These are matched (gap + corrective)

### 6.2 Initialization advantage

We start from V8 SFT checkpoint, NOT from base model. The body already has reasonable curve-prediction skill. Cross-ad SRCC fine-tunes the body toward better discrimination on top of an already-trained calibration. We are operating at the margin where RL is most effective: **incremental refinement of a pre-trained policy**, not from-scratch learning.

### 6.3 Variance control

The IBS anchor (α=0.2) provides dense per-rollout gradient signal even when SRCC is noisy. This is the standard ML risk-management pattern: dominant signal for the contribution + small dense signal for stability.

### 6.4 What we explicitly do NOT promise

- We do not promise to beat V8 SFT on **per-ad IBS**. SFT optimizes IBS directly; RL adds variance.
- We promise to beat V8 SFT on **cross-ad SRCC at fixed t** (the business-relevant metric).
- Combined: if V8 SFT has IBS = X and SRCC = Y, we expect E3 to have IBS ≈ X (slightly higher acceptable) and SRCC > Y (the contribution).

### 6.5 What's risky

| Risk | Mitigation | Tripwire |
|---|---|---|
| Reward variance dominates signal | IBS anchor + monitor per-batch SRCC variance | Early-stop if SRCC variance > 5× IBS variance |
| Parse-cliff drift (format compliance drops) | Format reward γ=0.1 + parse-rate monitor | Early-stop if parse rate < 60% |
| Body learns to emit median curves (gaming the rank by sticking to the middle) | Per-rollout LOO scale factor + variance monitor | Investigate if SRCC plateaus at 0.7-0.8 |
| Distribution shift away from SFT's IBS-good region | IBS anchor pulls back | Monitor IBS; early-stop if degrades > 20% |
| ms-swift GRPO trainer doesn't support batch-coupled reward | Use leave-one-out form (Form 2a); no trainer changes needed | Day 1 read of `swift/trainers/grpo_trainer.py` confirms |

---

## 7. Experimental design

### 7.1 Three experiments (this is what the report compares)

| ID | Setup | Role | Already running? |
|---|---|---|---|
| E1 | V8 SFT (hazard head + CoT distillation), continues to convergence | Primary baseline | YES (on 16×H100) |
| E2 | V7-style GRPO with `r = 1−IBS` only | "Redundant RL" control | Code already exists; can re-run from V8 ckpt |
| E3 | GRPO with composite reward (this proposal), starting from V8 SFT checkpoint | The contribution | TO BUILD |

### 7.2 Eval protocol (consistent across E1/E2/E3)

Three metrics on val_200:
1. **Per-ad IBS** (mean across 200 ads) — calibration quality
2. **Cross-ad SRCC at t ∈ {3, 6, 15, 30}** — business-relevant ranking at standard view horizons
3. **Average cross-ad SRCC over t ∈ [1, 30]** — overall discrimination

Plus operational metrics:
4. **Parse rate** — fraction of rollouts that produce parseable curves
5. **Reward variance per batch** — stability diagnostic

Plus comparison baseline:
- **B1 = per-second train-mean**: the constant-predictor floor

### 7.3 Expected outcomes (commitments)

| Metric | E1 (SFT) | E2 (1−IBS RL) | E3 (composite) |
|---|---|---|---|
| Per-ad IBS | best or tied | tied with E1 | within 5% of E1 |
| Cross-ad SRCC | mid | tied with E1 | **higher than E1 by ≥5pp** (commitment) |
| Parse rate | n/a | high | ≥80% |

If E3 does not beat E1 on cross-ad SRCC by ≥5pp, we have failed the empirical bar and should report it honestly in the discussion. The math contribution stands either way.

---

## 8. 4-day execution plan

| Day | Date | Goal | Risk |
|---|---|---|---|
| Day 0 (TODAY) | 2026-05-28 | Eval V8 ckpt-75 IBS + cross-ad SRCC vs B1; read `swift/trainers/grpo_trainer.py` to confirm Form 2a fits trainer | LOW |
| Day 1 | 2026-05-29 | Implement `ttcc_composite_reward_plugin.py` (~120 LOC + unit tests); single-GPU smoke train (4 ads × 4 rollouts × 50 steps) | **HIGHEST** — trainer integration |
| Day 2 | 2026-05-30 | Launch E3 on 16×H100 starting from V8 SFT ckpt; live monitor (parse rate, SRCC, IBS, reward variance every 5 min) | MEDIUM — monitor stability |
| Day 3 | 2026-05-31 | Mid-training eval; if E3 trending positive, continue; if not, diagnose | LOW |
| Day 4 | 2026-06-01 | Final eval on val_200 for E1/E2/E3; build comparison plots | LOW |
| Day 5 | 2026-06-02 | Report writing | LOW |
| Day 6 | 2026-06-03 (deadline 19:30 BJ) | Submit | LOW |

### 8.1 Decision gates

**End of Day 0**: V8 IBS vs B1 result must show V8 IBS << B1 IBS (SFT actually learned). If V8 ≈ B1, SFT failed and we have a different problem to debug.

**End of Day 1**: Smoke train must show (a) parse rate > 80%, (b) SRCC reward responding to policy updates (not flat at random level), (c) no NaN/Inf loss. If any fail, halt and diagnose.

**End of Day 2**: First 100 steps of full E3 training. If parse rate degrades or IBS degrades > 30%, early-stop and fall back to pure cross-ad SRCC (drop α anchor) or revert to V8 SFT as final.

**End of Day 3**: Mid-training eval. If E3 SRCC > E1 SRCC, continue training. If E3 SRCC ≤ E1 SRCC, run one more day then stop; report what was observed.

---

## 9. Fallback

If E3 doesn't train by end of Day 4:

> "We mathematically prove and empirically verify (reproducible scripts in verification/) that:
> (i) the current `1−IBS` reward is gradient-redundant with SFT MSE,
> (ii) within-ad Spearman is degenerate under any monotone predictor (machine-precision across all 200 val ads),
> (iii) cross-ad SRCC has gradient orthogonal to SFT MSE,
> (iv) peak-normalized retention curves carry strong cross-ad signal in t ∈ [1, 30].
> We propose a composite reward formulation and provide a reference implementation. Empirical validation deferred to future work due to project-window constraints."

This is a defensible CS224R submission on the math contribution alone. The empirical RL result is the demonstration, not the contribution.

---

## 10. What we deliberately drop

- **Within-ad Spearman as a metric or reward.** F2 + F3 killed it. It's a function of truth ties, not predictor structure.
- **"Beat V8 SFT on per-ad IBS" framing.** This is the wrong metric for the customer decision; we commit to beating it on cross-ad SRCC instead.
- **Pure cross-ad SRCC (no anchor).** Riskier on training stability; the 5pp gain we want is more reliable with the composite design.
- **Differentiable RL through the head.** Would just be SFT with CoT autoregressive sampling renamed; misleading to report as RL.
- **From-scratch training.** Always initialize from V8 SFT checkpoint.

---

## 11. Required confirmations from Leon before launch

- [ ] Composite reward (α=0.2, β=0.7, γ=0.1) is the design — not pure SRCC, not pure 1−IBS.
- [ ] Cross-ad SRCC at fixed t (not within-ad) is the business-relevant metric.
- [ ] Commitment is "≥5pp SRCC improvement at t ∈ {3, 6, 15, 30}", not "beat IBS."
- [ ] Day-by-day timeline is realistic.
- [ ] Tripwires + early-stop conditions are acceptable.
- [ ] Fallback (math-only contribution) is acceptable as worst-case.

---

## 12. First three actions after confirmation

1. Evaluate V8 ckpt-75 on val_200: IBS, cross-ad SRCC, parse rate (eval_ibs.py extended). Output: this is the E1 baseline number we must beat on SRCC.
2. Read `swift/trainers/grpo_trainer.py` and confirm `ttcc_composite_reward` Form 2a (per-rollout leave-one-out) integrates without trainer changes.
3. Write `ttcc_composite_reward_plugin.py` + unit tests.

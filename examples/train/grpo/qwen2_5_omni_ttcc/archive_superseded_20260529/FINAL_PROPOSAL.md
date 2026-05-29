# V8 Reward Design — Final Proposal (4-day plan)

**Author**: Claude (for Leon)
**Date**: 2026-05-28
**Status**: Draft pending Leon confirmation
**Deadline**: CB expires 2026-06-03 19:30 BJ (4 calendar days, ~6 working days incl. weekend)
**Companion docs**:
- [VERIFICATION_WALKTHROUGH.md](VERIFICATION_WALKTHROUGH.md) — math verification, Rounds 1 & 2
- [REWARD_DESIGN_ANALYSIS.md](REWARD_DESIGN_ANALYSIS.md) — claim-by-claim analytical analysis
- [verification/verify_math_v3.py](verification/verify_math_v3.py) — reproducible verification script
- [verification/verify_math_v3.out](verification/verify_math_v3.out) — actual output from 2-card box

---

## The thesis (this is what the report sells)

> **The 1−IBS reward is gradient-redundant with SFT MSE on the body and leaves the head W untrained; the genuine RL signal lives on the cross-ad ranking pathway, which is gradient-orthogonal to SFT (cos sim = −0.0116) — we propose and validate that pathway.**

This is an *insights paper*, not a SOTA-beating submission. The CS224R rubric explicitly rewards insights. We do not need Path B's RL to beat V8 SFT to have a valid submission; we need the *story* to be defensible — and it is, regardless of empirical outcome.

## Evidence backing the thesis (machine-precision verified)

| Claim | Source | Confidence |
|---|---|---|
| `_masked_mse` ≡ IBS metric per ad (mean-mode) | Round 2 / B1 | bit-identical |
| Within-ad Spearman is degenerate under any monotone predictor across all 200 val ads | Round 2 / A1+A5 | std = 0 across 200 ads × 200 preds |
| Head gradient of SFT ≡ head gradient of (1−IBS) reward (hypothetical differentiable form) | Round 1 / B3 | cos sim = 1.0 |
| GRPO reward is detached `list[float]` from parsed text; head W never updated by GRPO | Round 2 / B3' | source code direct |
| Cross-ad rank reward gradient ⊥ SFT MSE gradient | Round 1 / B4 | cos sim = −0.0116 |

## What we ship (3 experiments)

1. **E1 — V8 SFT (hazard head + CoT distillation)**.
   - Status: already running on 16×H100 cluster as of 2026-05-28.
   - Role: the SFT baseline.
   - Eval metric: IBS on val_200 vs. B1 (per-second-mean) baseline.

2. **E2 — V7-style GRPO with `r = 1−IBS`** (the "redundant RL" baseline).
   - Status: code already implemented in `examples/train/grpo/plugin/ttcc_ibs_plugin.py`; V7 has historical results.
   - Role: empirical baseline showing what current-reward GRPO does (or fails to do).
   - Eval metric: same as E1.

3. **E3 — GRPO with cross-ad per-second SRCC reward** (the new contribution).
   - Status: TO BUILD.
   - Role: validate the orthogonal-signal thesis empirically.
   - Eval metric: same as E1 + cross-ad SRCC per second + comparison to E1/E2.

## Timeline

| Day | Calendar date | Goal | Risk |
|---|---|---|---|
| Day 0 | 2026-05-28 (today) | Close math gap F (cross-ad truth tie distribution per t, ~30 LOC); finalize Path B reward spec; evaluate V8 ckpt-75 vs B1 on val_200 | LOW |
| Day 1 | 2026-05-29 | Implement `ttcc_cross_ad_srcc_reward` (~80 LOC + unit tests); read `swift/trainers/grpo_trainer.py` to confirm integration LOC estimate; smoke-test on 4 ads × 4 rollouts × 50 steps | **HIGHEST risk** — trainer extension may be deeper than estimated |
| Day 2 | 2026-05-30 | Launch E3 on 16×H100 (V8 SFT done by this point); monitor; in parallel: probe ckpt-75 with randomization_probe.py | MEDIUM |
| Day 3 | 2026-05-31 | E3 mid-training eval; B1/V8/E3/V7-GRPO numbers on val_200; build comparison plots | LOW |
| Day 4 | 2026-06-01 | Final eval pass; complete plots and tables; start report draft | LOW |
| Day 5 | 2026-06-02 | Report writing | LOW |
| Day 6 | 2026-06-03 (deadline 19:30 BJ) | Submit | LOW |

## Specific code changes

All paths relative to `/Users/marvl/Documents/stanford/cs224r/projects/go_viral`.

### New files

1. `examples/train/grpo/plugin/ttcc_cross_ad_srcc_plugin.py` (NEW, ~80 LOC)
   - Implements `TTCCCrossAdSRCCReward(ORM)` class.
   - Reward shape (preferred — Form 2a): per-rollout leave-one-out SRCC contribution. Each rollout gets `r_i = SRCC_full - SRCC_without_i`. Decomposable. Works with existing GRPO trainer unchanged.
   - Fallback shape (Form 2b): batch-level SRCC broadcast to all rollouts. Requires minimal trainer patch to compute reward at batch granularity. **Only used if 2a's leave-one-out signal is too weak in smoke test.**
   - Unit tests:
     - perfect-prediction batch → reward = high for all rollouts
     - anti-rank batch → reward = low for all
     - random batch → reward ≈ 0 mean, distributed around it
     - parse-fail rollout → reward = 0, doesn't poison batch

2. `examples/train/grpo/qwen2_5_omni_ttcc/grpo_v3_srcc.sh` (NEW)
   - Launch script for E3.
   - Inherits from `grpo_v2cot_full.sh`, swaps `--reward_funcs ttcc_ibs_reward ttcc_format` → `--reward_funcs ttcc_cross_ad_srcc_reward ttcc_format`.
   - Increases per-step batch size if needed for N≥16 ads (constraint from §"Engineering caveats" below).

3. `examples/train/grpo/qwen2_5_omni_ttcc/verification/test_f_cross_ad_tie_distribution.py` (NEW, ~30 LOC)
   - For each t ∈ {0..60}: compute std(R_i(t) across i=1..200 in val_200) and the number of unique ranks.
   - Output: per-t spread/tie summary. If middle-t has good spread and early/late-t is degenerate, we restrict reward to discriminative t-range.
   - Closes gap F from the walkthrough.

### Modified files (if needed)

4. `swift/trainers/grpo_trainer.py` — patch only if Form 2b is required. **First action of Day 1: read this file and confirm the 80 LOC estimate is realistic.**

5. `examples/custom/qwen2_5_omni_retention/eval_ibs.py` — add cross-ad SRCC metric column alongside per-ad IBS. ~20 LOC.

### Unchanged

- `register.py` — no change. Hazard head + CoT-distillation loss already correct per Round 2 B1.
- `ttcc_ibs_plugin.py` — keep as-is. We need it for E2 (the redundant-RL baseline).
- All V8 SFT scripts — keep running.

## Verification gates before each step

Before launching E3 training:
1. Gap F closed (`test_f_cross_ad_tie_distribution.py` shows informative spread on at least the discriminative t-range)
2. Reward unit tests pass (~4 cases)
3. Single-GPU smoke train (4 ads × 4 rollouts × 50 steps) shows reward trends up
4. No format-cliff catastrophic collapse: parse rate stays above 80% in smoke

Before reporting E3 results as final:
1. Eval pipeline identical to E1 and E2 (same val_200, same B1 baseline, same metrics)
2. At least 3 independent random seeds for E3 (variance bars on reward curve)
3. Failure-mode analysis: parse rate over training, reward distribution per batch

## Engineering caveats independent of math

| Caveat | Mitigation |
|---|---|
| Batch coupling: SRCC needs N≥16 distinct ads in same batch for stable signal | Set `per_device_train_batch_size × gradient_accumulation_steps × world_size` so that each *reward computation* sees ≥16 ads. With current V8 setup (16 GPUs × 1 ad each × 4 rollouts) this is already 16 ads × 4 rollouts = 64 rollouts per batch — sufficient. |
| Format-cliff: parse-fail → reward = 0; can dominate if policy drifts | Carry `ttcc_format` reward (already in our config) at ratio `0.2 × format_bonus`. Monitor parse rate. Early-stop if it drops below 60%. |
| ms-swift GRPO trainer assumes per-rollout independent reward | Form 2a (leave-one-out) is per-rollout decomposable → no trainer change. Form 2b only if needed. |
| Cross-ad rank reward variance with K=4 rollouts per ad and N=16 ads | Empirical question; smoke test will tell us. If variance dominates, increase K to 8 or N to 32. |

## What we deliberately DROP

- **Path C (differentiable RL via head)**. It collapses to SFT-with-CoT renamed; misleading to report as RL. Already SFT IS this, just without CoT autoregressive sampling — and CoT is teacher-forced anyway.
- **"Beat V8 SFT on IBS" framing**. Insights paper, not SOTA chase. If E3 fails to beat E1 on IBS, that itself is a finding: "the cross-ad signal target is orthogonal but harder to optimize than the dense SFT signal."
- **Within-ad Spearman as a metric**. Round 2 A1/A5 killed it. Report as a *negative result*, not a method.
- **Retraining from scratch**. E3 starts from V8 SFT checkpoint (the strongest SFT we have).

## Open questions for Leon BEFORE I start

1. **Reward form**: Form 2a (leave-one-out, no trainer change) or Form 2b (batch-coupled, ~80 LOC trainer patch)? My recommendation: start with 2a; swap to 2b if signal is weak.

2. **Risk tolerance**: If Day 1 trainer-read reveals 2b is actually 300+ LOC, do we (a) commit to 2a and accept slightly weaker signal, (b) push to 2b and accept timeline risk, or (c) fall back to "math-only contribution" as described below?

3. **Wanjia coordination**: Wanjia's hazard-head SFT on Modal is the parallel path. Is there value in running E3 from her checkpoint too (more data points, two SFT baselines × one RL → 2 comparisons), or focus only on Leon's V8?

## Fallback if E3 doesn't build in time

If E3 doesn't reach training by end of 2026-05-31, the report becomes:

> "We mathematically prove and empirically verify (verify_math_v3.py, 200 val ads, machine precision) that 1−IBS reward is gradient-redundant with SFT MSE and leaves the retention head untrained during GRPO. We measure that cross-ad SRCC reward is gradient-orthogonal to SFT MSE (cos sim = −0.0116) and propose it as the corrective. *Empirical validation of cross-ad SRCC RL deferred to future work due to project-window constraints; reference implementation provided.*"

This is a defensible CS224R submission on its own. The math + the orthogonality measurement + a publication-quality reference implementation are the contribution; the running RL would be the demonstration, but it is not strictly required.

## Decision needed from Leon

Confirm or modify any of:

- [ ] Thesis statement above is the right framing for the report.
- [ ] 3-experiment structure (E1 / E2 / E3) is correct.
- [ ] Form 2a (leave-one-out per-rollout reward) for first attempt.
- [ ] Timeline is realistic given parallel obligations.
- [ ] Fallback plan is acceptable as a worst-case.

Once confirmed, my first three actions in order:
1. Run `test_f_cross_ad_tie_distribution.py` to close gap F (~10s).
2. Run V8 ckpt-75 eval vs B1 on val_200 (~minutes).
3. Read `swift/trainers/grpo_trainer.py` and confirm Form 2a path doesn't need trainer changes.

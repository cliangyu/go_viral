# CoT-GRPO reward: maximum, per-term ceilings, and the *optimizable* headroom

Source of truth (read 2026-05-31): `rl/head_reward.py`, `verification/cross_ad_reward.py`,
`configs/rl_cot_grpo.yaml`, live `logging.jsonl` of run `v4-20260531-162616`.

## The reward (exact form)

GRPO total reward per rollout (config `reward_funcs:[ttcc_head, ttcc_cot_format]`, `reward_weights:[1.0, 0.15]`):

```
r_total = 1.0 · r_head  +  0.15 · r_fmt
r_head  = β · R_rank  +  α · R_acc            (β = COT_BETA = 1.0,  α = COT_ALPHA = 0.25)
```

Component ranges (all in [0,1]):
- `R_rank = 1 − mean_t |F_t(R̂(t)) − F_t(R_true(t))|`, t∈[1,30], F_t = percentile vs the **fixed train CDF**.  Max **1.0** when the predicted curve lands in the same percentile bucket as the true curve at every second (does NOT require R̂=R_true).
- `R_acc  = max(0, 1 − mean_t (R̂(t) − R_true(t))²)` = 1 − MSE.  Max **1.0** at R̂=R_true.
- `R_fmt  ∈ {0,1}` — 1 iff a valid `<cot>…</cot>` structure. Max **1.0**.

## Maximum and per-term ceilings

| term            | weight | max component | max contribution |
|-----------------|--------|---------------|------------------|
| β·R_rank        | 1.0    | 1.0           | **1.00**         |
| α·R_acc         | 1.0    | 0.25·1.0      | **0.25**  ← α-limit |
| 0.15·R_fmt      | 0.15   | 1.0           | **0.15**         |
| **TOTAL MAX**   |        |               | **1.40**         |

- **Maximum reward = 1.40.**
- **α-limit (the calibration term's ceiling) = α·max(R_acc) = 0.25·1.0 = 0.25.**

## Where we actually are (live, run v4, temp 0.8)

```
reward            = 1.168          # = head 1.018 + 0.15·(format 1.0) = 1.018 + 0.150
TTCCHeadPlaceholder/mean = 1.018   # = β·R_rank + α·R_acc
TTCCCoTFormatReward/mean = 1.000   std = 0.000
reward_std        = 0.033          # within-group spread (the GRPO gradient)
```

Estimated split of the head term (R_acc not separately logged — a sub-signal logging gap):
`R_acc ≈ 0.97–0.99` (SFT already minimized MSE) → `α·R_acc ≈ 0.243–0.247` → **`R_rank ≈ 0.77`** (mean |ΔF| ≈ 0.23). *Confidence moderate; needs r_rank/r_acc logged separately to confirm.*

## The point: nominal headroom ≫ optimizable headroom

Naive headroom = 1.40 − 1.168 = **0.23 (~17%)**. That is misleading. What GRPO can actually optimize is far smaller:

1. **Format (0.15) is maxed at 1.0 with std = 0 → contributes ZERO gradient.** Every rollout already emits valid `<cot>…</cot>`. Optimization-dead.
2. **The α·R_acc term (≤0.25) is near-saturated AND gradient-redundant with the SFT MSE objective** (cf. V8 finding: `1−IBS` reward is gradient-redundant with SFT MSE; head W not effectively moved by it). It is a calibration **anchor**, not a source of gain — ~0 useful gradient.
3. **Only β·R_rank is a live optimization target**, and GRPO optimizes the **within-group spread**, not the absolute level — `cross_ad_reward.py:12`: *"the percentile floor cancels in GRPO's group-mean-subtracted advantage, so only within-group spread matters."* The advantage is `(r − group_mean)/group_std` over the G=8 rollouts of the **same** ad.
4. Therefore the real gradient magnitude ≈ **within-group `reward_std` = 0.033**, and it is **shrinking** (0.07 → 0.027 over the run). The 8 CoT rollouts of an ad barely move the head's prediction → the **single-last-token readout caps how much the CoT can change R̂** → caps `reward_std` → caps the GRPO gradient.

## Conclusion

- Hard ceiling: **1.40**; α-term ceiling **0.25**; format ceiling **0.15**.
- Of the 1.40, **0.15 (format) + ~0.245 (α·R_acc) ≈ 0.40 is gradient-dead/anchored**; the live lever is β·R_rank (current ~0.77, ceiling 1.0).
- But the **GRPO-accessible** signal is the within-group spread (~0.033, shrinking), which the single-token readout bounds. CoT-RL is operating near the ceiling of what *this reward × this readout* can extract — consistent with the temp-experiment (sampling didn't open spread) and the head-oracle saturation (~0.515). Widening the readout (attention-pool, re-SFT) is the move that raises the ceiling the reward can reach; tuning the RL knobs cannot.

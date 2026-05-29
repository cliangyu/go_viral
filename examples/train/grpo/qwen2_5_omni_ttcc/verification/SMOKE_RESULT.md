# Head-PG GPU Smoke Gate — RESULT (2026-05-29)

**Verdict: GO.** head-PG RL produces a real, usable learning signal on the actual
Qwen2.5-Omni-3B + ckpt-150 retention head. Evidence below; full log `smoke_result.log`.

Script: `smoke_head_pg.py` (backbone frozen, head-only PG on 6 real val ads, G=16,
reward = cross-ad percentile rank vs train-only CDF). This is a **mechanics gate**,
not the production lift — it proves the RL signal exists and PG can climb it; the
held-out cross-ad SRCC lift is measured in the scale run.

## GATE A — σ calibration (within-group rank variance must exist)
Clean, monotone, tunable control of exploration:

| σ | within-group reward std | reward_mean |
|---|---|---|
| 0.05 | 0.019 | 0.740 |
| 0.10 | 0.036 | 0.735 |
| 0.20 | 0.073 | 0.748 |
| 0.40 | 0.121 | 0.771 |
| 0.70 | 0.182 | 0.717 |

→ R1 (make-or-break) **PASS**. Rank-dimension variance is real and σ is the knob.

## GATE C — head-PG raises reward (24-config sweep at σ*∈{0.1,0.2,0.4})
**All 24 configs improved reward; zero collapses.** Best:

`σ=0.2, lr=1e-4, kl=0.04, adv=group_std` → reward **0.735 → 0.867 (Δ+0.132)**.

Representative: every σ≤0.4 / lr≤1e-4 / grad-clip=1.0 config climbed to ~0.81–0.87.
`group_center` (Dr.GRPO, no std) and `group_std` (vanilla GRPO) both worked.

## The earlier NO-GO and its fix (why the first run collapsed)
First attempt (`σ=0.7, lr=1e-3, kl=0, no clip`) **collapsed after one step**:
the large step drove `head.linear` into saturation → `exp(−cumsum(softplus(z)))`
underflowed to 0 for all rollouts → within-group std → 0 → advantage 0 → no gradient.
Classic no-trust-region + step-too-large. Fix (matches the GRPO HP research,
`GRPO_HPARAM_RESEARCH.md`): σ≈0.1–0.2 (calibrated, not assumed), head lr 1e-4,
grad-clip 1.0, KL≈0.04. All three knobs were too hot at once.

## Chosen starting config for the scale run
- σ = 0.2 (calibrate again on the full ad set; target within-group std ~0.07–0.1)
- head lr = 1e-4, backbone lr = 1e-6 (stage: backbone frozen first runs)
- KL β = 0.04 to SFT ref (consider adaptive, target d≈0.02)
- advantage = group_std baseline (try group_center / batch_std per Dr.GRPO / Lite-PPO)
- grad-clip = 1.0, G = 16 (memory-free; rollouts share one forward)
- init = ckpt-150, fresh optimizer, KL-ref = ckpt-150

## Caveats (honest)
- Backbone frozen + 6 ads + reward = training reward → mechanics only. The real
  metric is **held-out cross-ad SRCC during training** (Goodhart guard R3); track it.
- GATE B (reward↔quality) from the first run was **tautological** (qerr ≡ 1−reward),
  removed. The real validation of "RL helps ranking" is the held-out SRCC vs the
  1−IBS control in the scale run.

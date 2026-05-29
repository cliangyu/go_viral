# V8 RL Design — Cross-Ad Ranking via GRPO (authoritative)

**Date**: 2026-05-29
**Status**: design locked, implementation starting on the 2-card box (i-0821d01c4168eff62, us-east-1)
**Companion**: [EVAL_DESIGN_AND_LEAK_AUDIT.md](EVAL_DESIGN_AND_LEAK_AUDIT.md), [ML_RUN_HYGIENE_PLAYBOOK.md](ML_RUN_HYGIENE_PLAYBOOK.md), [METRIC_RELATIONSHIP.md](METRIC_RELATIONSHIP.md)

This captures the full design discussion (2026-05-29). The 8-card SFT run keeps training; RL is built/validated on the 2-card box.

---

## 0. The goal (unchanged)

Use **RL to improve cross-ad retention ranking** (the customer's decision: pick the best creatives). Target metric = **cross-ad SRCC** (Spearman across ads at each second, averaged over t∈[1,30]). Leak-free SFT baseline (ckpt-150, bypass) = **SRCC 0.43**. RL must push it higher.

---

## 1. Key decisions reached

### 1.1 NO CoT (Chain-of-Thought is dropped as a load-bearing component)
Evidence CoT is decorative for prediction:
- Training abandoned CoT (token_acc 0.50→0.005) while IBS kept improving — the model voted with its feet.
- Bypass (empty `<cot></cot>`, zero reasoning content) already beats baselines and uses video (ckpt-150).
- The retention head reads the **last input token** (the `</cot>` anchor matcher is dead — `[522,64498,29]` never matches in-context; verified bypass-trailing == training-trailing).

Deeper RL reason: the "generate CoT → head reads it → reward" design only works if CoT changes the head output. It doesn't (decorative) → no gradient. So CoT-based RL is a dead lever.

### 1.2 SoftRank vs RL — what they are
- **SoftRank** = supervised: a differentiable approximation of ranking (pairwise sigmoid of predicted-value differences) added as a loss term; direct gradient descent. Efficient for a differentiable objective. NOT used as the main method (project requires RL) but is the conceptual basis of the rank reward.
- **RL (GRPO)** = sample rollouts, score with a scalar reward, policy-gradient toward higher reward. Required by the project. Needs generation (sampling).

### 1.3 RL operates on GENERATED CURVE TEXT, not the head, not CoT
Because RL needs the policy's *action* to have a lever on the reward, and the head output is deterministic (no sampling → no RL), the policy must **generate the retention curve as text** `{"R": [...]}`. The reward scores the parsed curve. This requires:
- Fixing generation (the `get_rope_index second_per_grid_ts` crash in `model.generate`).
- A **cold-start SFT** that teaches the model to emit parseable R-text (distill the head's curve into a text target) — the current head-based model does NOT emit R-text.

### 1.4 We do NOT retrain a no-CoT SFT baseline from scratch
The current model in bypass IS our no-CoT predictor and it works (SRCC 0.43, uses video, in-distribution since bypass-trailing == training-trailing). Retraining is a costly full run we can't afford pre-CB (2026-06-03) with no guarantee of improvement. The CoT-was-decorative finding is reported as an **insight**, not a defect. The only new SFT needed is the targeted **R-text-emission** SFT for RL (§1.3).

---

## 2. The 5 conditions for RL to work (GRPO)

GRPO learns from `advantage_i = (r_i − group_mean) / group_std`, computed WITHIN one ad's G rollouts.

| Cond | Requirement | Obstacle | Solution |
|---|---|---|---|
| **A** | policy generates the rewarded artifact | head-based model doesn't emit R-text (cold start) | R-text SFT (distill head→text) + fix generation |
| **B** | non-trivial within-group reward variance | confident model → identical rollouts → advantage≈0 | temperature τ≈0.8–1.0, G≥8; reward with within-group spread |
| **B′** | the variance is in the RANKING dimension, not just IBS | IBS-dim variance ⟂ rank gradient (cos≈−0.01) → reward moves, rank doesn't | the rank reward's within-group spread = rollouts' disagreement on the ad's rank; smoke-gate it |
| **C** | no parse-cliff collapse | malformed text → reward 0 → gradient hijacked to format | R-text SFT (high parse from step 0) + format reward γ |
| **D** | stability | mode collapse / reward hacking | KL anchor to the R-text SFT reference |
| **E** | reward correlates with true target, not gameable | percentile proxy diverges from SRCC; median-gaming | train-only CDF (no val overlap); eval real SRCC on held-out; length-aware baseline check |

**The make-or-break is B′.** If the G rollouts of one ad differ only in overall magnitude (IBS) but agree on the ad's relative rank, the rank reward has zero within-group spread → RL cannot learn ranking.

---

## 3. Rollout design

- **State (prompt)**: `system` + `"This ad is {T}s long. Predict per-second retention."` + video. (same as SFT)
- **Action**: generate `{"R": [1.0, v1, …, vT]}` — T+1 values, no CoT.
- **Sampling**: G = 8–16 rollouts/ad, temperature τ ≈ 0.8–1.0 (tune for within-group spread).
- **Group / batch**: 1 ad = 1 GRPO group; batch = N≈16 ads × G rollouts.
- **Parser** (reuse `_parse_curve` from `ttcc_ibs_plugin.py`): extract R, force R(0)=1, running-min monotonize, clip [0,1] → R̂_i.

---

## 4. Reward design

Per rollout i of ad a:

```
r_i = β·R_rank(i) + α·R_acc(i) + γ·R_fmt(i) − η·KL(i)
```

- **R_fmt** (cond C): 1 if completion parses to a valid length-(T+1) curve, else 0.
- **R_rank** (cond E, MAIN; percentile vs FIXED train-population CDF F_t):
  `R_rank = 1 − mean_{t∈[1,30]} |F_t(R̂_i(t)) − F_t(R_a(t))|`
  Per-rollout, no batch coupling. Its **within-group spread** = how much the G rollouts disagree on where ad a ranks in the population → the ranking signal (cond B′).
- **R_acc** (cond B anchor, add ONLY if smoke shows within-group signal too weak): `1 − IBS_i`. Dense within-group spread, but in the IBS dimension (⟂ rank) → keep small so it doesn't drown the rank direction.
- **KL** (cond D): anchor to the R-text SFT reference.

**Starting weights**: β=1.0, α=0 (Occam — add only if needed), γ=0.1, η=0.02.

**Why the percentile floor doesn't hurt**: GRPO's advantage subtracts the group mean → the high floor (≈0.67) cancels; only within-group spread matters.

---

## 5. Risk register (pre-mortem)

🔴 **R1 — within-group variance in the wrong dimension (IBS not rank)** → reward moves, SRCC doesn't. *Make-or-break.* → smoke-gate R_rank within-group std.
🔴 **R2 — within-group variance ≈ 0** (rollouts too similar) → no learning. → temperature/G; smoke-gate.
🔴 **R3 — reward↑ but SRCC flat (Goodhart / reward-metric mismatch)** → fake success. → track real held-out SRCC DURING training.
🔴 **R4 — reward leak** (train-CDF overlaps val; truth leaks into reward) → V7 disease. → train-only CDF, assert val disjoint.
🔴 **R5 — reward hacking** (median curves, format-only) → high reward, no skill. → hacking detectors + length-aware baseline check.
🔴 **R6 — cold-start SFT degrades video-use** (like CoT SFT killed language) → worse start. → assert R-text SFT still beats length-aware baseline before RL.
🟡 **R7** wrong init checkpoint · **R8** SFT-vs-RL compared with different eval/B1 → invalid · **R9** declaring success from noise (1 seed, no variance bar).
🟡 **R10** RL checkpoints lost (ephemeral) · **R11** silent divergence unnoticed · **R12** racy status reads · **R13** in-place edits to shared code · **R14** restarting a healthy run.

(R4/R8/R10/R11/R12/R13 are failures we ALREADY committed this session — solutions are known.)

---

## 6. Getting it right the first time — the GATE SEQUENCE

You cannot guarantee RL produces a good result first try (empirical, high-variance). You CAN guarantee you don't waste the window or report a fake result — via cheap gates, each validated before the expensive next step:

1. **Reward unit tests (no GPU)** — feed known curves; assert: perfect→high, anti-rank→low, monotone-invariance, parse-fail→0, floor behavior. Catches reward bugs before any training.
2. **Prerequisite validation** (verifier-first):
   - R-text SFT → assert **parse rate >95% AND still beats length-aware baseline** (didn't break video-use, R6). Don't proceed otherwise.
   - generation fixed → real sampling runs.
   - train-CDF → assert train-only, **zero ad_id overlap with val** (R4).
3. **Locked eval harness** — leak-free; fixed val + content hash; **identical eval config + B1 (same train file) for SFT and RL** (R8).
4. **Smoke gate (~1 hr, few ads, ~50 steps)** — 4 checks: parse>80% · **R_rank within-group std > 0 (directly tests R1/B′)** · reward↔quality correlation>0 · reward trends up. Catches R1/R2/R7 in an hour, not a day.
5. **Pre-registered success criterion** — held-out SRCC +X pp **with variance bars over ≥3 seeds**; no goalpost-moving (R9).

---

## 7. Detecting mistakes early — live monitoring

**Core rule: track the TRUE target (held-out cross-ad SRCC) DURING training, not just the reward.** If reward↑ but SRCC flat → Goodhart (R3) caught at step K, not at the end.

- Per-K-step held-out SRCC eval (leak-free).
- Live: R_rank/R_acc/R_fmt trajectories, parse rate, KL, within-group advantage std, NaN/grad-norm.
- Reward-hacking detector: periodically sample + eyeball rollouts (all-median? entropy collapse? parse-rate anomaly?).
- **Verified monitor** (babysitter lesson): inject a synthetic divergence line, confirm it alerts.
- Process: RL checkpoints via the fixed auto-restart S3 watcher; atomic status writes (.tmp→replace); edits on isolated branch; no mid-flight restart of a healthy run.

---

## 8. Build sequence + timeline

1. **Reward unit tests** (gate 1) — today, no GPU. ← STARTING HERE
2. **Fix generation** (`get_rope_index` `second_per_grid_ts`) — 0.5–1 day.
3. **R-text SFT** (distill head→text; validate parse + video-use) — 0.5–1 day.
4. **GRPO + composite reward + KL; smoke gate (4 checks)** — 0.5–1 day. ← go/no-go in ~1hr of this step.
5. Pass gate → scale up + held-out SRCC eval — 0.5–1 day.

Total 2–4 days; the smoke gate (step 4) tells us in ~1 hour whether RL can learn — the key protection for the CB-2026-06-03 deadline.

---

## 9. Expected results (calibrated)

- SFT baseline SRCC ≈ 0.43 (ckpt-150; maybe higher at 225).
- RL with composite ranking reward: **uncertain (~50–60% to beat baseline meaningfully)** due to R1/R2. Success = +5pp+ SRCC.
- V7-style 1−IBS-only RL control: SRCC ≈ baseline (confirms the rank term is what matters).
- If RL doesn't move SRCC, that itself is a reportable finding (within-group signal analysis); the leak-free eval methodology + "model uses video" evidence + reward design carry the report regardless.

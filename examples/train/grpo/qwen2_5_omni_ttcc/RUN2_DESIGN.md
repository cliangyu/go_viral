# Run-2: cross-ad ranking reward (head-PG REINFORCE)

**Goal:** improve **cross-ad Spearman (SRCC)** of predicted retention curves, well above the
SFT baseline. ckpt-225 baseline (video 8192, audio off, n=158) = **SRCC 0.5142**.

## Why run-1 failed (empirically confirmed)

Run-1 (`rl_head_pg.yaml`, wandb `fedu0gpx`) used reward = per-ad **percentile-match vs a fixed
CDF** (`cross_ad_reward.r_rank`). Per-checkpoint held-out SRCC @ video 8192:

| checkpoint | SRCC | vs baseline |
|---|---|---|
| ckpt-225 (start) | 0.5142 | — |
| ckpt-25 | 0.5082 | −0.006 |
| ckpt-50 | 0.4930 | −0.021 |
| ckpt-75 | 0.4938 | −0.020 |

Monotone *down*. Two compounding bugs (full analysis: `REWARD_DESIGN_FOR_RANKING.md`,
`HPG_REWARD_ANALYSIS_fedu0gpx.md`; converged across a 4-angle research workflow + primary
sources + the V8 cos −0.0116):

- **Bug A (structural):** G=16 rollouts of *one ad* + within-ad advantage `(r−mean)/std`. The
  per-ad mean baseline is shift-invariant per ad, so the gradient never compares ad A to ad B.
  Cross-ad ranking is a *between-ad* quantity → **no per-ad reward can teach it** under this grouping.
- **Bug B:** `r_rank` is a *calibration* reward at its no-skill floor (a constant-median predictor
  scores ~0.78 ≈ the live ~0.79). Calibration ⟂ discrimination (Murphy decomposition; survival
  C-index-vs-Brier; LambdaRankIC). "Low MSE/IBS ⇒ correct rank" holds only in the vacuous MSE→0
  limit, not the partial regime RL traverses.

## The fix (run-2) — minimal delta, first-principles

**Only the reward changes.** Same ckpt-225 init, ZeRO-3 full-FT, KL anchor, sigma/G/lr/video-env.

Reward = **cross-ad pairwise concordance** of each rollout against a **detached running buffer**
of recent ads (the only way to inject cross-ad signal under the hard `bs=1` constraint — the head
reads the literal last token, so >1 ad per forward right-pads → train/eval mismatch; and ≥16 ads
at video-8192 won't fit one forward anyway):

```
reward_{A,g} = mean_{t in [1,30]} mean_{B in buffer, valid} σ( β · (R_Ag(t) − R_B(t)) · sign(R_true_A(t) − R_true_B(t)) )
```

- `R_Ag` = ad A's rollout-g curve; `R_B` = buffer ad B's policy-mean curve; truths known.
- Bounded [0,1]; 0.5 = chance; →1 as A is ordered correctly vs the population. This is a smooth
  soft-C-index = the per-second cross-ad rank metric the eval measures (reward ≈ eval metric, RLVR).
- The within-ad advantage now encodes *"which rollout ranks A correctly vs the population"* — a
  genuine cross-ad SRCC gradient through the **existing** REINFORCE path. No trainer-core surgery:
  one new reward fn (`verification/cross_ad_rank_reward.py`) + a `deque` buffer, switchable via
  `HPG_REWARD` (`crossad` | `rrank`).
- KL-to-frozen-SFT-head is the **sole** calibration anchor (no extra terms — keep it simple).

## Verification gate (passed before any GPU spend)

`verification/gate_cosine_crossad.py` (synthetic, CPU): cosine between the reward's REINFORCE
gradient on the head and the SRCC-improving direction.

| reward | cos(grad, SRCC-dir) | within_grp_std | dead groups |
|---|---|---|---|
| run-1 `rrank` | **+0.000** (orthogonal) | 0.000 | 100% |
| run-2 `crossad` | **+0.593** | 0.004 | 3% |

Reproduces run-1's failure and proves run-2's gradient points at SRCC. **Gate must pass before launch.**

## How to run

```
# config: configs/rl_head_pg_v2.yaml  (HPG_REWARD=crossad, output_dir=rl_head_pg_v2)
# both nodes (EFA), MASTER_ADDR = node0 private IP:
NODE_RANK=0 MASTER_ADDR=172.31.1.226 bash launch_rl_2node.sh configs/rl_head_pg_v2.yaml   # node0
NODE_RANK=1 MASTER_ADDR=172.31.1.226 bash launch_rl_2node.sh configs/rl_head_pg_v2.yaml   # node1
```

## How to evaluate (the real verifier)

Per saved checkpoint, out-of-loop on the eval box (us-east-1), video 8192 / audio off:
`rl/srcc_eval.py --checkpoint <ckpt> --val-jsonl data/val_present.jsonl --attn-impl flash_attn`.
Compare cross-ad SRCC to the **0.5142** baseline. Success = SRCC clearly and durably above 0.5142.
Keep a step→ckpt→SRCC ledger; the in-loop reward is *not* the success metric (Goodhart guard).

## Knobs to iterate (if SRCC doesn't move)

`HPG_BETA` (concordance sharpness → signal density), `HPG_SIGMA` (exploration spread),
`HPG_BUF` (buffer size → cross-ad estimate variance), `learning_rate`, `HPG_KL` (loosen if the
SFT anchor over-constrains). Re-run the cosine gate after any reward-shape change.

# RL Investigation Log — cross-ad retention-curve ranking (V8)

**Goal:** beat the SFT baseline on **cross-ad Spearman (SRCC)** of predicted short-video
retention curves, using RL. Model = Qwen2.5-Omni-3B + a retention HEAD (hazard layer reading
the last-token hidden). Eval = leak-free **bypass** forward (assistant overwritten to empty
`<cot></cot>`, head reads last token), cross-ad SRCC over t∈[1,30] on `val_present.jsonl`
(n=158), video 8192 / audio off, flash_attn pinned.

**SFT baseline (ckpt-225): val SRCC = 0.5142.** This is the number to beat.

This log records, in order, every RL run, diagnosis, and decision from today. Detailed
sub-reports: `RUN2_DESIGN.md`, and in `~/Documents/stanford/audio_video_rl/`:
`RL_CEILING_DIAGNOSIS.md`, `REWARD_DESIGN_FOR_RANKING.md`, `RL_PREREQUISITES.md`,
`PATH_EVALUATION.md`, `HPG_REWARD_ANALYSIS_fedu0gpx.md`.

---

## 1. The three RL runs (head-PG REINFORCE) — all failed to beat 0.514

Per-checkpoint held-out val SRCC (video 8192, same eval for all):

| run | reward / change | ckpt-25 | ckpt-50 | ckpt-75 | outcome |
|---|---|---|---|---|---|
| **run-1** (wandb `fedu0gpx`) | per-ad **percentile-match** vs fixed CDF (`r_rank`, a *calibration* reward) | 0.508 | 0.493 | 0.494 | **flat**, ≈baseline, no gain |
| **run-2** (`fc0y48bs`) | **cross-ad pairwise concordance** vs a running buffer, β=10, no warmup-skip | 0.487 | 0.387 | — | **collapsed** |
| **run-3** | same, **β=50, σ=0.3, + warmup-skip** (zero adv until buffer warm) | 0.503 | 0.472 | 0.452 | **monotone decline** |

Cross-cutting fact: **in-loop TRAIN reward climbed (run-3: 0.60→0.65) while held-out VAL SRCC
fell.** Classic **train→val reward over-optimization** (Gao et al. 2210.10760: gold reward vs
KL "resembles early stopping" — proxy rises, gold peaks then falls).

Per-run mechanism:
- **run-1:** per-ad reward + within-ad advantage `(r−mean)/std` is shift-invariant per ad → the
  gradient never compares ad A to ad B → cannot optimize a *cross-ad* ranking. (cosine of its
  reward-gradient vs the SRCC direction ≈ 0.0.)
- **run-2/3:** cross-ad concordance reward fixed the direction (cosine +0.5 to +0.59) but
  over-optimizes the noisy train-ad orderings. Warmup-skip (run-3) delayed but didn't prevent it.

## 2. The head-ceiling oracle — 0.514 is the *bypass-feature* ceiling

`verification/head_oracle.py`: ridge per-second (h_anchor → R(t)), K-fold CV **fit directly on
val** (no train→val transfer):

| features | CV-SRCC | note |
|---|---|---|
| last-token (bypass) | **0.5152** | == baseline; **fit-all = 1.0** (d=2048 ≫ n) ⇒ readout is NOT the bottleneck |
| mean-pooled | 0.436 | naive pooling adds nothing |
| audio-on (untrained) | 0.496 | ckpt-225 wasn't audio-trained |

⇒ **No reward/optimizer/readout change on the frozen bypass features can beat ~0.515.** The
generalizable cross-ad signal in ckpt-225's video features caps there.

## 3. The label analysis — the task is NOT noisy; the model under-extracts the LEVEL

`verification/label_analysis.py` (GPU-free, on the retention curves):

- **Rank stability:** rank(t) vs rank(t+1) Spearman = **0.998**, vs rank(t+10) = 0.96.
  → **labels are clean and extremely structured; NOT noisy.**
- **The ranking is ~1-dimensional (overall retention LEVEL)** — privileged baselines:
  rank by true R(1) → **0.78**; by R(3) → 0.90; by mean retention → **0.95**.
- **Model (from video, no privileged R) → only 0.514.**

⇒ Bottleneck is **NOT** data quantity (35,793 train ads), **NOT** label noise, **NOT** task
difficulty. It is that **the model predicts the retention LEVEL poorly from the video.** Headroom
is large (0.514 → 0.78–0.95). *Caveat:* 0.78–0.95 is "if you knew true R"; the **video-achievable**
ceiling is unknown (retention is partly set by non-video factors) and lies somewhere in
[0.515, ~0.95].

## 4. Key challenges raised (Leon) + resolutions

- **"Rewards are climbing — what's the problem?"** → that's the over-opt *symptom*: the climbing
  reward is the *train* proxy; *val* SRCC falls. The reward dial is decoupled from the goal.
- **"If the ceiling is high and clean, RL should learn it — where's the problem?"** → the high
  ceiling is *privileged* (uses true R). The model extracts only 0.515 into its features; RL on a
  frozen-feature readout **can't add video-understanding the features don't contain** (RLVR
  elicits, doesn't add).
- **"But full-FT RL backprops into the whole model — can't it reshape the video features?"** →
  *Correct, my "frozen features" framing was imprecise.* run-2/3 were full-FT (3.4B trainable);
  RL *can* move the backbone. It didn't help because (a) KL≈0.03 + lr 1e-6 → it barely moved,
  (b) the non-generalizing reward → it moved toward fitting train-noise → val declined, (c) no
  best-checkpoint selection. So 0.514 is *not* a hard wall for full-FT RL — the prior failure was
  reward + dynamics, not impossibility.
- **"Is it data量 / signal / dynamics / other?"** → not 量 (35k ads; and the oracle fit *on val*
  still caps at 0.515, so it's not a transfer/quantity issue); not dynamics-alone (those set how
  fast it over-fits, not the ceiling). It's **signal (reward over-fits) + representation (the
  model under-extracts the level)**.
- **"Save every checkpoint and pick the best at eval, vs early-stop?"** → **save-all + select on
  held-out is better** (decoupled eval + ledger; sees the full trajectory, robust to noise dips).
  Adopted.
- **"Is moving the backbone to learn features the key — or can't the dynamics learn it?"** →
  learning better features IS the key, **but RL is a weak, high-variance tool for *learning new
  features*** (elicits-not-adds; run-3 moved the backbone and val fell). **We have labels, so
  SUPERVISED is the effective, sample-efficient tool for feature-learning;** RL's proper role is
  to *elicit reasoning* or *sharpen*.

## 5. Path evaluation (Path 1 vs Path 2 + others)

- **Path 1 (make CoT meaningful):** the SFT trains a `<cot>…</cot>` reasoning trace (hook/decay/
  reward-reveal = the engagement *level*), but the bypass discards it and the `</cot>` anchor is
  dead. Favored *conditionally*; literature: Time-R1 (RL+CoT for forecasting), Rank-GRPO.
- **Path 2 (CoT-free SFT + head-RL):** stays at the proven 0.515 head ceiling; only the fallback.
- **Other:** audio-on (only NEW-information lever), rank-aware/level SFT, richer readout.
- The decisive generate-cot test (does reasoning beat 0.514) **could not be run** — the
  multimodal generate→head path in `eval_ibs.py` is broken (TMRoPE generation the team had
  abandoned); and it isn't even the clean Path-1 test (head was SFT'd on empty CoT).

## 6. DECISION (proposal, accepted)

First-principles: the bottleneck is video→retention-LEVEL extraction; the right tool to *learn
features* is **supervised** (we have labels → dense gradient ≫ RL); RL is for elicit/sharpen.

- **Step 1 (now): full-FT SUPERVISED soft-Spearman / pairwise-rank SFT** from ckpt-225 (cross-ad
  via a running buffer, + small MSE for calibration, **margin** to drop noisy near-ties), **save
  ALL checkpoints, select on held-out.** This is BOTH the most effective feature-learning AND the
  decisive, cheap test of **whether the video features can be reshaped past 0.515** (bounds the
  whole problem). If >0.515 → reshapable → proceed; if ≈0.515 → video saturated → audio is the lever.
- **Step 2 (RL win, contingent):** on the reshaped representation, **CoT-RL** (GRPO, cross-ad SRCC
  outcome reward — RL's unique value: improve the generated reasoning) or **RL-sharpen**.
- **Contingent lever:** audio-on SFT (gated on Step 1 + the AUDIO_OOB history).

Risks held in view (MECE): supervised-first risks being "not RL enough" → mitigated by Step 2 +
the RL-centric diagnosis; the deepest unknown is the video-achievable ceiling (~0.55?) → Step 1
measures it cheaply; n=158 is noisy (95% CI ±0.116) → report **paired-bootstrap** CIs, not point
estimates.

---

## 7. Execution log (Step 1 onward)

*(appended as work proceeds)*
- [pending] Implement soft-Spearman/pairwise full-FT SFT trainer (adapt head_pg scaffold: r_pred
  is grad-connected; differentiable BPR/soft-rank loss vs the cross-ad buffer; no rollouts).
- [pending] Cosine gate (supervised gradient vs SRCC direction) → smoke → launch on 2×8.
- [pending] Per-checkpoint SRCC ledger vs 0.5142; paired-bootstrap on the best.

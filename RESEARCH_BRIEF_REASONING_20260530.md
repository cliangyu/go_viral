# Research Brief: Is reasoning/CoT a lever, or is audio? (per-second R(t), stuck at 0.514 SRCC)

Date: 2026-05-30. Deadline: 2026-06-03. Audience: Leon (CS224R RL final project + Vispie thesis).
Author: Claude Code (research synthesis + 4 verified primary sources this session).

**Source tiers:** T1 = primary / peer-reviewed / official; T2 = reputable secondary; T3 = general web;
T4 = my own prior, flagged "not verified". Numbers marked [verified this session] were read from the
paper HTML/abstract during this task, not from the upstream research dump.

**Scope guard up front:** No paper trains a *per-second retention-curve* head with CoT. Every number
below is from the nearest task class (aesthetic/quality scoring, short-video engagement scalars,
LLM-as-judge, reranking, contextual time-series). Transfer to the exact target is a T2 analogy and is
flagged where it matters. Internal repo facts (dead `</cot>` anchor, head_oracle ceilings) are taken
as given from `DECISION_COT_VS_RL_20260530.md` and `ADVERSARIAL_PREMISE_BREAK_PATH1.md`.

---

## 1. PRIOR: P(each lever meaningfully beats the 0.514 SRCC ceiling)

"Meaningfully" = a CI-clean beat over the 0.5142–0.5152 readout ceiling that survives GroupKFold-by-ad.

### (a) Functional CoT / reasoning — **P ≈ 0.05–0.10** (very low)

This is the lowest-EV lever, and the literature and your own code agree on *why*.

- The bypass null is the **expected, published** outcome, not a bug. "Understanding Pure Textual
  Reasoning for Blind IQA" (arXiv:2601.02441, **T1**) documents the exact mechanism — "models may
  bypass the intermediate text altogether when predicting scores" — and shows image→reason→score CoT
  *drops* SRCC 0.907→0.790. Your `ΔIBS≈0` is this phenomenon by name.
- Sprague et al., "To CoT or not to CoT?" (ICLR 2025, arXiv:2409.12183, **T1**): meta-analysis of
  1,218 comparisons / 110 papers — CoT buys +12–14pp on math/symbolic but **+0.7pp on everything
  else**. A perception→ordinal-score head has no symbolic execution step → expected ~null.
- Aes-R1 (arXiv:2509.21871, **T1**): plain SFT-on-CoT *hurts* aesthetic scoring; only an RL
  rank+regression objective recovers it. (Abstract verified this session: backbone PLCC/SRCC +47.9%/
  +34.8% comes from the RL stage, not the CoT tokens. The per-row "0.549→0.408→0.619" figures live in
  Table 3, which I could not re-extract from the abstract page — flagged **T2** for the exact triple,
  **T1** for the direction.)
- **Code-level kill (your repo, dominant over the literature):** the `</cot>` anchor is dead — the
  head reads `h[L-1]` in *both* train and eval (`register._locate_anchor_positions` → L-1 fallback;
  `DECISION_…md:25-40`). CoT content is **never routed into the head**, so "make CoT functional" is an
  architecture + retokenize + broken-generate-path repair, not a tuning knob. Three serial unknowns
  vs a 4-day clock.
- **Information ceiling:** distilled CoT = f(video) → I(CoT; R | video) = 0 by data-processing
  inequality. CoT can only re-express bits the backbone already has at its measured ceiling; it cannot
  add exogenous signal. (`ADVERSARIAL_…md:35-43`.)

**Honest disconfirmer (why P is 0.05–0.10, not 0):** MG-IQA (arXiv:2604.09704, **T1**, verified this
session) shows reasoning IS causally load-bearing for a perception→continuous-score task — removing
multi-attribute reasoning drops SRCC **0.798→0.779 (−1.9%)**, "the largest single-component drop." So
"CoT cannot help perception→scalar" is too absolute. **But the conditions that make it work are
exactly the ones you lack:** (i) gain is averaged over 8 datasets (OOD-skewed — VQAThinker shows
reasoning gives only +0.015 *in-domain* vs +0.048 OOD; arXiv:2508.06051, **T1**); (ii) fully-trainable
backbone under a read-through-forcing GRPO/RL2R objective; (iii) authors credit "more informative
gradient signals during training" — i.e. auxiliary supervision, not inference reasoning. Your setup is
in-domain cross-ad SRCC, a single bypassable token, a dead anchor. The disconfirmer raises the prior
off zero but does not clear it, because none of its load-bearing conditions hold for you.
**Key citation: arXiv:2601.02441 (bypass null) + arXiv:2604.09704 (the bounded counterexample).**

### (b) Audio — **P ≈ 0.35–0.45** for *any* CI-clean beat; **P ≈ 0.05** for a *breakout* (>+0.05)

The higher-leverage lever, but calibrate the size: the evidence says "real, small, additive," not
"breakout."

- SnapUGC base paper (arXiv:2410.00289, ECCV 2024, **T1**, **verified this session**): vision-only
  VQA baseline **0.625 → +audio 0.636 = +0.011 SROCC**. Audio is the **4th-ranked** of 6 features.
- SnapUGC LMM paper (arXiv:2508.02516, **T1**): VideoLLaMA2 (audio) 0.691 vs Qwen2.5-VL (no audio)
  0.665 → **+0.026 SROCC**; authors call audio "important." This is the *upper* end.
- **Honest range: +0.011 to +0.031 SROCC.** Your own `head_oracle` audio-on *untrained* probe already
  reads 0.496, and `EVAL_TRAINING_ALIGNMENT` flags audio-OFF as the largest eval distribution shift —
  consistent with a real-but-modest input-side signal.
- Mechanistically it is the *only* lever that adds bits the frozen video lacks (I(audio; R) > 0),
  which is precisely what CoT provably cannot do.

**Key citation: arXiv:2410.00289 (+0.011, audio is 4th lever) and arXiv:2508.02516 (+0.026, upper
bound).** The probability is capped below 0.5 because (i) two independent ablations bound the gain at
~+0.01–0.03 on an *easier* scalar task, and (ii) on the restricted-range *ad* population the
achievable correlation headroom is compressed (see §4).

### (c) Nonlinear / richer readout — **P ≈ 0.30–0.45** (the most under-explored lever)

Both of your saturated bounds (best-linear-on-frozen and full-3B dense-rank) read essentially a
**single pooled hidden state**. That is a shared confound, not two independent ceilings.

- In SnapUGC the **single biggest gain came from richer perceptual/readout features**: video
  captioning / mid-layer features **0.651→0.689 = +0.038** [verified this session] — 3.5× the audio
  gain. The lever that moved the peer task most is *feature/readout richness*, which your two bounds
  never probed (both collapse to one token).
- Probing theory (Belinkov 2022, Comp. Linguistics 48(1), **T1**; Pimentel ACL 2020, **T1**): if a
  *linear* probe equals the strongest probe's MI lower bound, info is linearly accessible and
  exhausted *for that input*. Your linear==dense-rank==0.5152 is the textbook **input-limited**
  signature — *for the single-token readout*. A multi-token / mean-pool / attention-pooled readout is
  a different input to the probe and is **not** foreclosed by either saturated bound.

**Key citation: arXiv:2410.00289 Fig 3/Table 3 (+0.038 from richer features ≫ audio) [verified].**
This is the lever the evidence most supports actually *testing*, and it is cheap (no training; reuse
ckpt-225).

| Lever | P(meaningful beat) | P(breakout >+0.05) | Key citation (tier) |
|---|---|---|---|
| (a) Functional CoT | **0.05–0.10** | ~0.02 | 2601.02441 bypass / 2604.09704 bounded counterexample (T1) |
| (b) Audio | **0.35–0.45** | ~0.05 | 2410.00289 (+0.011) / 2508.02516 (+0.026) (T1) |
| (c) Nonlinear/richer readout | **0.30–0.45** | ~0.10 | 2410.00289 (+0.038 from features) (T1) |

*Probabilities are my calibrated synthesis (T4 — my prior over T1 evidence), not measured. The single
cheap probe in §2 converts (b) and (c) from prior to measurement in ~1–2 GPU-hr.*

---

## 2. Does the literature change the oracle-sweep test design?

Your design — {video, video+audio, video+GT-CoT, shuffled-CoT control, video-masked control} ×
{linear, MLP} — is already well-aligned with best practice. The literature makes **four refinements**,
none requiring a redesign:

**ADD (1) — MDL / online-code learning curve, not just point CV-SRCC.** Voita & Titov (EMNLP 2020,
arXiv:2003.12298, **T1**) give a *hyperparameter-free* decision rule that is purpose-built for your
exact ambiguity ("is the signal IN the representation, or is the probe/backbone fitting it?"). Run an
online-code curve over data fractions {0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100}% and
compare **codelength/compression** of video vs video+audio vs the control. If video+audio gives
strictly shorter codelength than video AND both beat the control, audio carries extractable bits the
video features lack → green-light audio with evidence, not vibes. This is the single most decision-
relevant *method* you are not yet using. (Caveat **T4**, ~70% conf: all four probing pillars use
*classification* targets; you need a Gaussian/ordinal-likelihood codelength + a rank-preserving
control. The adaptation is sound but unverified in the literature.)

**ADD (2) — explicit selectivity number, not just "MLP beats linear."** Hewitt & Liang (EMNLP 2019,
arXiv:1909.03368, **T1**): report selectivity = (real-task SRCC) − (control-task SRCC) for the MLP. A
high-SRCC MLP with *low selectivity* proves probe capacity, not representation content — this is the
trap that would make a nonlinear "win" illusory. Your video-masked + shuffled-CoT controls already
give you the control arm; just compute the difference explicitly.

**REORDER — make the readout the early, cheap fork.** Per §1(c), the single-token readout is the
shared confound across both saturated bounds. Run **video × {linear, MLP, mean-pool, attention-pool}**
*first* (training-free on ckpt-225) — it is the lever with live, evidence-backed upside (+0.038 analog)
that neither saturated bound touched. Resolve "is 0.514 a readout artifact?" before spending audio
effort.

**KEEP / STRENGTHEN — the shuffled-CoT control is load-bearing, keep it.** Ramnath et al. (EMNLP-F
2024, arXiv:2406.14511, **T1**) show *shuffled* CoT-after-label retains ~all the benefit (CSQA 69.56
vs 70.92) → CoT-as-text is a **regularizer, not reasoning transfer**. If your video+GT-CoT arm beats
bypass but video+shuffled-CoT *also* beats bypass by a similar margin, you have proven the gain is a
distillation/label-leak artifact, not functional reasoning — a clean, publishable diagnosis. Do not
drop this arm.

**DROP — nothing.** Every arm earns its place. (Optional trim only if GPU-bound: the video+GT-CoT
*MLP* cell is lowest-value, since the dead anchor means even a positive there is non-actionable
without the architecture fix.)

**HYGIENE (non-negotiable, T1):** GroupKFold by `ad_id` (no ad in both train/test — else cross-ad
SRCC is inflated by within-ad memorization); nested CV for any probe hyperparameter; fit all
standardization inside the train fold only (scikit-learn nested-CV docs; arXiv:2311.04179).

---

## 3. The single most decision-relevant paper to read in full

**Voita & Titov, "Information-Theoretic Probing with Minimum Description Length," EMNLP 2020 —
arXiv:2003.12298 (code: github.com/lena-voita/description-length-probing).**

Why this one over the flashier IQA/engagement papers: your entire decision reduces to one question —
*is there extractable retention signal that the current video-only single-token readout is missing,
and does audio (or a richer readout) supply it?* MDL probing is the **only tool in the literature that
answers that with no probe-tuning confound and a stable, seed-robust bits number** (their control task
has *higher accuracy* but ~2× longer codelength — accuracy alone could not tell signal from
memorization; codelength could). It directly converts your "which lever" guess into a measured
bits-of-information comparison you can run tonight on the idle H100, training-free. It also gives you
the negative-result framing if it comes back flat: "video-only retention signal is linearly saturated;
audio/CoT add no extractable bits under MDL probing" — a defensible, literature-grounded negative.

Runner-up to read second: **arXiv:2601.02441 (Pure Textual Reasoning for Blind IQA)** — it names your
exact bypass mechanism *and* lists the only two literature-supported ways to make reasoning causal
(self-consistency keeping visual access: 0.713→0.762; or score-conditioned generation), which is the
honest "future work" section of your write-up.

---

## 4. Is 0.514 plausibly already at the engagement-prediction noise ceiling for this data?

**Partly — and the honest comparator says you are closer to the ceiling than the 0.69 headline
implies.** Two-sided answer:

**Why 0.69 is NOT your target (the 0.69-vs-0.514 gap is mostly task mismatch, not model quality):**
SnapUGC's ~0.69 differs from your task on four axes, every one making SnapUGC *easier* —
(1) **target**: SnapUGC predicts one scalar per video (NAWP/ECR); you predict a per-second *curve*
shape; (2) **population**: SnapUGC is organic UGC with huge content variance (high achievable rank
correlation); you have paid *ads* — a narrow, homogeneous population whose **restricted range
mechanically compresses SRCC**; (3) **label noise**: SnapUGC aggregates 2,000+ users/video (very low
noise); per-ad retention has far fewer views + documented non-deterministic watch-time noise;
(4) **cross-section**: your SRCC is a cross-ad rank *at each second* averaged over t — a smaller,
harder cross-section. **The like-for-like anchor is video-*ad* CTR prediction, which lands at
correlation ~0.487 — right at your 0.514** (T2: recovered from a search snippet in the upstream dump,
PDF unparseable; flagged). The VQualA 2025 challenge (arXiv:2509.02969, **T1**) is corroborating: 8
teams on the *same easier* SnapUGC test set cluster tightly at top SROCC 0.707 / baseline 0.660 —
tight top-clustering under heavy method diversity is the signature of a **task/label ceiling near
~0.70 for the easier scalar-UGC task**, so the harder ad-retention task ceiling is plausibly in the
low-mid 0.5s.

**Why it is NOT a *proven hard* ceiling (the live upside):** your 0.514 is set by two bounds that
**both read a single pooled hidden state**. In the SnapUGC ablation the *biggest* gain (+0.038) came
from richer perceptual/readout features, not the objective and not audio [verified this session]. That
shared weak-readout confound means part of 0.514 could be a **single-token-readout artifact**, not a
true information ceiling. So: CoT is dead at the ceiling (adds no bits, head bypasses it), but the
**readout + audio** levers were never actually probed by either saturated bound and retain genuine
upside.

**Citations:** SnapUGC arXiv:2410.00289 + 2508.02516 (T1, verified); VQualA challenge arXiv:2509.02969
(T1); ad-CTR ~0.487 anchor (T2, snippet-only — the weakest link, disclosed). **No published
inter-annotator / formal noise-ceiling SRCC for short-video retention exists** (failed search,
disclosed) — your own dense-exact-gradient saturation at ~0.514 remains the best internal proxy, but it
is a single-token-readout proxy, so it bounds *that readout*, not the task.

---

## 5. Bottom line for the 4-day deadline: research-first vs probe-first vs ship-SFT

**Verdict: run the probe FIRST (tonight, training-free), with ship-SFT as the LOCKED fallback. NOT
research-first; NOT a CoT retrain.** This matches your existing `DECISION_COT_VS_RL_20260530.md`
decision tree — the research strengthens it rather than changing it.

Ordering and EV:

1. **Probe-first (tonight, ~1–2 GPU-hr, no training).** On frozen ckpt-225, run the §2 sweep with the
   MDL/online-code curve + selectivity, GroupKFold-by-ad, focusing the cheap early fork on
   **{linear, MLP, mean-pool, attention-pool} on video** and **video+audio**. This is the highest-EV
   spend of the deadline because it *measures* the two live levers (audio, readout) and definitively
   forecloses CoT — for the cost of idle-H100 time you are paying anyway. Decision rule: if video+audio
   or a richer readout beats 0.5152 CI-clean with positive selectivity → that is your Day-2 train
   target. If both land within CI of 0.5152 → STOP, ship SFT + negative result.

2. **Ship-SFT + rigorous negative result is the guaranteed deliverable, lock it Day 1.** This is
   *literature-backed and publishable*, not a consolation prize: you would be replicating the CoT-bypass
   phenomenon (2601.02441), the shuffled-CoT-regularizer finding (Ramnath 2406.14511), and the
   non-symbolic CoT null (Sprague 2409.12183) **in a new modality (per-second video retention)**, plus a
   clean dense-exact-gradient *upper-bound* proof that the rank objective has no headroom. That is a
   stronger CS224R RL narrative than a marginal +0.02 win.

3. **Research-first is the wrong call** — the research is effectively *done* (this brief + your repo
   docs converge), and four T1 lines already predict the CoT null. More reading is emotional-laziness
   substitution for the one measurement that forks the decision.

4. **Do NOT spend the 4 days making CoT functional.** It is the lowest-EV option by every line of
   evidence: dead anchor (architecture + retokenize + broken-generate repair = 3 serial unknowns),
   I(CoT;R|video)=0 ceiling, +0.7pp non-symbolic prior, and the one positive counterexample (MG-IQA)
   needs conditions you structurally lack. If you touch reasoning at all, the *only* literature-supported
   causal path is on-policy / score-conditioned generation (2601.02441) — a research bet, post-deadline.

**One-line recommendation:** Lock the SFT negative-result deliverable Day 1; spend the idle H100
*tonight* on the MDL probe over {readout richness, audio}; train only the lever that clears its
CI-clean gate; if none clears, ship the negative result — it is the publishable, evidence-consistent
outcome. **Audio > nonlinear readout > (far behind) CoT.**

---

## Source ledger

| # | Source | Venue/Year | Tier | Load-bearing claim | Verified this session |
|---|---|---|---|---|---|
| 1 | arXiv:2409.12183 To CoT or not to CoT? | ICLR 2025 | T1 | CoT +0.7pp on non-symbolic | from dump |
| 2 | arXiv:2601.02441 Pure Textual Reasoning BIQA | 2026 | T1 | bypass null; 0.907→0.790; self-consistency 0.713→0.762 | from dump |
| 3 | arXiv:2509.21871 Aes-R1 | 2025 | T1 dir / T2 triple | SFT-CoT hurts, RL recovers | abstract verified |
| 4 | arXiv:2604.09704 MG-IQA | 2026 | T1 | reasoning causal: −1.9% on removal (OOD-skewed) | **verified** |
| 5 | arXiv:2508.06051 VQAThinker | 2025 | T1 | reasoning +0.015 in-domain vs +0.048 OOD | from dump |
| 6 | arXiv:2410.00289 SnapUGC base | ECCV 2024 | T1 | audio +0.011 (4th lever); features +0.038 | **verified** |
| 7 | arXiv:2508.02516 SnapUGC LMM | ICCV VQualA 2025 | T1 | audio +0.026 (upper) | from dump |
| 8 | arXiv:2509.02969 VQualA challenge | 2025 | T1 | top cluster 0.707 → task ceiling ~0.70 (easier task) | from dump |
| 9 | arXiv:2003.12298 MDL Probing | EMNLP 2020 | T1 | hyperparameter-free signal test (the method to run) | from dump |
| 10 | arXiv:1909.03368 Control Tasks/Selectivity | EMNLP 2019 | T1 | selectivity = real − control | from dump |
| 11 | arXiv:2406.14511 CoT-Augmented Distillation | EMNLP-F 2024 | T1 | shuffled-CoT still helps = regularizer | from dump |
| 12 | arXiv:2102.12452 Belinkov Probing survey | Comp Ling 2022 | T1 | correlation≠causal use (the bypass pitfall) | from dump |
| — | ad-CTR ~0.487 anchor | — | T2 | like-for-like comparator at 0.514 | snippet only, disclosed |

**Failed searches / gaps (disclosed):** (1) No paper ablates CoT on a *per-second retention curve*
specifically — all conclusions are by analogy from the perception→scalar class (the main residual
uncertainty). (2) No published inter-annotator/noise-ceiling SRCC for short-video retention. (3) The
ad-CTR ~0.487 anchor is a search snippet (PDF unparseable) — the weakest link in §4. (4) Aes-R1's exact
0.549→0.408→0.619 triple lives in Table 3, not re-extractable from the abstract page this session
(direction T1, exact numbers T2). (5) Tavily MCP was the requested tool but returned plan-usage-limit
errors across the upstream research session; this session used WebFetch on arXiv HTML/abstract pages.

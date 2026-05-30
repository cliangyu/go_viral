# DECISION: CoT-functional-SFT vs RL-rerun vs CoT-RL vs ship-SFT (2026-05-30, deadline 2026-06-03)

## VERDICT
**Ship the SFT bypass baseline as the deliverable + write the rigorous RL negative result.
Spend ZERO new GPU on Path 1 / Path 2 / hybrid until ONE cheap probe (below) clears its gate.**
The probe is the only thing that can resurrect any CoT path; absent a clean positive, every CoT
path is refuted by code already in the repo.

## The decision tree (what fires what)

```
TONIGHT: head_oracle.py NONLINEAR + teacher-forced-GT-CoT probe on frozen ckpt-225 (no training)
   |
   |-- CV-SRCC stays ~0.515 (CI overlaps 0.5152)  --> CoT/readout DEAD. Ship SFT + negative result.
   |                                                   (modal outcome, predicted by linear oracle)
   |
   |-- CV-SRCC > 0.515 CI-clean (GT-CoT beats bypass) --> extractability channel is REAL
            |
            +-- THEN and only then: Day-2 mini-SFT with FIXED ANCHOR + GradNorm alpha,
                re-measure with --generate-cot. Hybrid CoT-RL is post-deadline future work.
```

## Why each path is what it is (code-grounded)

### Path 1 (functional-CoT SFT) — REFUTED by construction, NOT a tuning problem
1. **The `</cot>` anchor is DEAD.** `register._locate_anchor_positions` (register.py:176-177) falls
   back to `L-1` when `</cot>` doesn't match; `tokenizer.encode('</cot>')=[522,64498,29]` never
   matches in-context (NORTH_STAR.md:17). The head reads the LAST input token in BOTH training and
   eval. So raising token_acc changes WHICH tokens fill the assistant span but routes ZERO CoT
   content into `h_anchor`. token_acc is causally disconnected from the readout.
2. **The token_acc collapse is real but second-order.** RL_STRATEGY §9 (code-confirmed):
   `L = loss_curve(MSE) + 1e-3*loss_cot(CE)`; the CoT term is <1% early, ~3% late of the gradient;
   full-FT curve-MSE reshapes the shared backbone for the head and the LM head collapses
   (token_acc 0.557 -> 0.004). Fixing it needs GradNorm/uncertainty weighting or a phased schedule,
   NOT a bigger scalar alpha. But even a PERFECT token_acc lands back at ~0.44 because of (1).
3. **Even with a fixed anchor, the readout is saturated.** `head_oracle.py` (RL_INVESTIGATION_LOG
   §2): linear ridge on the bypass last-token features hits CV-SRCC **0.5152 == baseline 0.5142**,
   fit-all = 1.0 (d=2048 >> n). The readout is NOT the bottleneck. And distilled CoT is a
   re-description of the same video the backbone already encodes -> by data-processing-inequality it
   adds no exogenous signal, only re-expresses features already at their measured ceiling.

### Path 2 (re-run head-RL "along the previous path") — empirically DEAD, do not re-run
- `metrics_baseline_vs_rl.json`: rank_sft (Node A, dense EXACT-gradient, full 3B trainable) =
  SRCC 0.4553, delta +0.0161, **95% CI [-0.0151, +0.0465] (includes 0 = NULL)**, IBS DOUBLED
  0.00672 -> 0.01307 (CI [+0.0057,+0.0071], significantly WORSE). reinforce (Node B) = SRCC 0.4265
  (null/negative), IBS 0.00888 (worse).
- rank_sft is the exact-gradient UPPER BOUND on the rank objective; REINFORCE is a higher-variance
  unbiased estimator of the same gradient -> it provably cannot beat what the exact gradient already
  failed to find. Rank objectives are level-invariant -> they structurally rot calibration (the IBS
  doubling). **Only legitimate residual: held-out-best trajectory eval of existing Node-A/B ckpts**
  (cheap, no training) — a measurement, not a re-run.

### HYBRID (RL on CoT generation, retention reward) — most principled, INFEASIBLE in 4 days
- It is the ONLY framing whose optimum REQUIRES the CoT to carry retention signal — genuinely new,
  discrete action space the rank_sft upper-bound never touched. Strongest narrative.
- BUT it inherits ALL of Path 1's blockers FIRST and adds its own:
  (a) dead anchor (CoT still must reach the head — needs re-architecture + retokenize ~39k rows);
  (b) the multimodal generate->head eval path is **BROKEN** (RL_INVESTIGATION_LOG.md:103-105 —
      TMRoPE generation the team abandoned) — you cannot compute the reward;
  (c) needs a coherent CoT generator to sample from (token_acc currently 0.004 = degenerate action
      space, RL_STRATEGY §9 calls preserved generation a HARD prerequisite);
  (d) credit assignment over ~280 discrete CoT tokens with a sparse scalar IBS reward is strictly
      higher-variance than the 60-dim continuous head-PG that ALREADY failed null;
  (e) swift GRPO is text-only; a head-conditioned CoT-RL trainer is NEW code (NORTH_STAR §10).
- Six coupled unknowns, each capable of eating a full day, gated on a headroom the linear oracle
  already measures as ~zero. Park as future work.

## THE ONE DECISIVE EXPERIMENT (tonight, idle H100, NO training, ~1-2 GPU-hr)

**Extend `verification/head_oracle.py` on frozen ckpt-225, K-fold CV on the full val set, compare:**
- (L0) linear ridge on empty-CoT bypass features  -> expect ~0.5152 (reproduce the known ceiling)
- (N0) **2-3 layer MLP probe** on the SAME bypass `h_anchor` features  -> linear-vs-nonlinear gap
- (G0) linear ridge on `h_anchor` under **teacher-forced GROUND-TRUTH Gemini CoT** (assistant span
       = the real distilled CoT, single forward, NO generate, NO broken TMRoPE path)
- (P0) optional: MLP probe on mean/attention-pooled ALL final-layer states (info-present upper bound)

This sidesteps BOTH dead infra: no `model.generate` (avoids the broken TMRoPE eval), no training
(reuses frozen ckpt-225). Teacher-forcing the real CoT into the assistant span IS supported
(eval_ibs.py teacher-forced forward without --generate-cot / without --strip-assistant). It directly
forks the decision by separating "can the model PRODUCE good CoT" (irrelevant here) from "does good
CoT in-context MOVE the readout" (the only question that matters).

### GO / NO-GO
- **NO-GO (ship SFT, modal):** N0 and G0 both land within CI of 0.5152. Then the retention signal is
  neither linearly nor nonlinearly extractable beyond the saturated readout, AND a perfect CoT does
  not move it -> every CoT path is dead -> ship the bypass baseline + negative result. STOP. No retrain.
- **GO (greenlight one Path-1 retrain):** G0 (teacher-forced GT-CoT) beats 0.5152 with a CI-clean
  positive margin AND IBS not worse. Then in-context CoT genuinely carries signal -> the readout-bypass
  is the bottleneck, not the features -> one time-boxed Path-1 retrain (fixed anchor + GradNorm) is
  justified, scored with the now-meaningful generate path.

## DAY-BY-DAY PLAN (guarantees a shippable deliverable; maximizes P(beat SFT))

**Day 0 (tonight, 2026-05-30):** Run the probe above on the idle H100 (else it burns ~$30-40/hr
idle, RL_STRATEGY §7.5). In parallel, cheap & training-free: held-out-best trajectory eval of the
existing Node-A/B RL ckpts (catch any earlier sweet spot the last-ckpt overshoot hid). Both finish
overnight.

**Day 1 (2026-05-31): LOCK THE FALLBACK FIRST.** Freeze ckpt-225 + the leak-free bypass eval as the
guaranteed deliverable: SRCC 0.4393, IBS 0.00672 (beats climatology 0.00814 AND a length-aware
baseline -> "uses video", NORTH_STAR §3.4). Write the negative-result core now: rank_sft dense
exact-gradient UPPER BOUND found null rank headroom; both RL nodes null/worse; level-invariance ->
calibration rot; head_oracle linear ceiling 0.5152. This is a defensible CS224R RL deliverable
regardless of the probe.
- If probe = NO-GO: stop chasing CoT; if any GPU remains, point it at **audio-on SFT** (the only
  NEW-information lever — head_oracle audio-on untrained already shows 0.496; EVAL_TRAINING_ALIGNMENT
  flags audio-OFF as the largest eval distribution shift). Audio adds I(audio;R) the video lacks —
  the one thing CoT provably cannot.

**Day 2 (2026-06-01):** Only if probe = GO. One time-boxed Path-1 retrain from ckpt-225: (i) register
a REAL `</cot>` special token + fix the anchor so the head reads a post-CoT position, (ii) GradNorm /
phased-alpha so token_acc recovers AND head MSE holds, (iii) save-all + held-out select. Hard stop
at end of day. If token_acc does not recover or eval does not beat 0.4393 CI-clean -> revert to the
Day-1 fallback. Do NOT start the hybrid trainer.

**Day 3 (2026-06-02):** Final eval pass on the best artifact with the paired-bootstrap pipeline
(5000x) already in eval_dumps. Lock all numbers. If Day-2 produced a CI-clean beat -> ship it as the
headline + SFT as the honest baseline. If not -> ship SFT baseline + negative result + audio-on /
CoT-RL as "future work" with the upper-bound argument.

**Day 4 (2026-06-03, deadline):** Write-up only. No new runs. Report contains: the leak-free eval
construction (the real contribution), the dense-exact-gradient upper-bound proof that head-RL has no
rank headroom, the level-invariance calibration finding, the head_oracle ceiling, and the probe
result (whichever way it fell). Genuine RL content is satisfied by the head-PG REINFORCE runs + the
upper-bound analysis, independent of whether anything beat SFT.

## STRONGEST RESIDUAL RISK + KILL-CRITERION
**Risk:** burning Days 1-3 on a Path-1 retrain that arrives at ~0.44 because the dead anchor +
saturated readout were the real block, not token_acc — landing 6/03 with a rushed deliverable AND a
lost fallback. **Kill-criterion (binding):** Day-1 fallback is LOCKED before any retrain; the Day-2
retrain only starts if the Day-0 probe is GO (G0 beats 0.5152 CI-clean); the retrain is hard-stopped
at end of Day-2 and any non-CI-clean result reverts to the locked fallback. No retrain may touch
ckpt-225 or the bypass eval.

## WHERE THE EXPERTS WERE RIGHT/WRONG
- **Skeptic (Phase-1 #4) was RIGHT and decisive:** the dead `</cot>` anchor (head reads L-1 in train
  AND eval) refutes "token_acc up => CoT useful" at the mechanism level. Verified in register.py +
  NORTH_STAR.md:17. lean = ship_sft_negative_result.
- **Info-theory refutation (Phase-2 #1) was RIGHT:** CoT = f(video) -> I(CoT;R|video)=0; the
  extractability fallback is bounded by head_oracle (linear 0.5152) + rank_sft (exact-gradient null).
  Correctly demands the nonlinear probe as the only un-foreclosed test.
- **Hybrid refutation (Phase-2 #3) was RIGHT** that the hybrid is a 4-day trap (severed channel +
  measured-zero headroom + broken reward infra).
- **Refutation #2 was WRONG on one fact:** it claimed "ckpt-225 token_acc is stable at 0.55, the
  0.004 is from archived v18 lineage." FALSE. The 0.55-0.65 token_acc numbers it likely keyed on live
  in `docs/source/BestPractices/Metax-support.md` — an UNRELATED upstream ms-swift Metax tutorial
  (94-step run), not our with-cot SFT. The 0.557->0.004 collapse IS our run, code-confirmed in
  RL_STRATEGY §9. Its CONCLUSION (ship SFT + negative result) was right anyway, for the right reasons
  (anchor + oracle), so the error doesn't flip the decision.
- **Mechanistic & CS224R lenses (Phase-1 #1, #3) leaned "hybrid"** on the premise that the readout is
  the bottleneck and CoT-as-action is orthogonal to the rank_sft saturation. PARTLY wrong: they
  under-weighted the dead anchor (which severs the CoT-action channel back to h[L-1], the exact
  channel rank_sft exhausted) and head_oracle (readout already saturated). Their core valid point —
  CoT-RL is the only GENUINE test of the reasoning thesis — survives only as POST-deadline future
  work, gated on the probe.
- **Empirical & pragmatic lenses (Phase-1 #2, #5) were RIGHT** that the question is unresolved by
  argument and the teacher-forced-CoT probe forks it cheaply; lean = need_more_data /
  ship_sft_negative_result. This is the synthesis.

## HOW LEON'S PREMISE WAS PARTLY WRONG
- "token_acc up => CoT useful for retention" is FALSE by construction (dead anchor: CoT never reaches
  the head). The premise mis-diagnoses a STRUCTURAL anchor break as a TUNING problem.
- "the MSE term dominates and the model abandoned CoT" is CORRECT (RL_STRATEGY §9, code-confirmed)
  but is the SECOND-order issue. Fixing alpha alone (Path 1 "needs some tuning") lands back at ~0.44.
- "the reasoning thesis was never tested" is CORRECT — but for TWO reasons, not one: CoT generation
  is broken AND the eval deletes/never-routes the CoT (bypass + dead anchor). Both must be fixed for
  a test to exist; the probe tests the only one that matters (does in-context CoT move the readout)
  without needing either fix.
```

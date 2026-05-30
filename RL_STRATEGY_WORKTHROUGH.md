# RL Strategy Work-Through — TTCC retention head (2026-05-30)

What we were trying to do, what worked, what didn't, the math behind why, and the
decision. Evidence base: leak-free bypass eval, same val set (n=1332), fps=1.0,
both metrics computed by the decoupled `retention_metrics.py` from the per-ad
dumps. Figure: `eval_dumps/report.png`. Numbers: `eval_dumps/metrics_baseline_vs_rl.json`.

---

## 0. TL;DR (the verdict)

- **SFT (ckpt-225) is still the best model on both axes.** SRCC 0.4393, IBS 0.00672
  (beats the climatology baseline 0.00814 → it is doing real curve work).
- **Two RL/rank methods, both with the full 3B backbone trainable, both with a
  gradient orthogonal to SFT, both FAILED to beat SFT on cross-ad SRCC and both
  significantly DEGRADED calibration (IBS).**
- The decisive fact: `rank_sft` is the **dense exact-gradient upper bound** on the
  cross-ad rank objective. It moved SRCC by **+0.016 (95% CI [−0.015, +0.047] →
  includes 0, not significant)** while **doubling IBS (0.0067 → 0.0131, CI entirely
  positive)**. If the *exact* gradient on a *fully trainable* model can't find
  significant SRCC headroom, the noisy sampled version (REINFORCE) cannot either —
  and it didn't (SRCC 0.4265, also null; IBS also worse).
- **Working hypothesis: ~0.44 cross-ad SRCC is at/near the achievable ceiling on
  this data+eval for a rank objective, and rank-only objectives' only consistent
  effect is to rot calibration.** One cheap check stands between us and confirming
  this (held-out trajectory selection — §6).

---

## 1. The objective and the win condition

Predict a per-second retention curve `R(t)` for a TikTok ad from video (audio off).
Two metrics, answering different questions:

- **cross-ad SRCC** (the RL target): at each second `t`, Spearman across ads between
  predicted `R(t)` and true `R(t)`, averaged over `t∈[1,30]`. *"Do we rank ads
  correctly?"* Higher = better.
- **IBS** (Integrated Brier Score): per ad, `mean_t (R_pred(t) − R_true(t))²`,
  averaged over ads. *"Is the predicted curve numerically calibrated?"* Lower = better.

Win condition (CS224R RL deliverable + Vispie thesis): **RL beats the SFT baseline on
cross-ad SRCC**, with a paired-bootstrap CI excluding 0, without wrecking IBS.

---

## 2. The experiment ladder (what we actually ran)

All three start from the same SFT ckpt-225. `freeze_vit=true, freeze_aligner=true`;
the **LM backbone is trainable in both RL runs** (`HPG_FREEZE_BACKBONE=0`,
`tuner_type: full`). So this is **not** a frozen-representation story.

| run | objective | gradient | anchor | lr / steps |
|---|---|---|---|---|
| **SFT ckpt-225** | curve MSE | dense exact | (is the anchor) | — |
| **Node A `rank_sft`** | cross-ad pairwise-rank loss | **dense exact** | 0.1·MSE | 5e-6 / ~559 (1 ep) |
| **Node B `reinforce`** | cross-ad concordance reward (head-PG) | **sampled (REINFORCE)** | 0.03·KL to frozen SFT head | 1e-6 / ~560 (2 ep) |

Why a rank objective at all: the earlier `reward = 1 − IBS` was **gradient-redundant
with SFT** (maximizing it = minimizing the MSE that SFT already minimized → the policy
sits at the reward optimum, gradient ≈ 0, "head W never updated"). The cross-ad rank
gradient is **orthogonal to the SFT MSE gradient** (measured cosine ≈ −0.0116), i.e.
genuinely new signal — that is the entire reason to expect RL could add anything.

---

## 3. Results — both metrics (the core evidence)

(eval: n=1332, fps=1.0, leak-free bypass; paired bootstrap 5000×.)

| ckpt | SRCC | ΔSRCC (95% CI) | IBS_full | ΔIBS (95% CI) | calibration |
|---|---|---|---|---|---|
| **SFT ckpt-225** | 0.4393 | — | 0.00672 | — | beats climatology (0.00814) |
| Node A `rank_sft` (ckpt-300) | 0.4553 | +0.0161 **[−0.0151, +0.0465]** → null | 0.01307 | +0.00635 **[+0.0057, +0.0071]** → **WORSE** | worse than climatology |
| Node B `reinforce` (ckpt-75) | 0.4265 | −0.0128 **[−0.0346, +0.0078]** → null | 0.00888 | +0.00216 **[+0.0015, +0.0028]** → **WORSE** | worse than climatology |

**The SRCC/IBS divergence (why both metrics matter — Leon's point made concrete):**
On SRCC alone both runs read as a harmless null. On IBS both are *significantly worse*
(CIs entirely positive). The SRCC-only view **hid real calibration damage**.

Figure panel (3) is the smoking gun: Node A's **mean predicted curve crashes to ~0 by
t=5** while the true curve plateaus at ~0.05–0.10. The reshape kept cross-ad *ranking*
alive at t=2–13 (panel 1) while predicting **absurdly over-confident, miscalibrated
curves** — a model can rank ads correctly and still be numerically wrong everywhere.
Both RL curves are even worse-calibrated than just predicting the population-mean curve.

---

## 4. The math — why it didn't work

**(a) `rank_sft` is the upper bound, and the upper bound is ~noise.**
`rank_sft` applies a *dense, exact* gradient of the cross-ad rank loss directly to a
*fully trainable* backbone. This is the best case for "can features be reshaped to
improve cross-ad rank beyond SFT?" The answer: ΔSRCC = +0.016, CI includes 0. Since
REINFORCE is an unbiased-but-high-variance *estimator of the same gradient*,
`Var[∇̂_REINFORCE] ≫ 0` while `∇_rank_sft` is exact ⇒ **REINFORCE cannot beat what the
exact gradient already fails to find.** Node B confirms (SRCC null, slightly negative).

**(b) REINFORCE variance through a σ-perturbation on a 60-dim head output.**
`z' = μ_z + σε`, `ε∼N(0,I)`, σ=0.3, G=16. The score-function gradient is
`∇_θ log π = (z'−μ_z)/σ² · ∇_θ μ_z`, variance `∝ 1/σ² · Var(A) · ‖∇μ‖²`. Reshaping a
3B backbone through this indirect, sampled, 60-dim continuous-action channel at lr=1e-6
is an extremely low-SNR path. ~560 steps moved the model *off* SFT but not *toward*
generalizable rank gain.

**(c) Goodhart on the cross-ad buffer.** The concordance reward is computed against a
running buffer (HPG_BUF=512), a proxy for the true held-out cross-ad SRCC. The live
signature is textbook over-optimization: train reward ↑ ~0.70, `within_grp_std`
collapses to ~0.02 (→ REINFORCE gradient self-extinguishes), `dead_grp_frac` ↑, and
`muz_absmean` drifts 3.83→3.94 — the policy climbs the *proxy* while held-out SRCC stays
flat/down. The run had **`eval_strategy: no`** so this was invisible during training
(exactly what the held-out SRCC guard we built fixes).

**(d) Rank objectives are level-invariant ⇒ calibration rots.** Cross-ad Spearman and
pairwise concordance depend only on the *ordering* of `R(t)` across ads, not its *level*.
So the objective gives **zero signal pinning the curve magnitude.** The only leashes were
`rank_sft`'s 0.1·MSE and `reinforce`'s 0.03·KL-to-SFT-head — both empirically **too weak**:
IBS rose 32% (Node B) to 94% (Node A). This is not a tuning detail; it is structural —
**a rank-only objective will always be free to wander in the calibration null-space.**

**(e) So the bottleneck is the signal, not the optimizer or the capacity.** Backbone was
trainable (not a frozen ceiling). Gradient was orthogonal to SFT (not redundant). Exact
and sampled gradients both tried. All land at 0.44±noise. The parsimonious reading:
**the cross-ad rank signal has ≤noise generalizable headroom over SFT on this eval.**

---

## 5. What SFT already buys (don't throw it away)

SFT ckpt-225 IBS 0.00672 < climatology 0.00814 ⇒ the head produces genuinely calibrated
curves, and it ties the best SRCC anyone reached. **It is the production model and the
calibration champion.** Any RL "win" must be measured as *SRCC up AND IBS not worse* —
a bar neither run cleared.

---

## 6. What we do NOT yet know (honest gaps)

1. **Held-out trajectory selection — the one cheap check left.** Both configs say
   "save-all + held-out selection captures the peak even if it overshoots." I evaluated
   the **last** ckpt of each (Node A-300, Node B-75), not the held-out-*best*. Node
   A-300 is the 1-epoch overshoot end (IBS catastrophic); an **earlier** Node A ckpt may
   sit at SRCC↑ with IBS preserved — the actual intended deliverable. **This is the
   immediate next experiment** (§7.1) and it is the fork between "ceiling, ship SFT" and
   "sweet spot exists, one more run."
2. **Label-noise ceiling: still uncomputable.** The 51-col dataset has no per-viewer /
   sample-N / test-retest, so the irreducible-noise ceiling on SRCC cannot be measured
   directly (verified infeasible earlier). We infer the ceiling behaviorally (two
   methods converging at 0.44), not analytically.

---

## 7. What to do — ranked, evidence-gated

**7.1 — DECISIVE & CHEAP: held-out trajectory eval (do first).**
Eval the Node A {75,150,225} and Node B {150,300} ckpts on val_full, **both metrics**,
through the decoupled pipeline (prediction on the eval box → npz → `retention_metrics`).
Outcome rule:
  - If the *entire* trajectory sits at SRCC ≤ 0.44+noise → **ceiling confirmed → go 7.4.**
  - If some ckpt shows SRCC↑ (CI excludes 0) **and** IBS not worse → **sweet spot → go 7.2.**

**7.2 — IF room exists: one calibration-anchored, guarded RL run.**
Fix the two measured failures at once:
  - **Composite reward**: `r = concordance − λ·IBS_penalty` (or raise the MSE/KL anchor
    by 5–10×). Directly closes the level-invariance hole (§4d).
  - **Guard ON**: the held-out SRCC guard (built, step-1-verified) → over-opt visible
    live + best-ckpt selection. Kills the `eval_strategy:no` blindness (§4c).
  - **Bigger / global cross-ad pool** to shrink the Goodhart proxy gap.
  This is the *only* run worth more GPU, and only if 7.1 shows a hint of headroom.

**7.3 — Reframe the target metric (optional, if SRCC is truly capped).**
Node A *did* lift t=2–5 SRCC to ~0.60 (from ~0.55–0.59) while crashing t=1. If the
business value is the first-3-seconds hook, an **early-horizon, calibration-constrained**
metric may have headroom that the t∈[1,30] average washes out. Worth a look only after 7.1.

**7.4 — IF ceiling confirmed: ship SFT + write the negative result.**
This is a legitimate, well-evidenced CS224R RL outcome: *"rank-only RL objectives — both
exact-gradient LTR and sampled REINFORCE, each with a fully trainable backbone and an
SFT-orthogonal gradient — fail to exceed the SFT cross-ad SRCC (≈0.44) and consistently
degrade calibration, because the rank signal saturates and is level-invariant."* The
figure + paired CIs + the dense-gradient-upper-bound argument make it publishable-grade.

**7.5 — Money / housekeeping (independent, do now).**
- **i-0f1d (us-east-2, p5.48xlarge) is IDLE** — Node A finished. ~$30–40/hr burning for
  nothing → **stop it.**
- **i-06cb is still running the un-guarded, un-anchored `reinforce`** the evidence says is
  over-optimizing toward a worse model. Unless 7.1 surprises us, **kill it** (don't pour
  GPU into confirming a known failure mode). Both decisions are yours (experiments end
  manually) — flagging the ~$60–80/hr.

---

## 8. One-paragraph answer to "what should we do?"

Stop trying to beat 0.44 SRCC with a rank-only objective — the dense exact-gradient
version already proved there's ≤noise headroom, and both versions rot calibration. Run
the **one cheap held-out trajectory eval** to confirm the ceiling vs find a sweet spot.
If a sweet spot exists, do exactly **one** more run with a **calibration-anchored
composite reward + the guard on**. If not, **ship SFT and write the negative result** —
it's rigorous and complete. Either way, **stop the idle box now and stop pouring GPU into
the un-guarded reinforce run.**

---

## 9. ROOT CAUSE of the token_acc collapse (code-confirmed, 2026-05-30)

**Leon's observation ("only the retention-head loss descends, the CoT loss doesn't") is literally correct.** Mechanism, from `register.py:724-738` + `configs/sft_retention_hazard_full_with_cot.yaml:53`:

    L_total = loss_curve(MSE on R(t))  +  alpha * loss_cot(LM cross-entropy on assistant span)
    alpha = RETENTION_COT_ALPHA = 1e-3     # config-set; code default is 0.0

- `loss_curve` runs ~29 (init) → ~0.1 (end). `alpha*loss_cot` ≈ 1e-3 * ~3 ≈ **3e-3** throughout.
- So the CoT term is **<1% early (≈0.01%) and ~3% late** of the total loss/gradient. Under
  **full fine-tuning**, the dominant curve-MSE gradient reshapes the *shared backbone* purely for
  curve prediction; the ~1%-weighted CoT-CE cannot keep the LM head intact ⇒ CoT generation
  **collapses (token_acc 0.557 → 0.004)**. `loss_cot` doesn't descend — it *rises* (CE up as acc
  falls) — but its gradient is swamped, so it never steers.
- **Why a fixed scalar alpha was structurally doomed:** `loss_curve` drifts ~40–290× over training
  while alpha is constant. The LM is *least* protected exactly when the backbone is reshaping most
  violently (early). A hand-tuned fixed weight cannot balance two losses with different scales and
  dynamics. → need **gradient-balanced multi-task weighting** (GradNorm / Kendall uncertainty /
  PCGrad), or normalize `loss_curve`, or a phased schedule — not a bigger scalar alone.
- **Secondary co-factors to check:** full-FT LR too high for LM preservation; no KL-to-base-LM
  regularizer; curve-MSE scale. (alpha imbalance is the dominant, code-confirmed cause.)

**Separate issue — data quality (Leon):** the CoT targets are Gemini-generated and may not match the
video. This is NOT what collapses token_acc (a bad-target run would *plateau* acc at a moderate
level, not destroy it to 0.4%). But it sets (a) the *achievable* token_acc ceiling and (b) whether a
*coherent* CoT is *causally useful* for retention. Audit a sample for video-consistency. For an
RL-on-CoT path, imperfect Gemini CoT is an acceptable bootstrap (RL corrects it); for pure-SFT-CoT it
is a hard ceiling (you cannot SFT past wrong targets).

**Does RL need token_acc maintained? Depends on the RL:**
- *Head-RL* (on h_anchor, what we ran): does NOT need token_acc — but it saturates at 0.44. Dead end
  for leveraging reasoning.
- *Reasoning-RL* (RL on the CoT generation, retention reward): token_acc maintained is a **hard
  prerequisite** — RL refines an existing capability; it cannot bootstrap coherent generation from a
  destroyed LM (token_acc 0.004 = degenerate action space). You need preserved generation so RL has a
  real action space to sculpt toward retention-predictive reasoning.

**The target (what "where should we get to" means):**
1. SFT goal: a checkpoint where **BOTH** losses descend — head predicts the curve well (IBS ≤ current,
   SRCC ≥ 0.44) **AND** token_acc maintained (≈ base 0.5+, "维持住"). Fix the multi-task balance.
2. Diagnostic gate: with a coherent CoT, run `--generate-cot` vs `--cot-bypass` ΔIBS. If generating
   the CoT now *helps* (ΔIBS>0), the reasoning thesis is finally alive → reasoning-RL has headroom.
3. RL goal: optimize the CoT generation so the produced reasoning maximizes held-out prediction —
   beating the SFT-bypass 0.44 by making h_anchor conditioned on RL-sculpted reasoning that extracts
   more from the video than the raw representation.

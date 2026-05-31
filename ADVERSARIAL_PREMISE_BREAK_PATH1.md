# Adversarial premise-break: "CoT token_acc up => CoT useful for retention"

Lens: skeptic / premise-breaker. Date 2026-05-30. All claims grounded in repo files.

## The premise dies at the anchor, before token_acc even matters

Leon's premise has a hidden precondition: that the head can *read* CoT content.
It cannot. Two independent breaks, both fatal to Path 1 as stated:

### Break 1 (FATAL, mechanism): the `</cot>` anchor is dead in BOTH train and eval
- `HEAD_PG_RL_RUN.md:17` (verified): `tokenizer.encode('</cot>') = [522, 64498, 29]`.
  The `</` piece (522) is context-dependent and **never matches in-context**.
- `register._locate_anchor_positions` (register.py:154-178): on no match it falls
  back to `L-1` (the literal last input token).
- `head_pg_trainer.py:16-21` (assumption A1): "register._locate_anchor_positions
  returns L-1 when the </cot> matcher misses (it always does — dead anchor), in
  training AND eval. So the head's input IS the last column."
- Consequence: the head reads `h[L-1]`, NOT a hidden state positioned *after* the
  reasoning span. **Even a perfect, fully-coherent CoT is never routed into the
  head.** CoT content is structurally not load-bearing.

Therefore: token_acc -> 1.0 changes WHICH tokens occupy the assistant span, but the
head still reads the last column. There is no mechanism by which improving CoT
generation moves retention-predictive info into `h_anchor`. The premise is false by
construction. Fixing token_acc alone lands back at ~0.44 with near-certainty.

### Break 2 (the cited "smoking gun"): token_acc collapse 0.557 -> 0.004
- `RETENTION_COT_ALPHA = 1e-3` (sft_retention_hazard_full_with_cot.yaml:53);
  `total = loss_curve + alpha * loss_cot` (register.py:728-735).
- Leon's reading (MSE dominates, model abandons CoT) is *consistent* with the loss
  balance, BUT it is the second-order problem. Break 1 means even fixing this does
  nothing for retention. The token_acc collapse is a symptom of an objective that
  (correctly, given the dead anchor) places no value on CoT for the head.

## (1) Distilled CoT => no NEW causal info beyond video
The CoT targets are Gemini-distilled descriptions of the SAME video the backbone
already sees (EVAL_TRAINING_ALIGNMENT_AUDIT.md:18 — CoT cites visual+audio cues from
the clip). CoT is a *re-description* of the input, not exogenous signal. Best case,
a functional CoT is a lossy bottleneck on information the backbone already has direct
access to. It cannot raise the information ceiling; it can only re-encode. So even
with a live anchor, the upside is "elicit reasoning the model already could do," not
"add new predictive signal." For a regression head reading a pooled hidden, the
direct-feature path strictly dominates the via-text path on information content.

## (2) The shortcut problem: high token_acc still leaves CoT decorative
Suppose we ALSO fix the anchor (move it to a real post-CoT token) AND get token_acc
high. The head can STILL read video features directly via the backbone's residual
stream at the anchor position and ignore the CoT tokens' content. Nothing forces the
head to *route through* CoT. What would force non-shortcut: (a) a representational
bottleneck — anchor on a position whose only path to retention info is THROUGH the
generated text (e.g. generate the curve AS text, or sample CoT and reward downstream
IBS so gradient flows only via sampled tokens); (b) ablation pressure — train with
CoT dropped on a fraction of steps so the head cannot rely on it, then measure
generate-cot vs bypass ΔIBS as the live target. Path 1 as stated has neither. So even
the optimistic version of Path 1 has no force against shortcut -> CoT stays decorative
at high token_acc. This is exactly what the ΔIBS≈0 diagnostic already measured.

## (3) The 4-day burn scenario (concrete)
Day 1: rebalance alpha (e.g. 1e-3 -> 1e-1), relaunch 10-epoch SFT from base. ~real
training time on the shared cluster. Day 2: token_acc recovers to ~0.5-0.7. Run
eval_ibs --generate-cot --cot-bypass to measure ΔIBS. ΔIBS≈0 again, because the
anchor is dead (Break 1) — CoT never reaches the head regardless of token_acc. Day 3:
realize the anchor, not the CoT, is the block. Attempt to fix the anchor matcher
(register a real special token, retokenize the 39k train set, re-verify the head
reads post-CoT h). This is a data + tokenizer + retrain cycle, not a tuning knob.
Day 4: even if the anchor is fixed, shortcut (problem 2) means generate-cot still ties
bypass; and generate-cot eval path is independently BROKEN (HEAD_PG_RL_RUN.md:103-105,
eval_ibs multimodal generate->head TMRoPE path abandoned), so you cannot even MEASURE
whether CoT helps without first repairing generation. Deadline hits at ~0.44, no
deliverable improvement, and the negative result is now rushed instead of clean.

## (4) Minimal evidence to JUSTIFY Path 1, and KILL-CRITERIA

MINIMAL GREEN-LIGHT EVIDENCE (all required, cheap, in priority order — none needs the
4-day retrain):
- E1 (anchor is live): print `_locate_anchor_positions` output on 10 real rows and
  confirm it returns a position INSIDE the `<cot>...</cot>` span (not L-1) for >0 rows.
  If it returns L-1 everywhere, Path 1 is dead on arrival. [~30 min, no training.]
- E2 (CoT carries info the video-only head lacks): head_oracle-style probe —
  ridge/linear fit R(t) from h_anchor under (a) bypass empty CoT vs (b) teacher-forced
  REAL distilled CoT, on the EXISTING ckpt-225, K-fold CV. (verification/head_oracle.py
  pattern, RL_INVESTIGATION_LOG.md:41.) If teacher-forced-CoT CV-R(t) does not beat
  bypass by a CI-clean margin, the distilled CoT has no extractable retention signal
  beyond video -> Path 1 cannot help. [~1-2 hr, no training, requires E1 fixed.]
- E3 (generate-cot eval works): the eval_ibs --generate-cot path must actually run end
  to end on >=20 ads (currently broken per HEAD_PG_RL_RUN.md:103-105). Without E3 you cannot
  measure the deliverable claim, so Path 1 is unfalsifiable -> do not commit.

KILL-CRITERIA (abort Path 1 immediately if ANY fires):
- K1: E1 shows anchor = L-1 on all rows AND you cannot register a real `</cot>` special
  token + retokenize within Day 1. (anchor unfixable in budget)
- K2: E2 shows teacher-forced-CoT oracle ΔSRCC or ΔIBS CI includes 0. (no info in CoT)
- K3: After alpha-rebalance retrain, generate-cot vs bypass ΔIBS CI includes 0 on
  val_full. (CoT still decorative = original symptom reproduced)
- K4: Any single attempt against the same verifier fails twice (per executor policy).
- Time-box: if E1+E2+E3 are not all green by end of Day 1, ship the SFT negative result.

## Verdict
Path 1's premise ("token_acc up => CoT useful") is FALSE by construction: the dead
anchor (Break 1) means CoT content never reaches the head, so token_acc is causally
disconnected from retention. Path 1 is not a tuning job; it is an architecture change
(anchor) + a data change (retokenize) + a broken-eval repair (generate path) +
an unforced-shortcut risk — four serial unknowns against a 4-day clock, with the
ceiling on upside capped by the fact that distilled CoT adds no info beyond the video.
Path 2 (re-run RL) is separately near-dead: rank_sft is the dense exact-gradient upper
bound and found <=noise SRCC headroom while doubling IBS. The honest, deliverable-grade
move is ship the SFT result + write the negative result, with the one cheap held-out
trajectory eval (RL_STRATEGY §7.1) as the only remaining check. If anything gets GPU,
it is audio-on SFT (the only NEW-information lever, EVAL_TRAINING_ALIGNMENT_AUDIT.md:27)
— not CoT.

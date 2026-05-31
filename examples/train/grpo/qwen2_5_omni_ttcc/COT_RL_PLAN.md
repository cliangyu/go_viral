# CoT-RL Plan — GRPO on CoT sequences (V8 retention)

**Created 2026-05-31.** Forward direction after head-PG RL came out null (see `HEAD_PG_RL_RUN.md`).
Goal unchanged: improve **cross-ad retention SRCC** (Spearman across ads, t∈[1,30]).

CoT-RL = the model **generates a `<cot>…</cot>` reasoning sequence**, the **retention head reads the
post-CoT last-token hidden state** → curve, and **GRPO** rewards curves that rank ads correctly.
It is the *only* framing whose optimum **requires** the CoT to carry retention signal — a genuinely
different (discrete, larger) action space than head-PG's 60-dim continuous hazard noise that ran null.

---

## 0. HONEST STATUS — code side vs config-research side vs gate

| Axis | State | Detail |
|---|---|---|
| **Config research** | **partial** | Reward math (`verification/cross_ad_reward.py`, Gate-1 verified), GRPO hyperparameters (`grpo.sh` + `verification/GRPO_HPARAM_RESEARCH.md`), and CoT length→`max_completion` (measured p99=430) are done. **Not settled:** the head-forward reward *channel* for CoT-RL, the exploration temperature needed for within-group **curve** diversity, and the make-or-break gate. |
| **Code** | **NOT built** | No CoT-RL trainer/reward exists. What exists: (a) `rl/head_pg_trainer.py` — *different* formulation (continuous hazard noise, ran null); (b) `grpo.sh` + `plugin/ttcc_ibs_plugin.py` + `plugin/ttcc_format_plugin.py` — the **abandoned text-channel** GRPO (parses `R=[...]` from text, per-ad IBS reward); (c) `/tmp/generate_eval_fixed.py` (box) — the channel-test eval = the *rollout + head-readout half*, **not wired into a GRPO loop**. The head-conditioned CoT-RL reward (forward the head on the generated CoT) is **unwritten**. |
| **Gate** | **running** | Channel test v2 on i-0f1d: does the generated CoT move the head's curve beyond bypass? (~48/168 ads as of 2026-05-31 09:20Z; DELTA SRCC prints at the end.) |

---

## 1. Formulation

- **Init** from the **SFT-with-CoT (shift-fixed)** checkpoint (training now on i-06cb,
  `configs/sft_retention_hazard_lora_v8_with_cot.yaml`). Post-shift-fix this model **generates
  coherent, grounded CoT** (channel-test samples: "0s | man on table … strong hook"; "0s | animated
  character … strong visual hook") — the degenerate-generator blocker (token_acc 0.004) is cleared.
- **Per ad** (state = video + "this ad is T s long, predict retention"): sample **G** CoT completions
  via `generate` at temperature τ. Each completion conditions the head's last-token hidden state →
  curve `R_g(t) = exp(−cumsum(softplus(z_g)))`.
- **GRPO update**: within-group advantage `A_g = (r_g − mean_g)/std_g` over the G rollouts; policy
  gradient on the **CoT tokens'** logprobs `∇ log π(CoT_g)·A_g`; KL to the SFT reference.

## 2. Reward — 3 terms, each justified  (read the head curve, NOT parsed text)

`r_g = β·R_rank + α·R_acc + γ·R_fmt`   (KL handled by the trainer)

| Term | What | Weight | Evidence / why |
|---|---|---|---|
| **R_rank** (PRIMARY) | cross-ad percentile match vs a **fixed train-population CDF**, `1 − mean_t|F_t(R̂)−F_t(R_true)|`, t∈[1,30] (`cross_ad_reward.r_rank`) | **β=1.0** | The shared CDF makes the within-group spread encode **cross-ad** rank. **Lesson (verified the hard way):** a *per-ad* reward cannot move cross-ad SRCC under GRPO because the group-mean-subtracted advantage cancels the per-ad level. Gate-1: within-group spread is rank-specific (std 0.23 rank-diverse vs 0.0004 magnitude-only). |
| **R_acc** (CALIBRATION ANCHOR) | `1 − IBS` (`cross_ad_reward.r_acc`) | **α=0.2–0.3** | **KEPT, not dropped.** head-PG's rank-only objective **rotted calibration**: IBS *doubled* 0.0067→0.0131 (`eval_dumps/metrics_baseline_vs_rl.json`, level-invariance). α anchors calibration without dominating the rank signal (rank ⟂ IBS, cos≈−0.01). |
| **R_fmt** (STRUCTURE GUARD) | valid `<cot>…</cot>`, non-empty interior, length≤512 | **γ=0.1–0.2** | Anti-degeneration insurance early. **The existing `ttcc_format` checks for `R=[...]` text — WRONG channel** for CoT-RL (our completion is reasoning; the curve comes from the head). Must be rewritten for CoT structure. |

**Channel correction (the one thing not to get wrong):** R_rank/R_acc must be computed from the
**head curve** obtained by forwarding the model on `(prompt + generated CoT)` and reading the
last-token hidden → head. The repo's `parse_curve` / `ttcc_ibs` are **text-channel** (parse a curve
out of the words) — that was the abandoned R-text path. CoT-RL rewards the **head**, conditioned on
the CoT.

## 3. Hyperparameters — text-GRPO formulation → `grpo.sh` is the correct evidence base

> Explicit, to avoid repeating a past error: CoT-RL is **text-generation GRPO**, so its HP evidence is
> `grpo.sh` (project-tuned text-GRPO) — **NOT** head-PG's σ/lr/kl (that is a *different* formulation,
> continuous Gaussian on hazards; citing it here would be the wrong-formulation mistake).

| Knob | Value | Source |
|---|---|---|
| `--rlhf_type` | `grpo` | swift GRPO (text) |
| init adapter | SFT-with-CoT shift-fixed | warm-start (grpo.sh `--adapters $SFT_CKPT`) |
| `--num_generations` G | **8** | grpo.sh used 4 ("literature min"); raise to 8 for group rank statistics + the curve-variance gate |
| `--temperature` / `--top_p` | **0.4 / 0.95** *(provisional)* | grpo.sh. **Flag:** τ sets CoT diversity → within-group **curve** variance; if curves collapse (std≈0) raise τ. Calibrate at the smoke (R1-CoT gate). |
| `--max_completion_length` | **512** | measured CoT tokens p99=430, max 527 (clean_v2, n=1500). grpo.sh's 1024 was the longer R-text channel. |
| `--beta` (KL) | **0.001** | grpo.sh ("0.04 was 16× too strong"); KL→SFT ref keeps CoT coherent / anti-reward-hack |
| `--learning_rate` | **5e-6** | grpo.sh, LoRA |
| tuner | LoRA r16/α32, `all-linear`, `--freeze_vit/--freeze_aligner true` | grpo.sh; text-decoder only |
| precision / mem | bf16, grad-checkpointing, `--deepspeed zero2`, vLLM colocate `mem_util 0.35` | grpo.sh |
| eff-batch | **recon the fleet + match bs×ga×world_size** before launch | run-hygiene lesson (eff-batch must be matched, ga is derived) |
| reward_funcs | **new** `ttcc_head_rank` (RM plugin) + rewritten `ttcc_cot_format` | replaces `ttcc_ibs_reward ttcc_format` |

## 4. Rollout + the new code (what to build)

1. swift GRPO generates G CoT per ad (vLLM colocate) — reused verbatim.
2. **NEW: head-forward reward = an RM plugin** (the `DefaultRMPlugin` pattern, `swift/rewards/rm_plugin.py`:
   re-encode `(prompt+CoT)` → forward `self.model` under inference_mode → last-token h → head curve →
   `cross_ad_reward(R_g, R_true, train_cdf)`). A pure-text ORM only sees the completion string and
   cannot read the head — so it must be the RM-plugin (model-access) path.
3. **NEW: `ttcc_cot_format`** ORM — `<cot>…</cot>` structure, not `R=[...]`.
4. The rollout+readout half already runs in `generate_eval_fixed.py` (channel test) → reuse its
   LoRA-load + gen-batch-strip + `head_curve()` for the plugin.

> Trainer choice: prefer **swift GRPO + the head RM plugin** (no trainer edit) over forking
> `head_pg_trainer.py` (that trainer has no text-generation path). If the RM-plugin forward can't share
> the policy forward and is too slow, fall back to a custom GRPO-trainer subclass per `HEAD_PG_RL_RUN.md` §10.

## 5. Validation / observability (Goodhart + calibration + R1-CoT guards)

- **Held-out cross-ad SRCC DURING training** (not just reward) — R3 Goodhart guard. head-PG's reward
  rose 0.735→0.867 while held-out SRCC stayed null; never trust reward-up alone.
- **Held-out IBS during training** — calibration guard (head-PG rotted it).
- **Within-group curve std** logged from step 1 — the **R1-CoT make-or-break**: if different CoTs give
  near-identical curves (std≈0), advantages vanish → no learning. This is the channel-test question, online.
- Same leak-free **bypass eval + B1** as SFT; ≥3 seeds + variance bars for any headline.

## 6. Full vs LoRA

Start **LoRA** (warm-start adapter, cheaper, RL-stable, grpo.sh precedent). Merge→full only if the head
ceiling binds — but note **nodeA already ran full-3B exact-gradient and found null rank headroom**, so
full-FT is not obviously the unlock here.

## 7. THE GATE + honest risk — read before spending GPU

**Decisive question:** does the CoT move the head's readout **beyond bypass**?

- **Channel test (running):** generated-CoT REASONED vs empty-CoT BYPASS SRCC on held-out val.
  *Alignment control first:* confirm the BYPASS arm reproduces the known `srcc_eval` on the same ckpt
  before trusting the REASONED−BYPASS delta.
- **Cleaner (recommended):** teacher-forced **ground-truth** CoT probe (`DECISION_COT_VS_RL_20260530.md`
  G0) — assistant span = real distilled CoT, single forward, **no `generate`** (sidesteps the noisier
  generate path). Separates "can it produce good CoT" (yes) from "does good CoT move the readout" (open).

**Cautionary evidence (code-grounded, 2026-05-30) — the disagreement I owe you:**
- `head_oracle.py`: linear readout ceiling on bypass features **CV-SRCC 0.5152 ≈ baseline 0.5142** → the
  readout is already **saturated**.
- head-PG (dense exact-gradient *upper bound* + REINFORCE) both **null on rank, worse on IBS**.
- Info-theory: CoT = f(video) ⇒ `I(CoT; R | video) = 0` — distilled CoT re-describes the video the
  backbone already encodes, so it adds **no exogenous signal**. The only new-information lever is **audio**.
- **If these hold, CoT-RL is bounded at the video ceiling** → it will not beat SFT on ranking.

**Why CoT-RL is still worth the gate:**
- Larger action space than head-PG: the CoT steers the backbone's last-token features, not just the
  60-dim hazards the rank-SFT upper bound exhausted.
- The **shift-fix is new since that decision doc** — a coherent CoT generator now exists (was degenerate
  token_acc 0.004), clearing one of the six blockers.
- It is the only genuine test of the reasoning thesis and the strongest CS224R RL narrative.

**DECISION RULE (gate-first):**
- **GO** — REASONED-CoT (or GT-CoT) beats BYPASS **CI-clean** on held-out SRCC **and** IBS not worse →
  run the consolidated config above.
- **NO-GO** — within CI of bypass → CoT-RL is upside-bounded; redirect GPU to **audio-on SFT** (the only
  lever that adds information CoT provably cannot), and write CoT-RL up as the principled upper-bound result.

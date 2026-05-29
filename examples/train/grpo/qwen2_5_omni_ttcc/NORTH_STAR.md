# NORTH STAR — V8 Retention RL (single source of truth)

**Updated**: 2026-05-29. Supersedes the docs in `archive_superseded_20260529/` (kept only for reasoning trail). Companion: `ML_RUN_HYGIENE_PLAYBOOK.md`, `verification/` (scripts + outputs = evidence).

---

## 1. Goal
Use **RL to improve cross-ad retention RANKING** — the customer's real decision: "of these creatives, which retain best?" Report deliverable for CS224R. Target metric = **cross-ad SRCC** (Spearman across ads at each second t, averaged over t∈[1,30]).

## 2. Big picture (where we are)
- We built a **leak-free eval** and got the **first trustworthy number**: SFT **ckpt-150** reaches **cross-ad SRCC 0.43** and **beats a length-aware baseline** → it genuinely **uses the video** (not just the ad-length scalar). This is the positive result we never had.
- Plan: from this SFT head, **RL** pushes ranking higher. RL is well-motivated because **value-accuracy (MSE) and ranking are orthogonal objectives** (measured cos≈−0.01) — so SFT-MSE structurally cannot improve ranking past a point; RL optimizes the true ranking metric directly.
- The RL design is the **clean one**: keep the head, make its hazard output a Gaussian policy, policy-gradient on sampled curves. **No CoT, no text generation, no generation-bug, no cold-start.**

## 3. Verified findings (evidence in `verification/`)
1. **V7 was a leak.** Its famous IBS=0.000092 was fake — ground-truth R was in the assistant span; 3 audits showed it ignored video. DON'T trust any V7 number. (INFERENCE.md on the box.)
2. **The retention head reads the LAST input token.** The `</cot>` anchor matcher is dead — `tokenizer.encode('</cot>')=[522,64498,29]` never matches in-context (the `</` token is context-dependent). True in training AND eval. So CoT content is not load-bearing.
3. **Leak-free eval = `--generate-cot --cot-bypass`** (assistant overwritten to `<cot></cot>`, single forward, no `model.generate`). Verified bypass trailing tokens == a real training row's → in-distribution. (`--strip-assistant` empty-assistant is OFF-distribution; its "B1 wins" was invalid.)
4. **Checkpoint sweep (bypass, n=134):** ckpt-75 (step75): SRCC 0.26, loses to length-aware baseline. **ckpt-150 (step150): SRCC 0.43, IBS 0.0150 < B1 0.0180, BEATS length-aware baseline 0.0150<0.0158 → uses video.** Later=better (bypass doesn't need language). 34/168 skipped (max_length>32768, long videos).
4b. **Cross-ad SRCC vs per-ad IBS:** different objectives; the mean baseline B1 gets within-ad Spearman 0.97 but cross-ad SRCC ≈ 0 (can't rank). IBS↑ ⟂ rank↑ (cos≈−0.01).
5. **Reward verified (Gate 1, `verification/test_*`):** the cross-ad rank reward's within-group spread is **rank-specific** (std 0.23 for rank-diverse rollouts vs 0.0004 for magnitude-only) — validates the make-or-break design property.
6. **Data quantities:** train 39,375 / val 4,906 / test 4,941 (clean 80/10/10 of ~49.2k, ad_id-disjoint). `val_200_no_cot.jsonl` is an arbitrary `--limit 200` smoke subset (effectively 168 — partial video mirror on the 2-card box). Real eval should use the full val split + B1 from the model's actual train file.

## 4. The model (unchanged)
Qwen2.5-Omni-3B backbone + **retention head**: `z=linear(h_last)` → `λ=softplus(z)` → `R(t)=exp(-cumsum(λ))` (monotone by construction). Input = video + text prompt; output = R(t) curve via the head.

## 5. Data design
- Source: `liangyuch/ttcc-v0_2_0` (video + true curves) ⋈ `ttcc-cot` (CoT — **now unused**).
- Each example: `{system, user("This ad is Ts long, predict retention"), video, R(true curve), T}`. Assistant span unused (no CoT).
- **train-CDF F_t**: per-second value distribution from the 39,375 train curves (`rl/train_cdf.npz`), train-only, zero val overlap → leak-safe percentile reference.
- RL trains on TRAIN ads (reward uses each ad's true R + the train CDF). Eval on held-out VAL.

## 6. Pipeline (input → output)

| Stage | Input (state) | Action / output | Supervision / reward |
|---|---|---|---|
| **SFT** (done, 8-card) | video + prompt | head → curve μ | MSE vs true R → ckpt-150 (SRCC 0.43, uses video) |
| **RL** (head Gaussian policy gradient) | video + prompt | sample z' = μ_z + ε (ε~N(0,σ²) **on the hazards**), curve = exp(−cumsum(softplus(z'))); G≥8/ad | r = β·R_rank(percentile vs train-CDF) + α·R_acc(1−IBS, anchor if needed) |
| **Eval** | val video + prompt | head μ (no noise) | cross-ad SRCC + IBS + vs length-aware baseline |

**RL loop**: forward head → sample G hazard-perturbed curves per ad → reward each → within-ad advantage `A=(r−mean)/std` → policy gradient `∇log π(z'|s)·A` (Gaussian log-prob on hazards) + KL/PPO-clip to the SFT head. Update head+backbone. **No text, no generate(), no parse, no CoT.**

**Sampling choice**: Gaussian on the **hazards** (unconstrained → preserves monotone constraint, exact log-prob, σ = exploration knob). Alternatives: learned-σ (upgrade), discretized categorical (if multimodal needed). NOT on the curve directly (would break monotonicity + bias the gradient).

## 7. RL motivation — honest
- **Strong** for "improve ranking beyond SFT": ranking is orthogonal to MSE (cos≈−0.01), so SFT-MSE provably can't get there; RL optimizes the true (non-differentiable) SRCC reward directly.
- **Moderate** vs SoftRank: a differentiable soft-rank loss would also target ranking, supervised and cheaper. RL's distinct value: optimizes the EXACT metric (no surrogate), and is the course requirement. The head-PG formulation makes it tractable (no generation issues), so it's now a clean, low-overhead RL.

## 8. Risks + gates (full register in archive/RL_DESIGN; the live ones)
- 🔴 **R1 within-group variance in the RANK dimension** (make-or-break): smoke-gate the sampled curves' R_rank std on real rollouts.
- 🔴 **R3 reward↑ but SRCC flat (Goodhart)**: track held-out SRCC DURING training, not just reward.
- 🔴 **R4 reward leak**: train-CDF train-only, val-disjoint (verified).
- 🟡 R8 invalid SFT-vs-RL comparison (same eval/B1); R9 noise-as-result (≥3 seeds, variance bars); R10–14 infra (backup watcher fixed; atomic status; branch isolation; no mid-flight restart).

## 9. Status + next
- ✅ Leak-free eval + ckpt-150 baseline (SRCC 0.43, uses video); incident closed (backup watcher fixed); reward built + unit-tested (Gate 1, B′ validated); train-CDF ready.
- ✅ **GPU SMOKE GATE PASSED (2026-05-29) → GO.** On the real model+ckpt-150: σ-calibration shows tunable within-group rank variance (σ 0.05→std .019 … 0.7→.182); head-PG raised the percentile-rank reward in **all 24 sweep configs, zero collapses**; best `σ=0.2, lr=1e-4, kl=0.04, group-std` reward **0.735→0.867**. Evidence: `verification/SMOKE_RESULT.md` + `smoke_result.log` + `smoke_head_pg.py`. (First attempt `σ=0.7/lr=1e-3/kl=0/no-clip` collapsed via head saturation — fixed by trust region + calibrated σ + grad-clip, matching `GRPO_HPARAM_RESEARCH.md`.)
- ▶ NEXT: build the **swift HeadPGTrainer** (see §10) and **scale on 2×8 H100**, tracking **held-out cross-ad SRCC during training vs the 1−IBS control** (Goodhart guard R3). Chosen start: σ=0.2 (recalibrate on full set), head lr 1e-4 / backbone lr 1e-6 (backbone frozen first), KL β 0.04 → ckpt-150, grad-clip 1.0, G=16.
- (Dropped vs earlier plans: CoT, generate-R-text, the `get_rope_index` generation fix, R-text cold-start SFT — all unnecessary under head-PG.)

---

## 10. RL IMPLEMENTATION DECISION + ACTION PLAN (2026-05-29 cont.)

### Verified fact that drives the design
`swift/rlhf_trainers/grpo_trainer.py` (2717 LOC, `GRPOTrainer(RolloutTrainerMixin, SwiftMixin, HFGRPOTrainer)`) is **purely text-generation**: vllm/transformers `generate` of completions, `num_generations`, `max_completion_length`, per-token logps, temperature sampling. **No continuous/Gaussian/head support.** → head-PG (continuous Gaussian on hazards) **cannot use swift's GRPO trainer**; using it forces the text path (generation-fix + R-text SFT + parse-cliff + abandons head).

### DECISION: custom Trainer SUBCLASS inside ms-swift (NOT a standalone loop, NOT swift GRPO)
Integrate head-PG as a **custom Trainer subclass** of swift's SFT/Seq2Seq trainer (override `compute_loss` / the training step to do: forward → hazards z → sample → curve → reward → REINFORCE). This **reuses swift's model+template loading, multimodal data collator, distributed/multi-GPU, optimizer, checkpointing, logging** — only the RL loss is new. (Confirmed by Leon: training can use the 8-card cluster → multi-GPU is desirable → reusing swift's distributed beats a hand-rolled single-GPU loop. A4 dropped.)
- **Investigate (Track A, in flight)**: exact swift extension point — how `rlhf_type`/CLI selects a trainer, the `SwiftMixin.compute_loss` signature, whether a custom trainer can be registered/passed. Fold result here.

### Change / Keep / Reset / Danger
- **KEEP (don't touch)**: `register.py` (model+head; read-only — load-bearing for SFT+eval), ckpt-150, the bypass leak-free eval, `cross_ad_reward.py` (Gate-1 verified), data.
- **CHANGE (new code)**: a custom trainer subclass (`compute_loss` = head-PG REINFORCE), reusing swift plumbing. New: hazard sampling + reward call + advantage + KL. `reinforce_core.py` holds the verified math.
- **RESET (归零)**: RL inits from ckpt-150 weights + **fresh optimizer** + KL-ref=ckpt-150. **Ignore the old text-GRPO scripts** (`grpo.sh`, `grpo_v2cot_full.sh`, `ttcc_ibs_plugin.py` reward) — they are the abandoned text path. Tokenizer/model structure NOT reset.
- **DANGER**: (a) modifying `register.py` → breaks predictor; (b) hand-rolled RL math bugs → unit-test in isolation FIRST; (c) **gradient leak — reward/advantage/action MUST be detached**, grad only via logpi(mu_z); (d) sample on hazards z NOT the curve (else biased); (e) don't overwrite ckpt-150; back up RL ckpts via the fixed watcher; (f) eval RL with the SAME bypass+B1 on held-out val.

### Assumptions + verification status
- A1 (swift allows a clean custom Trainer subclass / compute_loss override) — **VERIFIED against ms-swift source**. The swift-idiomatic way to add a new training algorithm = a Trainer subclass wired like swift's own GRPO/DPO (`TrainerFactory.TRAINER_MAPPING`, `swift/trainers/trainer_factory.py:11`). The official *loss* plugin (`docs/.../Architecture.md` → `BaseLoss.__call__(outputs, labels)→scalar`, sft/pretrain/reranker/embedding only) is **insufficient for RL**: it only sees (outputs, labels) and can't sample G rollouts or compute logπ of sampled actions (those need the training step + model). So a Trainer subclass is required, not a loss plugin.
  - **CHOSEN MECHANISM (shared-fork-safe, zero swift-core edits):** a custom entry script in `examples/.../rl/` that (1) imports `register.py` (model_type/template/forward already registered), (2) defines `HeadPGTrainer(Seq2SeqTrainer)` overriding `compute_loss(self, model, inputs, ...)` to do head-PG REINFORCE: `model(**inputs)` → `h_anchor = model._retention_h_holder.last[:,-1]` → `mu_z = model.retention_head.linear(h_anchor).float()` → sample G hazard-noised z' → `cross_ad_reward` per rollout → within-ad advantage → `reinforce_core` loss + KL→ckpt-150, (3) defines `HeadPGSft(SwiftSft)` overriding only `run()` (copy of `swift/pipelines/train/sft.py:172`, swapping `trainer_cls = HeadPGTrainer` at line 188), (4) `HeadPGSft(args).main()`. Everything else — model load, multimodal template/collator, dataset, DeepSpeed/torchrun multi-GPU, checkpointing, logging, the `train()` wrapper — is reused verbatim from `SwiftSft`. Launch mirrors `sft.sh` (NNODES/NODE_RANK/MASTER_ADDR → torchrun) for 2×8 H100.
  - NOTE: this mechanism is from reading ms-swift source/docs (Track A), **not** from the GRPO-hyperparameter research report (which only covered HPs).
- A2 (get hazards `z = head.linear(h_anchor)`, backprop-able) — **VERIFIED** from register.py (head.linear is a Linear; h_anchor = h_last[last token]; no register.py change needed).
- A3 (REINFORCE loss correct, reward detached, gradient direction) — `reinforce_core.py` written; **unit test pending (Gate 2)**.
- A4 (single-GPU) — **DROPPED**; multi-GPU via swift trainer (8-card) is the target.

### Files (RL, on box `/opt/dlami/nvme/v8_eval/rl/` + repo `verification/`)
- `cross_ad_reward.py` (+ `test_cross_ad_reward.py`) — Gate-1 verified reward.
- `reinforce_core.py` (+ test, pending) — isolated REINFORCE math.
- `train_cdf.npz` — train-only percentile CDF (val-disjoint).
- (`ttcc_train_rtext.jsonl` — R-text SFT data; UNUSED under head-PG, kept in case.)

### Next (parallel)
1. Track A: nail the swift custom-trainer extension point.
2. Track B: unit-test `reinforce_core` (advantage sign, reward detached, grad flow, KL≥0, monotone curve) — no GPU.
3. Write the custom trainer subclass using A1's mechanism + reinforce_core.
4. Smoke gate (real rollouts, ~50 steps, multi-GPU OK): within-group R_rank std>0, reward↔quality, reward↑.
5. Scale + live held-out SRCC vs 1−IBS control.

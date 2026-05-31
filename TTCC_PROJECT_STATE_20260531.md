# TTCC V8/V9 — Project State, Findings & Plan (2026-05-31)

Master/handoff doc. Consolidates the session's work. Confidence tags: **[V]** verified
(code + experiment), **[M]** measured, **[I]** inferred, **[?]** pending.
Companion docs: `V8_ROOT_CAUSE_AND_FIX_20260530.md`, `V8_CLEAR_WALKTHROUGH_20260530.md`,
`V8_PARADIGM_AND_VALIDATION_20260531.md`.

---

## 0. TL;DR

- **The token_acc collapse was a one-line bug**: the custom CoT cross-entropy (`loss_cot`)
  did not shift labels. **Fixed** (`torch.roll(labels,-1)`); A/B-confirmed token_acc
  0.005 → 0.745. **[V]**
- **The team's `cot_token_acc` metric is also unshifted** → it measures a *copy* shortcut;
  the 93.55% "CoT learns" reading is the bug succeeding, not learning. Same 1-line fix in
  the metric. **[V]**
- **Fixing the CoT did NOT (yet) improve retention** — because the retention HEAD reads a
  **bypassed** `</cot>` at eval, so generated reasoning never reaches the curve. **[V]**
- **The model DOES generate grounded CoTs** (verbatim on-screen text match to the video). **[M]**
- **Path forward** (aligned with Leon): SFT learns retention + coherent CoT (done); **RL**
  makes the CoT causally help via **generate-then-read** (model generates CoT → head reads
  the *generated* `</cot>` → reward = 1−IBS). Gate: does the head respond to the CoT
  (channel test, **running**).

---

## 1. Root cause + fix — the label shift  [V]

`register.py` `RetentionLoss.loss_cot` computed `CE(logits[i], labels[i])` — **no shift**.
ms-swift hands the loss RAW labels; every correct consumer shifts to the next-token target
(`swift/trainers/utils.py:103` `torch.roll(labels,-1)`; `seq2seq_trainer.py:176/208`;
`metrics/acc.py:23`). Externally identical to HF `ForCausalLMLoss` (pad+shift).

Because SFT response tokens are both input and label, unshifted CE is a trivial **copy**
task: `loss_cot → ~0` while genuine next-token prediction is destroyed → `token_acc → 0.005`.

**Fix:** `shift_labels = torch.roll(labels, shifts=-1, dims=-1)` then CE on `shift_labels`.
Byte-identical to swift's `per_token_loss_func`. Adversarial audit: **correct** (wrap-around
safe — `template/base.py:1464` forces `labels[0]=-100`; no `</cot>` leak; only nit = a bf16
`logits.float()` upcast, immaterial).

**A/B (one variable):** control (unshifted) token_acc **0.005** / loss_cot 0.003; fixed
(shifted) token_acc **0.745** / loss_cot 0.92 (live on i-06cb).

## 2. The metric trap — `cot_token_acc` measures copy  [V]

`cot_token_acc` (Wanjia's metric, `ttcc-v8` register.py:738) is **unshifted**:
`argmax(logits[i]).eq(labels[i])` = "did I predict the *current* token" = copy-accuracy.
It read **93.55%** on a copy-collapsed checkpoint that cannot generate coherent CoT. It is
INVERTED as a quality signal (high = bug; the correct fix makes it DROP). Synthetic smoke
proves: shifted metric scores a copy model **0.0** (not 93%), a true next-token predictor 1.0.
→ The fix is **two spots**: shift `loss_cot` AND `cot_token_acc`.

## 3. Architecture — two channels, two families  [V]

**Two prediction channels:**
- **Channel H (head):** a regression head reads the `</cot>` anchor hidden state → curve.
  Eval (`srcc_eval.py:137`) **bypasses** the CoT (assistant overwritten to `<cot></cot>`),
  so the head never sees reasoning. The `</cot>` matcher is **dead** → head always reads the
  **last token**. All our SRCC numbers are this channel.
- **Channel T (text):** the model generates the curve as JSON text; the RL reward
  (`ttcc_ibs_plugin.py`) parses it from the completion.

**Two model families:**
- **HEAD** (`sft_retention_hazard_*`, custom `RetentionLoss` = the shift bug; `rl_head_pg`).
- **TEXT-LM** (`sft_lm_full_*`, standard LM loss = no bug, curve-as-text).

The reasoning thesis needs reasoning to *condition* the prediction — only possible if the
readout (head or text) consumes the **generated** CoT, which today it does not.

## 4. Experimental results  [M]

| metric | control (unshifted) | fixed (shifted) |
|---|---|---|
| train token_acc | 0.005 | **0.745** |
| train loss_cot | 0.003 (copy) | 0.92 (real) |
| val eval_loss_curve | ~0.013 | ~0.016 |
| cross-ad SRCC (head, bypass eval, val_200) ckpt-100 | +0.170 | −0.119 |
| cross-ad SRCC ckpt-500/600 | +0.148 | +0.054 |
| V8 recorded baseline (different val set) | 0.514 / 0.4393 **[?]** | — |

Reading: the shift fixes *generation*, but on the bypass-head curve the **real CoT scores
*below* the broken-CoT control** (it competes for capacity; the head can't use it). Both far
under V8's recorded 0.514 — but that comparison is **not apples-to-apples** (our `val_200_no_cot`
+ long-ad skips vs V8's `val_present`), unresolved.

**Generation is grounded [M]:** the shift-fixed model generates real per-second CoTs;
ad1's `0s | woman massages man's neck, "Find Migraine Massage Services" text` matches the
actual frames (scene + verbatim on-screen text). Not hallucination.

## 5. Canonical version + divergence  [V]

Wanjia's `ttcc-v8` and our box version **diverged in both directions** (structurally
identical otherwise): Wanjia has `cot_token_acc` (unshifted); ours has the shift fix + the
`mu_z`/policy-mean RL hook + component-loss logging. **Neither is a superset** → recurring
"which is fixed?" confusion.

- **`register_canonical.py`** = our version + shifted `cot_token_acc` (the union). Built via
  an assertion-guarded patcher (8 edits each asserted once + `ast.parse` + synthetic smoke).
- **`wj_register_fixed.py`** = Wanjia's exact file + the 2-spot shift (preserves her file).
- **Resolution (Leon):** ONE canonical version as a git **branch + PR**, not S3 files. Open
  decisions: full-FT vs LoRA; which fork is home (cliangyu/go_viral vs wanjiaZhao1203).

## 6. Target paradigm — SFT learns both, RL makes CoT help

**SFT's job:** learn the retention head AND learn to generate a coherent CoT — *both*, not
"CoT must help yet". (Shift fix enables the 2nd leg.) **RL's job:** make the generated CoT
causally improve retention, via **generate-then-read** + a curve reward.

Current (bug) vs target: today the eval/RL feed the head an *empty* `</cot>` (bypass); the
target generates the CoT and the head reads *that*. See `V8_PARADIGM_AND_VALIDATION_20260531.md`
for the diagram.

## 7. The channel test (the gate)  [?]

`generate_eval.py` (fixed: `set_mode`, LoRA loading, strip generate kwargs) runs
generate-then-read on the shift-fixed ckpt: generate CoT → head reads generated `</cot>` →
**REASONED vs BYPASS cross-ad SRCC delta**. Running on i-0f1d. **If delta > 0**, the head
responds to reasoning → RL viable. **If ≈ 0**, the single-token readout ignores the CoT →
**widen the readout (mean-pool) before RL.** *Result pending.*

## 8. RL readiness — prerequisites + build plan

**Prereqs (backwards from "RL works"):** ① coherent generation ✅; ② SFT init ✅;
③ generate-then-read forward ✅ (exists in `generate_eval.py`); **④ THE GATE: head responds
to CoT — pending (channel test)**; ⑤ generate-then-read reward — to build; ⑥ RL loop wired —
to build (existing head-RL is REINFORCE-on-noisy-head-logits + bypass = wrong paradigm).

**Readiness checklist (Leon's 5 + 6 missed):**
1. full vs LoRA (decision) · 2. GRPO config (adapt `grpo.sh`) · 3. head-IBS reward · 4. rollout
(swift GRPO; multimodal `generate()` kwargs fixed) · 5. generate-then-read guard ·
**6. the gate/readout** · 7. GRPO dataset (video + CoT-eliciting prompt + `R_true`/`T` cols;
curve NOT in text) · 8. KL reference policy · 9. the reward MODEL (head-wrapped frozen copy) ·
10. exploration temp + reward variance (group std>0 or no gradient) · 11. Goodhart guard
(train 1−IBS ≠ eval SRCC).

**Build plan [from sub-agent, file:line-verified]:** extend **swift GRPO** (token policy +
group advantage already there) with a **custom reward-model-plugin** (subclass
`DefaultRMPlugin`, `swift/rewards/rm_plugin.py:29-38`) that re-forwards prompt+generated-CoT
through the head-wrapped model and returns `1−IBS`. The reward-model path is the **only**
swift channel with model access (plain ORM reward funcs get text only,
`grpo_trainer.py:368`). Reference forward to lift: `generate_eval.head_curve`. Do **not**
rewrite `head_pg_trainer.py` (continuous-Gaussian-on-logits, no generation).
Build order once gate passes: dataset → reward plugin (+reward model) → adapt `grpo.sh`
(+ref/KL/temp) → guards (generate-then-read held-out + group-collapse + Goodhart).

## 9. Open decisions (Leon)

1. **Full-FT vs LoRA** for the canonical/RL policy.
2. **Home repo/branch** for the canonical (cliangyu/go_viral vs wanjiaZhao1203) + branch conventions.
3. **Readout**: keep single-token or widen (mean-pool) — decided by the channel-test gate.
4. Clean V8-on-val_200 baseline (to make the 0.514 comparison real).

## 10. SWE / repo state

- Local repo = `github.com/cliangyu/go_viral` (fork of ms-swift), branch `ttcc-rl`.
- Session work committed to branch **`cc/v8-cot-shift-fix`** (5 logical commits: the fix /
  configs / tooling / docs / gitignore). Working tree clean. **Not pushed** (awaiting the
  home-repo + branch-convention decision).
- Boxes: i-06cb training the shift-fixed LoRA run; i-0f1d running the channel test; gpu-box idle.
- **Lesson (Leon):** git is the source of truth — commit in logical units proactively; no
  S3/box/tmp file dumps as the canonical. Done going forward.

## 11. Key lessons (for the team)

- A custom LM loss/metric must shift labels exactly like swift (`torch.roll(labels,-1)`); an
  **unshifted accuracy measures copy** and is an inverted quality signal.
- A fixed proxy (token_acc) is not the goal — verify the **target metric** (retention SRCC)
  on held-out data; and verify which **channel** the objective/reward actually reads.
- Coherent ≠ grounded — eyeball the source modality (frames vs CoT) before trusting generation.
- The binding RL prerequisite is usually the **gate** (does the readout respond to the
  rollout), not the parts everyone lists.

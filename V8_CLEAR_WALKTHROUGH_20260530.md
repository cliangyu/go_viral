# V8 / CoT-SFT / RL — clear walkthrough (2026-05-30)

Confidence tags: **[V]** verified (code + A/B + external), **[M]** measured this session,
**[I]** inferred, **[?]** uncertain / needs a check.

---

## 0. The one thing that caused all the confusion: there are TWO prediction channels

The model can output the retention curve R(t) two completely different ways, and we
(I) kept conflating them:

- **Channel H — the regression HEAD.** A small head reads the hidden state at the
  `</cot>` anchor token and emits the curve. Trained by MSE (`loss_curve`). **[V]**
  At eval (`srcc_eval.py`) the CoT is *bypassed*: the assistant is overwritten to the
  empty string `<cot></cot>` and the head reads that bare anchor — it never sees any
  reasoning. (`srcc_eval.py:137`, head read `register.py:298-301`, no `.generate()`.) **[V]**
- **Channel T — the generated TEXT.** The model autoregressively generates
  `<cot>reasoning</cot> … curve`, and RL rewards the curve **parsed from the generated
  text**. (`ttcc_ibs_plugin.py`: `reward = 1 − IBS` of `_parse_curve(completion)`;
  `grpo.sh:48 --reward_funcs ttcc_ibs_reward --num_generations`.) **[V]**

**RL uses Channel T. The SRCC eval I kept quoting measures Channel H.** They are not
the same prediction. This is the root of my mis-scoped "drop the CoT" advice.

---

## 1. What we set out to fix

V8 SFT-with-CoT had a **token_acc collapse** (next-token accuracy → ~0). Goal: make the
reasoning channel actually train.

## 2. Root cause (the real win) — **[V]**

`loss_cot` (the CoT cross-entropy in `register.py`) did **not shift the labels**:
`CE(logits[i], labels[i])` instead of `labels[i+1]`. Since SFT response tokens are both
input and label, that's a trivial **copy** task → `loss_cot → ~0` while genuine
next-token generation is destroyed → `token_acc → 0.005`.

Evidence: swift hands raw labels and every correct consumer shifts (`utils.py:103`
`torch.roll(labels,-1)`; `seq2seq_trainer.py:193/176/208`; `metrics/acc.py:23` shifts);
externally triangulated against **HF `ForCausalLMLoss`** (pad+shift) and **upstream
ms-swift** (roll) — my fix is byte-identical to ms-swift's own loss. The math tell:
loss_cot 0.12 nats vs token_acc 0.005 is impossible over aligned positions.

## 3. The fix + A/B — **[V][M]**

Added `torch.roll(labels,-1)`. One-variable A/B:

| | control (buggy) | fixed (shift) |
|---|---|---|
| token_acc | 0.005 | **0.72** |
| loss_cot | 0.003 (copy) | 0.99 (real next-token) |

**The model now generates coherent reasoning instead of copying.** This is real and is a
prerequisite for *anything* that uses generated text (i.e., RL Channel T).

## 4. The twist — fixing the CoT *hurt* the HEAD's curve — **[M]**

I evaluated Channel H (SRCC, bypass eval) across checkpoints:

| cross-ad SRCC (val_200) | fixed (real CoT) | control (broken CoT) |
|---|---|---|
| ckpt-100 | −0.119 | +0.170 |
| ckpt-500/600 | +0.054 | +0.148 |
| V8 recorded | — | 0.514 / 0.4393 **[?] (different val set — not apples-to-apples)** |

Real CoT < broken CoT on Channel H, both far under V8. **Because Channel H bypasses the
CoT** — the generated reasoning is never read by the head, so a real CoT only competes
for capacity and creates a train(real-CoT)/eval(empty-CoT) mismatch. **[V]** This says
nothing bad about reasoning; it says the *head readout ignores reasoning*.

## 5. Your challenge, answered: "why drop CoT? then how does RL work?"

I was wrong to say "drop CoT." That conclusion was from Channel H, which RL doesn't use.

**RL (Channel T) works like this:** for each ad the model samples N completions
(`--num_generations`), each = reasoning + a curve; `ttcc_ibs_reward` parses the curve
from the *generated text* and rewards `1 − IBS`; GRPO pushes the policy toward reasoning
that yields better curves. Here the curve text **follows** the reasoning (causal), so
**reasoning genuinely conditions the prediction** — this is the real "reasoning improves
retention" loop, and **the shift fix is essential to it** (a copy-collapsed model emits
garbage text → nothing to reward). So the overnight work is *on* the RL critical path,
not orthogonal.

## 6. The honest complication — two RL formulations, opposite implications — **[V] they exist / [?] which is "the" one**

- **Text-GRPO** (`grpo.sh`/`rloo.sh` + `ttcc_ibs_reward`): uses generated CoT → reasoning
  can help → shift fix essential. **But** the cleaned SFT data (`clean_v2`) ends at
  `</cot>` with **no curve in text** **[M]**, so a model trained on it won't emit a
  parseable curve → reward parse fails. Data/format must be aligned.
- **Head-PG** (`rl_head_pg_*.yaml`, `mu_z` policy, `register.py:216`): the head's curve
  output is the policy; PG with IBS reward, reading the **bypass** anchor → generated CoT
  can't condition it → reasoning can't help via the head. (This matches the 2026-05-29
  red-team: "the readout is the bottleneck; single-token head; CUT CoT-RL.")

## 7. Where we actually are

- token_acc bug: **found, fixed, verified, externally triangulated.** Real result.
- Channel H (head): LoRA ~0.15 vs V8 0.514 (caveat: eval-set mismatch unresolved). Head
  ignores CoT by construction.
- RL (Channel T): can express reasoning→retention, needs (a) coherent generation
  [done ✅], (b) SFT data that emits the curve as text, (c) a decision between
  text-GRPO and head-PG.

## 8. The real fork (corrected) + next experiment

Not "drop CoT vs keep." It's **which channel is the product**:

- **Reasoning thesis / RL** → keep CoT; use the shift-fixed SFT; align the SFT data to
  emit the curve as text after `</cot>`; run **text-GRPO**. Channel H / the bypass-SRCC
  was a detour.
- **Just the retention number, fast** → full-FT head regressor, no CoT (old "Solution
  A"). Abandons RL.

**Decisive cheap next experiment (Channel T, the thing RL actually optimizes):** take the
shift-fixed checkpoint, **generate** on ~20 val ads, and check (i) is the reasoning
coherent, (ii) does it emit a parseable curve, (iii) what's the IBS of the *generated*
curve. That measures the channel RL uses — unlike every SRCC number above, which measured
the head. If the generated curve is sane → RL has a real substrate. If not → fix the
SFT data/format before any RL.

(Open data check: confirm whether the SFT-with-CoT data is supposed to include the curve
as text after `</cot>` — `clean_v2` does not, but `build_ttcc_jsonl.py` reportedly wrote
`</cot>\n{R_json}`. This inconsistency decides whether text-GRPO can even score the
current checkpoint.)

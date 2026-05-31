# V8 — current vs target paradigm, and the validation experiment (2026-05-31)

## 1. CURRENT paradigm — the CoT is decorative (never feeds the curve)

```
 SFT (train), teacher-forced:
   video+prompt + <cot> GROUND-TRUTH reasoning </cot>          (the real CoT is fed in)
                         │                    │
                         │                    └─► h[</cot>] ─► HEAD ─► R̂  ─┐ loss = MSE(R̂,R)
                         └──────────────────────────────────► LM  ─► CoT ─┘      + α·CE(CoT)
                                                                                 (shift-fix ✓: real CoT, not copy)

 EVAL (srcc_eval.py:137), bypass:
   video+prompt + <cot></cot>            ◄── EMPTY. the reasoning is deleted.
                       │
                       └─► h[</cot>=empty] ─► HEAD ─► R̂      ─►  SRCC ≈ 0.43
                              ▲
                              └─ head reads ONE token, sees ZERO reasoning.

  ✗ train feeds a real CoT, eval feeds an empty one   (covariate shift on the head's only input)
  ✗ the model's OWN generated CoT is never produced, never read
  ✗ ⇒ "does reasoning help retention?" is structurally UNanswerable in this setup
```

## 2. TARGET paradigm — CoT causally feeds the curve; RL makes it help

```
 SFT — learn BOTH capabilities (not "CoT must help yet"):
   video+prompt + <cot> reasoning </cot>
                        │
            ┌───────────┴────────────┐
            ▼                        ▼
        HEAD ─► R̂              LM ─► CoT
     (learn retention)     (learn to GENERATE coherent CoT, shift-fix ✓)

 EVAL / RL — generate-then-read:
   video+prompt ─► model GENERATES ─► <cot> ITS OWN reasoning </cot>
                                              │
                                              ▼
                               mean-pool over the CoT span      ◄── WIDENED readout
                                              │
                                              ▼
                                            HEAD ─► R̂
                                              │
                                              ▼
                                  reward = curve IBS  ─►  RL (GRPO)
                                  ↳ pushes the GENERATED reasoning toward better R̂

  ✓ the generated CoT actually conditions the curve
  ✓ RL has a gradient: better reasoning → better curve → higher reward
```

## 3. THE DIFFERENCE — three concrete changes

| | CURRENT | TARGET |
|---|---|---|
| **CoT source at eval/RL** | empty `<cot></cot>` (bypass) | the model's **own generated** CoT |
| **CoT's role** | ignored — decorative | **causally feeds** head + reward |
| **Readout** | single last token (matcher is dead) | **mean-pool** over the CoT span (fat channel) |

SFT barely changes (it already learns head + CoT generation). The change is **at eval/RL**:
produce the CoT and read it, through a wider readout.

---

## The validation experiment (before building any RL)

**Question it answers:** is there a CHANNEL at all — does the head's prediction even *respond*
to the CoT content? If R̂ never moves when the CoT changes, RL has zero gradient and cannot
make reasoning help; we'd have to widen the readout first. This is the cheapest, most decisive
check, and it gates everything downstream.

### Exp 1 — Channel test (teacher-forced, no generation needed) — ~10 min
On the shift-fixed checkpoint, for ~40 val ads, run the head forward under **3 CoT inputs**:
- (a) empty `<cot></cot>`  — the current bypass (baseline)
- (b) the real (clean) CoT  — upper bound of "CoT helps"
- (c) a **random other ad's** CoT — counterfactual / wrong reasoning

Measure: (i) Δ‖R̂‖ across (a)/(b)/(c) = **how wide the channel is**; (ii) SRCC/IBS of each
vs ground-truth R. Reads:
- R̂ ~identical across all three  → head **ignores** the CoT → widen the readout BEFORE any RL.
- R̂ moves, and (b) > (a) and (b) > (c) → **reasoning helps** → RL has a real lever.
- R̂ moves but no SRCC gain → channel exists, SFT head uses it poorly → RL can shape it.

(No generation in Exp 1 — just 3 forwards per ad — so it's fast and isolates "does the head
USE a CoT" from "can the model generate a good CoT".)

### Exp 2 — Generate-then-read (only if Exp 1 shows a channel)
Model **generates** its own CoT, head reads the generated `</cot>`, SRCC/IBS vs bypass on val_200.

### Hardware
**8-GPU i-0f1d (idle, H100, flash_attn)** — kernel-consistent `h_anchor` with the existing
SRCC baseline (the 2-GPU Blackwell box lacks flash_attn → `sdpa` → not comparable), and
8-way sharded so Exp 1 finishes in minutes. i-06cb run untouched.

### What gets built
A small `generate_then_read.py` (adapts `srcc_eval.py`): instead of overwriting the assistant
to `<cot></cot>`, it (Exp 1) swaps in the 3 candidate CoTs, and (Exp 2) calls `model.generate()`
to produce the CoT, then runs the head over the result. ~1 new file.
```

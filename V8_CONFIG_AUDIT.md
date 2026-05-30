# V8 SFT — full config + W&B audit → what to change (2026-05-30)

Sources: `configs/sft_retention_hazard_full_with_cot.yaml` + resolved `args.json` (ckpt-225) +
W&B run `liangyuch/ttcc/b5rhw2te` (the v19 run that produced ckpt-225).

## A. The W&B verdict (run b5rhw2te)
- **The run CRASHED** (`state: crashed`) at step 279, after ~18.4 h, at **epoch 0.98** — i.e. it
  never finished even 1 of the planned 10 epochs. ckpt-225 is a checkpoint *before a crash*, not a
  chosen stopping point.
- **token_acc collapse is a CLIFF, not a drift:** 0.557 (s0) → 0.52 (s54) → 0.43 (s90) →
  **0.18 (s108) → 0.02 (s126) → 0.004 (s162+)**. The cliff at s108–126 coincides with **the end of
  warmup** (warmup_ratio 0.03 ≈ 94 steps): once LR reached full 5e-6, the curve-MSE-dominated
  gradient reshaped the shared backbone and the LM died.
- **grad_norm is wild:** 3686 at step 0 (random head + huge initial curve MSE=29), then oscillating
  **25–450 the whole run** while `max_grad_norm=1.0` clips it → effective step size swings 25–450×.
  Unstable landscape; the random head shocks the backbone.
- **`loss_curve` / `loss_cot` were NEVER logged** (the register.py custom_metrics patch didn't fire).
  We trained the whole run **blind to the MSE-vs-CoT balance** — the single thing that mattered.
- **eval/token_acc = 0** (val is `val_200_no_cot` — wrong format for a with-cot model; the metric
  was meaningless).

## B. Current config (the parts that matter)
| field | value | verdict |
|---|---|---|
| init | base Qwen2.5-Omni-3B | ✅ keep (so "从头" = redo THIS; "从225" = continue the corpse) |
| loss | `MSE + 1e-3·CE` | ❌ **the collapse cause** |
| head | hazard, reads h at **single** `</cot>`≈last token | ⚠️ single-token readout = the saturated bottleneck |
| data | `ttcc_train_with_cot.jsonl` (audio refs + `gain` labels) | ❌ clean (de-audio, de-gain) |
| val | `val_200_no_cot.jsonl` | ❌ wrong format → eval_token_acc=0 |
| lr / sched | 5e-6 / cosine, warmup 0.03, wd 0.1 | ⚠️ collapse aligns with warmup-end |
| eff-batch | 1 × grad_accum 8 × world 16 = **128** | ok |
| epochs | 10 planned (crashed at 0.98) | — |
| max_length | 32768, truncation=delete (~15% long ads dropped) | cleaning shortens CoT → fewer drops |
| precision/kernel | bf16, **FA3 train** (eval uses flash_attn) | ⚠️ train/eval kernel mismatch |
| ZeRO | 3 (required: ZeRO-2 triggers the audio_tower dummy-tensor FA3 bug with audios=[]) | keep |
| fps / video tokens | **NOT in config/args.json/wandb — unversioned env vars** | ❌ pin them |

## C. What to change (prioritized)

**CRITICAL — fix the collapse (do all three):**
1. **Phased / gradient-balanced training.** Root cause = a fixed `alpha=1e-3` made CoT-CE ~1–3% of the
   gradient AND the random head shocked the backbone. Fix:
   - **Phase 1:** freeze the LM backbone, train the **head only** on the frozen base features (LM stays
     perfect, token_acc ~0.557; head reaches ~the linear ceiling ~0.514). Tames the grad_norm 3686 shock.
   - **Phase 2:** unfreeze, joint training with **gradient-balanced** weighting (GradNorm / Kendall
     uncertainty-weighting), not a fixed scalar — so neither loss is starved as the curve MSE drifts.
2. **Log the component losses (`loss_curve`, `loss_cot`) + token_acc from step 1.** We were blind. Wire
   the existing custom_metrics patch (it didn't fire) or add an explicit callback. No more flying blind.
3. **LM-preservation guard:** monitor token_acc live; optionally a KL-to-base-LM term or LoRA the LM
   path so the head can't overwrite the LM. Stop/rollback if token_acc drops below a floor.

**DATA:**
4. **Clean the CoT** — remove audio/voiceover content (audio off) + remove `gain` labels (R(t) is
   monotone non-increasing → "gain" is impossible / a soft outcome-leak). [verifying monotonicity]

**HYGIENE / REPRODUCIBILITY:**
5. **Pin fps / video-token budget IN the config** (currently unversioned env). Confirm the actual fps
   the v19 run used — it is not recorded anywhere, which is a real reproducibility hole.
6. **Train/eval kernel parity** — use the same attention kernel for SFT and eval (FA3 vs flash_attn
   drifts h_anchor).
7. **Eval val set** — use a with-cot-format val (or drop token_acc eval); the no-cot val gave 0.
8. **Stability** — the head-warmup (Phase 1) removes the random-head grad shock; also consider lower
   peak LR / longer warmup for Phase 2.

**STRUCTURAL (research-informed — fold in since we're retraining anyway):**
9. **Richer readout.** Both saturation bounds (head_oracle, rank_sft) read a *single* token. The
   research's highest-upside lever (+0.038 in the closest analog, 3.5× audio) is a **mean-pool /
   attention-pool / multi-token** readout over the CoT span — which is *also* exactly how reasoning
   would help (the head reads the whole reasoning, not one token). Small change, high expected value.

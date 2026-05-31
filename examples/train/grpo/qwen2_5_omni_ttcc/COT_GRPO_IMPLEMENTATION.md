# CoT-GRPO — what this version of the code does (2026-05-31)

**Principle:** subclass the **official** swift `GRPOTrainer` and change **one** thing — the reward.
Everything else (multimodal vLLM rollouts, group-relative advantage, KL to the SFT ref, per-token
logps, DeepSpeed/torchrun multi-node) is upstream GRPO, untouched.

`grey` = official swift (unchanged).  `★` = the code we wrote.

---

## 1. One CoT-GRPO training step

```
  ttcc_train_bypass.jsonl                                  train_cdf.npz  (TRAIN-only,
  { video, prompt, <cot></cot>, R, T, ad_id }              val-disjoint "grading curve")
            │                                                       │
            ▼   swift: sample G rollouts per ad (vLLM colocate)     │
  ┌───────────────────────────────────────────────────────┐        │
  │ GRPOTrainer._generate_completions                      │        │
  │   video+prompt ─► generate ─► <cot> reasoning g1 </cot>│        │
  │                               <cot> reasoning g2 </cot> │  ...×G │
  └─────────────────────────┬─────────────────────────────┘        │
                            ▼                                        │
  ┌─────────────────────────────────────────────────────────────┐  │
  │ ★ CoTGRPOTrainer._compute_rewards_per_func   (the only change)│  │
  │     for each generated CoT g:                                 │  │
  │       re-encode (prompt + <cot>g</cot>) in TRAIN mode         │  │
  │       forward the POLICY  ─►  retention HEAD  ─►  R̂_g(t) ◄─────┼──┘
  │            (head reads the post-CoT last token = head_curve)  │   head reads the
  │       r_g = β·R_rank(R̂_g vs CDF)              ◄── cross-ad rank│   GENERATED CoT
  │           + α·(1−IBS(R̂_g, R_true))            ◄── calib anchor │
  │           + 0.15·fmt(<cot>…</cot>)            ◄── structure    │
  │       log within-group curve std             ◄── R1 gate      │
  └─────────────────────────┬───────────────────────────────────┘
                            ▼   swift: A_g = (r_g − mean_g)/std_g   (group-relative)
  ┌─────────────────────────────────────────────────────────────┐
  │ GRPOTrainer loss:  −Σ_g A_g · logπ(CoT_g)  +  β·KL(π ‖ SFT)  │
  │      gradient flows into the CoT-GENERATION policy            │
  └─────────────────────────────────────────────────────────────┘

  ✓ reward rises ONLY if a better CoT ⇒ a better-ranking R̂  ⇒ RL teaches the reasoning to help.
  ✗ a per-ad reward would cancel under (r−mean)/std → we grade vs the SHARED train CDF (cross-ad).
```

## 2. The reward (3 terms — head channel, not text)

```
  r_g  =  β · R_rank   +   α · R_acc        +   γ · R_fmt
          (1.0)            (0.25)               (0.15)
          │                │                    │
  cross-ad percentile      1 − IBS              well-formed <cot>…</cot>
  of R̂_g vs the TRAIN      (calibration         (non-empty, ≤512;
  CDF, t∈[1,30].           anchor so rank       guards degenerate /
  WITHIN-GROUP spread      doesn't rot          runaway CoT).
  = the ranking signal.    calibration like     pure-text ORM.
  ★ head_reward.py +       head-PG did).        ★ ttcc_cot_format.
  cross_ad_reward.r_rank   ★ ..r_acc            R̂ from the HEAD, not parsed text.
```

## 3. Files (★ = written this version)

| File | Role | Base / reference |
|---|---|---|
| ★ `rl/cot_grpo_trainer.py` | `CoTGRPOTrainer(GRPOTrainer)` — overrides only `_compute_rewards_per_func`; `_head_curve` re-encodes + forwards + reads `r_pred` | **official** `swift.rlhf_trainers.GRPOTrainer`; readout mirrors `generate_eval.head_curve` |
| ★ `rl/head_reward.py` | head `R̂` → scalar (`β·R_rank + α·R_acc`) | `verification/cross_ad_reward.py` (Gate-1 math) |
| ★ `plugin/ttcc_head_plugin.py` | placeholder ORM that reserves the `ttcc_head` reward column (trainer fills it) | swift ORM |
| ★ `plugin/ttcc_cot_format_plugin.py` | `<cot>…</cot>` structure reward | swift ORM |
| ★ `rl/train_cot_grpo.py` | entry — `SwiftRLHF` + swap `trainer_cls`, inject train CDF | mirrors `rl/train_head_pg.py` (HeadPG) |
| ★ `configs/rl_cot_grpo.yaml` | G=8, temp 0.4, max_completion 512, β=0.001, lr 5e-6, LoRA, warm-start SFT-with-CoT, bypass data + CDF | `grpo.sh` HPs + head-PG yaml shape |
| ★ `rl.sh` (1-line `RL_ENTRY`) | reuse the proven 2-node torchrun launcher | head-PG `rl.sh` |

## 4. How it differs from the two prior RL attempts

```
  head-PG (ran NULL)          text-GRPO (abandoned)        CoT-GRPO (this version)
  ───────────────────         ─────────────────────        ───────────────────────
  action = Gaussian noise     action = write R=[...]        action = generate <cot>…</cot>
           on hazards z                as TEXT
  no CoT, no generation       parse curve from words        head READS the generated CoT
  reward on head curve        reward on PARSED curve         reward on head curve (cross-ad)
  → null + IBS doubled        wrong channel (parse-cliff)    → genuine reasoning test
```

## 5. Status

- **Written + committed** (`b761ca53`): all 7 files. Static checks pass (`py_compile`, `bash -n`, `head_reward` regression).
- **Running:** an adversarial preflight audit of every dimension (reward-column fill, head-forward under
  GRPO/vLLM wrapping, completion type + **prompt boundary**, adapter warm-start of the head, config sanity,
  reward semantics) vs the real swift source — **before** the smoke.
- **Next:** fix any blocking audit findings → 1–2 GPU smoke on idle i-0f1d (assert: generation runs, head
  reward computes, `r_pred` reachable, within-group curve std > 0, prompt boundary correct) → **2-node run**
  (gated on the SFT finishing on i-06cb, which RL warm-starts from).
```

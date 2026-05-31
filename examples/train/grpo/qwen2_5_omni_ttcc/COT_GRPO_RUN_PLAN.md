# CoT-GRPO — 2-node production run plan (monitoring + contingency + stopping)

**Standard from 2026-05-31: every CoT-GRPO run is a 2-node (2×8 H100) production run, launched
from the repo (`launch_cot_grpo.sh`), no `/tmp` scripts.** Single-node is only the pre-SFT-finish
diversity/config shakedown. This doc is the operating plan; update it, don't fork it.

---

## 0. Launch (2×8, repo-canonical)

Prereqs on BOTH nodes (i-06cb, i-0f1d), us-east-2:
- env shims: `pip install msgspec`; peft `save_and_load.py` `EmbeddingParallel` shim (`.bak_cotgrl` backup) — peft 0.19.1 vs transformers 4.56.2 skew.
- code synced to `/opt/dlami/nvme/go_viral` (md5-verified), owned by `ssm-user`.
- `use_vllm=false` (HF generate; vLLM can't load our custom head, see COT_GRPO_IMPLEMENTATION.md).

Warm-start from the **final** SFT-with-CoT checkpoint (not a mid-training snapshot). Launch as
`ssm-user`, one invocation per node:
```
# node 0 (MASTER):  ADAPTERS=<final SFT ckpt>  on i-06cb
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29514 GA=8 \
  ADAPTERS=/opt/dlami/nvme/ssm-out/sft_retention_hazard_lora_v8_with_cot/<final>/checkpoint-<N> \
  nohup bash launch_cot_grpo.sh >/dev/null 2>&1 &
# node 1:  same, NODE_RANK=1, same MASTER_ADDR/PORT
```
Effective batch = world(16) × bs(1) × GA(8) = **128 ads/step × G=8 = 1024 gens/step**. Match this
eff-batch on any comparison run (the eff-batch lesson). MASTER_PORT 29514 (SFT held 29500; single-node held 29513).

---

## 1. Monitoring — what to watch, where, how often

**In-loop (tensorboard + `<out>/logging.jsonl`, every step):**
| signal | healthy | meaning |
|---|---|---|
| `cot_head_fail_frac` | **= 0** | r_pred attaches; >0.5 at step 0 = ABORT (auto, built-in) |
| `reward` / `rewards/TTCCHeadPlaceholder/mean` | rising slowly | the head reward (R_rank + IBS) |
| `reward_std` / head `std` | **> ~0.03** | within-group spread = the GRPO gradient signal (the R1 channel) |
| `cot_curve_std` | **> 0** | within-group CURVE diversity (do CoTs move the head?) |
| `kl` | stable, not runaway | drift from the SFT ref; climbing = hacking/divergence |
| `grad_norm` | bounded (~1–5) | stability; spikes = instability |
| `completions/clipped_ratio` | low | CoTs fitting max_completion=512 |
| `frac_reward_zero_std` | low | fraction of groups with no gradient |

**Out-of-loop — THE TRUE METRIC (per saved checkpoint, save_steps=25):**
- **Held-out cross-ad SRCC** (+ IBS) via the leak-free bypass eval, **REASONED (generated CoT) vs BYPASS**,
  vs the SFT baseline — the GO/NO-GO. Run `rl/srcc_eval.py` on each checkpoint on a spare GPU / after the run.
- This is the Goodhart guard: in-loop reward ↑ means nothing unless held-out SRCC ↑.

**Cadence:** diff-based check every ~10–15 min; **alert only on material change** (a metric crosses a
threshold below), otherwise stay silent. Eyeball 2–3 `completions.jsonl` CoTs per checkpoint (coherence).
**`monitor_cot_grpo.sh`** (repo) tails + diffs; **`eval_ckpts_srcc.sh`** (repo) evals new checkpoints.

---

## 2. Contingency / tuning playbook — if X then Y (evidence-based)

| symptom | first action (evidence) | escalation |
|---|---|---|
| **Low within-group curve std** (<~0.03 sustained) → weak gradient | **raise temperature 0.4→0.8** (0.4 was a *text-channel* parse-stability artifact, moot for the head channel) | if CoTs diversify but curves don't → **mean-pool readout** (head_oracle 0.5152 = last-token saturated; paradigm doc) |
| **reward_std tiny AND advantages noisy/unstable** | switch **`scale_rewards=group`→`none`** (Dr.GRPO: std-norm amplifies noise at low spread) | lower lr |
| **KL runaway** (climbs, ≫ baseline) → drift/hacking | **raise β (KL)** 0.001→0.005–0.01 | lower lr 5e-6→2e-6 |
| **IBS rising** (calibration rot, the head-PG failure) | **raise α (IBS anchor)** 0.25→0.4 | re-weight reward |
| **reward ↑ but held-out SRCC flat/down** (Goodhart) | **STOP** — reward is gamed; re-examine reward, do not keep training | NO-GO path |
| **grad_norm spikes / loss NaN** | lower lr, tighten grad-clip, drop bad batch | restart from last good ckpt |
| **OOM** (watch at G=8/512/1024-gens) | lower GA, then G, then max_completion; ensure grad-ckpt on | reduce per-device load |

All tuning changes go through the **config / launcher in the repo** + a one-line note here; never a /tmp edit.

---

## 3. Stopping criteria (the monitor flags; the stop is MANUAL — experiments never auto-stop)

- **SUCCESS → stop + lock best ckpt:** held-out **reasoned** cross-ad SRCC beats the SFT baseline
  **CI-clean** (paired bootstrap) **AND** IBS not worse, sustained over **≥2–3 consecutive checkpoints**.
- **NO-GO → stop + write negative result:** after **~150–200 steps / ~6–8 checkpoints**, held-out SRCC
  stays **within CI of the baseline** (no improvement) → CoT-RL is bounded at the video ceiling (the
  gate's bearish prior — head_oracle saturation + I(CoT;R|video)=0 — confirmed). Pivot the GPUs to
  **audio-on SFT** (the only new-information lever).
- **DIVERGENCE → stop:** KL runaway / reward-hacking / instability that the playbook (§2) doesn't fix
  within ~2 tuning attempts against the same symptom.
- **Cost guard:** 2×8 H100 ≈ \$/hr is real; a run with flat SRCC for 6+ checkpoints is the NO-GO stop, not "let it run."

**Both nodes stop together** (kill NODE_RANK 0; rank 1 exits on the torchrun rendezvous drop). Back up
the best checkpoint before any cleanup (the V8 backup incident).

---

## 4. Why 2-node (Leon, 2026-05-31)
Everything is a production 2-node run from now on — it doubles throughput (1024 gens/step across 16 GPUs),
so we reach the SUCCESS/NO-GO decision faster. The single-node run was the stopgap to start observing;
it stops when the SFT frees i-06cb and we have the final checkpoint.

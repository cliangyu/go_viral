# GRPO / Continuous-Action Policy-Gradient Hyperparameter Research

**For:** V8 retention-head RL — Qwen2.5-Omni-3B + retention head, continuous-Gaussian-on-hazards policy gradient with a cross-ad percentile-ranking reward.
**Date:** 2026-05-29.
**Author:** research pass for the RL design decision in `NORTH_STAR.md §6/§10`.

---

## Source / tool note (read first)

The task requested Tavily MCP search. **Tavily was hard rate-limited the entire session** (`"This request exceeds your plan's set usage limit"` on every call, including a single-query retry). Per the research-routing principle ("best tool for the job, then figure out access"), I fell back to the built-in `WebSearch` + `WebFetch` and pulled **primary sources directly** (arXiv HTML/PDF, the official TRL doc, the ms-swift doc, OpenAI Spinning Up). Every claim below carries a URL. arXiv PDFs that rendered as binary were re-read via the PDF reader or the `arxiv.org/html` / `ar5iv` mirror to confirm exact numbers. **Residual risk:** I could not run a second independent search engine on a couple of the secondary claims (flagged inline with confidence).

Confidence tags: **[primary]** = quoted from the paper/official doc; **[training-data]** = from model knowledge, not re-verified this session, with a % confidence.

---

# PART 1 — How others do LLM GRPO, and what transfers

## 1.1 The canonical recipe

GRPO (DeepSeekMath, Shao et al. 2024) drops PPO's value/critic network and replaces the per-token advantage baseline with a **group-relative baseline**: for each prompt, sample a group of `G` outputs, score each, and normalize the reward within the group. DeepSeek-R1 reuses it verbatim and is explicit about *why* the critic is dropped and why the reward is rule-based (avoid reward hacking at scale). [primary: https://huggingface.co/docs/trl/main/en/grpo_trainer ; https://arxiv.org/html/2501.12948v1]

**Advantage** (DeepSeekMath / TRL):
$$\hat{A}_{i} = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}$$
group mean subtracted, divided by group std. [primary: https://huggingface.co/docs/trl/main/en/grpo_trainer]

**KL** uses the Schulman k3 unbiased estimator (always ≥ 0):
$$D_{KL}[\pi_\theta\|\pi_{ref}] = \tfrac{\pi_{ref}}{\pi_\theta} - \log\tfrac{\pi_{ref}}{\pi_\theta} - 1$$
[primary: https://huggingface.co/docs/trl/main/en/grpo_trainer ; http://joschu.net/blog/kl-approx.html]

**Loss** (clipped surrogate, with `μ` inner updates per generation batch):
$$\mathcal{L} = -\frac{1}{\sum_i|o_i|}\sum_i\sum_t\Big[\min\big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon_{lo},1+\epsilon_{hi})\hat A_i\big) - \beta D_{KL}\Big]$$
where `ρ` is the per-token probability ratio. With `μ=1` the clip is inactive (ratio ≡ 1) and it reduces to plain REINFORCE-with-baseline. [primary: https://huggingface.co/docs/trl/main/en/grpo_trainer]

## 1.2 Hyperparameter table (primary values)

| Hyperparameter | Typical value(s) | Source |
|---|---|---|
| Group size `G` (samples/prompt) | **64** (DeepSeekMath); **16** (DAPO); **8–16** common in TRL/swift recipes | [DeepSeekMath §4.2 quote: "we sample 64 outputs"](https://arxiv.org/html/2402.03300v3); [DAPO](https://arxiv.org/html/2503.14476) |
| KL coef `β` | **0.04** (DeepSeekMath); **0.0 / off** is now the common default (TRL default `β=0`; DAPO removes KL; Open-Reasoner-Zero, Dr.GRPO drop it) | [DeepSeekMath §4.2: "KL coefficient is 0.04"](https://arxiv.org/html/2402.03300v3); [TRL: "we use β = 0.0 by default"](https://huggingface.co/docs/trl/main/en/grpo_trainer); [DAPO §2.3: "exclude the KL term"](https://arxiv.org/html/2503.14476) |
| Clip `ε` | **0.2** symmetric (PPO/GRPO classic); **DAPO asymmetric: ε_low=0.2, ε_high=0.28** ("clip-higher", avoids entropy collapse) | [PPO Schulman 2017](https://arxiv.org/pdf/1707.06347); [DAPO: "ε_low to 0.2 and ε_high to 0.28"](https://arxiv.org/html/2503.14476) |
| Policy LR | **1e-6** (DeepSeekMath GRPO and DAPO; constant, ~20-step linear warmup in DAPO); TRL `GRPOConfig` default **1e-6**; TRL VLM example uses **1e-5** | [DeepSeekMath §4.2: "learning rate of the policy model as 1e-6"](https://arxiv.org/html/2402.03300v3); [DAPO: "constant 1×10⁻⁶ … linear warm-up over 20 rollout steps"](https://arxiv.org/html/2503.14476); [TRL "learning_rate: Defaults to 1e-6"](https://huggingface.co/docs/trl/main/en/grpo_trainer) |
| Advantage norm | group-mean − group-std (classic). **Dr.GRPO: drop std**. **Lite-PPO: group-mean, batch-std** (most robust per ablation). TRL exposes `scale_rewards=False` (no std) and `scale_rewards="batch"` (batch std). | [Dr.GRPO](https://arxiv.org/pdf/2503.20783); [Lite-PPO/"Tricks or Traps" 2508.08221](https://arxiv.org/abs/2508.08221); [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer) |
| Iterations `μ` / batch | DeepSeekMath **μ=1** ("single update following each exploration"); TRL default `num_iterations=1`; swift default `1` | [DeepSeekMath §4.2](https://arxiv.org/html/2402.03300v3); [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer) |
| Batch / prompts | DeepSeekMath: train batch **1024** (≈16 prompts × 64); DAPO: **512 prompts × 16** | [DeepSeekMath §4.2](https://arxiv.org/html/2402.03300v3); [DAPO](https://arxiv.org/html/2503.14476) |
| Reward scaling | rewards are raw rule-based {0,1}-ish, then group-normalized; **no separate reward scaling** beyond the std normalization | [DeepSeek-R1](https://arxiv.org/html/2501.12948v1) |
| Sampling temperature | **1.0** (swift/TRL rollout default); diversity comes from temperature, not an explicit entropy bonus | [swift GRPO doc, pseudocode temperature=1.0](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html) |
| Dynamic sampling | DAPO: drop prompts whose group is all-correct or all-wrong (reward std = 0 ⇒ zero gradient) | [DAPO dynamic sampling](https://arxiv.org/html/2503.14476) |

PPO's **adaptive-KL** alternative (the trust-region knob we care about for Part 3): keep KL near a target `d_targ` by **β←β/2 if d < d_targ/1.5, β←β×2 if d > d_targ×1.5**; the paper notes a healthy first update has "a KL divergence of about 0.02 from the initial policy." [primary, PPO §4, read from PDF: https://arxiv.org/pdf/1707.06347]

## 1.3 What transfers vs what is text-specific

**Transfers cleanly to our continuous-Gaussian setting:**
- Group-relative advantage `(r − mean)/std` — this is exactly what `reinforce_core.group_advantage` does, and it is **action-space agnostic** (it operates on scalar rewards). ✔
- The Dr.GRPO and Lite-PPO critiques of the *std* term (below) — applies to any group-normalized PG.
- KL-to-reference as a trust region (the *idea*; the *estimator* differs — see below).
- The DAPO "kill zero-variance groups" insight — if a group's `G` rewards are all equal, advantage is 0/0 and the step is wasted. Directly relevant: this is your 🔴 R1 make-or-break gate.
- Single inner update (`μ=1`) ⇒ no ratio, no clip needed. This is your case.

**Text-specific, does NOT transfer:**
- **Token-level ratio `ρ_{i,t}` and the clip on it.** Our policy is a single Gaussian draw per ad on a 60-dim vector; there is no token sequence, no per-token importance ratio. With `μ=1` the ratio is identically 1 and clip is a no-op anyway. Clip-higher (ε_high=0.28) is meaningless here.
- **Length normalization / token-level loss aggregation (DAPO, Dr.GRPO length-bias fix).** There is no variable-length response; our "action" is fixed-dimension (60 hazards). The length-bias literature is irrelevant — but the *companion* finding (don't let the std term distort the signal) does carry over.
- **The k3 KL estimator over token logprobs.** For a Gaussian-vs-Gaussian with shared σ, KL has a **closed form** (`Σ(μ−μ_ref)²/2σ²`) — already in `reinforce_core.kl_to_ref`. No sampling estimator needed.
- **vLLM/generation, temperature sampling, format/parse rewards** — all gone in the head-PG path (the whole point of the design).

---

# PART 2 — If we stayed in the text-generation paradigm (we are NOT)

The text-GRPO route would have the model **emit the curve as text** (e.g. `{"R":[1.0,0.98,...]}`), parse it, and reward the parsed curve — exactly the `cross_ad_reward.parse_curve` / `r_fmt` path that already exists for the abandoned approach. Standard GRPO (swift's `grpo_trainer.py`, 2717 LOC, confirmed pure text-generation per `NORTH_STAR §10`) would apply directly. Known failure modes:

1. **Parsing cliffs.** Reward is gated on parse success (`r_fmt==0 ⇒ reward 0`). A single malformed bracket or a 61st number zeroes the whole rollout; the gradient sees "bad output" for what is actually a formatting slip, not a ranking error. Your own `parse_curve` does heroic coercion (pad/truncate to T+1, force monotone) precisely because raw text output is unreliable. [evidence: `cross_ad_reward.py:22-71`]
2. **Format collapse / reward hacking the format term.** With a `γ·R_fmt` term, the policy can farm the easy format reward (always emit valid JSON of a trivial constant curve) while ignoring ranking — Goodhart on the cheap sub-reward. DeepSeek-R1 explicitly switched to rule-based rewards because "the neural reward model may suffer from reward hacking." [https://arxiv.org/html/2501.12948v1]
3. **Monotonicity is not free.** A text curve can violate `R(t) ≤ R(t−1)`; you must post-hoc clamp (you do), which means the gradient is taken w.r.t. text tokens that don't correspond to the rewarded (clamped) curve — a biased signal.
4. **Generation bugs + cold-start.** `NORTH_STAR` lists the `get_rope_index` generation fix and R-text cold-start SFT as costs that only exist on this path. Long-video rollouts also hit `max_length` truncation (34/168 skipped in eval).
5. **Sample inefficiency.** Each of `G` rollouts needs a full autoregressive `generate()` of ~60+ numbers; the head path gets all `G` from **one forward pass** (`head_pg_compute_loss` expands `mu_z` into `G` cheaply).

**Why the continuous head is cleaner:** exact log-prob (no parse), monotonicity by construction (`exp(-cumsum(softplus))`), `G` is memory-free (shared forward), closed-form KL, no format reward to hack, no generation/cold-start. The only thing you give up is the LLM's ability to "reason in text," which the eval already showed is **not load-bearing** (the head reads the last input token; CoT content doesn't matter — `NORTH_STAR §3.2`). This is the right call.

---

# PART 3 — How to do it right: continuous-action PG on a Gaussian over a latent vector

## 3.1 Exploration σ: fixed vs learned, init, schedule

- **Parameterization.** Standard continuous-control practice (PPO, SAC, Spinning Up) parameterizes a **diagonal Gaussian** as a network mean `μ` plus **log-std** parameters, because "log stds are free to take on any values in (−∞, ∞), while stds must be nonnegative … easier to train." The log-std is commonly a **state-independent** standalone vector, not a network output. [primary: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html ; https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/]
- **Init.** CleanRL/SB3 PPO initialize **log_std = 0 ⇒ σ = 1** (state-independent). [primary: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/] But σ=1 is a *normalized-action* default; the right σ for us is set in **hazard units** (see Part 4 — our hazards live on a scale where softplus(z) is the per-second drop). **[training-data, confidence 70%]:** for a head whose pre-softplus logits are O(0.1–1), σ in the 0.05–0.3 range is a sane starting bracket; this must be **calibrated empirically** against the 🔴 R1 within-group `R_rank` std gate, not assumed.
- **Fixed vs learned.** A fixed σ is the safe start. PPO is known to **prematurely shrink exploration variance** in continuous spaces and stall in local optima (the motivation for PPO-CMA, which *adaptively expands* variance). [primary: https://arxiv.org/pdf/1810.02541] Implication for us: if you make σ learned and it collapses, exploration dies and the gradient vanishes. **Start with fixed σ; only make it learned after the loop is stable.**
- **Annealing.** Common practice is to **decay σ** as the policy sharpens (more exploit, less explore late in training). Given our short horizon (signal expected in ~50 steps) and the collapse risk, a **fixed σ with optional mild linear decay** in the back half is the conservative choice; a learned σ is an explicit upgrade path (matches `NORTH_STAR §6` "learned-σ (upgrade)").
- **Entropy bonus.** SAC's lesson: the entropy/temperature `α` is the single most sensitivity-inducing knob — too high ⇒ near-uniform policy ignores reward, too low ⇒ deterministic collapse to poor optima; SAC v2 **auto-tunes α** to hit a target entropy. [primary: https://ar5iv.labs.arxiv.org/html/1812.05905] For us, σ *is* the entropy knob (entropy of a Gaussian is monotone in log σ). An explicit entropy term is redundant with σ; prefer to control exploration via σ directly.

## 3.2 Advantage normalization & variance reduction for continuous PG

- **Group-relative baseline = your variance reduction.** Subtracting the group mean is the critic-free baseline (DeepSeekMath's core trick) and is unbiased. ✔ already implemented.
- **The std term is contested.** Dr.GRPO shows dividing by group std introduces a **question-level difficulty bias**: groups with small reward spread (near 0 or near 1 mean) get their advantages blown up, over-weighting easy/hard prompts. They **remove the std normalization**. [primary: https://arxiv.org/pdf/2503.20783 ; TRL `scale_rewards=False`] Lite-PPO's ablation goes further with the most robust recipe: **group-level mean, batch-level std** — normalize advantages by the std computed over the *whole batch* rather than per-group, and reports it "surpassing strategies like GRPO and DAPO." [primary: https://arxiv.org/abs/2508.08221 ; TRL `scale_rewards="batch"`]
  - **Direct consequence for us:** our `group_advantage` divides by per-group std (`reinforce_core.py:40`). With a bounded `[0,1]` ranking reward, a group that happens to land in a tight percentile band gets its tiny differences amplified to unit variance — exactly the difficulty bias Dr.GRPO warns about. **Recommend switching to batch-std (Lite-PPO) or no-std (Dr.GRPO).** See Part 4.

## 3.3 KL / trust region for a Gaussian policy

- **Closed form.** For two diagonal Gaussians with the same σ, `KL = Σ(μ−μ_ref)²/(2σ²)` — exact, cheap, always ≥ 0, gradient flows only through `μ`. Already in `reinforce_core.kl_to_ref`. No k3 estimator. ✔
- **Fixed vs adaptive β.** Two valid regimes from the literature:
  - **Fixed small β** (DeepSeekMath used 0.04; many modern recipes use 0). [https://arxiv.org/html/2402.03300v3]
  - **Adaptive β to a KL target** (PPO §4): `β←β/2 if KL < d_targ/1.5; β←β×2 if KL > d_targ×1.5`, with a healthy update sitting "about 0.02" KL from the reference. [primary: https://arxiv.org/pdf/1707.06347] This is more robust when you don't know the right β a priori — which is our situation, since the per-ad KL magnitude `Σ(μ−μ_ref)²/2σ²` depends on σ and the 60-dim scale and is hard to guess.
- **Note the σ coupling:** because KL scales as `1/σ²`, your effective trust region tightens as σ shrinks. If you anneal σ you are *also* changing the KL penalty strength — keep this in mind, or hold σ fixed to keep β interpretable.

## 3.4 Learning rate: head-only vs backbone

- The policy distribution is *entirely* a function of `μ_z = Linear(h_last)`. Gradient reaches the backbone only through `h_last`. Two regimes:
  - **Head-only first** (freeze backbone): the head Linear is tiny; you can use a **larger LR** (e.g. 1e-4–1e-3) and get fast, low-risk signal on whether the reward is learnable at all. This is the cheapest smoke test.
  - **Head + backbone** (the `NORTH_STAR §6` target "update head+backbone"): use the **GRPO-standard tiny LR (1e-6)** for the backbone to avoid destroying the SFT features, optionally a higher LR group for the head. TRL's own VLM GRPO example uses 1e-5 with LoRA. [https://huggingface.co/docs/trl/main/en/grpo_trainer]
- **[training-data, confidence 75%]:** RLHF/GRPO backbones are routinely fine-tuned at 1e-6–1e-5; going higher on a 3B multimodal backbone risks catastrophic forgetting of the SFT-learned video features (which are the whole reason ckpt-150 beats the length baseline). Recommend **two param groups: head ~1e-4, backbone ~1e-6**, or stage it (head-only → unfreeze).

## 3.5 RL for ranking rewards (learning-to-rank)

The reward is a **cross-ad percentile/ranking** objective, which is a learning-to-rank problem. Relevant literature:
- **Differentiable Spearman exists.** Blondel et al., *Fast Differentiable Sorting and Ranking* (ICML 2020), give an O(n log n) differentiable rank operator (projection onto the permutahedron) and explicitly "showcase … differentiable Spearman's rank correlation coefficient." [primary: https://arxiv.org/abs/2002.08871] **SoDeep** (CVPR 2019) learns a sorting surrogate net to optimize rank metrics including Spearman. [primary: https://arxiv.org/abs/1904.04272] **SoftRank/NeuralSort/PiRank** give soft-NDCG via soft permutation matrices. [https://arxiv.org/html/2508.14180v2]
  - **Why this matters for the RL-vs-supervised question (honest framing, matches `NORTH_STAR §7`):** a differentiable soft-Spearman loss would let you optimize ranking **supervised, cheaper, lower-variance** than RL. RL's distinct justification is: it optimizes the **exact** non-differentiable percentile metric (no surrogate gap) and is the course requirement. Worth stating in the report that soft-Spearman is the strongest baseline/ablation to compare against.
- **Listwise vs pairwise reward.** PG-RANK-style methods optimize expected reward over a Plackett-Luce distribution via REINFORCE. [https://arxiv.org/html/2508.14180v2] Our reward is **listwise-ish but evaluated per-ad against a fixed population CDF** — i.e. you've turned a cross-ad ranking into a *per-ad percentile-matching* reward, which is what makes the group-relative (per-ad) GRPO structure legal. The subtlety (your own `cross_ad_reward.py` docstring nails it): **the population percentile floor cancels in the group-mean subtraction, so only within-group spread carries signal.** That is correct and is the crux of the 🔴 R1 gate.
- **Reward-shaping caution for rank rewards:** a bounded `[0,1]` reward with a fixed population reference means most of the reward range is unused for any single ad (an ad sits in a narrow percentile band). This is exactly why **the std-normalization choice (3.2) matters so much** and why **G must be large enough to see rank-order spread within the group**.

---

# PART 4 — Synthesis & concrete recommendation for our system

Opinionated starting config for the head-PG (continuous Gaussian on 60-dim hazards, cross-ad percentile reward, 2×8 H100, ms-swift custom trainer, init ckpt-150, KL-ref = SFT).

| Knob | Recommendation | Why / source | Confidence |
|---|---|---|---|
| **Group size G** | **16** per ad (try 8 in smoke, 32 if variance starved). | DAPO uses 16; DeepSeekMath 64 (text, far cheaper there). G is **memory-free** here (one shared forward), so favor the high end — more rollouts = better within-group rank-spread estimate, directly feeding the R1 gate. [DAPO; DeepSeekMath] | High |
| **σ init** | **Fixed σ ≈ 0.1**, then sweep {0.05, 0.1, 0.2} against the R1 gate. **Calibrate, don't assume.** | σ is the entropy knob; must produce non-zero within-group `R_rank` std on real rollouts. Fixed first (PPO variance-collapse risk; PPO-CMA). [PPO-CMA 1810.02541; Spinning Up] | Med (70%) — scale-dependent |
| **σ schedule** | **Hold fixed** for the first runs. Add mild linear decay (×0.5 over training) only after stable. Learned-σ = later upgrade. | Collapse kills the gradient; KL∝1/σ² couples σ to trust region. | Med |
| **KL β** | **Adaptive to a target**, `d_targ ≈ 0.02` per PPO's rule (β/2 if KL<d_targ/1.5, β×2 if KL>d_targ×1.5). Init β=0.04. If you must fix it, **β=0.04** (DeepSeekMath). | We can't guess the right fixed β (KL scale depends on σ and 60-dim). Adaptive is robust and cheap (closed-form KL). [PPO §4; DeepSeekMath] | Med-High |
| **Advantage norm** | **Group mean, BATCH std** (Lite-PPO), i.e. switch `group_advantage` to divide by batch std. Acceptable fallback: **no std** (Dr.GRPO). Do NOT keep per-group std. | Per-group std amplifies tight-band groups (difficulty bias) — acute with a bounded percentile reward. Lite-PPO reports it beats GRPO/DAPO. [Dr.GRPO 2503.20783; Lite-PPO 2508.08221] | High |
| **LR (head)** | **~1e-4** | Tiny Linear; safe to move fast. | Med (75%) |
| **LR (backbone)** | **~1e-6** (GRPO standard), or **freeze backbone for the first runs**. | Protect SFT video features (the reason ckpt-150 wins). [DeepSeekMath/DAPO 1e-6; TRL VLM 1e-5] | Med-High |
| **Stage** | **Head-only first** (fast, cheap signal it's learnable) → unfreeze backbone. | Lowest-risk path to a green smoke gate. | High |
| **Ads/step (batch)** | **128–256 ads/step** (16/GPU × 16 GPUs, grad-accum to taste) × G=16 = 2048–4096 rollouts/step. ZeRO-2/3 as memory dictates. | Mirrors DAPO's 512-prompt scale, scaled to 2×8 H100; G memory-free so the cost is the single forward per ad. | Med |
| **μ (inner updates)** | **1** | DeepSeekMath default; with μ=1 there is **no ratio and no clip needed** (our continuous case). [DeepSeekMath; TRL] | High |
| **Clip ε** | **None / N/A** | No token ratio; μ=1 ⇒ ratio≡1. Clip-higher is text-specific. | High |
| **Reward** | β·R_rank only to start (α=0); add α·R_acc (1−IBS) **only if** within-group spread is starved. Drop R_fmt entirely (no text). | `NORTH_STAR §6`; the percentile floor cancels in advantage, so R_rank spread is the signal. | High |
| **Steps to signal** | Expect reward-trend signal in **~50 steps** (your own gate); held-out SRCC movement over **100–300 steps**. | `NORTH_STAR §9` smoke gate. | Med |

### Top 3 failure modes + the metric that detects each

1. **Zero within-group variance (gradient vanishes) — 🔴 R1, make-or-break.** If σ is too small or the head is too confident, all `G` sampled curves land in the same percentile band ⇒ `R_rank` identical ⇒ advantage = 0/0 ⇒ no learning. DAPO's dynamic-sampling exists exactly for this. **Detector:** `reward_within_group_std` (already logged in `head_pg_compute_loss.metrics`) and DAPO-style `frac_reward_zero_std`. Gate: must be **> 0** (target ≫ 0.004 magnitude-only floor, ~0.2-ish per your Gate-1 finding). If it's near zero → raise σ. [DAPO; your `head_pg_compute_loss.py:53`]
2. **Reward ↑ but held-out SRCC flat (Goodhart) — 🔴 R3.** The reward is a percentile-matching surrogate for the real cross-ad SRCC; they can decouple, especially if the std-normalization distorts the signal. **Detector:** track **held-out cross-ad SRCC during training** (not just training reward) with the same bypass+B1 eval; alarm if reward rises while SRCC is flat/declining over a window. [DeepSeek-R1 reward-hacking caution; `NORTH_STAR §8 R3`]
3. **Backbone catastrophic forgetting / policy collapse away from SFT.** Too-high backbone LR or too-weak KL lets the policy drift off the SFT manifold, destroying the video features that make ckpt-150 win, or σ collapses and exploration dies. **Detector:** the **closed-form KL-to-ref** (already logged as `kl`) — should stay near `d_targ≈0.02`, not blow up; and watch σ (if learned) / entropy not collapsing to ~0. If KL spikes → raise β (adaptive handles this); if SRCC drops below the ckpt-150 0.43 baseline → backbone LR too high, freeze it. [PPO adaptive KL; SAC entropy-collapse]

### One strong recommendation beyond hyperparameters

Run a **differentiable soft-Spearman (Blondel 2020) supervised baseline** alongside the RL run as your ablation. It targets the same ranking objective, is cheaper and lower-variance, and is the honest control that tells you whether RL's exact-metric optimization actually buys anything over a soft surrogate — exactly the "moderate vs SoftRank" caveat in `NORTH_STAR §7`. [https://arxiv.org/abs/2002.08871]

---

## Confidence & residual risk summary

- **High confidence (primary-sourced exact values):** GRPO recipe & equations, DeepSeekMath/DAPO/Dr.GRPO/Lite-PPO/PPO numbers, the transfer analysis, advantage-norm recommendation, μ=1/no-clip, differentiable-Spearman existence.
- **Medium confidence (scale-dependent, needs in-env calibration):** σ init/schedule and head LR — these are *brackets to sweep against the R1 gate*, not settled values, because they depend on the hazard-logit scale of ckpt-150 which I did not measure this session.
- **Could not verify this session:** Tavily was rate-limited (fell back to WebSearch/WebFetch); a couple of continuous-control "common practice" claims (σ init ranges, RLHF backbone LR norms) are tagged training-data with explicit confidence. The hazard-scale-dependent knobs (σ, head LR) **must** be empirically calibrated on real rollouts before the scaled run — the R1 within-group-std gate is the verifier.

### Primary sources
- DeepSeekMath (GRPO origin): https://arxiv.org/abs/2402.03300 · https://arxiv.org/html/2402.03300v3
- DeepSeek-R1: https://arxiv.org/abs/2501.12948 · https://arxiv.org/html/2501.12948v1
- DAPO: https://arxiv.org/abs/2503.14476 · https://arxiv.org/html/2503.14476
- Dr.GRPO (Understanding R1-Zero-Like Training): https://arxiv.org/abs/2503.20783
- Lite-PPO / "Tricks or Traps?": https://arxiv.org/abs/2508.08221
- PPO (Schulman 2017, adaptive KL + clip): https://arxiv.org/abs/1707.06347
- TRL GRPOTrainer/GRPOConfig (defaults, loss types, scale_rewards): https://huggingface.co/docs/trl/main/en/grpo_trainer
- ms-swift GRPO doc: https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html
- OpenAI Spinning Up (diagonal Gaussian policy, log-std): https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
- PPO 37 implementation details (log_std=0 init, state-independent): https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- PPO-CMA (variance collapse in continuous PPO): https://arxiv.org/abs/1810.02541
- SAC v2 (auto-tuned entropy temperature): https://arxiv.org/abs/1812.05905
- Schulman KL approximators (k3): http://joschu.net/blog/kl-approx.html
- Blondel et al. Fast Differentiable Sorting & Ranking (diff. Spearman): https://arxiv.org/abs/2002.08871
- SoDeep (rank-metric surrogate net): https://arxiv.org/abs/1904.04272
- RewardRank (LTR utility, soft NDCG survey): https://arxiv.org/abs/2508.14180

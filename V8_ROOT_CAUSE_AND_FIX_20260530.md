# V8 token_acc collapse — root cause found and fixed (2026-05-30, overnight)

## TL;DR

The token_acc collapse that the whole V8 SFT-with-CoT effort has been chasing (and
that we blamed on "alpha imbalance") was caused by **one missing line**: the custom
CoT cross-entropy in `register.py` did **not shift the labels** to the next-token
target. That turned the CoT objective into a trivial *copy* task, which silently
drove `loss_cot → 0` while **destroying** the model's real next-token prediction
(`token_acc → ~0.005`).

**The fix is one line** (shift the labels with `torch.roll(labels,-1)`), and a
controlled A/B confirms it:

| metric | **Control** (unshifted, i-0f1d, step 129) | **Fixed** (shifted, i-06cb, step 40) |
|---|---|---|
| **token_acc** | **0.0048** (collapsed, flat) | **0.563** (holds, even rising from 0.50) |
| **loss_cot**  | **0.0028** (degenerate copy → ~0) | **2.08** (healthy real next-token CE, from 2.69) |
| loss_curve    | 0.013 | 0.015 (curve head learns in both) |

Same config, same data, **only** the label-shift differs. The reasoning (CoT)
channel is now genuinely training. This is the real "调通 V8".

> Confidence: the token_acc collapse is **fixed** (definitive, code-verified +
> empirical A/B). Two things are NOT yet directly verified (both GPU-blocked, see
> Open items): (a) that the generated CoT text is *coherent* reasoning, (b) whether
> the corrected CoT *improves the retention SRCC*. loss_cot=2.08 @ token_acc=0.56 is
> fully consistent with coherent generation, but I have not generated yet.

## The root cause (code evidence)

ms-swift hands the loss function **raw, unshifted** labels; every correct consumer
shifts to the next-token target:
- `swift/trainers/utils.py:103` — `per_token_loss_func`: `labels = torch.roll(labels, shifts=-1, dims=-1)`
- `swift/trainers/seq2seq_trainer.py:176` — channel-loss masks on `torch.roll(labels,-1)`
- `swift/trainers/seq2seq_trainer.py:208` — label_smoother path passes `shift_labels=True`
- `swift/trainers/seq2seq_trainer.py:193` — our custom loss is called `compute_loss_func(outputs, labels, ...)` with that **same raw labels**
- `swift/metrics/acc.py:23-24` — `compute_acc` (token_acc) **does** shift (`labels[...,1:]` vs `preds[...,:-1]`), so token_acc was the *honest* metric all along.

The buggy code (`register.py` RetentionLoss):
```python
loss_cot = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           labels.view(-1), ignore_index=-100)   # NO shift
```
This computes `CE(logits[i], labels[i])`. In SFT the response token **is** the input
at position `i`, so the hidden state `h[i]` already contains `labels[i]` → predicting
it is a trivial **copy**. The model learns the copy, `loss_cot → ~0`, and in doing so
**overwrites** the pretrained next-token behaviour → `token_acc` (correctly shifted)
collapses.

The tell that cracked it: `loss_cot=0.119` nats ⇒ ~0.89 prob on the true token ⇒
greedy argmax should match ~85%+, yet `token_acc=0.005`. Impossible over the same
aligned positions ⇒ a shift mismatch.

The fix:
```python
shift_labels = torch.roll(labels, shifts=-1, dims=-1)   # match swift's own convention
loss_cot = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           shift_labels.view(-1), ignore_index=-100)
```

## Why this reframes the whole saga

Every prior lever was treating a symptom:
- The "alpha imbalance" diagnosis (gradient probe 45–384×) was real *arithmetic* but
  the wrong *cause* — the CoT gradient was large because the **objective was
  malformed** (a copy task), not merely mis-weighted.
- The alpha sweep (1e-3 → 0.1) and the LoRA escalation could never fix it: **no
  hyperparameter fixes a misaligned objective.** alpha=0.1 full-FT still went
  0.50→0.286; LoRA still went 0.50→0.005.
- The eval has always **bypassed** the CoT (`<cot></cot>`, head reads the anchor)
  precisely because the CoT text was trained on the broken objective. With the shift
  fixed, the CoT is real and the bypass may no longer be necessary.

## Current state (as of ~step 40 fixed / ~step 129 control)

- **Fixed run** — i-06cb, single-node 8×H100, LoRA r16 + ZeRO-3, W&B `306pvu20`
  (`v8_lora_shiftfix_1node`). token_acc 0.56 and rising, loss_cot 2.08 falling,
  loss_curve ~0.015, ~46 s/it, no OOM, 0 collapse alerts. Box-persistent babysitter
  running.
- **Control run** — i-0f1d, same setup, unshifted (buggy), W&B `v8_lora_clean_1node`.
  token_acc 0.005, loss_cot 0.003. Kept alive deliberately as the live A/B control
  (per "never auto-stop"); has `checkpoint-50` and `checkpoint-100`.
- Both runs are 3 epochs / 1680 steps (~20 h wall at 46 s/it). Neither will be stopped
  autonomously — your call in the morning.

## Open items (GPU-blocked — both boxes are training)

1. **Curve-channel SRCC** is not yet measured. The eval (`srcc_eval.py` via
   `run_lora_eval.sh`) hit a device bug on the control's ckpt-50: the 3B model loaded
   spread across GPUs 0–3 (device_map auto) → `RuntimeError: indices should be on the
   same device as cuda:3` on 199/200 ads (the 1 other was a legit
   MaxLengthError 35962>32768). **Fix is to force single-GPU load** (`CUDA_VISIBLE_DEVICES=0`
   / `device_map={'':0}`). Couldn't re-run because both boxes are now fully training.
2. **Generation probe** (does the fixed model emit *coherent* CoT?) — the ultimate
   proof the reasoning is real — also needs a free GPU.

## Recommended next steps (your call)

1. **Free one box** (or wait for a checkpoint) and run, on the fixed run's ckpt:
   (a) the single-GPU-fixed SRCC eval (does corrected CoT change the curve SRCC vs the
   0.514/0.4393 bypass baseline?), (b) a generation probe (coherent reasoning text?).
2. With the objective now correct, **re-open the alpha question honestly**: sweep
   alpha (0.1 / 0.3 / 1.0) *with the shift* to balance CoT vs curve — this is now a
   real tuning question, not a workaround for a bug.
3. Decide full-FT vs LoRA for the production run now that the CoT trains correctly
   (LoRA proved token_acc holds; full-FT may push loss_cot lower / CoT quality higher).
4. This fix should land on `main` register.py (it currently lives on i-06cb + locally;
   i-0f1d intentionally still has the buggy version as the control).

## Overnight infra notes (resolved, for completeness)

- LoRA single-node hit two OOMs, both fixed: NVLS multicast (`NCCL_NVLS_ENABLE=0` +
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) then LM-head logits OOM on long
  samples → switched to the config's documented `deepspeed: zero3` fallback (shards the
  base, keeps the validated `max_length=32768`).
- Single-node NVLink ZeRO-3 (~46 s/it) beats 2-node EFA ZeRO-3 (~256 s/it) for this
  model — the inter-node base-param all-gather is the bottleneck, so single-node wins.

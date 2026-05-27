# V8 Launch Runbook — Modal (Wanjia)

Companion to [V8_LAUNCH_RUNBOOK.md](V8_LAUNCH_RUNBOOK.md) (the AWS H100 path).
This doc covers the Modal LoRA path. You already have a working V7-hazard-LoRA
Modal pipeline; this is the V8 **delta**, not a from-scratch rebuild.

## TL;DR — what's different in V8

| Aspect | V7 (your old run) | V8 (this run) |
|---|---|---|
| **Training data** | `ttcc_train_sft.jsonl` (R(t) in assistant span — LEAKY) | `ttcc_train_with_cot.jsonl` (assistant = `<cot>…</cot>` only) |
| **Architecture** | hazard head reads h[last input token] | hazard head reads h[last `</cot>` token] |
| **Joint loss** | MSE only | MSE + α · LM-CE on CoT span |
| **α (CoT weight)** | n/a | **1e-3** (was 0.1 in earlier hazard-LoRA config — outdated) |
| **Init** | from base or V7 LoRA | **from base Qwen2.5-Omni-3B** (never from V7 ckpts — they have the leak path baked in) |
| **Yaml** | `sft_retention_hazard_lora_no_cot.yaml` (deprecated) | `sft_retention_hazard_lora_with_cot.yaml` (needs the patch below) |

**Why V8 exists**: V7's training data put ground-truth R(t) in the assistant
span. The hazard head reading h[last token] trivially echoed it. Three audits
(randomization probe, video-swap, constant-predictor) confirm V7's backbone
**did not use video**. V8 fixes this by (a) clearing R(t) out of the assistant
span and (b) replacing it with Gemini-distilled CoT reasoning.

See `INCIDENT_2026-05-26_EVAL_LEAK.md` in the `ttcc-eval` repo for the full
incident review.

## What experiment to run

**V8 LoRA + CoT** (matches `sft_retention_hazard_lora_with_cot.yaml`).

This is a complementary run to the AWS full-FT path:
- AWS Node A: V8 main — **full FT** + CoT (α=1e-3)
- Modal (yours): V8 LoRA + CoT (α=1e-3) — **tests whether LoRA suffices**
- AWS Node B (if available): V8 α=0 ablation — tests whether CoT supervision helps

If your LoRA result is within ~5–10% IBS of full FT, that's a big win for
cost/reproducibility. If it's much worse, full FT was necessary.

## Pre-launch — what to pull from HuggingFace

| Asset | HF path | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-Omni-3B` | Same as V7. ~6 GB. |
| Videos | `liangyuch/ttcc-v0_2_0` (dataset) | 61,789 rows × 51 cols with embedded video bytes. Same as your V7 pipeline. |
| **V8 training jsonl** | **TODO Leon: upload** — currently lives on the dead 8-GPU box | 39,375 rows, ~57 MB. Each row: `messages` (sys+user+assistant CoT), `videos`, `audios`, `T`, `R`. See "Data layout" below for schema. |
| Holdout val (leak-free) | TODO Leon: upload `val_200_no_cot.jsonl` | 200 rows, used for in-loop eval. |

**ACTION FOR LEON before launch**: push the V8 train jsonl + holdout val to
HuggingFace as a private dataset repo (`liangyuch/ttcc-v8-train` or similar),
then update this doc with the path. Until then, Wanjia rebuilds V8 jsonl from
V7 jsonl + CoT jsonl using `tools/build_v8_train_jsonl.py` (see fallback below).

## Required credentials (Wanjia's responsibility)

| Credential | Where on Modal | Why |
|---|---|---|
| HF token | Modal Secret (`hf-token` or similar), read into env as `HF_TOKEN` | Pull base model + dataset + push checkpoints |
| GitHub PAT | Modal Secret (only if cloning private `cliangyu/go_viral`) | Already public on the `ttcc-rl` branch, so PAT is optional |
| wandb API key | Modal Secret, read into env as `WANDB_API_KEY` | Run tracking |
| Vertex AI SA JSON | NOT needed | Only for generating more CoT data, which is already done |

You do **not** need AWS credentials. None of V8's required data lives on S3
once Leon uploads the jsonl to HF.

## Data layout — what one row of V8 jsonl looks like

```json
{
  "messages": [
    {"role": "system",    "content": "You are an expert in short-form video advertising..."},
    {"role": "user",      "content": "This ad is 15 seconds long. Estimate the per-second retention curve."},
    {"role": "assistant", "content": "<cot>The opening shot shows... viewers in the 18-24 segment...</cot>"}
  ],
  "videos": ["/path/to/ad.mp4"],
  "audios": ["/path/to/ad.mp4"],
  "T": 15,
  "R": [1.0, 0.84, 0.71, ...]
}
```

**Critical correctness invariants** (the V7 leak audits caught these — verify
on at least 20 random rows before launching):

1. `messages[-1]["content"]` starts with `<cot>` and ends with `</cot>` — nothing else.
2. No decimal numbers from `R` appear anywhere inside `<cot>...</cot>`. (The
   teacher LLM was prompted to avoid R values, but spot-check.)
3. `videos[0]` and `audios[0]` point to the **same** MP4 path (Qwen-Omni reads
   audio embedded in video).
4. `len(R) == T + 1` and `R[0] == 1.0`.
5. `R` is monotone non-increasing.

Quick check:
```python
import json, re
n_ok = 0
with open("ttcc_train_with_cot.jsonl") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        a = d["messages"][-1]["content"]
        R = d["R"]
        assert a.strip().startswith("<cot>") and a.strip().endswith("</cot>"), f"row {i}: assistant span malformed"
        for r in R[1:]:
            assert f"{r:.4f}" not in a and f"{r:.3f}" not in a, f"row {i}: R value leaked into CoT"
        assert d["videos"][0] == d["audios"][0], f"row {i}: video/audio mismatch"
        assert abs(R[0] - 1.0) < 1e-6 and len(R) == d["T"] + 1, f"row {i}: R/T mismatch"
        assert all(R[k+1] <= R[k] + 1e-6 for k in range(len(R)-1)), f"row {i}: R not monotone"
        n_ok += 1
        if n_ok >= 50: break
print(f"{n_ok} rows validated")
```

## The V8 LoRA yaml (patch from current `sft_retention_hazard_lora_with_cot.yaml`)

The yaml at `examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_lora_with_cot.yaml`
is the right starting point but **needs three small patches** to match V8:

```diff
 ENV:
   RETENTION_HEAD_TYPE: hazard
-  RETENTION_COT_ALPHA: '0.1'
+  RETENTION_COT_ALPHA: '1e-3'         # principled balance: CoT ~3e-3 vs curve ~0.1

-dataset: /home/ssm-user/work/data/ttcc_swift_v2cot/ttcc_train_sft.jsonl
+dataset: <YOUR MODAL PATH TO ttcc_train_with_cot.jsonl>
+val_dataset: <YOUR MODAL PATH TO val_200_no_cot.jsonl>
+eval_strategy: steps

-max_length: 32768
+max_length: 49152                     # eliminates the ~18% drop rate at 32768
```

Why these specifically:
- **α=1e-3**: with random-init head from base (curve MSE ~0.1 at start), this
  makes CoT-CE contribution ~3% of total loss — small enough to let the head
  converge, large enough to drive LM head learning. α=0.1 (the old value) was
  tuned for warm-start from a converged head and would over-weight CoT here.
- **max_length=49152**: at 32768, ~18% of training rows get dropped due to
  long ad videos exceeding 33K–45K tokens. 49152 gives 0 drops on the 300-ad
  sample we tested.
- **val_dataset + eval_strategy=steps**: critical for catching divergence
  early. Use the leak-free holdout (assistant span empty).

Everything else in the LoRA yaml (rank=16, α=32, `modules_to_save:
retention_head`, lr=1e-4, ZeRO-2) stays.

## Inference-time anchor — what changed under the hood

The hazard head now reads h at the **last `</cot>` token** instead of the
last input token. This is the anti-leak design: if some `<cot>` content gets
malformed in the future, the head still attends to a deterministic anchor
that's downstream of all reasoning but upstream of any structured output.

The plugin (`examples/custom/qwen2_5_omni_retention/register.py`) does this
automatically — no code change needed on your end. Fallback: if no `</cot>`
is found, it falls back to h[last input token] (same as V7) so this is
backward-compatible.

## Launch command (your Modal app)

In your existing Modal app, the training launcher should call:

```bash
# Inside the Modal container:
export RETENTION_HEAD_TYPE=hazard
export RETENTION_COT_ALPHA=1e-3
export PYTHONPATH=/path/to/ms-swift

NPROC_PER_NODE=$N_GPUS \
swift sft \
  --config_file examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_lora_with_cot.yaml \
  --dataset $V8_TRAIN_JSONL \
  --val_dataset $V8_VAL_JSONL \
  --max_length 49152 \
  --output_dir $OUT_DIR
```

CLI flags override yaml. This lets you patch dataset/val paths without
modifying the yaml itself.

## Fallback: if Leon hasn't uploaded the V8 jsonl yet

You can rebuild it locally on Modal from V7 jsonl + CoT jsonl. Both are
mirrored on HuggingFace under datasets Leon controls (or in private S3 if HF
isn't ready):

```bash
# Pull V7 train jsonl + CoT jsonl from wherever Leon staged them
# (replace these paths once Leon confirms HF upload)
huggingface-cli download liangyuch/<v7-data-repo> --repo-type dataset --local-dir /tmp/v7_data
huggingface-cli download liangyuch/<cot-data-repo> --repo-type dataset --local-dir /tmp/cot_data

# Build V8 (merges V7 jsonl rows with CoT entries by ad_id)
python /go_viral/examples/custom/qwen2_5_omni_retention/tools/build_v8_train_jsonl.py \
    --v7-jsonl /tmp/v7_data/ttcc_train_sft.jsonl \
    --cot-jsonl /tmp/cot_data/cot_v6_train.jsonl \
    --out-jsonl /tmp/ttcc_v8/ttcc_train_with_cot.jsonl

wc -l /tmp/ttcc_v8/ttcc_train_with_cot.jsonl   # should be 39,375
```

## Pre-launch sanity check

Same script as the AWS path:
```bash
bash examples/custom/qwen2_5_omni_retention/tools/validate_v8_launch.sh \
    /path/to/base/Qwen2.5-Omni-3B \
    /path/to/ttcc_train_with_cot.jsonl \
    /path/to/val_200_no_cot.jsonl
```

Exits 0 = safe to launch. Exits 1 = stop and debug.

## During training — what to watch

| Step range | Expected behavior | Abort if |
|---|---|---|
| 0 | head is random; curve MSE ~0.1, CoT-CE ~3 per token | NaN / inf in either |
| 50 | both losses dropping; r_pred starts to look like a curve | losses stuck or rising |
| 100–200 | curve MSE < 1e-2; eval IBS comparable | eval IBS > 0.1 (means video signal not propagating) |
| 500+ | curve MSE plateaus around 5e-4 to 1e-3 | divergence |

Save checkpoints every 50 steps (already set in yaml). Keep last 10.

## What to do with the trained model

1. Upload LoRA adapters + retention_head to HuggingFace:
   `liangyuch/ttcc-sft-qwen25omni-3b-lora-cot-v8` or your preferred name.
2. Make sure `modules_to_save: retention_head` was honored — the
   `adapter_model.safetensors` should contain a `retention_head.weight` key.
   If it doesn't, the head is gone and the checkpoint is useless.
3. **Tokenizer overlay** — same gotcha as V7. After saving, copy these 6
   files from base `Qwen2.5-Omni-3B/` into your ckpt dir before pushing:
   `added_tokens.json`, `merges.txt`, `special_tokens_map.json`, `vocab.json`,
   `chat_template.json`, `tokenizer_config.json`. Otherwise loading the
   ckpt later will fail with `Qwen2TokenizerFast has no attribute image_token`.
   The ms-swift codebase now has a vendor patch for this in
   `swift/trainers/mixin.py`, but verify it actually copied them by listing
   the ckpt dir before pushing.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `Qwen2TokenizerFast has no attribute image_token` on load | Tokenizer overlay missing | Copy the 6 files from base model into ckpt dir |
| LoRA checkpoint has only LoRA deltas, no `retention_head.weight` | `modules_to_save` lost in CLI override | Verify the yaml's `modules_to_save: [retention_head]` survived |
| Eval IBS stuck near 0.1 while train MSE drops | Model overfitting to train distribution / not using video | Suspicious — run the randomization probe (zero-out video pixels at inference; if IBS doesn't change much, model isn't using video) |
| Train loss diverges around step 50 | α too high relative to curve loss | Lower α to 5e-4 |
| Drop rate >0 at max_length | Some ads exceed 49152 tokens | Don't raise further; let those rows drop |

## Coordination with Leon's AWS run

Both runs share:
- Same training data (`ttcc_train_with_cot.jsonl`)
- Same base init (Qwen2.5-Omni-3B)
- Same α (1e-3)
- Same `</cot>` anchor for the head

Difference: LoRA vs full FT.

After both finish, the comparison: LoRA IBS vs full FT IBS on the
**leak-free** test set (use `eval_ibs.py --strip-assistant` — default ON in
the latest commit on `ttcc-rl`).

## When to escalate to Leon

- Tokenizer overlay broken at save time and you can't get the vendor patch to work
- CoT data validation fails on more than ~1% of rows (this would mean the
  Gemini distillation has a problem)
- Eval IBS still >0.05 after 1000 steps with leak-free protocol (means V8
  architecture is also failing, not just V7)
- Modal billing concern (LoRA on Modal is much cheaper than full FT on AWS,
  but full pass through 39K videos at H100 prices still adds up)

---

## Quick reference — what Leon owes Wanjia before 4am

1. Upload V8 train jsonl + holdout val to HF (private dataset repo)
2. Confirm whether `ttcc-rl` branch on `cliangyu/go_viral` has the latest
   `register.py` (the one with `</cot>` anchor + per-component loss logging)
3. Send wandb project name to log into (so AWS + Modal runs land in the
   same project for easy comparison)
4. Confirm HF token Wanjia is using has write access to the destination
   repo for checkpoint upload

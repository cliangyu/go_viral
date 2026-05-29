# Reproduce V8 SFT on Modal — handoff (Wanjia)

Run **exactly our V8 SFT** (the leak-free hazard-retention-head fine-tune that produced
`checkpoint-225`, val cross-ad SRCC **0.5142**) — on **Modal** instead of AWS.

**The setup is identical to ours in every respect except the compute/parallelism layer.** Same base
model, same data, same config, same hyperparameters, same loss, same leak-free design. The only thing
your coding agent needs to adapt for Modal is **§4 (parallelism)**.

Everything below is **verified** (§6): the data script's output was diffed byte-for-byte against our
trusted V8 data — **150/150** exact match on `R`, `T`, and the `<cot>` assistant.

Everything you need is in **this repo** (`cliangyu/go_viral`, branch **`ttcc-rl`**) — the training
engine, the config, the retention-head plugin, AND the data-build script — plus two public HF datasets.

---

## 1. What you need (all self-serve)

| # | Item | Source |
|---|------|--------|
| 1 | Base model | `Qwen/Qwen2.5-Omni-3B` (HF) — train **from base**, never a V7/V8 ckpt (leak) |
| 2 | Engine + config + head plugin + data script | **this repo** (`go_viral` @ `ttcc-rl`) |
| 3 | Video + retention data | `liangyuch/ttcc-v0_2_0` (public dataset) |
| 4 | CoT (`<cot>`) data | `liangyuch/ttcc-cot` (public dataset) |

> Provenance (verified 2026-05-29): ~87–100% of the V8 training ad_ids are in public `ttcc-v0_2_0`,
> and 100% are in `ttcc-cot` — so you rebuild the data entirely from these two public repos.

---

## 2. Build the data (one script — verified byte-identical to ours)

Script: **`examples/custom/qwen2_5_omni_retention/tools/make_v8_from_hf.py`** (in this repo).

```bash
# 0) pull the parquet (train + val; you do NOT need test-*). Large — embedded video bytes.
hf download liangyuch/ttcc-v0_2_0 --repo-type dataset \
    --include 'data/train-*.parquet' --include 'data/val-*.parquet' --local-dir /vol/data/hf_ttcc

# 1) TRAIN split (joins ttcc-cot's <cot>; extracts <ad_id>.mp4 into --video-dir)
python examples/custom/qwen2_5_omni_retention/tools/make_v8_from_hf.py --split train \
    --hf-ttcc-dir /vol/data/hf_ttcc --video-dir /vol/data/videos \
    --out-jsonl /vol/data/ttcc_v8/ttcc_train_with_cot.jsonl

# 2) VAL split (no CoT; empty assistant)
python examples/custom/qwen2_5_omni_retention/tools/make_v8_from_hf.py --split val --no-cot \
    --hf-ttcc-dir /vol/data/hf_ttcc --video-dir /vol/data/videos \
    --out-jsonl /vol/data/ttcc_holdout/val_200_no_cot.jsonl
```

Produces the exact V8 row schema: `{messages:[system,user,assistant], videos:[<mp4>], audios:[],
ad_id, T, R}` — short prompts, `audios=[]`, leak-free `<cot>` (train) / empty (val), `R` via
`c[0]`-normalization. (Transforms documented in the script docstring.)

**Preflight (run before training — must be 0 violations):**
```python
import json
for path, cot in [("/vol/data/ttcc_v8/ttcc_train_with_cot.jsonl", True),
                  ("/vol/data/ttcc_holdout/val_200_no_cot.jsonl", False)]:
    bad = 0
    for line in open(path):
        r = json.loads(line); a=[m["content"] for m in r["messages"] if m["role"]=="assistant"][0]
        R=[float(x) for x in r["R"]]; T=int(r["T"])
        ok = (a.startswith("<cot>") and a.rstrip().endswith("</cot>")) if cot else (a.strip()=="")
        ok &= r["audios"]==[] and len(R)==T+1 and abs(R[0]-1)<1e-6 and all(R[i]<=R[i-1]+1e-6 for i in range(1,len(R)))
        bad += (not ok)
    print(path, "violations", bad); assert bad==0
```

---

## 3. Train — the EXACT V8 config

Config: `examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_with_cot.yaml`
(in this repo). **Change only the paths**; the non-path fields are the V8 setting:

```yaml
model: <local>/Qwen2.5-Omni-3B            # base model (§1.1)  — PATH
torch_dtype: bfloat16
attn_impl: flash_attention_3
external_plugins: [examples/custom/qwen2_5_omni_retention/register.py]
model_type: qwen2_5_omni_retention
loss_type:  retention_loss
ENV: { RETENTION_HEAD_TYPE: hazard, RETENTION_COT_ALPHA: '1e-3' }   # loss: MSE + 1e-3·CoT-CE
dataset:     /vol/data/ttcc_v8/ttcc_train_with_cot.jsonl            # §2 output — PATH
val_dataset: /vol/data/ttcc_holdout/val_200_no_cot.jsonl           # §2 output — PATH
max_length: 32768                          # memory-dependent — see §4d
truncation_strategy: delete                # drop over-length rows (never chop the </cot> tail)
lazy_tokenize: true
remove_unused_columns: false               # REQUIRED: the head reads R from the raw row
tuner_type: full
freeze_vit: true
freeze_aligner: true
deepspeed: zero3                           # REQUIRED (NOT zero2 — audio-tower FA3 crash with audios=[])
gradient_checkpointing: true
vit_gradient_checkpointing: true
torch_compile: false
num_train_epochs: 10
learning_rate: 5.0e-6
warmup_ratio: 0.03
max_grad_norm: 1.0
per_device_train_batch_size: 1             # REQUIRED — keep at 1 (see §4a)
gradient_accumulation_steps: 8             # scale to your GPU count (see §4b)
eval_steps: 50
save_steps: 75
output_dir: /vol/out/v8_sft                # PATH
```

Loss = `masked_MSE(R_pred, R_true) + 1e-3·LM_CE(<cot> tokens)`; the head is a softplus **hazard**
head reading `h[</cot>]`, so `R(t)=exp(-cumsum(λ))` is monotone by construction.

Launch (same swift path we use):
```bash
swift sft --config examples/train/grpo/qwen2_5_omni_ttcc/configs/<your-modal-config>.yaml
```

---

## 4. ⚠️ Modal vs AWS — the ONLY real difference (for your coding agent)

We ran **8×H100 80 GB, single node, on AWS** (SSM + `torchrun`/swift). On Modal you pick GPU type+count,
so the **distributed/parallelism layer differs**. Adapt these — nothing else:

**4a. Keep fixed regardless of topology:**
- `per_device_train_batch_size: 1` — **required, do not raise.** The hazard head reads the hidden state
  at the last token; `bs>1` right-pads → that becomes a PAD position → wrong anchor → train/eval
  mismatch. Scale via grad-accum + GPU count, never batch size.
- `deepspeed: zero3` — **required** (ZeRO-2 calls the audio tower with a dummy tensor and hits a
  FlashAttention-3 `cu_seqlens` crash because `audios=[]`).

**4b. Match the effective batch size.**
- Ours: `bs(1) × grad_accum(8) × num_gpus(8) = 64` ads / optimizer step.
- On Modal with **N** GPUs: keep `bs=1`, set `gradient_accumulation_steps = 64 / N`
  (N=8→8, N=4→16, N=2→32, N=1→64) to preserve identical optimization dynamics.

**4c. Distributed launch.** Single-node multi-GPU: `torchrun --nproc_per_node=<N>` (or swift's
launcher) with `--nproc_per_node` = your GPU count; ensure `MASTER_PORT` is free. Prefer one node with
N GPUs (multi-node needs inter-node NCCL set up).

**4d. Memory (`max_length`).** `32768` was tuned for **H100 80 GB** (a 60 s ad ≈ 36 K video tokens).
On smaller GPUs (e.g. A100 40 GB) **lower it** (24576 / 16384). `truncation_strategy: delete` drops
over-cap rows (a bit less data, no crash). Never use `truncate` (it chops the load-bearing `</cot>`).

**4e. Modal scaffolding (your coding agent builds this):** a persistent **Volume** for `/vol/data`
(parquet→videos→JSONL) + `/vol/out` (ckpts); an **image** with this repo (`ttcc-rl` branch) + ms-swift +
`flash-attn` (FA3) + transformers(Qwen2.5-Omni) + `huggingface_hub` + `pyarrow`; a **GPU function**
(`gpu="H100:8"` or your choice) running the `swift sft` above; HF token inside the function.

---

## 5. What you get
`checkpoint-225` reproduced = val cross-ad SRCC **0.5142** (leak-free). That's the SFT baseline — it
runs correctly and leak-free but does not yet beat the privileged baselines; closing that gap
(rank-SFT / RL) is a separate research thread on our side.

## 6. Verification (2026-05-29)
- `make_v8_from_hf.py` output vs trusted `ttcc_train_with_cot.jsonl`: **150/150** byte-exact on `R`,
  `T`, `<cot>` assistant (0 mismatches), on ads present in both repos.
- Trusted-data invariants (35,793 train + 200 val): `<cot>`/empty assistant, no R-leak, R well-formed,
  `audios=[]`, `videos` basename `<ad_id>.mp4` — **0 violations**.
- Provenance: trusted ad_ids ⊂ `ttcc-cot` (100%), ⊂ `ttcc-v0_2_0` (~87–100%).

## 7. Known caveat
`make_v8_from_hf.py --split val --no-cot --limit N` (if you cap) takes the first N val ads in shard
order; our `val_200_no_cot.jsonl` is a specific 200-ad holdout. Fine for training/monitoring; for an
exact SRCC comparison against our 0.5142, use the same 200 ad_ids (ask us for the list). Default (no
`--limit`) emits the full val split.

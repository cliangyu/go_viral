# V8 Launch Runbook — Modal (Wanjia)

Companion to [V8_LAUNCH_RUNBOOK.md](V8_LAUNCH_RUNBOOK.md) (the AWS H100 path).

**Goal: replicate the AWS run exactly on Modal.** Same yaml, same data, same
architecture (full FT 8×H100, ZeRO-3, hazard head + CoT, α=1e-3). The only
delta is the platform (Modal container + Volume vs AWS SSM + NVMe), so the
two runs constitute a clean reproducibility A/B.

## What V8 is (and why this run exists)

V7's training data put ground-truth R(t) in the assistant span. The hazard
head reading h[last token] trivially echoed it. Three audits (randomization
probe, video-swap, constant-predictor sanity) confirm V7's backbone
**did not use video**. V8 fixes this by:

- Replacing the leaky assistant span with Gemini-distilled `<cot>…</cot>` reasoning
- Anchoring the hazard head at h[last `</cot>` token] (downstream of all
  reasoning, upstream of any structured output — anti-leak by design)
- Adding a joint LM-CE loss on the CoT span with α=1e-3

V8 trains from **base Qwen2.5-Omni-3B**, never from V7 ckpts (the leak path
is baked into V7's weights).

See `INCIDENT_2026-05-26_EVAL_LEAK.md` in `ttcc-eval` for the full review.

## Hardware target

**8×H100 single container** (same shape as AWS p5.48xlarge):
- `gpu="H100:8"` in Modal = 640 GB GPU RAM on one machine, NVLink
- Modal docs caveat: requesting >2 GPUs incurs longer wait times in the queue,
  so submit early
- Pricing: $3.95/hr × 8 = **$31.60/hr**
- Estimated wall-clock: 45–60h for V8 full FT (10 epochs × ~4400 steps)
- Estimated cost: **~$1,500–$1,900 per run**

Alternative: 8×H200 if your account has access ($4.54×8 = $36.32/hr, ~15% faster).

## The 24-hour boundary (Modal-specific gotcha)

Modal hard-caps single function executions at **24h**. V8 needs ~50h. The
official pattern (from [modal.com/docs/examples/long-training](https://modal.com/docs/examples/long-training)):

```python
@app.function(
    gpu="H100:8",
    timeout=86400,                                  # 24h max
    retries=modal.Retries(max_retries=10, initial_delay=0.0),
    volumes={"/vol": modal.Volume.from_name("ttcc-v8")},
)
def train():
    ...
```

When the 24h timer expires, Modal kills the container and the retry policy
spawns a fresh one. The new container mounts the same Volume → ms-swift's
`--resume_from_checkpoint` picks up from the last saved step. Save every 75
steps (already in yaml) so at most 75 steps × ~40s = ~50 minutes of work is
re-done after each boundary.

**Don't use `infinity` for `timeout`** — Modal will reject it for GPU jobs.

## Pre-launch — what to provision

### 1. Modal Secrets

Create these in the Modal dashboard (one-time, reused across all runs):

| Secret name | Contents | Used for |
|---|---|---|
| `hf-token` | `HF_TOKEN=<your hf token>` | Pull base model + dataset, push checkpoints |
| `wandb-key` | `WANDB_API_KEY=<your wandb key>` | Run tracking |

You do **not** need AWS credentials. You do **not** need a GitHub PAT (the
`ttcc-rl` branch on `cliangyu/go_viral` is public).

### 2. Modal Volume

```bash
modal volume create ttcc-v8
```

Layout it will hold (~1 TB total):
```
/vol/hf-cache/Qwen2.5-Omni-3B/        # base model, ~6 GB
/vol/data/ttcc_v8/                     # train + val jsonl, ~60 MB
/vol/data/videos/                      # 39K MP4s, ~935 GB
/vol/output/sft_retention_hazard_full_with_cot/   # ckpts, ~80 GB after 10 epochs
```

### 3. Data to pull from HuggingFace (inside container, on first run)

All inputs are PUBLIC on HF. No Leon-side upload required. Your V7
pipeline already pulls (1) and (2); (3) is the new ingredient for V8 and
also already public.

| Asset | HF path | Notes |
|---|---|---|
| (1) Base model | `Qwen/Qwen2.5-Omni-3B` | Same as V7. ~6 GB. |
| (2) Videos + retention curves | `liangyuch/ttcc-v0_2_0` (public dataset) | 61,789 rows × 51 cols with embedded video bytes; 935 GB. Same as V7. |
| (3) **CoT distillation** | **`liangyuch/ttcc-cot` (public dataset)** | 39,375 train rows. Schema: `{ad_id, cot, model, prompt_version}`. The `cot` field is already wrapped in `<cot>...</cot>` and leak-free. |

The V8 training JSONL is **built on Modal at runtime** by merging (2) + (3)
on `ad_id` via the helper
`examples/custom/qwen2_5_omni_retention/tools/build_v8_from_hf.py`. The
val/test JSONLs are built from (2) only with empty assistant content
(no CoT supervision at eval time). No pre-uploaded JSONL needed.

## Data layout — what one row of V8 jsonl looks like

```json
{
  "messages": [
    {"role": "system",    "content": "You are an expert in short-form video advertising..."},
    {"role": "user",      "content": "This ad is 15 seconds long. Estimate the per-second retention curve."},
    {"role": "assistant", "content": "<cot>The opening shot shows... viewers in the 18-24 segment...</cot>"}
  ],
  "videos": ["/vol/data/videos/<ad_id>.mp4"],
  "audios": ["/vol/data/videos/<ad_id>.mp4"],
  "T": 15,
  "R": [1.0, 0.84, 0.71, ...]
}
```

**Critical correctness invariants** — verify on at least 50 random rows
before launching ($1,800 mistakes hurt):

1. `messages[-1]["content"]` starts with `<cot>` and ends with `</cot>` — nothing else
2. No decimal numbers from `R` appear anywhere inside `<cot>...</cot>`
3. `videos[0]` and `audios[0]` point to the same MP4 path
4. `len(R) == T + 1` and `R[0] == 1.0`
5. `R` is monotone non-increasing

```python
import json
n_ok = 0
with open("/vol/data/ttcc_v8/ttcc_train_with_cot.jsonl") as f:
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

## The yaml to use

`examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_with_cot.yaml`

**Use this verbatim** — same one the AWS run uses. The only Modal-specific
overrides happen at the CLI:

```bash
swift sft \
  --config_file examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_with_cot.yaml \
  --model /vol/hf-cache/Qwen2.5-Omni-3B \
  --dataset /vol/data/ttcc_v8/ttcc_train_with_cot.jsonl \
  --val_dataset /vol/data/ttcc_v8/val_200_no_cot.jsonl \
  --output_dir /vol/output/sft_retention_hazard_full_with_cot
  # add --resume_from_checkpoint <ckpt-path> on retry runs (see below)
```

Config recap (do not change):
- `tuner_type: full`, `deepspeed: zero3`
- `learning_rate: 5.0e-6`, `num_train_epochs: 10`
- `max_length: 49152` (eliminates ~18% drop rate seen at 32768)
- `RETENTION_HEAD_TYPE=hazard`, `RETENTION_COT_ALPHA=1e-3`
- `save_steps: 75`, `save_total_limit: 10`

## Dependencies — build the Modal image correctly

The AWS path uses `pip install -e .` inside a cloned `go_viral` repo, which
installs all of ms-swift's `requirements/framework.txt`. Same idea on Modal.
Two gotchas:

1. **flash-attn must be installed AFTER torch** with `--no-build-isolation`,
   matching the CUDA + torch ABI. Doing it in a single `pip_install` call
   silently picks the wrong wheel.
2. **CUDA base image** — use Modal's `from_registry("nvidia/cuda:...")` or
   `modal.Image.from_registry("pytorch/pytorch:...")` so the right CUDA libs
   are present. `debian_slim` does NOT have CUDA.

```python
image = (
    # Pytorch 2.4 + CUDA 12.4 base — matches the wheel flash-attn 2.8.3 expects
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg", "libgl1", "build-essential", "ninja-build")
    # flash-attn first, with --no-build-isolation so it sees torch
    .pip_install(
        "flash-attn==2.8.3",
        extra_options="--no-build-isolation",
    )
    # Everything else
    .pip_install(
        "deepspeed==0.15.2",
        "huggingface_hub[hf_transfer]",            # parallel HF downloads
        "wandb",
        "qwen-omni-utils",
        "av",                                       # video decoding for Qwen-Omni
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Clone ttcc-rl branch + install ms-swift in editable mode.
    # This pulls in the rest (transformers, accelerate, peft, datasets, etc.)
    # from requirements/framework.txt.
    .run_commands(
        "git clone -b ttcc-rl https://github.com/cliangyu/go_viral.git /opt/go_viral",
        "cd /opt/go_viral && pip install -e '.[all]'",
    )
)

vol = modal.Volume.from_name("ttcc-v8", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100:8",
    timeout=86400,                                          # 24h ceiling
    retries=modal.Retries(max_retries=10, initial_delay=0.0),
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token"),
             modal.Secret.from_name("wandb-key")],
)
def train():
    os.environ["HF_HOME"] = "/vol/hf-cache"
    os.environ["RETENTION_HEAD_TYPE"] = "hazard"
    os.environ["RETENTION_COT_ALPHA"] = "1e-3"
    os.environ["WANDB_PROJECT"] = "ttcc-v8"
    os.environ["WANDB_NAME"] = f"v8_main_modal_{os.environ.get('MODAL_TASK_ID','')[:8]}"

    # --- Stage 1: ensure base model + data present (no-op on retry) ---
    base = Path("/vol/hf-cache/Qwen2.5-Omni-3B")
    if not (base / "config.json").exists():
        subprocess.run([
            "huggingface-cli", "download", "Qwen/Qwen2.5-Omni-3B",
            "--local-dir", str(base),
        ], check=True)

    train_jsonl = Path("/vol/data/ttcc_v8/ttcc_train_with_cot.jsonl")
    if not train_jsonl.exists():
        # Pull from HF (Leon should have uploaded by now); or rebuild from V7 + CoT
        raise SystemExit("V8 training jsonl missing — pull from HF or use fallback path")

    val_jsonl = Path("/vol/data/ttcc_v8/val_200_no_cot.jsonl")
    assert val_jsonl.exists(), "holdout val jsonl missing"

    # --- Stage 2: videos. Either staged once or streamed from HF every retry. ---
    videos_dir = Path("/vol/data/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    n_mp4 = len(list(videos_dir.glob("*.mp4")))
    if n_mp4 < 39000:
        # First-time staging. ETA: 2-6h depending on Modal's HF cache.
        # Step 1: pull parquet shards (~935 GB embedded video bytes).
        subprocess.run([
            "huggingface-cli", "download", "liangyuch/ttcc-v0_2_0",
            "--repo-type", "dataset",
            "--local-dir", "/vol/data/hf_ttcc",
        ], check=True)
        # Step 2: extract MP4s from parquet rows -> /vol/data/videos/<ad_id>.mp4
        subprocess.run([
            "python",
            "/opt/go_viral/examples/custom/qwen2_5_omni_retention/tools/extract_videos_from_hf.py",
            "--hf-dir",  "/vol/data/hf_ttcc",
            "--out-dir", str(videos_dir),
            "--split",   "train",
        ], check=True)
        vol.commit()                                        # persist before training

    # --- Stage 2b: rewrite video paths in the V8 jsonl to Volume paths. ---
    # The jsonl as uploaded references /home/ssm-user/... AWS paths.
    # On Modal those don't exist — rewrite to /vol/data/videos/<ad_id>.mp4.
    train_repathed = train_jsonl.with_suffix(".modal.jsonl")
    if not train_repathed.exists():
        import json
        with open(train_jsonl) as fin, open(train_repathed, "w") as fout:
            for line in fin:
                r = json.loads(line)
                ad_id = r["ad_id"]
                mp4 = f"/vol/data/videos/{ad_id}.mp4"
                r["videos"] = [mp4]
                r["audios"] = [mp4]
                fout.write(json.dumps(r) + "\n")
        vol.commit()
    train_jsonl = train_repathed

    # --- Stage 3: launch training. Auto-resume from latest ckpt. ---
    out_dir = Path("/vol/output/sft_retention_hazard_full_with_cot")
    latest_ckpt = None
    if out_dir.exists():
        ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            latest_ckpt = str(ckpts[-1])

    cmd = [
        "swift", "sft",
        "--config_file", "/opt/go_viral/examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_with_cot.yaml",
        "--model", str(base),
        "--dataset", str(train_jsonl),
        "--val_dataset", str(val_jsonl),
        "--output_dir", str(out_dir),
    ]
    if latest_ckpt:
        cmd += ["--resume_from_checkpoint", latest_ckpt]
        print(f"RESUMING from {latest_ckpt}")
    else:
        print("Fresh start from base Qwen2.5-Omni-3B")

    # --- Stage 4: start the HF-upload watcher in parallel. ---
    # Each new checkpoint-<step>/ dir gets pushed to HF as it's written.
    # The watcher is idempotent (marks uploaded with a sentinel file) so
    # surviving multiple container retries is safe.
    upload_proc = subprocess.Popen([
        "python",
        "/opt/go_viral/examples/train/grpo/qwen2_5_omni_ttcc/watch_and_upload_ckpts.py",
        "--output-dir", str(out_dir),
        "--hf-repo",    "liangyuch/ttcc-sft-qwen25omni-3b-v8-cot-modal",  # change as desired
        "--poll-sec",   "60",
    ], stdout=open("/vol/output/upload.log", "a"), stderr=subprocess.STDOUT)

    env = {**os.environ, "NPROC_PER_NODE": "8"}
    try:
        subprocess.run(cmd, env=env, check=True, cwd="/opt/go_viral")
    finally:
        # Give the watcher 5 minutes to drain pending uploads, then kill.
        # The next retry's watcher will pick up anything left.
        import time
        time.sleep(300)
        upload_proc.terminate()
        try:
            upload_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            upload_proc.kill()
        vol.commit()

@app.local_entrypoint()
def main():
    train.remote()
```

Run with:
```bash
modal run train_v8.py
```

The retries policy means you can detach after launching — Modal will
auto-relaunch up to 10 times across the 24h boundary. Total wall-clock
ceiling: ~240h, more than enough headroom for a 50h run.

## Pre-launch sanity — run on a cheap L4 before reserving 8×H100

Don't pay $32/hr to find out your data layout is broken. Add this function
to `train_v8.py` and run it once before launching:

```python
@app.function(
    image=image,
    gpu="L4",                                       # $0.80/hr
    timeout=1800,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token")],
)
def preflight():
    os.environ["HF_HOME"] = "/vol/hf-cache"
    # Ensure the validate script + data + model are present.
    subprocess.run([
        "bash",
        "/opt/go_viral/examples/custom/qwen2_5_omni_retention/tools/validate_v8_launch.sh",
        "/vol/hf-cache/Qwen2.5-Omni-3B",
        "/vol/data/ttcc_v8/ttcc_train_with_cot.jsonl",
        "/vol/data/ttcc_v8/val_200_no_cot.jsonl",
    ], check=True)
    # Also spot-check the leak-free invariants on 50 random rows.
    subprocess.run([
        "python", "-c",
        # (same 50-row leak invariants check as the AWS runbook step 5)
        "import json,random; rows=open('/vol/data/ttcc_v8/ttcc_train_with_cot.jsonl').read().splitlines();"
        "random.seed(0); sample=random.sample(rows, min(50,len(rows)));"
        "n=0\nfor i,line in enumerate(sample):"
        "  d=json.loads(line); a=d['messages'][-1]['content']; R=d['R'];"
        "  assert a.strip().startswith('<cot>') and a.strip().endswith('</cot>'), f'row {i}: assistant span malformed';"
        "  [(_ for _ in ()).throw(AssertionError(f'row {i}: R={r:.4f} leaked')) for r in R[1:] for dp in (2,3,4) if f'{r:.{dp}f}' in a];"
        "  assert d['videos'][0]==d['audios'][0]; assert abs(R[0]-1.0)<1e-6 and len(R)==d['T']+1;"
        "  assert all(R[k+1]<=R[k]+1e-6 for k in range(len(R)-1));"
        "  n+=1\nprint(f'{n} rows OK')",
    ], check=True)
    print("preflight: PASS")
```

Run:
```bash
modal run train_v8.py::preflight
```

Exits 0 = safe to launch. Exits non-zero = STOP and debug.

## First-checkpoint sanity (V7 post-mortem check)

After step 75 saves, run the randomization probe BEFORE letting the run go
for 50h. This is the V7 incident's primary learning: V7 trained to ~0
loss while completely ignoring video. The probe perturbs the video tensor
and checks that predictions change materially.

```python
@app.function(
    image=image,
    gpu="H100:1",                                    # 1 GPU is enough for forward-only
    timeout=3600,
    volumes={"/vol": vol},
)
def probe(checkpoint_step: int = 75, n_ads: int = 20):
    import glob
    pattern = f"/vol/output/sft_retention_hazard_full_with_cot/v*-*/checkpoint-{checkpoint_step}"
    matches = sorted(glob.glob(pattern))
    assert matches, f"no checkpoint at step {checkpoint_step}"
    ckpt = matches[-1]
    subprocess.run([
        "python",
        "/opt/go_viral/examples/custom/qwen2_5_omni_retention/tools/randomization_probe.py",
        "--ckpt", ckpt,
        "--val-jsonl", "/vol/data/ttcc_v8/val_200_no_cot.jsonl",
        "--n-ads", str(n_ads),
    ], check=True)
```

Run from your laptop once step 75 lands (watch wandb for the save event):
```bash
modal run train_v8.py::probe
# Exits 0 = PASS (model uses video). Exits 1 = FAIL → kill training, escalate.
```

If FAIL: the V8 architecture has the same blind-to-video problem as V7.
Don't burn 45 more hours. Kill with `modal app stop ttcc-v8` and escalate.

## During training — what to watch

Both wandb (live) and tensorboard (from Volume) work.

| Step range | Expected | Abort if |
|---|---|---|
| 0 | head random; `loss_curve` ~0.1, `loss_cot` ~3.0 per token | NaN/inf in either |
| 50 | both losses dropping; first val IBS reported | losses stuck or rising |
| 100–200 | `loss_curve` < 1e-2; val IBS comparable to train | val IBS > 0.1 |
| 500+ | `loss_curve` plateaus around 5e-4 to 1e-3 | divergence |
| ~4400 | end of 10 epochs; expected IBS < 0.005 (leak-free) | — |

The plugin logs `loss_curve` and `loss_cot` separately (commit
`d3399991`) — verify both show up in wandb.

## When a retry fires

Modal will email you when a container is preempted/timed-out. The next
container starts with `MODAL_TASK_ID` different from the prior one but the
same Volume → ms-swift sees the previous `checkpoint-*` dir and resumes.

Verify after the first retry: in wandb, the step counter should be
continuous (not reset to 0), and `loss_curve` should pick up near where it
left off.

## What to do with the trained model

1. Upload to HuggingFace:
   `liangyuch/ttcc-sft-qwen25omni-3b-v8-cot` (or whatever Leon names the V8 repo).
2. **Tokenizer overlay** — same V7 gotcha. After saving, copy these 6 files
   from base `Qwen2.5-Omni-3B/` into your ckpt dir before pushing:
   `added_tokens.json`, `merges.txt`, `special_tokens_map.json`, `vocab.json`,
   `chat_template.json`, `tokenizer_config.json`. Otherwise loading the ckpt
   later will fail with `Qwen2TokenizerFast has no attribute image_token`.
   ms-swift's `swift/trainers/mixin.py` now has a vendor patch for this, but
   verify by listing the ckpt dir.
3. Skip uploading `global_step*` DeepSpeed shards — they're only needed for
   resume, not inference, and they double the upload size.

## Data sourcing — important caveat

There are **two** scripts in the repo and only one is leak-free for V8:

| Script | Status | What it does |
|---|---|---|
| `scripts/data/build_ttcc_jsonl.py` | ⚠ **V7-LEAKY** as committed | Builds an SFT JSONL where assistant span = `<cot>(reasoning here)</cot>\n{"R": [...]}` — embeds R(t) in the answer, which is exactly the leak V8 fixes |
| `examples/custom/qwen2_5_omni_retention/tools/build_v8_train_jsonl.py` | ✅ Leak-free | Merges V7 jsonl (R + video paths) with CoT jsonl (Gemini distillation), produces assistant span = `<cot>...</cot>` only. **But this script lived only on the dead 8-GPU box's NVMe — it's NOT in the repo yet.** |

So: **Wanjia cannot rebuild V8 jsonl from scratch on Modal.** The leak-free
builder isn't in the repo, and `build_ttcc_jsonl.py` in its current form
would reintroduce the V7 leak.

**The only safe path is: Leon uploads the pre-built `ttcc_train_with_cot.jsonl`
+ `val_200_no_cot.jsonl` to HF as a private dataset.** Wanjia then pulls
them directly:

```python
# Inside the Modal function:
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="liangyuch/ttcc-v8-train",          # <-- Leon will tell you exact name
    repo_type="dataset",
    local_dir="/vol/data/ttcc_v8",
    allow_patterns=["*.jsonl"],
)
```

Then re-path the video field from each jsonl row (the AWS box paths
`/home/ssm-user/...` won't resolve on Modal):

```python
import json
from pathlib import Path
jsonl = Path("/vol/data/ttcc_v8/ttcc_train_with_cot.jsonl")
rows = [json.loads(l) for l in jsonl.read_text().splitlines()]
for r in rows:
    ad_id = r["ad_id"]
    r["videos"] = [f"/vol/data/videos/{ad_id}.mp4"]
    r["audios"] = [f"/vol/data/videos/{ad_id}.mp4"]
jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
```

Or pre-rewrite the jsonl on Leon's side before upload — even better. Either way, video paths must be Modal-resolvable.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `Qwen2TokenizerFast has no attribute image_token` on load | Tokenizer overlay missing in ckpt dir | Copy the 6 files from base model into ckpt dir |
| Retry started but step counter reset to 0 | `--resume_from_checkpoint` flag not passed | Check the `latest_ckpt` detection in the Modal function |
| Modal complains about GPU shape | Account doesn't have 8×H100 quota | Request quota via Modal support, or fall back to 8×A100-80GB ($2.50×8=$20/hr, slightly slower) |
| Train loss diverges around step 50 | α too high relative to curve loss | Lower α to 5e-4 (probably won't be needed at α=1e-3) |
| Val IBS stuck near 0.1 while train loss drops | Model not using video (same failure mode as V7) | Run randomization probe: zero out video pixels at inference; if IBS doesn't change, V8 also failed |
| Volume commit slow / fails | Modal Volume size approaching limit | Check `modal volume get ttcc-v8` size; default cap is 1 TB, request increase if needed |

## Coordination with Leon's AWS run

Both runs share:
- Same training data (`ttcc_train_with_cot.jsonl`)
- Same base init (Qwen2.5-Omni-3B)
- Same α (1e-3)
- Same yaml (`sft_retention_hazard_full_with_cot.yaml`)
- Same architecture (full FT, ZeRO-3, hazard head at h[`</cot>`])

This is a **clean reproducibility A/B**: AWS vs Modal, same everything else.
If IBS matches within ~1×10⁻⁴ on the leak-free test, we have a robust V8
result. If they diverge, that's interesting and needs investigation.

Log into the same wandb project (`ttcc-v8`) with run names `v8_main_aws_*`
and `v8_main_modal_*` for easy side-by-side.

## When to escalate to Leon

- Modal 8×H100 quota denied → falls back to 8×A100-80GB, ~30% slower but
  same correctness — no escalation needed unless A100 also denied
- Tokenizer overlay broken at save time and the vendor patch isn't picking
  it up
- CoT data validation fails on >1% of rows → means Gemini distillation has
  a problem, not a Modal problem
- Eval IBS still >0.05 after 1000 steps with leak-free protocol → means V8
  architecture is also failing, not just V7
- Wall-clock ETA going far over 60h → check if checkpoint-resume is
  re-doing too many steps each cycle

---

## Quick reference — what Leon owes Wanjia before 4am

1. Upload V8 train jsonl + holdout val to HF (private dataset, e.g.
   `liangyuch/ttcc-v8-train`)
2. Confirm `ttcc-rl` branch on `cliangyu/go_viral` has the latest
   `register.py` (with `</cot>` anchor + per-component loss logging)
3. Send wandb project name (suggest `ttcc-v8`) so both runs land together
4. Confirm Wanjia's HF token has write access to the destination repo

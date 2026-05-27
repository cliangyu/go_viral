# V8 Launch Runbook (H100, two-node parallel)

Designed for a fresh teammate to execute end-to-end without asking questions.
Target wall-clock from SSM-in to first training step: **~45 min** (most of which is the model + video downloads).

## What to launch — two experiments in parallel

If two p5.48xlarge nodes are available, run both in parallel. They share the same yaml; only `RETENTION_COT_ALPHA` differs.

| Node | Experiment | yaml | What it measures |
|---|---|---|---|
| **A** | V8 main: SFT-Hazard+CoT | `configs/sft_retention_hazard_full_with_cot.yaml` | The headline run. Model learns CoT generation + retention prediction jointly. |
| **B** | V8 α=0 ablation: SFT-Hazard, no LM loss | same yaml, override `RETENTION_COT_ALPHA: '0.0'` at CLI | Control. Same architecture, same data, but the LM head gets no supervision. Tests "does CoT supervision help, holding architecture fixed?" |

**If only one node is available, do A first.** The α=0 ablation can run later on a follow-up reservation.

## Why this experimental design

- Both runs use the same leak-free V8 training data (assistant span = `<cot>...</cot>` only)
- Both anchor the retention head at `h[</cot>]` (the proposal's leak-safe design)
- The only difference is whether the LM head is trained on the CoT span
- Combined with `--cot-bypass` at inference time, this gives a clean 2×2 ablation:
  `(CoT supervision: on/off) × (CoT generation at inference: on/off)`

## Pre-launch credentials checklist (Leon's responsibility)

The teammate needs all of these accessible from the H100 box before they start:

| Credential | Where stored | Form |
|---|---|---|
| AWS access | IAM role attached to instance OR ssm-user has aws cli config | `~/.aws/credentials` profile or instance-profile |
| HF token | `~/.cache/huggingface/token` (chmod 600) | Single line, hf token |
| GitHub PAT | `~/.git-credentials` | `https://cliangyu:<token>@github.com` |
| wandb API key | `~/.netrc` machine `api.wandb.ai` | Standard netrc format |
| Vertex AI SA JSON (optional, only if val/test CoT will be generated later) | `~/.gcp_sa.json` (chmod 600) | JSON file |

If the existing `bootstrap.sh` in `~/work/scripts/` is reachable, **running it once installs HF + git tokens automatically**. Otherwise paste them by hand.

## Step-by-step (each step is copy-paste-able)

### Step 0 — Confirm the box is reachable

```bash
# On your laptop (not the H100 box):
TARGET=<i-xxxxxxxxxxxx>     # capacity-block instance id
REGION=<us-east-2 OR whatever the block was reserved in>
aws --profile gpu-box --region $REGION ssm describe-instance-information \
    --query "InstanceInformationList[?InstanceId=='$TARGET'].PingStatus" \
    --output text
# Should print "Online"

# Open an interactive session:
aws --profile gpu-box --region $REGION ssm start-session --target $TARGET
```

### Step 1 — Take ownership of NVMe scratch

```bash
sudo chown -R ssm-user:ssm-user /opt/dlami/nvme
mkdir -p /opt/dlami/nvme/logs
df -h /opt/dlami/nvme    # should show ~3 TB free
nvidia-smi               # should show 8 H100s, ~80 GB each
```

### Step 2 — Run bootstrap.sh (installs HF + git tokens, clones helper repos)

```bash
ls ~/work/scripts/bootstrap.sh && bash ~/work/scripts/bootstrap.sh
# If bootstrap.sh isn't there, paste tokens by hand (see Pre-launch checklist).
```

### Step 3 — Clone go_viral on NVMe + install ms-swift

Background the install (it takes ~5 min) so the SSM session doesn't time out:

```bash
cd /opt/dlami/nvme
git clone -b ttcc-rl --depth 1 https://github.com/cliangyu/go_viral.git
cd go_viral
nohup setsid /home/ssm-user/work/venv/bin/pip install -e . \
    </dev/null >/opt/dlami/nvme/logs/swift_install.log 2>&1 &
echo "swift install pid=$!"
# Also install deepspeed + flash-attn separately (they aren't in setup.py):
nohup setsid /home/ssm-user/work/venv/bin/pip install deepspeed flash-attn --quiet \
    </dev/null >/opt/dlami/nvme/logs/ds_install.log 2>&1 &
```

Verify after ~5 min:

```bash
tail -3 /opt/dlami/nvme/logs/swift_install.log
tail -3 /opt/dlami/nvme/logs/ds_install.log
/home/ssm-user/work/venv/bin/python -c "import swift, deepspeed, flash_attn; print('OK')"
```

### Step 4 — Download base model + tokenizer overlay

```bash
# Base model (~9 GB; takes 2-3 min)
nohup setsid /home/ssm-user/work/venv/bin/huggingface-cli download \
    Qwen/Qwen2.5-Omni-3B --local-dir /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    </dev/null >/opt/dlami/nvme/logs/dl_base.log 2>&1 &
```

The V8 yaml's `model:` field points at this path. No checkpoint download needed — V8 trains from base.

### Step 5 — Stage training data

The training data lives on the dead 8-GPU box's S3 backup OR on HF.

```bash
# CoT-merged training data (53 MB, instant from S3)
mkdir -p /home/ssm-user/work/data/ttcc_v8
aws s3 cp s3://vio-juicefs-us-east-1/ttcc_cot/cot_v6_train.jsonl \
    /home/ssm-user/work/data/ttcc_cot/cot_v6_train.jsonl --region us-east-1

# V7 train JSONL (76 MB)
aws s3 cp s3://vio-juicefs-us-east-1/ttcc_v7_data/ttcc_train_sft.jsonl \
    /home/ssm-user/work/data/ttcc_v7/ttcc_train_sft.jsonl --region us-east-1

# Re-build V8 merged jsonl (39,375 rows)
python3 /opt/dlami/nvme/go_viral/examples/custom/qwen2_5_omni_retention/tools/build_v8_train_jsonl.py \
    --v7-jsonl /home/ssm-user/work/data/ttcc_v7/ttcc_train_sft.jsonl \
    --cot-jsonl /home/ssm-user/work/data/ttcc_cot/cot_v6_train.jsonl \
    --out-jsonl /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl
wc -l /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl   # ~39375

# Holdout val for in-loop eval
aws s3 cp s3://vio-juicefs-us-east-1/ttcc_v7_data/val_200_no_cot.jsonl \
    /home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl --region us-east-1
```

### Step 6 — Stage video files (BIGGEST single time cost)

The 39K training videos are ~935 GB. Two options:

**Option A (preferred): from the HF dataset `liangyuch/ttcc-v0_2_0`**

```bash
mkdir -p /home/ssm-user/work/data/videos/train
nohup setsid /home/ssm-user/work/venv/bin/huggingface-cli download \
    liangyuch/ttcc-v0_2_0 \
    --repo-type dataset \
    --local-dir /home/ssm-user/work/data/hf_ttcc \
    </dev/null >/opt/dlami/nvme/logs/dl_videos.log 2>&1 &
# Then a tiny extraction script symlinks the videos under /data/videos/train/
# (look in /home/ssm-user/work/scripts/ for build_full_valtest_hf.py)
```

ETA: 4-12 hours depending on H100 box's network.

**Option B (fallback): from S3** — only works if the videos are mirrored there. Check `aws s3 ls s3://vio-juicefs-us-east-1/ttcc_videos/` first.

While this is running, proceed to step 7 (the validation can wait until videos are staged).

### Step 7 — Pre-launch sanity (once videos are staged)

```bash
bash /opt/dlami/nvme/go_viral/examples/custom/qwen2_5_omni_retention/tools/validate_v8_launch.sh \
    /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl \
    /home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl
# Exits 0 = safe to launch. Exits 1 = do NOT launch; debug first.
```

### Step 8 — Launch V8 (the actual training)

The launcher script `sft.sh` lives at:
`/opt/dlami/nvme/go_viral/examples/train/grpo/qwen2_5_omni_ttcc/sft.sh`

```bash
# Node A — V8 main (with CoT supervision, α=1e-3, the headline run)
cd /opt/dlami/nvme/go_viral
WANDB_NAME="v8_main_$(date +%Y%m%d_%H%M)" \
nohup setsid bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
    examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_with_cot.yaml \
    </dev/null >/opt/dlami/nvme/logs/v8_main.log 2>&1 &
echo "v8 main launched: pid=$!"

# Node B — V8 α=0 ablation (CoT-suppression control)
# Same yaml but override α at CLI:
cd /opt/dlami/nvme/go_viral
WANDB_NAME="v8_alpha0_$(date +%Y%m%d_%H%M)" \
RETENTION_COT_ALPHA=0.0 \
nohup setsid bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
    examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_with_cot.yaml \
    --output_dir /opt/dlami/nvme/ssm-out/sft_retention_hazard_full_alpha0 \
    </dev/null >/opt/dlami/nvme/logs/v8_alpha0.log 2>&1 &
echo "v8 α=0 launched: pid=$!"
```

The yaml is set to use wandb. The runs appear at `https://wandb.ai/liangyuch/ttcc`.

### Step 9 — Monitor (first 200 steps)

```bash
# Health check
pgrep -af "swift sft" | head
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail -50 /opt/dlami/nvme/logs/v8_main.log | tr '\r' '\n' | tail -20

# Loss progression
grep -oE "'loss': [0-9.eE+-]+|'global_step/max_steps': '[0-9]+/[0-9]+'|'token_acc': [0-9.eE+-]+" \
    /opt/dlami/nvme/logs/v8_main.log | tail -30
```

**Success signals at step 100-200:**
- `train/loss` decreasing
- `train/grad_norm` settling below ~10
- `train/token_acc` rising above 0.55
- GPU memory stable, no OOM
- First wandb-logged eval step (step 50) shows non-NaN eval/loss
- First ckpt save (step 75) lands in `output_dir/v*-*/checkpoint-75/`

**Abort signals at step 100-200:**
- Training loss NaN or rising
- Grad norm sustained > 200 past step 100
- OOM
- Eval loss > 0.5 (suggests model isn't even fitting train)

### Step 10 — Verify first ckpt is leak-free

After ckpt-75 (saves at step 75):

```bash
CKPT=$(ls -d /opt/dlami/nvme/ssm-out/sft_retention_hazard_full_with_cot/v*-*/checkpoint-75 | head -1)
python3 /opt/dlami/nvme/go_viral/examples/custom/qwen2_5_omni_retention/tools/validate_ckpt.py "$CKPT"

# Quick randomization probe (should NOT show constant output anymore):
# ... (probe script in /home/ssm-user/work/scripts/randomization_probe.py if staged)
```

**Critical**: if the randomization probe shows V8 also doesn't use video (like V7), **abort the run** and revisit architecture. Don't burn 45 hours on a model that won't beat constant prediction.

## Wandb namespace

Project: `liangyuch/ttcc`
Run names will be `v8_main_<timestamp>` and `v8_alpha0_<timestamp>`.

## Expected total compute

| Phase | Time |
|---|---|
| Bootstrap (steps 1-5) | ~10 min |
| Video staging (step 6) | 4-12 hr (largest single cost; runs in parallel with everything else after step 6) |
| Pre-launch sanity (step 7) | ~5 min |
| Training V8 main (10 epochs, 39K rows) | ~45-60 hr |
| Training V8 α=0 (3-5 epochs is enough for the ablation) | ~15-25 hr |
| Monitoring | continuous |

If both nodes share a capacity block, the slower (V8 main) determines the deadline.

## Failure modes & responses

| Symptom | Likely cause | Response |
|---|---|---|
| `Qwen2TokenizerFast has no attribute image_token` | Tokenizer overlay missing on model dir | Copy 6 files from base `Qwen2.5-Omni-3B`: `added_tokens.json`, `merges.txt`, `special_tokens_map.json`, `vocab.json`, `chat_template.json`, `tokenizer_config.json` |
| OOM at first step | max_length=49152 too aggressive | Lower to 32768 in yaml |
| `MaxLengthError` skip rate >10% during in-loop eval | val rows too long | Raise eval max_length to match train |
| Train loss exploding past step 50 | LR too high for base init | Lower LR to 2e-6 |
| Grad norm spikes 100x | Bad row in batch | `truncation_strategy: delete` should handle; check log for parse errors |

## Single command summary (for the teammate's notes app)

```bash
# After SSM-ing into the H100 box:
sudo chown -R ssm-user:ssm-user /opt/dlami/nvme && mkdir -p /opt/dlami/nvme/logs
bash ~/work/scripts/bootstrap.sh
cd /opt/dlami/nvme && git clone -b ttcc-rl --depth 1 https://github.com/cliangyu/go_viral.git
cd go_viral && nohup setsid /home/ssm-user/work/venv/bin/pip install -e . deepspeed flash-attn </dev/null >/opt/dlami/nvme/logs/swift_install.log 2>&1 &
huggingface-cli download Qwen/Qwen2.5-Omni-3B --local-dir /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B
# … (data + videos stage in parallel) …
bash examples/custom/qwen2_5_omni_retention/tools/validate_v8_launch.sh \
    /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl
nohup setsid bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
    examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_with_cot.yaml \
    </dev/null >/opt/dlami/nvme/logs/v8_main.log 2>&1 &
```

That's the whole thing. Total bootstrap + launch wall-clock once H100 box is up: ~30-45 min for the non-video staging.

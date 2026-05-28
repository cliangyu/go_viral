# V8 Launch Runbook (H100, two-node parallel)

Target audience: a teammate with the same AWS-S3-SSM setup Leon uses
(`~/work/`, `/opt/dlami/nvme/`, AWS profile `gpu-box` with permissions extended
to the new H100 instance, S3 bucket `vio-juicefs-us-east-1` reachable).

Target wall-clock from SSM-in to first training step: **~45 min** (most of
which is the model + video downloads).

## ⚠ FILL IN BEFORE EXECUTING

Leon fills these once the H100 capacity block is allocated. Until then this
runbook is not directly executable.

| Variable | Where to find | Value |
|---|---|---|
| `H100_INSTANCE_ID` | `aws ec2 describe-instances ...` for the new capacity block | `i-xxxxxxxxxxxxxxxxx` |
| `H100_REGION` | AWS console — the region the capacity block was reserved in | `us-east-1` / `us-east-2` / … |
| `AWS_PROFILE_NAME` | name of the local `~/.aws/credentials` profile that can SSM-StartSession to the new instance | usually `gpu-box`, extend its IAM policy if new instance ARN isn't whitelisted |
| `S3_BUCKET` | Confirm the S3 bucket name that mirrors V7 jsonl + CoT jsonl + val jsonl + (optionally) videos | likely `vio-juicefs-us-east-1`; verify with `aws s3 ls` before launch |

Throughout this doc, those variables appear verbatim — substitute before pasting.

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
# On your laptop (not the H100 box). Fill in from the "FILL IN BEFORE EXECUTING" table above.
TARGET=H100_INSTANCE_ID
REGION=H100_REGION
PROFILE=AWS_PROFILE_NAME

aws --profile $PROFILE --region $REGION ssm describe-instance-information \
    --query "InstanceInformationList[?InstanceId=='$TARGET'].PingStatus" \
    --output text
# Should print "Online"

# Open an interactive session:
aws --profile $PROFILE --region $REGION ssm start-session --target $TARGET
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

The launcher (`sft.sh` → `_common.sh`) expects the venv at
`/opt/dlami/nvme/work/swift_venv`. If your existing venv is somewhere else
(e.g. `/home/ssm-user/work/venv`), set `VENV` in the environment before
invoking the launcher in step 8.

Background the install (it takes ~5 min) so the SSM session doesn't time
out. flash-attn needs `--no-build-isolation` so it picks up the installed
torch ABI:

```bash
cd /opt/dlami/nvme
git clone -b ttcc-rl --depth 1 https://github.com/cliangyu/go_viral.git
cd go_viral

# ms-swift + framework deps (transformers, accelerate, peft, datasets, …).
VENV=/opt/dlami/nvme/work/swift_venv             # the path _common.sh expects
nohup setsid $VENV/bin/pip install -e . \
    </dev/null >/opt/dlami/nvme/logs/swift_install.log 2>&1 &
echo "swift install pid=$!"

# DeepSpeed + flash-attn (NOT in ms-swift's setup.py). Order matters:
# install torch-aware flash-attn AFTER torch (which the swift install
# brings in via transformers) using --no-build-isolation.
wait $!                                            # let swift install finish first
nohup setsid bash -c "
  $VENV/bin/pip install deepspeed==0.15.2 --quiet &&
  $VENV/bin/pip install flash-attn==2.8.3 --no-build-isolation --quiet
" </dev/null >/opt/dlami/nvme/logs/ds_install.log 2>&1 &

# Also need yq for sft.sh's ENV: parsing.
sudo apt-get update -qq && sudo apt-get install -y -qq yq || \
    sudo wget -q https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 \
        -O /usr/local/bin/yq && sudo chmod +x /usr/local/bin/yq
```

Verify after ~5 min:

```bash
tail -3 /opt/dlami/nvme/logs/swift_install.log
tail -3 /opt/dlami/nvme/logs/ds_install.log
$VENV/bin/python -c "import swift, deepspeed, flash_attn; print('OK')"
command -v yq && yq --version
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

The training data on the dead 8-GPU box's NVMe is gone (NVMe is ephemeral on
capacity blocks). Use the S3 mirror if it exists; otherwise download from HF
and re-build.

**First: verify what S3 has.**

```bash
aws --region $REGION s3 ls s3://$S3_BUCKET/ttcc_v7_data/ttcc_train_sft.jsonl 2>&1
aws --region $REGION s3 ls s3://$S3_BUCKET/ttcc_cot/cot_v6_train.jsonl 2>&1
aws --region $REGION s3 ls s3://$S3_BUCKET/ttcc_v7_data/val_200_no_cot.jsonl 2>&1
aws --region $REGION s3 ls s3://$S3_BUCKET/ttcc_v8/ttcc_train_with_cot.jsonl 2>&1  # if pre-built
```

**Path A — S3 has the pre-built V8 jsonl (fast):**

```bash
mkdir -p /home/ssm-user/work/data/ttcc_v8 /home/ssm-user/work/data/ttcc_holdout
aws --region $REGION s3 cp \
    s3://$S3_BUCKET/ttcc_v8/ttcc_train_with_cot.jsonl \
    /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl
aws --region $REGION s3 cp \
    s3://$S3_BUCKET/ttcc_v7_data/val_200_no_cot.jsonl \
    /home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl
wc -l /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl   # expect ~39375
```

**Path B — S3 has V7 + CoT separately, rebuild V8 here:**

```bash
mkdir -p /home/ssm-user/work/data/{ttcc_v7,ttcc_cot,ttcc_v8,ttcc_holdout}
aws --region $REGION s3 cp \
    s3://$S3_BUCKET/ttcc_v7_data/ttcc_train_sft.jsonl \
    /home/ssm-user/work/data/ttcc_v7/ttcc_train_sft.jsonl
aws --region $REGION s3 cp \
    s3://$S3_BUCKET/ttcc_cot/cot_v6_train.jsonl \
    /home/ssm-user/work/data/ttcc_cot/cot_v6_train.jsonl
aws --region $REGION s3 cp \
    s3://$S3_BUCKET/ttcc_v7_data/val_200_no_cot.jsonl \
    /home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl

# Build leak-free V8 jsonl by replacing the V7 assistant span (which has R
# values) with the Gemini CoT (which doesn't).
$VENV/bin/python /opt/dlami/nvme/go_viral/examples/custom/qwen2_5_omni_retention/tools/build_v8_train_jsonl.py \
    --v7-jsonl  /home/ssm-user/work/data/ttcc_v7/ttcc_train_sft.jsonl \
    --cot-jsonl /home/ssm-user/work/data/ttcc_cot/cot_v6_train.jsonl \
    --out-jsonl /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl
wc -l /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl   # expect ~39375
```

**Spot-check the leak-free invariants** before launch (5 invariants, 50 rows,
~3 seconds):

```bash
$VENV/bin/python -c "
import json
n_ok = 0
with open('/home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl') as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        a = d['messages'][-1]['content']
        R = d['R']
        assert a.strip().startswith('<cot>') and a.strip().endswith('</cot>'), f'row {i}: assistant span malformed'
        for r in R[1:]:
            for dp in (2,3,4):
                assert f'{r:.{dp}f}' not in a, f'row {i}: R={r:.{dp}f} leaked into CoT'
        assert d['videos'][0] == d['audios'][0], f'row {i}: video/audio mismatch'
        assert abs(R[0] - 1.0) < 1e-6 and len(R) == d['T'] + 1, f'row {i}: R/T mismatch'
        assert all(R[k+1] <= R[k] + 1e-6 for k in range(len(R)-1)), f'row {i}: R not monotone'
        n_ok += 1
        if n_ok >= 50: break
print(f'{n_ok} rows validated, leak-free')
"
```

### Step 6 — Stage video files (BIGGEST single time cost)

The 39K training videos are ~935 GB. The HF dataset `liangyuch/ttcc-v0_2_0`
embeds them as `video_bytes` inside parquet shards; the extractor script
writes one `<ad_id>.mp4` per row.

```bash
mkdir -p /home/ssm-user/work/data/videos /home/ssm-user/work/data/hf_ttcc

# Pull parquet shards from HF (~935 GB, ETA 4-12 hours).
nohup setsid $VENV/bin/huggingface-cli download \
    liangyuch/ttcc-v0_2_0 \
    --repo-type dataset \
    --local-dir /home/ssm-user/work/data/hf_ttcc \
    </dev/null >/opt/dlami/nvme/logs/dl_videos.log 2>&1 &
echo "HF download pid=$!"
```

Once the parquet shards land, extract MP4s:

```bash
$VENV/bin/python /opt/dlami/nvme/go_viral/examples/custom/qwen2_5_omni_retention/tools/extract_videos_from_hf.py \
    --hf-dir  /home/ssm-user/work/data/hf_ttcc \
    --out-dir /home/ssm-user/work/data/videos \
    --split   train

# Verify ~39K MP4s on disk
ls /home/ssm-user/work/data/videos | wc -l
```

The `videos:` field in `ttcc_train_with_cot.jsonl` already references
`/home/ssm-user/work/data/videos/<ad_id>.mp4` paths (that's how it was built
on the original 8-GPU box) — no jsonl rewrite needed.

**S3 fallback**: if the new H100 box's network out to HF is slow, check
whether the videos were mirrored to S3:

```bash
aws --region $REGION s3 ls s3://$S3_BUCKET/ttcc_videos/ | head
# If present:
nohup setsid aws --region $REGION s3 sync \
    s3://$S3_BUCKET/ttcc_videos/ /home/ssm-user/work/data/videos/ \
    </dev/null >/opt/dlami/nvme/logs/s3_videos.log 2>&1 &
```

While videos are downloading, proceed to step 7's static validation
(model + data shape checks) — the full validation that touches videos waits
until step 6 completes.

### Step 7 — Pre-launch sanity (once videos are staged)

```bash
bash /opt/dlami/nvme/go_viral/examples/custom/qwen2_5_omni_retention/tools/validate_v8_launch.sh \
    /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl \
    /home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl
# Exits 0 = safe to launch. Exits 1 = do NOT launch; debug first.
```

### Step 7.5 — Post-incident defenses (added 2026-05-28)

Before launching, deploy the defenses that came out of the
[2026-05-28 incidents](INCIDENT_2026-05-28_TRAINING_CRASH_AND_MONITOR_GAP.md):
marker file, on-box health monitor, wandb env auto-source, and (on
shared boxes) `gpu-clean.sh` lockdown.

These are bundled in `ops/post_incident_setup.sh`. Idempotent — safe to
re-run.

```bash
cd /opt/dlami/nvme/go_viral
bash examples/train/grpo/qwen2_5_omni_ttcc/ops/post_incident_setup.sh \
    --owner $USER \
    --runid v8-$(date +%Y%m%d-%H%M%S) \
    --wandb-url https://wandb.ai/liangyuch/ttcc \
    --contact your_email@example.com \
    --lock-gpu-clean    # only on shared boxes where cleanup automation runs
```

The script:
1. Writes `WANDB_API_KEY` to `/opt/dlami/nvme/wandb_env.sh` from `~/.netrc`
   so swift can pick it up automatically (closes the wandb auth-failure
   crash from 2026-05-28 relaunch attempt #2).
2. Writes `/opt/dlami/nvme/RUNNING_<owner>_<runid>.lock` so any cleanup
   automation (like the `zane-ai-agent` incident at 07:23 UTC) can detect
   an active run and bail.
3. (With `--lock-gpu-clean`) Replaces `gpu-clean.sh` with a refusal banner
   and `chattr +i`'s it. Defense-in-depth alongside an IAM Deny on the
   actor running cleanup.
4. Starts `/opt/dlami/nvme/health_writer.sh` under `setsid nohup` so it
   survives SSH/SSM disconnects. Writes
   `/opt/dlami/nvme/health/v8_status.json` every 30s.

**Coordination protocol** (on shared boxes):
- IAM Deny is the primary defense. Add an inline policy to any cleanup
  agent that targets `Resource: arn:aws:ec2:<region>:<acct>:instance/<i-...>`
  for the instances you're running on. See
  `INCIDENT_2026-05-28_TRAINING_CRASH_AND_MONITOR_GAP.md` Part 3 for the
  exact JSON.
- Marker file is the secondary signal — visible to any human or agent
  that lists `/opt/dlami/nvme/RUNNING_*.lock`.
- The `gpu-clean.sh` chattr +i lock is belt-and-suspenders that survives
  IAM policy removal.

### Step 8 — Launch V8 (the actual training)

The launcher `sft.sh` is at
`/opt/dlami/nvme/go_viral/examples/train/grpo/qwen2_5_omni_ttcc/sft.sh`. It
sources `_common.sh` which defines `VENV=/opt/dlami/nvme/work/swift_venv` —
override `VENV` in the environment if your venv is elsewhere.

```bash
# Node A — V8 main (with CoT supervision, α=1e-3, the headline run)
cd /opt/dlami/nvme/go_viral
VENV=/opt/dlami/nvme/work/swift_venv \
WANDB_NAME="v8_main_$(date +%Y%m%d_%H%M)" \
nohup setsid bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
    examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_with_cot.yaml \
    </dev/null >/opt/dlami/nvme/logs/v8_main.log 2>&1 &
echo "v8 main launched: pid=$!"

# Node B — V8 α=0 ablation (CoT-suppression control). Same yaml + α override.
cd /opt/dlami/nvme/go_viral
VENV=/opt/dlami/nvme/work/swift_venv \
WANDB_NAME="v8_alpha0_$(date +%Y%m%d_%H%M)" \
RETENTION_COT_ALPHA=0.0 \
nohup setsid bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
    examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_with_cot.yaml \
    --output_dir /opt/dlami/nvme/ssm-out/sft_retention_hazard_full_alpha0 \
    </dev/null >/opt/dlami/nvme/logs/v8_alpha0.log 2>&1 &
echo "v8 α=0 launched: pid=$!"
```

The yaml has `report_to: [tensorboard, wandb]`. Runs appear at
`https://wandb.ai/liangyuch/ttcc`.

### Step 9 — Monitor (first 200 steps)

The `ops/post_incident_setup.sh` step started `health_writer.sh` which
writes a status JSON every 30s. Prefer reading that over re-greppping
the log on each check:

```bash
# Live status (always fresh, ≤30s old)
cat /opt/dlami/nvme/health/v8_status.json

# Alert log (only writes on PROCESS_DIED / STEP_STALE_>600s / RuntimeError)
tail /opt/dlami/nvme/health/v8_alerts.log

# Optional: legacy ad-hoc inspection
pgrep -af "swift sft" | head
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
grep -oE "'loss': [0-9.eE+-]+|'global_step/max_steps': '[0-9]+/[0-9]+'|'token_acc': [0-9.eE+-]+" \
    /opt/dlami/nvme/logs/v8_distributed.log | tail -30
```

The register.py loss-split patch (commit after 2026-05-28) also wires
`loss_curve` and `loss_cot` into swift's `custom_metrics` pipeline. They
appear in wandb as separate scalars; check there for "is retention head
actually learning vs is LM head drifting" — see register.py docstring
just below the `_is_packed_sequence` patch for the rationale.

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

### Step 10 — Verify first ckpt uses video (V7 post-mortem check)

After ckpt-75 (the first save at step 75) and BEFORE letting the run go for
hours: confirm the model isn't replicating V7's "ignores video" failure
mode.

```bash
CKPT=$(ls -d /opt/dlami/nvme/ssm-out/sft_retention_hazard_full_with_cot/v*-*/checkpoint-75 | head -1)

# A) Smoke: ckpt loads + 1 forward returns non-NaN r_pred
$VENV/bin/python /opt/dlami/nvme/go_viral/examples/custom/qwen2_5_omni_retention/tools/validate_ckpt.py "$CKPT"

# B) Randomization probe: zero out / swap the video, predictions MUST change.
$VENV/bin/python /opt/dlami/nvme/go_viral/examples/custom/qwen2_5_omni_retention/tools/randomization_probe.py \
    --ckpt "$CKPT" \
    --val-jsonl /home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl \
    --n-ads 20
# Exits 0 = PASS (zero/between and swap/between ratios > 0.3 each).
# Exits 1 = FAIL — V8 has the same blind-to-video failure as V7. STOP.
```

**Critical**: if the randomization probe FAILs at ckpt-75, **kill the run**
(`pkill -f "swift sft"`) and escalate to Leon. Don't burn 45 hours on a
model that won't beat constant prediction.

### Step 11 — Post-training: upload checkpoints to HF

Run the watcher in the background alongside training so each new
`checkpoint-<step>` directory uploads as it's written (NVMe is ephemeral —
when the capacity block ends, anything not on HF is gone).

```bash
# Node A (V8 main, α=1e-3):
nohup setsid $VENV/bin/python \
    /opt/dlami/nvme/go_viral/examples/train/grpo/qwen2_5_omni_ttcc/watch_and_upload_ckpts.py \
    --output-dir /opt/dlami/nvme/ssm-out/sft_retention_hazard_full_with_cot \
    --hf-repo liangyuch/ttcc-sft-qwen25omni-3b-v8-cot \
    --poll-sec 60 \
    </dev/null >/opt/dlami/nvme/logs/upload_main.log 2>&1 &

# Node B (V8 α=0 ablation):
nohup setsid $VENV/bin/python \
    /opt/dlami/nvme/go_viral/examples/train/grpo/qwen2_5_omni_ttcc/watch_and_upload_ckpts.py \
    --output-dir /opt/dlami/nvme/ssm-out/sft_retention_hazard_full_alpha0 \
    --hf-repo liangyuch/ttcc-sft-qwen25omni-3b-v8-alpha0 \
    --poll-sec 60 \
    </dev/null >/opt/dlami/nvme/logs/upload_alpha0.log 2>&1 &
```

**Tokenizer overlay** is handled at save time by the vendor patch in
`swift/trainers/mixin.py`: each `checkpoint-<step>/` dir gets the 6
multimodal tokenizer files (`added_tokens.json`, `merges.txt`,
`special_tokens_map.json`, `vocab.json`, `chat_template.json`,
`tokenizer_config.json`) copied from the base model. Verify on the first
ckpt:

```bash
CKPT=$(ls -d /opt/dlami/nvme/ssm-out/sft_retention_hazard_full_with_cot/v*-*/checkpoint-75 | head -1)
for F in added_tokens.json merges.txt special_tokens_map.json vocab.json chat_template.json tokenizer_config.json; do
    [[ -f "$CKPT/$F" ]] && echo "ok: $F" || echo "MISSING: $F"
done
```

If any file is missing, the patch didn't apply — manually copy from
`$MODEL` (the base Qwen2.5-Omni-3B dir) into the ckpt before pushing, or
inference will fail with `Qwen2TokenizerFast has no attribute image_token`.

**Don't upload the `global_step*/` directories** — they're DeepSpeed shards
needed for resume only, not inference, and they double upload size. The
watcher's `is_complete_ckpt` check skips them by default (only uploads dirs
matching `checkpoint-<step>/` with `trainer_state.json` + a model shard).

### Step 12 — Recovery if the SSM session drops

Training runs were launched with `nohup setsid` so they survive your
session disconnect. To reconnect:

```bash
# On your laptop:
aws --profile $PROFILE --region $REGION ssm start-session --target $TARGET

# On the box:
pgrep -af "swift sft"                 # confirm still running
tail -f /opt/dlami/nvme/logs/v8_main.log
```

If the box itself was preempted (capacity block ended), the ckpts on
`/opt/dlami/nvme/` are GONE (ephemeral NVMe). The HF repo (step 11) is the
only durable artifact. Resume from the last HF-uploaded checkpoint via
`--resume_from_checkpoint <local-path-after-redownload>`.

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

## Single-command summary (for the teammate's notes app)

```bash
# Substitute H100_INSTANCE_ID / H100_REGION / AWS_PROFILE_NAME / S3_BUCKET
# (see top of doc).

# 0. Connect.
aws --profile $PROFILE --region $REGION ssm start-session --target $TARGET

# 1. Inside the box: take ownership + bootstrap.
sudo chown -R ssm-user:ssm-user /opt/dlami/nvme && mkdir -p /opt/dlami/nvme/logs
bash ~/work/scripts/bootstrap.sh        # installs HF + git tokens

# 2. Clone + install deps. Order: ms-swift first, then flash-attn.
cd /opt/dlami/nvme
git clone -b ttcc-rl --depth 1 https://github.com/cliangyu/go_viral.git
VENV=/opt/dlami/nvme/work/swift_venv
cd go_viral && $VENV/bin/pip install -e . && \
    $VENV/bin/pip install deepspeed==0.15.2 && \
    $VENV/bin/pip install flash-attn==2.8.3 --no-build-isolation

# 3. Base model (background) + data + videos (background).
nohup setsid $VENV/bin/huggingface-cli download Qwen/Qwen2.5-Omni-3B \
    --local-dir /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    >/opt/dlami/nvme/logs/dl_base.log 2>&1 &

# (Data: see Step 5 — try S3 pre-built V8 first, else build from V7+CoT.)
# (Videos: see Step 6 — HF parquet + extract_videos_from_hf.py, or S3 sync.)

# 4. Pre-launch sanity (after model + data + videos staged).
bash examples/custom/qwen2_5_omni_retention/tools/validate_v8_launch.sh \
    /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B \
    /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl \
    /home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl

# 5. Launch + start ckpt watcher in parallel.
VENV=/opt/dlami/nvme/work/swift_venv WANDB_NAME="v8_main_$(date +%Y%m%d_%H%M)" \
    nohup setsid bash examples/train/grpo/qwen2_5_omni_ttcc/sft.sh \
    examples/train/grpo/qwen2_5_omni_ttcc/configs/sft_retention_hazard_full_with_cot.yaml \
    >/opt/dlami/nvme/logs/v8_main.log 2>&1 &

nohup setsid $VENV/bin/python \
    examples/train/grpo/qwen2_5_omni_ttcc/watch_and_upload_ckpts.py \
    --output-dir /opt/dlami/nvme/ssm-out/sft_retention_hazard_full_with_cot \
    --hf-repo liangyuch/ttcc-sft-qwen25omni-3b-v8-cot \
    >/opt/dlami/nvme/logs/upload_main.log 2>&1 &

# 6. After step 75 saves: run randomization probe. PASS = continue, FAIL = kill + escalate.
CKPT=$(ls -d /opt/dlami/nvme/ssm-out/sft_retention_hazard_full_with_cot/v*-*/checkpoint-75 | head -1)
$VENV/bin/python examples/custom/qwen2_5_omni_retention/tools/randomization_probe.py \
    --ckpt "$CKPT" --val-jsonl /home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl --n-ads 20
```

That's the whole thing. Total bootstrap + launch wall-clock once H100 box is up: ~30-45 min for everything except video staging (4-12 hr in parallel).

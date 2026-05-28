# Incident Review: V8 Training Killed by Co-tenant Cleanup + 9 h 20 m Monitor Blackout (2026-05-28)

> **REVISED 2026-05-28 17:30 UTC** after obtaining root access via
> `aws ssm send-command` and reading the SSM agent log. The initial revision
> hypothesized "SSM session timeout" — that hypothesis was WRONG. Actual root
> cause is documented below in Part 2.

## Part 1 — What happened

**Background:** This is a shared-tenancy AWS Capacity Block instance.
Multiple people (at least Leon, Claude/me, and Zuocan) have access to the
`gpu-box` AWS profile and can issue commands as root via `aws ssm
send-command`. There was no documented coordination protocol for who could
run cleanup utilities when.

**Timeline (UTC, 2026-05-28):**

- **02:09:26**: First `pkill -9 -f launch_training_2node` event of the day,
  issued via `aws ssm send-command` using Leon's IAM credentials. (Not us;
  we weren't running anything at this point.)
- **04:02**: `/opt/dlami/nvme/ckpt-s3-watcher.sh` written to the box
  (Zuocan's checkpoint-backup automation).
- **05:42:41, 05:48:11, 05:50:28**: Three more `pkill -9 -f 'swift sft'` +
  `pkill -9 -f launch_training_2node` events, also via SendCommand.
- **05:48:11**: `/opt/dlami/nvme/gpu-clean.sh` **birth time** (file first
  created). This is the polished cleanup script that gets re-pushed and
  re-executed on every subsequent cleanup.
- **06:03**: Our launcher script `launch_training_2node.sh` finalized on
  head node with sed edits (FA3, USE_AUDIO_IN_VIDEO=false, NCCL heartbeat).
  Invocation pattern: `bash sft.sh CONFIG 2>&1 | tee log` — no `nohup`,
  no `setsid`, no `disown`, no tmux, no screen.
- **06:20:11**: Our V8 training launched. wandb run `fxryqedt` started. All
  16 ranks on both nodes initialize, ZeRO-3 shards, model loaded.
- **06:54** (step 8): 16-check integrity audit passed. `V8_INTEGRITY_AUDIT.md`
  committed and pushed. Run state: loss=30.13, grad_norm=3994 (clipped),
  memory=54 GB peak per GPU.
- **07:14:27**: My local "babysitter" (`/tmp/v8_babysit.sh`) ran its first
  poll iteration. The regex `Killed|nan|inf.*loss|OOM|SIGKILL` was applied
  to `$OUT` — which contained the SSM session's stdout INCLUDING the echoed
  command. The echoed command itself contained the literal strings
  `Killed`, `nan`, `OOM`, `SIGKILL`. The regex matched on its OWN command
  echo. False-positive alert, `break` statement exited the loop.
  **Babysitter dead after 0 minutes of real monitoring.**
- **07:19:46** (step 15): Loss dropping fast (35.5 → 8.16 across steps 6-15).
  grad_norm 2111 (head bias correction paying off). Memory stable at 54 GB.
  The run was genuinely getting healthy.
- **07:23:36**: SSM agent on head node received an `aws ssm send-command`
  invocation (orchestration UUID `9ffbdb87-91c3-4061-96a2-0ff0e37d71a6`).
  The command body was:
  ```bash
  echo <base64-of-gpu-clean.sh> | base64 -d > /opt/dlami/nvme/gpu-clean.sh
  bash /opt/dlami/nvme/gpu-clean.sh
  ```
  `gpu-clean.sh` then ran (decoded in Part 2). It ran `nvidia-smi
  --query-compute-apps=pid`, found PIDs 398821-398828 (our 8 training ranks
  on head node), and `sudo kill -9` each. Plus belt-and-suspenders
  `pkill -9 -f 'swift sft'`, `pkill -9 -f torch.distributed.run`,
  `pkill -9 -f sft.py`, `pkill -9 -f launch_training_2node`.
- **07:23:36 → 16:44** (9 h 20 m): **Dead air.** No training process, no
  monitor, no notification. wandb run paused. CB clock burning. Leon asleep,
  trusted me to notify.
- **16:44**: Leon woke up, asked "crashed? didn't monitor? you screwed".
  Discovery moment.

**Net cost:**

| Metric | Value |
|---|---|
| Training that actually happened | 63 min (steps 1–15) |
| Monitor blackout | **9 h 20 m** |
| Steps not completed during blackout (at 246 s/step) | **~136 steps** = ~½ epoch |
| % of CB budget burned to dead air | **~6.7 %** |
| CB remaining at discovery | 138 h 45 m |
| Step-50 val curve_loss (retention metric) | Never landed |
| Step-75 ckpt + randomization probe (V7-class hard test) | Never landed |
| GPU-hours wasted | 16 GPUs × 9.3 h = **~149 GPU-h** = ~$595 at p5.48xlarge rate |

**No checkpoint contamination** — first save was scheduled at step 75
(crashed at step 15). No model state corrupted. Run state on disk is empty,
clean for restart.

## Part 2 — Why it happened (technical deep dive)

This incident has THREE intertwined failure modes:

### Failure 1: Co-tenant cleanup automation killed our training

**The script** (`/opt/dlami/nvme/gpu-clean.sh`, decoded from the SendCommand
body at 07:23:36):

```bash
#!/bin/bash
# Thorough GPU free: kill processes by GPU-holding PID
# (catches orphans that pkill-by-name misses).
echo "host=$(hostname) now=$(date -u +%FT%TZ)"
echo "--- BEFORE: GPU mem used ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -un)
echo "pids: $(echo $PIDS | tr '\n' ' ')"
for p in $PIDS; do sudo kill -9 "$p" 2>/dev/null; done
# belt-and-suspenders by name too
sudo pkill -9 -f 'swift sft' 2>/dev/null
sudo pkill -9 -f 'torch.distributed.run' 2>/dev/null
sudo pkill -9 -f 'sft.py' 2>/dev/null
sudo pkill -9 -f 'launch_training_2node' 2>/dev/null
sleep 10
echo "--- AFTER: GPU mem + util (want ~0 MiB all 8) ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- ckpt watcher alive? ---"
pgrep -f ckpt-s3-watcher.sh | tr '\n' ' '; echo
```

**Authorship:** Not mine. The script references `ckpt-s3-watcher.sh` (S3
checkpoint backup utility located at `/opt/dlami/nvme/ckpt-s3-watcher.sh`,
created 04:02 UTC) which is **Zuocan's infrastructure** — I have no such
utility in my codebase.

**Intent analysis (what was Zuocan trying to do, and why):**

The script's primary goal is "thorough GPU free" — return all 8 GPUs to
0 MiB used and 0% util so a fresh training run can launch into a clean
state. Reading the script's design choices reveals the implicit assumptions:

| Design choice | Implied prior experience |
|---|---|
| `nvidia-smi --query-compute-apps=pid` → kill by PID (not by name first) | Has experienced **zombie/orphan GPU processes** that don't match `pkill -f swift` but still hold GPU memory. The author has been burned by "memory occupied but no swift process found" before. |
| `# belt-and-suspenders by name too` (the comment exactly) | The author distrusts either method alone — wants both PID-based AND name-based cleanup. |
| `sleep 10` then verify | Knows that SIGKILL → kernel reap → GPU memory release can take several seconds. The verify step ("want ~0 MiB all 8") confirms the cleanup succeeded. |
| Final `pgrep -f ckpt-s3-watcher.sh` check | Explicitly checks that the **S3 checkpoint backup watcher survived the cleanup**. They want the backup utility to keep running even when training is reset. This implies they save checkpoints continuously and don't want to lose that capability. |
| No SIGTERM-first / no grace period | Optimizing for speed of cleanup over safety. Iterate-debug cycle, not production. |
| No exclusion list, no "is anyone else using this" check | **The script's mental model is single-user.** The author assumed they were the only tenant. |

**Inferred workflow** (Zuocan's iteration loop):
1. Launch training run (their own variant — different config, possibly different
   data, definitely different from ours)
2. Observe results in real time, decide whether to abort early
3. When ready for next iteration: run `gpu-clean.sh` to nuke GPU state
4. Verify the cleanup worked (the "AFTER" diagnostic block)
5. Verify `ckpt-s3-watcher.sh` is still alive (don't lose ckpt backup)
6. Launch next iteration

This is a **completely reasonable utility for solo iteration on a research
box.** The author solved a real problem they had — zombie GPU processes
that wouldn't die from `pkill -f` alone. The bug is not in the script's
logic; it's in the script's MENTAL MODEL: "all GPU processes are mine and
all are safe to kill."

**Recurrence pattern indicates this was a debug session.** Four
invocations at 02:09, 05:48, 05:50, 07:23 with irregular gaps (3 h → 5 min
→ 1.5 h). This is classic single-developer iteration:
- 02:09: Initial cleanup at start of work session
- 04:02–05:50: Focused 2-h iteration burst (write code → launch → kill →
  clean → relaunch loop)
- 05:50 → 07:23: 1.5-h gap (possibly went AFK, ate breakfast, took a meeting)
- 07:23: Returned, ran cleanup as the **first ritual of the new iteration
  attempt**, ready to launch their next variant

We happened to launch in that 1.5-h gap. When Zuocan came back and ran his
cleanup ritual to prepare for HIS next run, it killed OURS — because his
script has no way to know our run is different from his old zombie process.

**Why this is the right call to make peace with, not be angry about:**
The cleanup script is well-designed for its intended use. It addresses a
real reliability problem (zombie GPU procs). The fix is not "Zuocan should
not have run his script"; the fix is "shared-tenancy boxes need a
coordination protocol that solo iteration scripts can be taught to
respect." See Part 3.

**Recurrence:** This is **not a one-off event.** Auth log shows four
distinct invocations on the same day:

| Time (UTC) | Action |
|---|---|
| 02:09:26 | `pkill -9 -f launch_training_2node` |
| 05:48:11 | `pkill -9 -f 'swift sft'` + birth of `gpu-clean.sh` file |
| 05:50:28 | `pkill -9 -f 'swift sft'` |
| **07:23:36** | `pkill -9 -f 'swift sft'` ← **killed our V8 run** |

The irregular spacing (3 hr → 5 min → 1.5 hr) rules out a cron; this is
manual invocation. Zuocan was iterating his own runs through the morning;
between his iteration #3 (05:50) and iteration #4 (07:23), we launched V8
without coordinating. His iteration #4 killed us.

**Authorization context:** Anyone with credentials for the `gpu-box` AWS
profile can issue `aws ssm send-command` as root. The CloudTrail event
shows `SessionOwner = arn:aws:iam::590184069312:user/leon` — Leon's IAM
user — but those credentials are presumably shared between Leon, Zuocan,
and possibly others. We cannot distinguish individual humans from the
audit log; we can only say "someone with Leon's credentials, who has
Zuocan's automation scripts on disk." Inference: Zuocan.

**Why detached launcher (setsid nohup) does NOT solve this:** The kill
was a direct `kill -9 <pid>` enumerated from `nvidia-smi
--query-compute-apps=pid`. There is no shell ancestor to detach from; the
kernel's PID lookup finds the rank workers regardless of session
membership. Detachment protects against SIGHUP from shell death; it does
not protect against a privileged user with the PID.

### Failure 2: No co-tenancy coordination protocol

The deeper failure underneath Failure 1: this is a shared-tenancy box
without a documented "who's using the GPUs right now" protocol. Symptoms:
- No reservation system (calendar, file lock, anything)
- No "training in progress" marker file that cleanup scripts check
- No team chat thread saying "I'm running V8 from 06:20 for ~7 days"
- No protective convention like "don't pkill -9 -f swift unless it's yours"

**Both Zuocan and I lacked the information to coordinate.** I launched
without telling anyone. He cleaned up without checking who was running.

### Failure 3: My babysitter regex matched its own command echo

**The bug (`/tmp/v8_babysit.sh` line 17-18):**
```bash
OUT=$(... ssm_run.sh \
  "tail -200 /opt/dlami/nvme/logs/v8_train.log 2>&1 | tr -d '\r' | \
   grep -E \"global_step|Killed|Error|nan|inf|OOM|SIGKILL|saved checkpoint|eval_loss\" | \
   tail -15; echo ===PROC===; \
   ps -ef | grep -E 'swift sft|torchrun' | grep -v grep | wc -l" 2>&1)

if echo "$OUT" | grep -qE "Killed|nan|inf.*loss|OOM|SIGKILL"; then
  echo "[$NOW] !!! ALERT: failure signal in log:"
  echo "$OUT" | grep -E "Killed|nan|inf|OOM|SIGKILL|Error"
  break
fi
```

`ssm_run.sh` invokes `aws ssm start-session` and pipes the command into
the session's stdin. The SSM session **echoes the command line to stdout
before executing it.** So `$OUT` contains, near its top, a line that
literally reads the grep command — including the words `Killed`, `nan`,
`OOM`, `SIGKILL`. My failure-detection regex matched its own command echo.

This was deterministic — every single iteration would have triggered the
same false positive — but the `break` statement ensured we only saw it
once.

**Why the monitor stayed dead:**
- No persistence across Claude session boundaries — local bash background
  process tied to my Claude Code session
- No file-based status that anyone (Leon, me on restart, anyone) could
  poll independently
- `break` on first alert = exit-permanently semantics
- No external heartbeat to a file or webhook

### Why these three failures compounded

If only Failure 1 had occurred (without my broken monitor), Leon would
likely have noticed within an hour via his own wandb-checking habits.
~1 h cost.

If only Failure 3 had occurred (without the co-tenant kill), nothing bad
would have happened. The run would have continued healthy through the
night and Leon would have woken to working V8.

Both together produced 9 h 20 m because:
- I publicly committed to babysitting → Leon stopped checking wandb himself
- My broken babysitter gave Leon no indication anything was wrong
- I had no second layer (heartbeat staleness, on-box monitor, external page)

Same class as previous V8 incidents: **silent default behavior + no
surfaced loud signal**. SDPA silently overrode FA3. Audio tower silently
truncated PE. Now: a co-tenant cleanup script silently kills any training
on shared hardware, and my monitor silently dies on its own command echo.

## Part 3 — What we change

### Immediate (before relaunch)

**1. Coordinate with Zuocan BEFORE relaunching.**

Until we know when he's running and when he isn't, any relaunch is at risk.
Specific asks:
- "Are you running anything in the next 5 days?"
- "When are you running cleanup? Is it on a schedule or ad-hoc?"
- "Can your `gpu-clean.sh` check for a marker file before killing?"
- "Can we agree on a marker-file convention for 'training in progress'?"

This is a social fix and the most important one. Without it, all the
technical fixes below are necessary but insufficient.

**2. Marker-file convention for "training in progress."**

Convention: anyone running training writes
`/opt/dlami/nvme/RUNNING_<owner>_<runid>.lock` with their wandb URL and
contact info before launch. Anyone running cleanup checks for any
`/opt/dlami/nvme/RUNNING_*.lock` and bails out with a loud message if
present.

Proposed patch to `gpu-clean.sh`:
```bash
LOCKS=$(ls /opt/dlami/nvme/RUNNING_*.lock 2>/dev/null)
if [ -n "$LOCKS" ]; then
  echo "REFUSING to clean: active training locks present:"
  for f in $LOCKS; do echo "  $f:"; cat "$f"; done
  exit 1
fi
```

**3. Detach the launcher (setsid nohup) anyway.**

Even though detachment doesn't solve THIS failure mode, it solves the
class we feared initially (SSH disconnect, SSM session timeout, terminal
exit) and is cheap insurance:

```bash
setsid nohup bash sft.sh CONFIG > log 2>&1 < /dev/null &
echo "training pid=$!"
disown $!
```

**4. File-based monitor on the AWS box.**

Architecture:
- `health_writer.sh` runs on the AWS box (under `setsid nohup`), polls
  `/opt/dlami/nvme/logs/v8_train.log` every 60 s and writes
  `/opt/dlami/nvme/health/v8_status.json` with `{ts, step, loss,
  grad_norm, mem_gb, proc_alive, last_log_line}`.
- `health_reader.sh` runs on Leon's laptop / anywhere, single SSM
  SendCommand to `cat` the status file. Transient, no persistence needed.
- My role: invoke `health_reader.sh` whenever asked, OR wake up at
  decision-point intervals (step 50 eval, step 75 ckpt).

This separates "produce monitoring data" (persistent process on the box,
owns the data file) from "consume monitoring data" (transient query,
reads the file). No long-running local process required.

**5. Fix the regex bug class permanently.**

The lesson generalizes beyond this monitor: when grepping output that may
contain command-text or shell-echoed args, EITHER:
- Grep against a file directly (`grep PATTERN /path/to/log` — bypasses
  the command-echo problem), OR
- Use sentinel-delimited regions: `echo '===BEGIN==='; <command>;
  echo '===END==='` and `awk` to extract only the region between
  sentinels.

For this monitor specifically: have the on-box writer read
`/opt/dlami/nvme/logs/v8_train.log` directly. Never grep SSM session
stdout.

**6. Word-boundary the failure regex.**
Replace `grep -E "Killed|nan|inf|OOM|SIGKILL"` with
`grep -E "\bKilled\b|\bnan\b|\bOOM\b|\bSIGKILL\b"`.

**7. Never `break` on alert. Log + continue.**
The right semantics: alert, write status, keep polling. We want to know
whether the situation resolved itself OR whether subsequent state confirms
the alert.

### Process changes

**8. Pre-launch announcement.**
For any multi-hour run on shared hardware: announce in team chat
(Slack/WeChat/whatever) with start time, expected duration, wandb URL,
and how to reach you for stop/coordination. **No silent launches on
shared boxes.**

**9. Pre-launch detachment smoke.**
Before any multi-hour run, verify the launcher detaches correctly: launch,
immediately close the SSM session, wait 90 s, reconnect with a new SSM
session, verify training processes are still alive. Add to V8 launch
runbook as mandatory step.

**10. Two-layer monitoring contract.**
For any training run > 1 GPU-hour, require:
- **Layer 1**: wandb dashboard URL pinned in run notes
- **Layer 2**: file-based health-writer on the box producing a status file
  every 60 s
- **Layer 3**: my Claude wakeups at known decision points

No single layer is allowed to be the only source of truth.

**11. Honesty contract.**
If I cannot reliably monitor a run (e.g., because Claude sessions don't
persist across my own restarts), I must say so explicitly rather than
promising to "babysit". This incident's deepest failure mode was social,
not technical: I claimed coverage I couldn't actually provide.

### Reproducing the diagnosis (for future incidents)

```bash
# 1. Get root access to the box (any user with gpu-box AWS profile)
aws ssm send-command \
  --instance-ids <i-id> --document-name "AWS-RunShellScript" \
  --parameters "commands=[\"cat /var/log/auth.log | tail -100\"]" \
  --region <region> --profile <profile>

aws ssm get-command-invocation \
  --command-id <returned-cmd-id> --instance-id <i-id> \
  --region <region> --profile <profile>

# 2. Search for SIGKILL + pkill events in auth.log around the death time:
grep "07:23" /var/log/auth.log

# 3. Find the SendCommand script that ran the kill:
find /var/lib/amazon/ssm/<i-id>/document/orchestration \
  -name "_script.sh" -newermt "<death-time-minus-10s>" \
  ! -newermt "<death-time-plus-10s>"

# 4. Read each candidate to identify the kill script:
cat <orchestration>/awsrunShellScript/0.awsrunShellScript/_script.sh
```

## Affected files

- `/opt/dlami/nvme/launch_training_2node.sh` — needs `setsid nohup`
  wrapper added
- `/opt/dlami/nvme/gpu-clean.sh` — needs marker-file guard (Zuocan's
  script, requires coordination)
- `/tmp/v8_babysit.sh` — to be replaced by on-box `health_writer.sh` +
  off-box `health_reader.sh`
- `examples/train/grpo/qwen2_5_omni_ttcc/V8_LAUNCH_RUNBOOK.md` — add
  "announce + smoke + marker file" steps

## Related incidents

- [INCIDENT_2026-05-28_SDPA_OVERRIDE.md](INCIDENT_2026-05-28_SDPA_OVERRIDE.md) —
  silent attn_impl override defeated FA3 install
- [INCIDENT_2026-05-28_AUDIO_OOB.md](INCIDENT_2026-05-28_AUDIO_OOB.md) —
  audio tower silently truncated PE slice
- [INCIDENT_2026-05-26_EVAL_LEAK.md](../../../../ttcc-eval/INCIDENT_2026-05-26_EVAL_LEAK.md) —
  V7 R(t) leak in assistant span

**Common pattern across all four:** silent default value or silent action +
no surfaced loud signal. A silently-failing monitor is worse than no
monitor; a silently-killing cleanup is worse than no cleanup.

## Aphorisms

- "Shared hardware without a coordination protocol is private hardware
  with a roulette wheel."
- "A monitor without persistence is a wish. A monitor without
  cross-restart-survivability is a smoke detector that requires you to
  be awake to hear it."
- "Grep against files, not stdout streams that echo your own command back."
- "`break` on alert is the wrong semantics. Alert and keep watching — the
  next 60 seconds tell you whether you saw a glitch or a fire."
- "If you cannot promise to monitor, do not claim to babysit. The social
  failure is the deepest failure of this incident."
- "When debugging crashes on shared infrastructure: `auth.log` first,
  hypotheses second."
- "The fix for 'shared credentials, no human distinction' is not technical;
  it's a coordination protocol."

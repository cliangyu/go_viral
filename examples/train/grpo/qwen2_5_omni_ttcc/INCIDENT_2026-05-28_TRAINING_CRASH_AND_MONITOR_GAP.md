# Incident Review: V8 Training Crash + 9 h 20 m Monitor Blackout (2026-05-28)

## Part 1 — What happened

**Timeline (UTC, 2026-05-28):**

- **06:03**: Launcher script `launch_training_2node.sh` finalized on head node
  with sed edits (FA3, USE_AUDIO_IN_VIDEO=false, NCCL heartbeat). Invocation
  pattern: `bash sft.sh CONFIG 2>&1 | tee /opt/dlami/nvme/logs/v8_distributed.log`
  — no `nohup`, no `setsid`, no `disown`, no tmux, no screen.

- **06:20:11**: Training launched. wandb run `fxryqedt` started. All 16 ranks
  on both nodes initialize, ZeRO-3 shards, model loaded.

- **06:54** (step 8): 16-check integrity audit passed. `V8_INTEGRITY_AUDIT.md`
  committed and pushed. Run state: loss=30.13, grad_norm=3994 (clipped),
  memory=54 GB peak per GPU.

- **07:14:27**: My local babysitter (`/tmp/v8_babysit.sh`) ran its first
  poll iteration. The regex
  `grep -E "Killed|nan|inf.*loss|OOM|SIGKILL"` was applied to
  `$OUT` — which contained the SSM session's stdout INCLUDING the echoed
  command line. The echoed command itself contained the literal strings
  `Killed`, `nan`, `inf`, `OOM`, `SIGKILL` (as part of the grep arguments
  themselves). The regex matched on its OWN command echo. The babysitter
  printed `!!! ALERT: failure signal in log:` and hit `break` to exit the
  loop. **Babysitter dead after one (false) iteration. Total monitoring
  time: 0 minutes.**

- **07:19:46** (step 15): Loss dropping fast (35.5 → 8.16 across steps 6–15
  as the head learned the curve scale). grad_norm 2111 (smaller than earlier
  4000s — head bias correction was paying off). Memory still 54 GB. **The run
  was actually getting healthy.**

- **07:23:36**: Training crashed. Ranks 0–4 received SIGKILL (exit -9),
  ranks 5–7 received SIGTERM (exit -15) one second later (07:23:37). torchrun
  parent (PID 398525) died with `Killed`. All 8 GPUs on head node went to
  0 MiB / 0% util.

- **07:23:36 → 16:44** (9 h 20 m): **Dead air.** No process running. No
  monitor watching. wandb run paused. CB clock burning. I (Claude) was idle
  with no notification path. Leon was asleep, trusted that I would notify.

- **16:44**: Leon awoke, asked "crashed? didn't monitor? you screwed".
  Discovery moment.

**Net cost:**

| Metric | Value |
|---|---|
| Training that actually happened | 63 min (steps 1–15) |
| Monitor blackout | **9 h 20 m** |
| Steps not completed during blackout (at 246 s/step) | **~136 steps** = ~½ epoch |
| % of CB budget burned to dead air | **~6.7 %** |
| CB remaining at discovery | 138 h 45 m |
| Status of step-50 eval (val curve_loss — the retention metric we care about) | Never landed |
| Status of step-75 ckpt + randomization probe (V7-class hard test) | Never landed |
| GPU-hours wasted | 16 GPUs × 9.3 h = **~149 GPU-h** = ~$595 at p5.48xlarge rate |

**No checkpoint contamination** — first save was scheduled at step 75 (crashed
at step 15). No model state corrupted. Run state on disk = empty output dir,
clean for restart.

## Part 2 — Why it happened (technical deep dive)

This incident has TWO independent failure modes that intersected catastrophically.
Either alone would have been a near-miss; together they produced a 9-hour
blackout.

### Failure 1: Training process killed by SSM session lifecycle

**Process tree at launch:**
```
ssm-user's SSM shell (transient, parent)
└── bash launch_training_2node.sh
    └── bash sft.sh CONFIG
        └── tee /opt/dlami/nvme/logs/v8_distributed.log
            └── ${VENV}/bin/swift sft CONFIG  (PID 398525, master)
                └── python -m torch.distributed.run --nproc_per_node 8 ...
                    └── 8 × swift sft rank workers (PID 398821–398828)
                        └── 32 × dataloader workers
```

**The kill chain:**
1. The training launcher was invoked from a shell inside an `aws ssm
   start-session` interactive session. There was no `nohup`, `setsid`,
   `disown &`, tmux, or screen anywhere in the launch script. Verified by
   grep: `nohup|tmux|screen|disown|setsid` returns 0 matches in
   `launch_training_2node.sh`.

2. SSM Session Manager has an **idle session timeout** (default 20 minutes,
   sometimes configured higher; in this account behavior consistent with ~1 h
   based on the 06:20 → 07:23 = 63 min interval).

3. When the SSM session terminated, its shell process died. By POSIX semantics,
   the kernel sent `SIGHUP` to every process whose controlling terminal was
   that shell — which was the entire descendant tree.

4. The swift master (PID 398525) is a Python process with no SIGHUP handler
   installed. SIGHUP terminated it. swift's atexit handlers fired briefly,
   trying to send SIGTERM to all 16 ranks for orderly shutdown.

5. The TERM-then-KILL escalation took ~1 second on 5 of the 8 head-node ranks;
   the kernel reaped them with SIGKILL (exit -9) before TERM finished. The
   remaining 3 ranks managed orderly SIGTERM exit (-15) one second later.

**Evidence supporting this theory:**
- `dmesg` clean (no OOM-killer activity → not memory pressure)
- Host RAM: 159 GB used / 2 TB → not RAM pressure
- GPU memory stable at 54 GB across 15 steps → not GPU OOM
- The "ranks 0–4 SIGKILL, ranks 5–7 SIGTERM" pattern is exactly the signature
  of a kernel-reap-after-graceful-shutdown-timeout
- The 63-min interval matches typical SSM idle timeout windows
- `ssm-user` doesn't have `journalctl -u amazon-ssm-agent` permissions to
  confirm, but the elimination of all other plausible causes (OOM, NaN, GPU
  fault, manual kill) leaves SSM timeout as the dominant hypothesis

**Why the V7 / V8 SDPA-override / V8 audio-OOB incidents didn't expose this:**

Those incidents all crashed the training within minutes of launch, before the
SSM session timeout could fire. This was the **first** V8 run that survived
past ~5 minutes — and immediately ran into the next silent infrastructure
trap.

### Failure 2: Babysitter regex matched its own command echo

**The bug (`/tmp/v8_babysit.sh` line 17–18):**
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

**What goes wrong:**

`ssm_run.sh` invokes `aws ssm start-session` and pipes the command string into
the session's stdin. The SSM session **echoes back the command line to stdout**
before executing it. So `$OUT` contains, near its top, a line that literally
reads:

```
$ tail -200 /opt/dlami/nvme/logs/v8_train.log 2>&1 | tr -d '\r' | \
   grep -E "global_step|Killed|Error|nan|inf|OOM|SIGKILL|saved checkpoint|eval_loss" | \
   ...
```

The babysitter's failure-detection regex `Killed|nan|inf.*loss|OOM|SIGKILL`
then matched on the words `Killed`, `nan`, `OOM`, `SIGKILL` that appear inside
the echoed grep arguments. **The monitor flagged itself as failed.**

This was deterministic — every single iteration would have triggered the same
false positive — but the `break` statement on first alert ensured we only saw
it once.

**Why it stayed broken:**

1. **No persistence across Claude session boundaries.** The babysitter was a
   local bash background process tied to my Claude Code session. Even if the
   regex had been correct, when my Claude session restarted, the babysitter
   would have continued running on Leon's laptop with no path to alert me. I
   had no mechanism to be re-woken on event.

2. **No file-based status that I could poll.** A correctly designed monitor
   would write `/tmp/v8_status.json` every iteration with `{step, loss, mem,
   proc_alive}`, and any future check could just read that file. Instead the
   babysitter tried to be both producer and consumer in the same process —
   when the producer died, the data died with it.

3. **`break` on first alert was the wrong semantics.** Even for a real alert,
   the correct behavior is to keep monitoring (or restart the loop) so we see
   what happens AFTER the alert. Exiting permanently means we lose all
   information about whether the situation resolves itself.

4. **No external heartbeat.** The babysitter only printed when something
   interesting happened. There was no "every 30 min, regardless, print a
   heartbeat to a file with the current state". With heartbeats, the absence
   of a fresh heartbeat itself becomes a signal.

### Why these two failures compounded so badly

If only the training had crashed (without my broken monitor), Leon would
likely have noticed within an hour from his own wandb checking habits, and we
would have lost ~1 h.

If only my monitor had been broken (without the training crash), nothing bad
would have happened — the run would have continued healthy through the night
and Leon would have woken to an actually-working V8.

Both failures together produced a 9 h 20 m blackout because:
- I had publicly committed to babysitting (so Leon stopped watching wandb himself)
- My broken babysitter gave Leon no indication anything was wrong (it didn't
  alert because it was already dead from the false positive 9 min earlier)
- I had no second layer (no heartbeat-staleness check, no wandb-based
  watchdog, no cron on the box itself)

**This is the same class as the previous V8 incidents:** silent default
behavior + no surfaced loud signal. SDPA silently overrode FA3. Audio tower
silently truncated PE. Now: SSM silently kills training, and my monitor
silently dies on its own command echo. Pattern: any shared infrastructure
that fails silently is a load-bearing dependency.

## Part 3 — What we change

### Immediate (before relaunch)

**1. Detach the training from any shell session.**

Edit the launcher invocation to use `setsid` + `nohup`. This is the simplest
and most reliable detachment:

```bash
# Instead of:
bash sft.sh CONFIG 2>&1 | tee /opt/dlami/nvme/logs/v8_distributed.log

# Use:
setsid nohup bash sft.sh CONFIG > /opt/dlami/nvme/logs/v8_distributed.log 2>&1 < /dev/null &
echo "training started, pid=$!"
disown $!
```

`setsid` creates a new session detached from any controlling terminal.
`nohup` ignores SIGHUP. The redirected stdin (`< /dev/null`) prevents
read-from-terminal blocks. `disown` removes the job from the shell's job
table so even `exit` won't reach it.

Alternative: launch inside tmux/screen. Equally valid. tmux additionally
lets us `tmux attach` later to inspect.

**2. Build a file-based monitor that survives Claude session restarts.**

Architecture:
- A `health_writer.sh` runs on the AWS box (under `setsid nohup`), polling
  the training log every 60 s and writing `/opt/dlami/nvme/health/v8_status.json`
  with `{ts, step, loss, grad_norm, mem_gb, proc_alive, last_log_line}`.
- A `health_reader.sh` runs on Leon's laptop (or anywhere) when invoked,
  doing one SSM call to `cat` the status file. Single-shot, no persistence
  needed. Returns parseable JSON.
- My role: invoke `health_reader.sh` whenever I'm asked to check, or set up
  a Claude Code wakeup at decision-point intervals (step 50 eval, step 75
  ckpt).

This separates "produce monitoring data" (persistent process on the box, owns
the data file) from "consume monitoring data" (transient query, reads the
file). No long-running local process needed.

**3. Fix the regex bug class permanently.**

The babysitter regex matched its own command echo. The lesson generalizes:
when grepping output that may contain command-text or shell-echoed args,
either:
- Grep against a file directly (`grep PATTERN /path/to/log` — bypasses any
  command-echo problem), OR
- Use a sentinel-delimited region: `echo ===BEGIN===; <command>; echo ===END===`
  and `awk` to extract only the region between sentinels.

For this monitor specifically: read `/opt/dlami/nvme/logs/v8_train.log`
directly via `tail -n 200 /opt/dlami/nvme/logs/v8_train.log` and grep the
result, never the SSM session stdout.

**4. Word-boundary the failure regex.**

Replace `grep -E "Killed|nan|inf|OOM|SIGKILL"` with
`grep -E "\bKilled\b|\bnan\b|\bOOM\b|\bSIGKILL\b"` so partial matches
("info", "inference", "training") don't false-trigger.

**5. Never `break` on alert. Log + continue.**

The right semantics: alert, write status, keep polling. We want to know
whether the situation resolved itself, or whether subsequent state confirms
the alert.

### Process changes

**6. Pre-launch detachment smoke.**

Before any multi-hour run, verify the launcher detaches correctly: launch
the script, immediately close the SSM session, wait 90 s, reconnect with a
new SSM session, verify the training processes are still alive. Add to the
V8 launch runbook as a mandatory step.

**7. Two-layer monitoring contract.**

For any training run > 1 GPU-hour, require:
- **Layer 1**: wandb dashboard URL pinned in run notes (Leon can check
  directly without me)
- **Layer 2**: file-based health-writer on the box producing a status file
  every 60 s
- **Layer 3**: my Claude wakeups at known decision points (step 50, step 75,
  step 200, etc.) reading the status file

No single layer is allowed to be the only source of truth.

**8. Honesty contract.**

If I cannot reliably monitor a run (e.g., because Claude sessions don't
persist across my own restarts), I must say so explicitly rather than
promising to "babysit". This incident's deepest failure was social, not
technical: I claimed monitoring coverage I couldn't actually provide.

### Reproducing the diagnosis (for future incidents)

```bash
# Confirm SSM-timeout-kill pattern on a node:
TARGET=<i-id> REGION=<reg> AWS_PROFILE=<profile> ssm_run.sh \
  "tail -100 /opt/dlami/nvme/logs/v8_train.log | grep -B2 -A8 'Root Cause'"

# Look for the SIGKILL exit -9 / SIGTERM exit -15 mix pattern.
# If you see this pattern WITHOUT NaN/OOM/GPU errors, suspect external signal.

# Verify launcher detachment:
grep -E "nohup|setsid|disown|tmux|screen" /opt/dlami/nvme/launch_training_2node.sh
# 0 matches → launcher is undetached, vulnerable to SSM timeout
```

## Affected files

- `/opt/dlami/nvme/launch_training_2node.sh` — needs `setsid nohup` wrapper added
- `/tmp/v8_babysit.sh` — to be replaced by `health_writer.sh` (on box) +
  `health_reader.sh` (off box)
- `examples/train/grpo/qwen2_5_omni_ttcc/V8_LAUNCH_RUNBOOK.md` — add
  "verify detachment" smoke step

## Related incidents

- [INCIDENT_2026-05-28_SDPA_OVERRIDE.md](INCIDENT_2026-05-28_SDPA_OVERRIDE.md) —
  silent attn_impl override defeated FA3 install
- [INCIDENT_2026-05-28_AUDIO_OOB.md](INCIDENT_2026-05-28_AUDIO_OOB.md) —
  audio tower silently truncated PE slice
- [INCIDENT_2026-05-26_EVAL_LEAK.md](../../../../ttcc-eval/INCIDENT_2026-05-26_EVAL_LEAK.md) —
  V7 R(t) leak in assistant span

**Common pattern across all four:** silent default value + no surfaced loud
signal. The "any shared infrastructure that fails silently is a load-bearing
dependency" rule (from the AUDIO_OOB incident) now extends to: **the same is
true for any monitor.** A silently-failing monitor is worse than no monitor,
because it gives false confidence.

## Aphorisms

- "Grep against files, not stdout streams that echo your own command back."
- "A monitor without persistence is a wish. A monitor without
  cross-restart-survivability is a smoke detector that requires you to be
  awake to hear it."
- "`break` on alert is the wrong semantics. Alert and keep watching — the
  next 60 seconds tell you whether you saw a glitch or a fire."
- "If you cannot promise to monitor, do not claim to babysit. The social
  failure is the deepest failure mode of this incident."
- "SSM Session Manager is a remote shell, not a daemon. Anything launched
  from it dies with it. Plan for the disconnection that always comes."

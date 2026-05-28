# V8 Lessons Learned

Captured 2026-05-28 during V8 SFT-Hazard+CoT run on 2-node H100 cluster.
Coverage: the 24h sprint that included the V7 leak finding, three live incidents
(SDPA override, audio OOB, FA3 install), and the 16-check integrity audit.

The goal of this doc: when we come back to plan V9 / V10 / future runs, we
should NOT re-derive the same lessons. Each section is a takeaway, what
evidence produced it, and how it shapes the next run.

---

## Part 1 — What worked, keep doing

### 1.1 Pre-flight integrity audit (16 evidence-based checks)

**Pattern:** Before reading any loss number, prove the pipeline can produce a
trustworthy loss number. Each check has (a) hypothesis, (b) script, (c) raw
output, (d) verdict. No "looks right to me" — only "0/35,793 hits".

**Evidence:** `V8_INTEGRITY_AUDIT.md` catches V7-class leaks, label sanity, frozen
modules, FA3 runtime activity, model init source, anchor position safety, etc.
Total runtime: ~25 min including writing the doc.

**Why it works:** The audit's value is the doc itself. If we ever need to argue
"did V8 succeed?", the answer cites concrete file:line evidence, not memory.

**For V9:** Reuse the same 16 categories. Update only the specifics (paths,
versions, dataset hash). The template lives in `V8_INTEGRITY_AUDIT.md`.

### 1.2 Three-part incident docs (timeline → why → fix)

**Pattern:**
1. **What happened** — timestamps, GPU-hours wasted, who-discovered-when
2. **Why it happened** — technical root cause + adjacent silent-failure modes
3. **What we change** — immediate fix + process change + reproduction recipe

**Files:** `INCIDENT_2026-05-28_SDPA_OVERRIDE.md`,
`INCIDENT_2026-05-28_AUDIO_OOB.md`, `INCIDENT_2026-05-26_EVAL_LEAK.md`.

**Why it works:** Each incident teaches a generalizable lesson, not just a fix.
"max_source_positions is a hard contract with the data, not a hint." That
aphorism prevents a class of bugs, not one.

**For V9:** Same template. Document every silent-failure mode encountered.

### 1.3 Full-corpus data scans

**Pattern:** Don't sample-check critical invariants. Scan the whole corpus.
35,793 rows in 1.3 s — there's no excuse for 100-row spot-checks.

**Examples (`/tmp/full_scan.py`, `/tmp/audit_jkl.py`):**
- 0/35,793 train rows with R-leak pattern
- 0/35,793 with R label violations (monotone, range, length)
- 0 overlap between 35,793 train and 200 val ad_ids

**For V9:** All data-integrity checks must be 100% corpus scans, not samples.

### 1.4 `/proc/<pid>/maps` for runtime library verification

**Pattern:** When a launch-arg should mean "X library is active", verify the
library's `.so` is actually mapped into rank workers' memory. Don't trust
launch args or import warnings alone.

**Evidence:** FA3 install completed, launch arg looked correct, but the only
proof was `/proc/<rank-pid>/maps` showing `flash_attn_3/_C.abi3.so` loaded
across all 8 ranks. Without that, the previous SDPA override would have gone
unnoticed.

**For V9:** Add this check to launch smoke tests.

### 1.5 gzip+base64 SSM script transfer

**Pattern:** SSM Session Manager interactive shell mangles multi-line heredocs
(backslash escapes, line-length limits, dash vs bash). Reliable transfer:
1. Write the script locally as a `.py` file
2. `gzip -c script.py | base64 | tr -d '\n'`
3. `echo $B64 | base64 -d | gunzip > /tmp/script.py && python3 /tmp/script.py`
4. Use `LINGER=60` env to keep stdin open while big payloads decode

**Why it works:** Single-line stdin, no shell escaping, atomic file write
before execution. Saves repeated debugging of heredoc syntax errors.

**For V9:** First-class scripted operations should ship through this pattern,
not inline heredoc.

---

## Part 2 — What didn't work, change for V9

### 2.1 Stash without callback = silent observability gap

**The bug:** `register.py` line 673-691 stashes `loss_curve` and `loss_cot` on
`outputs` and `holder`, expecting "some callback" to read them. No callback was
ever registered. The values are computed every step and discarded.

**Cost:** Cannot see retention loss separately from LM loss in wandb. Only the
combined total is logged. For a multi-objective loss where you care about ONE
objective primarily (retention), this is half-blind.

**For V9:** Every stashed metric must have a matching `TrainerCallback` that
reads it on `on_log` and pushes to the trainer's log dict. No stash without
callback. Add to integrity audit as a check: "every output attribute set in
`compute_loss` must appear in at least one wandb run history."

### 2.2 ZeRO-3 + per_device_batch=1 + 32K seq = 26% TDP

**The waste:** GPUs averaged 184 W / 700 W cap, 38 °C, while showing 74% SM
util. The high SM% is NCCL kernels spinning on inter-rank comm, not tensor-core
matmul. Estimated effective throughput ≈ 5% of practical H100 peak.

**Why it happened:** ZeRO-3 partitions every parameter across DP world, so
forward needs all-gather and backward needs reduce-scatter for every layer.
With per_device_batch=1, there's not enough compute per all-gather to amortize
the comm cost.

**For V9 — testable hypotheses (rank by expected speedup):**

| Lever | Expected gain | Risk | Pre-launch test |
|---|---|---|---|
| ZeRO-2 (params unsharded) | 5–15% | Memory: +6 GB/rank, still safe at our 80 GB | Single-node smoke |
| `group_by_length: true` | 5–10% (kills rotating straggler) | swift multimodal may not support | Smoke test before full run |
| per_device_batch 1→2 + accum 8→4 | 10–20% raw, but padding cost halves it | OOM on long-seq batches | Memory profiler at batch=2 |
| Sequence packing | 15–25% | Anchor positions across packed boundaries (need template change) | LOC change + smoke test |

**Don't combine all at once** — stack one lever per V9-variant smoke.

### 2.3 Restart cost is real and compounds

**Math:** Each restart in this session cost ~30 min (kill, edit yaml, push,
pull, relaunch, get past warmup steps 1–3). Across 4 restarts we lost ~2 h.

**Risk math:** Each restart in this session also surfaced a NEW issue (SDPA
override, audio OOB, FA3 metadata, NCCL config). The empirical "new-issue rate
per restart" was ~50% in this session. Even if the real rate is 15–20%, the
expected cost of one more restart is ~30 min × 1.15–1.2 = ~35 min plus a real
chance of compounding into another incident.

**For V9:** Once a run is healthy AND the integrity audit passes, the bar to
restart is high. Specific triggers to restart:
- Verification gate fails (val MSE blows up at step 50)
- Randomization probe ratio < 0.3 at step 75
- Memory leak (peak grows past 70 GB over 100+ steps)
- Loss NaN/Inf

NOT triggers to restart:
- Found a nice-to-have config improvement
- Marginal compute efficiency gain
- "I'd be more comfortable if X were also enabled"

### 2.4 Default `ssm_run.sh` wrapper has shell-escaping issues

**The bug:** The shared `ssm_run.sh` wrapper uses heredoc with backslash
escapes, and SSM's interactive shell is dash, not bash. Multi-line awk, nested
quotes, and parens can corrupt mid-stream.

**Workaround used in this session:** gzip + base64 the script, decode on
target, run via `python3 /tmp/x.py`. See 1.5.

**For V9:** Either patch `ssm_run.sh` to support the gzip+base64 path
natively, or build a thin `ssm_script` helper that does the encoding.

---

## Part 3 — V9 must-haves (load-bearing for "is the run correct")

### 3.1 Loss-split logging callback wired from step 1
Add a `RetentionLogCallback(TrainerCallback)` in `register.py` whose `on_log`
reads `holder.loss_curve` and `holder.loss_cot` and merges into `logs` dict.
Verify before launch that wandb shows the three keys at step 1.

### 3.2 Integrity audit re-run pre-launch
Same 16 checks, updated for V9 dataset / model / config. Total runtime ~30
min. Required output: `V9_INTEGRITY_AUDIT.md` committed before training
launches.

### 3.3 Pre-launch single-row forward smoke
Per `INCIDENT_2026-05-28_AUDIO_OOB.md` process change: one row, one GPU,
isolated subblock forward must succeed. Catches:
- Audio tower OOB (would have caught audio OOB pre-launch)
- TMRoPE / FA3 patch wiring (would have caught cu_seqlens 3D pre-launch)
- Custom head attach + forward (would catch any swift internals change)

### 3.4 Runtime library verification
`/proc/<pid>/maps` check at step 5 confirms FA3 (or whatever attention impl is
in the yaml) is actually mapped into rank workers. Cheap; catches any silent
fallback.

---

## Part 4 — V9 nice-to-haves (speedups + extensions)

### 4.1 Audio reintroduction
Three feasibility-ranked options for re-enabling audio:
1. **PE interpolation** (~5 LOC change): linearly interpolate audio_tower's
   `[1500, 1280]` positional embedding to `[6000, 1280]` at load time. Standard
   ALiBi-style trick. Risk: audio encoder was trained at 1500 cap; quality may
   degrade. Cheap to test.
2. **Audio truncation**: keep first 15 s of audio at template stage. Captures
   the hook (most retention-decisive) and avoids PE overflow. ~10 LOC.
3. **Audio chunking**: 15 s windows, concatenate encoder outputs. ~50 LOC,
   needs patching `audio_tower.forward`. Higher quality, more work.

Recommend testing in order. If (1) gives reasonable IBS, ship.

### 4.2 Larger model variant
Qwen2.5-Omni-7B has the same architecture as 3B, just deeper. If V8 IBS
plateaus below the lower bound we expected, the 3B may be IBS-capacity-limited
and 7B is the next test. ~3× compute cost; only run after V8 results.

### 4.3 Sequence packing
Pack multiple short ads into one batch slot. Speedup ~15–25% from better
matmul shape on H100. Needs template changes to keep retention head anchors
correct across packed boundaries.

### 4.4 group_by_length test for multimodal
swift may or may not support `group_by_length: true` for multimodal inputs.
Run a 50-step smoke with this flag on; check for crashes; measure step-time
variance.

---

## Part 5 — Operational lessons (process, not code)

### 5.1 Don't ask for routine reversible work; just execute

This was a recurring inefficiency early in the session. Leon's 2026-05-18
blanket auth says: execute reversible/local commands, only escalate for
explicit boundaries. Asking "should I run the audit script?" is the kind of
permission-seeking that wastes turns. **For V9 work specifically:** anything
inside the go_viral repo + reading state from the AWS nodes is reversible
and pre-authorized.

### 5.2 The right question for compute waste is "what's the bottleneck"

Initial framing was "ZeRO-3 vs ZeRO-2 perf diff" — the wrong question. The
right question: "what's actually preventing us from using 100% of compute?"
Answer turned out to be communication + small per-device batch + activation
checkpointing recompute, not ZeRO-stage choice. Always profile before
prescribing.

### 5.3 Restart with a verification gate, not a feeling

"I'd be more comfortable if we restarted with X" is not a good restart
trigger. "Val MSE diverged at step 50" is. Pre-commit to the verification
gates before launch so we don't argue in the middle.

### 5.4 Distrust agent self-reports (already in memory)

Verified: codex-rescue and other subagents often declare "completed" after
only dispatching a sub-task. Always verify the artifact, not the summary.
Same applies to "checked X" claims — show the evidence (file:line + command
output).

### 5.5 Capture incidents in the moment

Three incident docs were written 30 min – 2 h after the incident, while the
state was still fresh. Writing them 2 days later would have lost the GPU-time
numbers and the specific stack trace addresses. **For V9:** any GPU-time-burning
incident gets its three-part doc within 4 h of resolution.

---

## Part 6 — Genuinely open questions (no answer yet)

1. **Does V8 condition on video signal?** Resolved at step 75 by
   randomization probe.
2. **Does V8 IBS beat V7?** Resolved by offline eval after ckpt harvest.
3. **Does audio matter for IBS?** Open until V9 with audio reintroduced.
4. **Is 3B IBS-capacity-limited?** Open until 7B comparison.
5. **Is the train/val anchor mismatch (post-CoT vs post-prompt) costing us
   monitoring fidelity?** Open until first eval lands.

---

## Part 7 — Files this session produced (for future reference)

| File | Purpose |
|---|---|
| `V8_INTEGRITY_AUDIT.md` | 16-check pre-launch / mid-run integrity audit |
| `V8_LESSONS_LEARNED.md` | This file |
| `INCIDENT_2026-05-28_SDPA_OVERRIDE.md` | Silent SDPA override killed FA3 install |
| `INCIDENT_2026-05-28_AUDIO_OOB.md` | Audio tower PE overflow for ads > 15 s |
| `FLASH_ATTENTION_INSTALL.md` | FA3 install playbook for cu130 + torch2120 |
| `V8_LAUNCH_RUNBOOK.md` | Bootstrap-to-launch on 2-node H100 |
| `V8_LAUNCH_RUNBOOK_MODAL.md` | Wanjia's Modal LoRA path |
| `HEAD_COMPARISON.md` | Path-A (language head) vs Path-B (hazard head) |
| `register.py` (custom plugin) | model_type=qwen2_5_omni_retention with TMRoPE patch |
| `configs/sft_retention_hazard_full_with_cot.yaml` | The live V8 config |

---

## Closing

V8 is a step forward not because the model is necessarily better than V7
(unknown until eval), but because we now have:
- A definitively closed leak class
- An integrity audit pattern
- An incident-documentation pattern
- A list of compute-efficiency levers, ranked
- Operational discipline around restarts

If V8 fails the step-75 probe, V9 starts from a stronger position than V8 did.
If V8 succeeds, V9 starts from a stronger position than V8 did. The work
captured here is the actual asset.

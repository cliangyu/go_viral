# Incident Review: SDPA Override Silently Defeated FA3 Install (2026-05-28)

## Part 1 — What happened

**Timeline (UTC, 2026-05-28):**

- **02:00 — 04:00**: After ~2 hours of FA install attempts (4 separate OOM
  failures with FA2 root setup.py and FA3 hopper subdir), we located a working
  prebuilt: `varunneal/flash-attention-3` HF kernel for cu130+torch2120.

- **04:30**: FA3 installed cleanly on both nodes:
  - `from flash_attn_interface import flash_attn_func` ✓
  - `importlib.util.find_spec('flash_attn_3')` returns valid spec ✓
  - transformers `is_flash_attn_3_available()` would pass

- **04:35**: Patched yaml `attn_impl: flash_attn → flash_attention_3`, pushed
  to ttcc-rl (commit `e7093ef1`).

- **04:45**: Launched Zuocan's `launch_training_2node.sh` on both nodes.
  16 H100 GPUs activated, memory loaded, training started.

- **04:55**: While monitoring, noticed the actual training command logged by
  ms-swift contains:
  ```
  ... --attn_impl flash_attention_3 ... [other args] ... --attn_impl sdpa
  ```
  The yaml-derived `flash_attention_3` appears early; a second `--attn_impl sdpa`
  appears at the **very end** of the CLI args. Python argparse keeps the LAST
  occurrence, so the model loaded with SDPA, NOT FA3.

- **04:58**: Investigated. Found in `/opt/dlami/nvme/launch_training_2node.sh`
  (Zuocan's launcher) at lines 72-76:
  ```bash
  # NOTE: override --attn_impl sdpa (flash-attn compile pathologically slow on torch2.12+cu130;
  ...
      --attn_impl sdpa \
  ```
  Zuocan had hit the same FA compile OOM during his bootstrap and chose to
  hardcode SDPA as a permanent fallback. The override was undocumented in any
  V8_LAUNCH_RUNBOOK and not surfaced when we coordinated.

- **05:00**: Killed training on both nodes (no actual training steps had
  completed — caught mid-warmup).

**Net cost of incident**:
- ~15 min of GPU compute on 16 H100s ≈ $80 wasted (training warmup that ran on
  SDPA instead of intended FA3)
- ~30 min of investigation time
- Discovered before any checkpoint, so no model-quality impact

## Part 2 — Why it happened (root causes)

### Root cause 1: silent hardcoded override defeats the explicit yaml

Zuocan's launcher adds `--attn_impl sdpa` as a CLI arg AFTER the yaml-derived
args. ms-swift's argument plumbing (`swift sft <yaml> [extra args]`) makes
the LATER value win. The yaml's `attn_impl: flash_attention_3` was visible,
logged, looked authoritative — but was silently overridden.

This is a classic "convenience override" anti-pattern: a fix for yesterday's
problem becomes a silent footgun for today's. The override should have been:
- (a) in the yaml itself (so it shows up where attn_impl is declared), OR
- (b) gated by an env var (`FORCE_SDPA=true bash launch...`), OR
- (c) removed once FA install was solved

None of these were done.

### Root cause 2: assumed yaml = truth, didn't trace the actual CLI

When I edited the yaml from `flash_attn` to `flash_attention_3`, I assumed
the change would propagate. I did not:
- Read the full Zuocan launcher source before launching it
- Verify the live CLI command after launching (it WAS logged in the training
  log right at the top, I just didn't read carefully)
- Add an integration check: "in the training process, what attention class
  is actually being used?"

If I had grep'd the launch_training_2node.sh for `attn_impl` before launching,
I would have seen the hardcoded `sdpa` override immediately.

### Root cause 3: undocumented workaround in a shared launcher

Zuocan added the SDPA override around 2026-05-27 during his bootstrap when he
hit the same FA compile OOM. The launcher is shared infrastructure (used by
both Leon and Zuocan), but the override was:
- Commented inline with `# NOTE: ...` (visible only if you read the file)
- NOT mentioned in his Lark runbook, NOT in V8_LAUNCH_RUNBOOK.md
- NOT communicated when Leon picked up FA install

Workarounds in shared code without a heads-up are tech debt that bites the
next person.

### Root cause 4: no SWE process for "we changed a critical default"

There's no PR review on Zuocan's `launch_training_2node.sh`. It's a script
sitting on the H100 box's NVMe. Edits don't have audit trail. We have no
"hey, I added a sdpa fallback because FA install kept failing" change log.

## Part 3 — What we change going forward

### Process changes (immediate)

1. **All shared launchers must live in the repo, not on individual boxes.**
   Move `launch_training_2node.sh` from `/opt/dlami/nvme/launch_training_2node.sh`
   into `examples/train/grpo/qwen2_5_omni_ttcc/launchers/` and source it via
   `git pull`. Edits become PRs.

2. **Hardcoded overrides MUST be in the yaml or env-var-gated.**
   Update Zuocan's launcher to either:
   - Read `ATTN_IMPL` from env (default to yaml value if unset)
   - Inject via yaml override, not CLI append
   Specific rule: **no `--<flag>` appended after yaml that contradicts the yaml**.

3. **Pre-launch sanity must include "verify effective config".**
   Add a step to `validate_v8_launch.sh`: dump the resolved CLI command,
   diff against yaml, fail if any flag is overridden silently.

4. **Cross-person workarounds get a `WORKAROUND_*.md` note.**
   When Zuocan hits FA compile OOM and decides to fallback, that decision
   needs a 5-line markdown next to the launcher: "Why this fallback exists,
   what would let us remove it, who owns the followup." Without that, the
   next person debugs from scratch.

### Verification (next launch)

Before pressing go on V8 training again:

- [ ] Read `launch_training_2node.sh` end-to-end on both nodes
- [ ] Remove or env-var-gate the `--attn_impl sdpa` override
- [ ] Confirm `--attn_impl` is NOT mentioned twice in the resolved CLI
- [ ] Spot-check the training log within first 30s: it should print
  "Using attn_implementation: flash_attention_3" (or equivalent) BEFORE
  the model loads
- [ ] grep the training log for "flash_attn_3" or "Selected Provider is EFA"
  to confirm FA3 dispatch path is actually used

### Tech-debt items (next iteration)

- Migrate `launch_training_2node.sh` into repo as
  `examples/train/grpo/qwen2_5_omni_ttcc/launchers/multi_node_2.sh`
- Add config-effective-dump to `validate_v8_launch.sh`
- Write a `LAUNCHER_OVERRIDES.md` documenting any non-yaml CLI overrides
- Cross-link the FLASH_ATTENTION_INSTALL.md from this incident so future
  readers know the FA3 install path

## Affected files / commits

- Yaml change that was effectively a no-op (until override removed):
  `e7093ef1` — `V8 yaml: attn_impl flash_attn -> flash_attention_3 for H100`
- FA3 install documentation (still useful):
  `c018b584` — `FLASH_ATTENTION_INSTALL.md`
- Zuocan's launcher with the override: `/opt/dlami/nvme/launch_training_2node.sh`
  on each node (lines 72-76)

## Related incidents

- [INCIDENT_2026-05-26_EVAL_LEAK.md](../../../../ttcc-eval/INCIDENT_2026-05-26_EVAL_LEAK.md) —
  V7 R(t) leak. Different surface, same underlying class: a subtle code
  default silently produced wrong results without surface error.

## Lesson aphorisms

- "An undocumented override is a future incident."
- "Trust the yaml until you've verified the CLI."
- "Workarounds for yesterday's bug become silent footguns for today's."

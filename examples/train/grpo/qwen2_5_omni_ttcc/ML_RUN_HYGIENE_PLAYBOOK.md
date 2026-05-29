# ML Run Hygiene Playbook (web research, 2026-05-29)

> Deep research on best-in-class ML engineering practice, mapped to this project's 6 real failures. 5 parallel Tavily tracks + synthesis.

This is a synthesis-and-writing task grounded in research that's already been done and handed to me. No new searching needed — the research is comprehensive and the 6 failures are concrete. Let me produce the playbook directly.

# ML Run Hygiene Playbook
**For a small ML team. Grounded in primary-source research, mapped to this week's 6 real failures.**

---

## 1. TOP 10 RULES (ordered by leverage)

| # | Rule | Why (one line) | Source |
|---|------|----------------|--------|
| 1 | **Audit every feature for "is this available at prediction time?" — kill label-in-input / proxy leakage.** | A great offline number from a leaked label is worth zero; this is exactly how V7 hit IBS=0.000092 while ignoring the video. | https://www.ibm.com/think/topics/data-leakage-machine-learning · https://reproducible.cs.princeton.edu |
| 2 | **Nets fail silently — build simple→complex, verify loss@init, overfit one batch, change one thing at a time.** | No crash, no stack trace; a too-powerful net fits *around* your bug and trains to a plausible loss while learning the wrong thing. | http://karpathy.github.io/2019/04/25/recipe |
| 3 | **Never write checkpoints only to ephemeral/node-local disk — drain to durable multi-node/object storage.** | Local NVMe vanishes on preempt/crash; a checkpoint that exists only on the dead node is gone (our V8 near-loss). | https://aws.amazon.com/blogs/storage/architecting-scalable-checkpoint-storage-for-large-scale-ml-training-on-aws · https://oneuptime.com/blog/post/2026-02-17-how-to-use-preemptible-vms-with-gpus-for-cost-effective-machine-learning-training/view |
| 4 | **Test the full save→kill→resume loop before the real run; make resume automatic and idempotent.** | A working recovery path turns a 1-hour failure into a 5-minute hiccup; an untested one turns it into a multi-day rebuild. | https://blog.prompt20.com/posts/checkpoint-storage-and-recovery · https://ar5iv.labs.arxiv.org/html/2407.21783 (§3.3.4) |
| 5 | **Alert on the absence of an event — every backup/uploader job must actively prove it's alive (dead man's switch).** | A silently-dead uploader emits no error; it just stops. Invert it: page when the expected heartbeat is missing. | https://updog.watch/learn/what-is-dead-mans-switch · https://oneuptime.com/blog/post/2026-03-02-how-to-monitor-cron-job-execution-and-alerting-on-ubuntu/view |
| 6 | **Evaluate on the real held-out split, against a dumb baseline, touching the test set exactly once.** | A metric on an arbitrary 168-row smoke subset that doesn't beat a naive baseline isn't a result; repeated peeking is a p-hack leak. | http://karpathy.github.io/2019/04/25/recipe · https://scikit-learn.org/stable/common_pitfalls.html |
| 7 | **Make eval match the deployment/inference path exactly — and validate that path runs before trusting any number.** | Offline metrics are fake if the eval code differs from the served path; our intended generate-CoT path was broken and never exercised. | https://developers.google.com/machine-learning/guides/rules-of-ml (Rule #29/#37) · http://dswok.com/General-ML/Training-serving-skew |
| 8 | **Split BEFORE anything else; train the model and any baseline on the *same* train file; fit all preprocessing train-only.** | Different train files for model vs. baseline make the comparison meaningless; fitting on full data leaks test stats (0.91→chance). | https://scikit-learn.org/stable/common_pitfalls.html |
| 9 | **Pin code SHA + data hash + resolved config + seed + env to every reported number; one run ID = one provenance unit.** | A metric you can't trace to exact artifacts isn't trustworthy; ML accumulates silent config/data-dependency debt. | https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf · https://mlflow.org/blog/mlflow-autolog |
| 10 | **Don't trust loss curves alone; monitor grad-norm/NaN + system signals, and meta-monitor (test the monitor on a schedule).** | SDC makes loss "appear well behaved while masking corruption"; an untested monitor that self-matched its own regex gives false silence. | https://arxiv.org/html/2503.20263v1 (L4) · https://grafana.com/blog/how-we-use-metamonitoring-prometheus-servers-to-monitor-all-other-prometheus-servers-at-grafana-labs |

---

## 2. THE 6 FAILURES → PREVENTING PRACTICE + MINIMAL FIX

### Failure 1 — V7 IBS=0.000092 was fake (label-in-input leakage; model ignored video)
- **Best practice that would have prevented it:** Feature-availability audit + Karpathy's **input-independent baseline** test. Zero out / ablate the input — if the model still scores well, the input carries no signal (or there's a leak). (IBM data-leakage; Karpathy recipe.)
- **Minimal fix now:**
  1. Add a CI gate: **zero-ablate the video input** and assert IBS *degrades to chance*. If it doesn't, fail the run.
  2. Add a **permutation/shuffle test** on the suspect label-adjacent feature (`sklearn.inspection.permutation_importance`, `n_repeats≥5`) and a **random "noise" feature** as a control; any feature that shouldn't matter ranking above noise = leak flag.
  3. **Treat any suspiciously-good score (IBS ≈ 0) as a leakage hypothesis to disprove, not a win.** Source: https://www.sciencedirect.com/science/article/pii/S2666389923001599

### Failure 2 — V8 checkpoints not backed up (manual decoupled uploader died silently; ephemeral NVMe only)
- **Best practice:** Tiered durability (never ephemeral-only) **+** dead man's switch on the drain job. (AWS multi-level; dead-man's-switch.)
- **Minimal fix now:**
  1. Replace the decoupled manual uploader with an **in-process async drain** that writes `.tmp` → fsync → atomic `os.replace()` → push to object storage (S3/GCS), so "latest" only advances on a complete durable write.
  2. Wrap whatever upload still runs out-of-band with a **heartbeat ping on success** (`upload.sh && curl -fsS https://hc-ping.com/<uuid>`); an external monitor (Healthchecks.io, free ≤20 checks) pages when the ping is missing within the grace window.
  3. Acceptance check before trusting it: `df -h $(realpath ckpt_dir)` confirms the target is the mounted durable PD, not NVMe.

### Failure 3 — Eval and save on different schedules (eval@50/101/152/203, save@75/150/225) → reported metrics map to no saved checkpoint
- **Best practice:** Reproducibility/provenance — **every reported number must be traceable to a recoverable artifact**; a metric with no checkpoint behind it is unrecoverable by definition.
- **Minimal fix now:** **Couple the two schedules**: eval *only* on save steps (or save *at* every eval step). Single config value: `eval_interval == save_interval` (or `eval_steps ⊆ save_steps`). Then **stamp the checkpoint path/step into the eval log line** so every metric row names the exact `.ckpt` it came from. Source: https://research.google.com/pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf (Infra: reproducible/traceable)

### Failure 4 — Evaluated on arbitrary "first 200" (really 168, partial mirror), not the real 4906-ad val split; B1 baseline from a different train file
- **Best practice:** Eval must match the deployment data; split-by-the-right-key once; **model and baseline trained on the same train file**; fit nothing on a partial mirror. (scikit-learn pitfalls; eval-deployment match.)
- **Minimal fix now:**
  1. Hard-code the canonical val split (full **4906 ads**) by **content hash**, and **assert `len(val)==4906` and the hash matches** at eval start — fail loudly if the local mirror is partial (catches the 168 silently).
  2. **Pin the single train-file hash** and assert the model run and the B1 baseline both consumed *that exact file* (log both hashes; assert equal). Source: https://scikit-learn.org/stable/common_pitfalls.html
  3. Reserve the full val split as touch-once; smoke subsets are for plumbing only and must be **labeled "SMOKE — not a result."**

### Failure 5 — Intended generate-CoT inference path never actually run / broken; leak-free path unvalidated until now
- **Best practice:** **Make eval match the inference path and validate that path runs** (Rules of ML #29/#37: log/served-path skew; Karpathy: end-to-end skeleton before anything else). An eval that never exercises the served path is observational fiction.
- **Minimal fix now:** Add an **end-to-end smoke that runs the real generate-CoT path on 2-3 examples and asserts non-empty, well-formed CoT output** as a *precondition gate* before any metric is computed. Wire it into CI so a broken inference path fails the run instead of silently reporting a number from the wrong path. Source: https://developers.google.com/machine-learning/guides/rules-of-ml

### Failure 6 — Racy status reads; "monitor" self-matched its own regex → false silence
- **Best practice:** **Test the monitor itself on a schedule; meta-monitor with an always-firing watchdog; page on symptom absence.** An untested monitor is worse than none (false confidence). (Grafana meta-monitoring; oneuptime test-the-switch; smoke-test-monitor lesson.)
- **Minimal fix now:**
  1. **Char-by-char check the babysitter regex for self-match** (anchor it; exclude the monitor's own log lines) and confirm it reads the *correct* log path — the two exact bugs that cost 9h20m on V8 day 1.
  2. **Inject a synthetic alert** (write a fake NaN/divergence line) and assert the monitor pages — i.e., deliberately break the chain to prove a human gets paged.
  3. Make status reads atomic: write `status.tmp` → `os.replace()`, never read a file mid-write. Source: https://oneuptime.com/blog/post/2026-02-06-heartbeat-dead-man-switch-opentelemetry-pipeline/view

---

## 3. ONE-PAGE ML RUN HYGIENE CHECKLIST (apply to every run)

**PRE-FLIGHT (before the real run starts — gates, not suggestions)**
- [ ] **Leak gate:** zero-ablate the primary input → assert metric degrades to chance. Permutation test on label-adjacent features + a random noise-feature control.
- [ ] **Data gate:** val split = canonical full set; assert `len==expected` AND content hash matches. No partial-mirror eval.
- [ ] **Baseline gate:** model and baseline consume the *same* train file; assert train-file hashes equal. Baseline beats naive (popularity/majority) before claiming lift.
- [ ] **Inference-path gate:** run the real served path (generate-CoT) on 2-3 examples; assert well-formed output. No metric computed until this passes.
- [ ] **Sanity:** loss@init ≈ theoretical value; overfit a single batch to ~0 loss.
- [ ] **Save→kill→resume smoke:** kill mid-run, auto-resume, confirm loss curve continues and step/data index don't reset.

**CHECKPOINT DURABILITY**
- [ ] Contents = weights + optimizer + LR scheduler + RNG + **data-loader index** + global step.
- [ ] Write path: GPU→pinned-CPU (sync) → async sharded flush; never block GPU on disk.
- [ ] Tiered: NVMe → durable object/shared storage; **never ephemeral-only** (`df -h $(realpath ckpt)` proves the durable mount).
- [ ] Atomic finalize: `.tmp` → fsync → `os.replace()`; "latest" advances only on complete write.
- [ ] Retain rolling N checkpoints (rollback past contamination); GC only after confirmed durable.
- [ ] **Eval and save on the SAME schedule**; every eval row stamped with the exact checkpoint path/step it came from.

**REPRODUCIBILITY / PROVENANCE**
- [ ] Every reported number pins: git SHA + data hash + resolved config + seed + env.
- [ ] Seeds: `random`/`numpy`/`torch`/`cuda_all` + `use_deterministic_algorithms(True)` + `cudnn.benchmark=False` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + DataLoader `generator`/`worker_init_fn`.
- [ ] Report mean ± variance over **≥5 seeds**; never the single best run, never tune the seed.
- [ ] Config single-source (Hydra/OmegaConf); log the fully-resolved config; deps pinned (lock/Docker).

**MONITORING / INCIDENT**
- [ ] Dead man's switch on the checkpoint-drain job (heartbeat on success → external monitor pages on miss).
- [ ] Monitor grad-norm / NaN-Inf / divergence **+ system signals** (GPU util, disk I/O, node heartbeat) — not loss alone.
- [ ] **Test the monitor on a schedule:** inject a synthetic alert, confirm a human gets paged; regex anchored + excludes monitor's own log lines + correct log path.
- [ ] Page on symptoms ("no checkpoint in 6h", "loss diverged"), not causes; prune noisy alerts.
- [ ] Atomic status writes (`.tmp` → `os.replace()`); never read a status file mid-write.
- [ ] On divergence: roll back to last-good checkpoint, lower LR, resume; log every intervention in a run logbook.

---

## 4. THE SINGLE HIGHEST-LEVERAGE CHANGE TO MAKE FIRST

**Institute a mandatory PRE-FLIGHT GATE that every run must pass before any metric is reported — and make the first gate the leak/input-ablation check.**

Rationale: Of the 6 failures, the most expensive by far was **V7 (the flagship model was fake)** — a leaked label produced IBS=0.000092 while the model ignored the video entirely. Durability, schedule-coupling, and monitoring fixes protect *real* runs from losing work; but the leak gate protects you from the catastrophic failure of **shipping a number that was never real in the first place.** Karpathy's input-independent baseline + a zero-ablation assertion is ~20 lines of code and would have caught V7 on day one. The reproducibility-crisis literature is explicit: **treat any suspiciously-good score as a leakage hypothesis to disprove, not a win** (Kapoor & Narayanan, ≥294 affected papers across 17 fields). https://www.sciencedirect.com/science/article/pii/S2666389923001599 · http://karpathy.github.io/2019/04/25/recipe

Concretely, ship one `preflight.py` that runs as a CI gate on every run: (1) input zero-ablation → assert chance-level; (2) val-split length + hash assert; (3) train-file hash equality (model vs. baseline); (4) real inference-path smoke. Four asserts. They would have caught failures 1, 4, 5 outright and surface 3 the moment an eval row can't name its checkpoint.

---

*No files changed; this is a synthesis deliverable. All claims cite the primary/authoritative sources supplied in the research brief (frontier-lab papers, official PyTorch/Google/scikit-learn docs, Karpathy, Sculley et al., Kapoor & Narayanan, Google SRE). Confidence high — every rule is triangulated across the four research syntheses provided.*
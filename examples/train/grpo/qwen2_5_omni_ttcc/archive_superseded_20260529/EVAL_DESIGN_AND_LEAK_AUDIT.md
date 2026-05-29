# V8 Eval Design + Adversarial Leak Audit (workflow output, 2026-05-28)

Source: deep-read workflow over register.py (667L) + eval_ibs.py (414L) + tools/randomization_probe.py, with an adversarial leak-hunt critic. Files fetched + md5-verified to /tmp/ttcc_eval_src.

**Corrected paths** (the workflow agents had no box access and guessed wrong):
- CKPT75 = `/opt/dlami/nvme/v8_eval/ckpt-75` (verified: v19-20260528 with_cot run, step 75, token_acc 0.498)
- VAL = `/opt/dlami/nvme/v8_eval/data/val_present.jsonl` (168 ads; bypass overwrites assistant → empty-assistant is fine)
- TRAIN = `/opt/dlami/nvme/v8_eval/data/ttcc_train_with_cot.jsonl`
- PLUGIN = `examples/custom/qwen2_5_omni_retention/register.py`

---

## VERIFIED DATA FLOW (register.py)

video+audio+text → Qwen2.5-Omni thinker → `forward_pre_hook` on `thinker.lm_head` captures `h_last (B,L,d)` (register.py:173-178, 358-359) → anchor = last token of last `</cot>` span (`_locate_anchor_positions`, :66-90), fallback = last input token if no `</cot>` (:78-79, 88-89) → `h_anchor = h_last[arange(B), anchor_idx]` (:204-206) → `r_pred = head(h_anchor)` → `(B,60)`, hazard: softplus→cumsum→exp (:120-132). **head's ONLY input is the one hidden vector; r_true/r_mask never touch head(), only the loss (:558-564, 628).** Literal-echo leak (R in input) is structurally closed in bypass.

## MINIMAL LEAK-FREE EVAL = cot-bypass (Mode 3)

`--generate-cot --cot-bypass`:
- (a) leak-free: assistant overwritten to literal `<cot></cot>` (eval_ibs.py:102), discards any answer in the span regardless of --strip-assistant
- (b) runs: single `model(**batch)` forward, NOT generate → avoids the `get_rope_index` `second_per_grids None` bug that kills Mode 2
- (c) correct anchor: real in-distribution `</cot>`, empty reasoning → strictly dominates Mode 1 (off-distribution last-token)

generate-cot (actual generation) NOT needed for the first number; the `--generate-cot` flag is only needed to reach `--cot-bypass`.

---

## ADVERSARIAL VERDICT: NOT TRUSTWORTHY AS-IS

The bypass design closes the V7 literal-echo leak, but the critic found the eval still does not PROVE video use. Six items stand between us and a reportable number:

### 🔴 R-2 / R-8 — THE T-SCALAR LEAK (headline)
The user prompt says **"This ad is {T} seconds long"**, and **T = len(R) − 1** = the drop-off time. A model can output a plausible monotone curve keyed purely on T (longer ad ⇒ flatter normalized curve) and **beat B1 without ever using the video**. This is V7's failure class in subtler form: not echoing R, but regressing on a scalar correlated with R's shape. Neither IBS-vs-B1 nor SRCC controls for it.
→ **MUST add a video-ablated-but-T-preserved arm; the model's edge over B1 must COLLAPSE when video is removed but T kept. That collapse (not IBS-beats-B1) is the real proof of video use.**

### 🔴 R-3 — the probe does not bind the eval
`randomization_probe.py` runs WITHOUT `set_mode('train')`; `eval_ibs.py:292` uses `set_mode('train')`. Different template mode → different input_ids → different anchor positions. A probe PASS does not certify the bypass tensor path.
→ **MUST run probe with `set_mode('train')` + assistant overwritten to `<cot></cot>`, matching eval_ibs.py:102 exactly.**

### 🔴 R-4 / R-5 — probe arms too weak
- Zero-ablation (`videos=[]`) is confounded by sequence-length change → passes for a model that merely "notices vision tokens exist." Demote to **non-evidence**, not weak evidence.
- Swap gate is `≥0.3` but the docstring target is `>>0.5`; `between-ad L2 > 0` is the wrong bar (ratio unstable when predictions near-constant — the V7 collapse signature).
→ **MUST: swap gate ≥0.5; require absolute `between` floor tied to true-curve spread; video-swap is the LOAD-BEARING arm.**

### 🟡 R-10 — train/val disjointness (must-confirm)
ad_id-level: already verified 0 overlap (earlier leak audit). **Advertiser-level near-duplicates NOT yet checked** — same advertiser's near-identical creatives in both splits = partial memorization.
→ **SHOULD confirm advertiser-level disjointness.**

### 🔴 R-11 / R-12 — mechanical bugs
- eval_ibs.py:109 (bypass branch) reads `getattr(out,'r_pred',None)` with **no holder fallback** (unlike :159-165, :353-355) → if a wrapper drops the attr, row silently skips → IBS over a biased subset.
- `</cot>` token-id assumption: if `<cot></cot>` doesn't tokenize to a clean matchable `</cot>` id-sequence, anchor silently falls back to last-token with NO error.
→ **MUST add holder fallback at :109; dump `anchor_ids` (logged :374-375) + confirm a non-fallback `</cot>` match on a real `<cot></cot>` row.**

### 🟡 R-7 — SRCC not implemented
No `srcc`/`spearman` anywhere in the code. eval_ibs.py emits only IBS/ΔIBS. Cross-ad SRCC must be computed post-hoc from `--dump-curves`. And R-8: post-hoc SRCC is itself T-gameable → compute **T-stratified** or regress T out.

### 🟡 R-1 — latent footgun
In bypass, r_true/r_mask are still computed + stashed on the holder (register.py:514-518). Not a current leak (head ignores them), but if anyone later "improves" the head to read holder.r_true, the leak silently returns. Clean design: drop `R` from the row before bypass-encode.

---

## MINIMUM BAR TO A REPORTABLE NUMBER (the trust gate)
1. Probe re-run with `set_mode('train')` + `<cot></cot>` assistant (R-3)
2. Swap gate ≥0.5, zero demoted to non-evidence, absolute `between` floor (R-4, R-5)
3. **Video-ablated-T-fixed arm: edge over B1 must collapse** (R-2) ← the real video-use proof
4. Advertiser-level train/val disjointness (R-10)
5. holder.r_pred fallback at :109 + dump anchor_ids confirming non-fallback `</cot>` (R-11, R-12)
6. T-stratified SRCC, or drop SRCC from the gate (R-7, R-8)

Only if ALL hold (and --no-strip-assistant never used) is the bypass IBS reportable. A passing gate + model beating B1 *with the T-control* ⇒ the head genuinely conditions on video; then ΔIBS≈0 ⇒ CoT decorative, ΔIBS>0 ⇒ reasoning adds signal.

All implementation changes go on an ISOLATED copy/branch first (lesson: [[init-git-first]] generalized).

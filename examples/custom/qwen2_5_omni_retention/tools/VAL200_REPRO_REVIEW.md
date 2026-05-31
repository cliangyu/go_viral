# make_v8_from_hf.py — val200 / reproducibility review

Lens: will `--split val --no-cot --limit 200` reproduce the SAME 200 ad_ids as the
trusted `val_200_no_cot.jsonl`, and is row-emission order deterministic?

## Verdict: fix-first (one MAJOR reproducibility gap; train path is fine)

The byte-match VERIFICATION (150/150) only covered ads present in BOTH repos for the
(R, T, `<cot>`) fields. It did NOT verify that the val SELECTION (which 200 ad_ids)
matches. It does not.

## Finding 1 (MAJOR): `--limit 200` will NOT reproduce the trusted 200 ad_ids

Provenance of the trusted file (not from this script):
- `V8_LAUNCH_RUNBOOK.md:172,188` — trusted file is copied from
  `s3://$S3_BUCKET/ttcc_v7_data/val_200_no_cot.jsonl`, i.e. a **V7-pipeline** artifact.
- `HEAD_PG_RL_RUN.md:22` — "`val_200_no_cot.jsonl` is an arbitrary `--limit 200` smoke
  subset (effectively 168 — partial video mirror on the 2-card box)."

So the trusted 200 ad_ids were determined by (a) the V7 `ttcc_train_sft.jsonl` row
order and (b) which `<ad_id>.mp4` files happened to already be mirrored on a specific
AWS box at build time. Neither is reproducible from the public parquet shard order.

This script (`make_v8_from_hf.py:142,166-217`) selects the FIRST 200 rows that survive
filters in `sorted(glob(val-*-of-*.parquet))` -> `iterrows()` order. There is no ad_id
allowlist (`grep` for allow/ad_ids/filter in the file returns nothing). The resulting
200 ad_ids will almost certainly differ from the trusted set.

Impact: fine for monitoring (any 200 leak-free val rows work as a smoke metric), but
NOT valid for exact per-row comparison against historical val_200 numbers (IBS/SRCC
in V8_INTEGRITY_AUDIT.md §B, verify_gap_f.py, verify_crossing.py all key on these ads).

Fix (pick one):
1. Add `--ad-id-allowlist FILE` (one ad_id per line, or read from an existing
   `val_200_no_cot.jsonl`): keep a row only if `ad_id in allowlist`, and stop when all
   allowlisted ads are emitted. This is the only way to reproduce the exact set. The
   teammate extracts the allowlist once via
   `jq -r .ad_id val_200_no_cot.jsonl > val_200.ids`.
2. If exact reproduction is not required, change the docstring (`make_v8_from_hf.py:42-45`)
   to stop implying parity: replace "cap to 200 to mirror val_200_no_cot.jsonl" with
   "emit an arbitrary first-200 val subset for monitoring; this will NOT match the
   historical val_200_no_cot.jsonl ad_ids — use --ad-id-allowlist for that."

## Finding 2 (MINOR): cross-machine order depends on dataset revision / shard layout

Within a fixed local parquet download, emission order IS deterministic:
`sorted(glob)` (line 142) + Arrow's order-preserving `to_pandas()` + ordered
`iterrows()` (line 169) + a no_video drop that reads bytes from the parquet struct
(line 183-184), not the filesystem. So re-running on the same machine is stable, and
`--limit` is not affected by prior `<ad_id>.mp4` mirror state. Good.

The cross-machine caveat: if `liangyuch/ttcc-v0_2_0` is re-sharded or re-ordered upstream,
or a teammate downloads a different revision, the first-200 set shifts. Mitigation:
pin the dataset revision in the `hf download` command in the docstring (line 34-35),
or document that `--limit` output is only stable for a fixed revision. With Finding 1's
allowlist fix, this becomes moot for the val set.

## Not an issue
- Train path: `--limit` is not used for train; full-split emission order does not affect
  which rows exist, only their JSONL line order. Byte-match already covered content.
- Determinism of filters themselves (horizon/normalize_curve/check_R) is pure-functional.

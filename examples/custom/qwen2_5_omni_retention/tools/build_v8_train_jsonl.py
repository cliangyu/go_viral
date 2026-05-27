#!/usr/bin/env python
"""Build the leak-free V8 training JSONL by merging V7 SFT rows with
Gemini-distilled CoT entries.

V7's training data put the ground-truth retention curve `R(t)` directly
into the assistant span (as `{"R": [1.0, 0.84, ...]}`). The hazard head
reading h[last token] trivially echoed it; three audits confirmed the
backbone did not use the video signal.

V8 fixes the leak by REPLACING the assistant span with a Gemini CoT
that contains qualitative reasoning only — no `R` decimals — wrapped in
`<cot>...</cot>` markers. The retention head now anchors at h[last
`</cot>` token] (see register.py); the LM loss applies only over the
CoT span.

Inputs
------
--v7-jsonl   The V7 SFT JSONL — one row per ad, with `messages`,
             `videos`, `audios`, `ad_id`, `T`, `R`. The assistant turn
             is LEAKY (`<cot>(reasoning here)</cot>\\n{"R": [...]}` or
             pure `{"R": [...]}`).

--cot-jsonl  Gemini distillation output — one row per ad. Required
             fields:
                 ad_id : str
                 cot   : str   # raw reasoning text, NO `<cot>` markers,
                               # NO R-value decimals
             Optional fields (preserved if present): reasoning_model,
             distill_version, distill_ts.

--out-jsonl  Output path. One V8 row per ad, schema:
                 messages:   [system, user, assistant]
                             assistant.content == "<cot>{cot_text}</cot>"
                 videos:     unchanged from V7
                 audios:     unchanged from V7
                 ad_id:      unchanged from V7
                 T:          unchanged from V7
                 R:          unchanged from V7  (ground truth — used by
                             the retention head's _data_collator only,
                             never appears in the LM context)

Anti-leak invariants (validated row-by-row, drops rows that violate):
  I1   assistant.content matches r"^<cot>.+</cot>$" (no trailing JSON)
  I10  no R-value decimal (rounded to 2/3/4 dp) appears inside the CoT
  I3   videos[0] == audios[0]
  I4   len(R) == T + 1  and  R[0] == 1.0
  I5   R is monotone non-increasing (within 1e-6 numerical slack)

Usage
-----
    python build_v8_train_jsonl.py \\
        --v7-jsonl  /path/to/ttcc_v7/ttcc_train_sft.jsonl \\
        --cot-jsonl /path/to/ttcc_cot/cot_v6_train.jsonl \\
        --out-jsonl /path/to/ttcc_v8/ttcc_train_with_cot.jsonl

The script is idempotent: re-running with the same inputs produces
byte-identical output (rows are emitted in the order they appear in the
V7 jsonl).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARN: {path}:{i+1}: {e}", file=sys.stderr)
    return rows


def _cot_to_assistant_span(cot_text: str) -> str:
    """Wrap raw CoT in <cot>...</cot> markers. Strips any existing markers
    in case the upstream distillation already emitted them."""
    text = cot_text.strip()
    for marker in ("<cot>", "</cot>", "<COT>", "</COT>"):
        text = text.replace(marker, "")
    return f"<cot>{text.strip()}</cot>"


def _check_no_R_leak(cot_span: str, R: list[float]) -> str | None:
    """Return None if no leak; else a string describing the leak."""
    for r in R[1:]:                          # R[0] == 1.0 always, exclude
        for dp in (2, 3, 4):
            s = f"{r:.{dp}f}"
            if s in cot_span and s not in ("0.00", "0.000", "0.0000",
                                            "1.00", "1.000", "1.0000"):
                return f"R value {s} appears in CoT span"
    return None


def _check_R_well_formed(R: list[float], T: int) -> str | None:
    if len(R) != T + 1:
        return f"len(R)={len(R)} but T+1={T+1}"
    if abs(R[0] - 1.0) > 1e-6:
        return f"R[0]={R[0]} (expected 1.0)"
    for i in range(1, len(R)):
        if R[i] > R[i-1] + 1e-6:
            return f"R not monotone at i={i}: R[i-1]={R[i-1]} R[i]={R[i]}"
        if R[i] < -1e-6 or R[i] > 1.0 + 1e-6:
            return f"R[{i}]={R[i]} outside [0,1]"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-jsonl",  required=True, type=Path)
    ap.add_argument("--cot-jsonl", required=True, type=Path)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--cot-field", default="cot",
                    help="field name for CoT text in --cot-jsonl (default: 'cot')")
    ap.add_argument("--ad-id-field", default="ad_id",
                    help="field name for ad_id in BOTH input jsonls (default: 'ad_id')")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if ANY row drops")
    args = ap.parse_args()

    if not args.v7_jsonl.exists():
        print(f"ERROR: --v7-jsonl not found: {args.v7_jsonl}", file=sys.stderr)
        return 2
    if not args.cot_jsonl.exists():
        print(f"ERROR: --cot-jsonl not found: {args.cot_jsonl}", file=sys.stderr)
        return 2
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading V7 jsonl ... {args.v7_jsonl}")
    v7_rows = _load_jsonl(args.v7_jsonl)
    print(f"  {len(v7_rows)} rows")

    print(f"loading CoT jsonl ... {args.cot_jsonl}")
    cot_rows = _load_jsonl(args.cot_jsonl)
    print(f"  {len(cot_rows)} rows")

    cot_by_ad = {}
    for r in cot_rows:
        ad_id = str(r.get(args.ad_id_field, ""))
        cot   = r.get(args.cot_field, "") or ""
        if ad_id and cot:
            cot_by_ad[ad_id] = cot
    print(f"  {len(cot_by_ad)} unique ad_ids with non-empty CoT")

    drops = {"no_cot_for_ad": 0, "leak_check_failed": 0,
             "R_malformed": 0, "schema_malformed": 0}
    n_written = 0

    with open(args.out_jsonl, "w") as fout:
        for row in v7_rows:
            ad_id = str(row.get(args.ad_id_field, ""))
            if not ad_id:
                drops["schema_malformed"] += 1
                continue
            if ad_id not in cot_by_ad:
                drops["no_cot_for_ad"] += 1
                continue

            messages = row.get("messages", [])
            videos   = row.get("videos", [])
            audios   = row.get("audios", [])
            T        = row.get("T")
            R        = row.get("R")

            if (not isinstance(messages, list) or len(messages) < 3
                    or not videos or not audios
                    or not isinstance(R, list) or T is None):
                drops["schema_malformed"] += 1
                continue

            R_err = _check_R_well_formed(R, int(T))
            if R_err:
                drops["R_malformed"] += 1
                continue

            assistant_span = _cot_to_assistant_span(cot_by_ad[ad_id])
            leak = _check_no_R_leak(assistant_span, R)
            if leak:
                drops["leak_check_failed"] += 1
                continue

            new_messages = list(messages[:-1])           # keep system + user
            new_messages.append({"role": "assistant", "content": assistant_span})

            out_row = {
                "messages": new_messages,
                "videos":   videos,
                "audios":   audios,
                "ad_id":    ad_id,
                "T":        int(T),
                "R":        [float(x) for x in R],
            }
            fout.write(json.dumps(out_row) + "\n")
            n_written += 1

    print()
    print(f"wrote {n_written} rows to {args.out_jsonl}")
    print(f"  drops: {drops}")
    if args.strict and sum(drops.values()) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

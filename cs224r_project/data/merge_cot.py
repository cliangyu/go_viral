"""
Merge Liangyu's CoT distillation into the existing splits.

Reads a CoT manifest (JSONL or JSON dict) keyed by ad_id, and writes
splits/{train,val,test}_with_cot.jsonl by injecting `_meta.cot` and rendering
it inside the assistant message between <cot>...</cot>.

The val and test rows get cot="" — only train uses the distilled chain (per
the milestone, distillation runs over the train split only).

Run:
  python cs224r_project/data/merge_cot.py \
      --cot_manifest cs224r_project/data/cot/ttcc_train_cot.jsonl \
      --splits_dir   cs224r_project/data/splits
"""

import argparse
import json
from pathlib import Path


def load_cot_manifest(path: str) -> dict[str, str]:
    """Accepts:
      - JSONL with `{ad_id, cot}` or `{ad_id, raw}` per line (Liangyu's
        ttcc-cot dataset uses `raw`),
      - JSON dict `{ad_id: <str or {cot|raw: str}>}`.
    """
    p = Path(path)
    text = p.read_text()
    out: dict[str, str] = {}
    if path.endswith(".jsonl"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cot = row.get("cot") or row.get("raw") or ""
            out[str(row["ad_id"])] = cot
    else:
        raw = json.loads(text)
        for k, v in raw.items():
            if isinstance(v, str):
                out[str(k)] = v
            else:
                out[str(k)] = v.get("cot") or v.get("raw") or ""
    return out


def rewrite_split(in_path: Path, out_path: Path, cot_map: dict[str, str],
                  inject_cot: bool):
    n_with, n_without = 0, 0
    with open(in_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            ad_id = str(row["_meta"]["ad_id"])
            cot = cot_map.get(ad_id, "") if inject_cot else ""
            row["_meta"]["cot"] = cot
            # Re-render the assistant message.
            for msg in row["messages"]:
                if msg["role"] == "assistant":
                    msg["content"] = f"<cot>{cot}</cot>"
            f_out.write(json.dumps(row) + "\n")
            if cot:
                n_with += 1
            else:
                n_without += 1
    print(f"  {out_path.name}: {n_with} with CoT, {n_without} without")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot_manifest", required=True,
                    help="JSONL with {ad_id, cot} per line, or JSON {ad_id: cot}")
    ap.add_argument("--splits_dir", default="cs224r_project/data/splits")
    args = ap.parse_args()

    cot_map = load_cot_manifest(args.cot_manifest)
    print(f"loaded CoT for {len(cot_map)} ads")

    splits_dir = Path(args.splits_dir)
    for split, inject in [("train", True), ("val", False), ("test", False)]:
        in_path = splits_dir / f"{split}.jsonl"
        out_path = splits_dir / f"{split}_with_cot.jsonl"
        if not in_path.exists():
            print(f"  skipping {in_path} (not found)")
            continue
        rewrite_split(in_path, out_path, cot_map, inject_cot=inject)


if __name__ == "__main__":
    main()

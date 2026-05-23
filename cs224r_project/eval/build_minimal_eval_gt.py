"""
Convert our test.jsonl (with `_meta.duration_s`, `_meta.retention_curve`,
`_meta.ad_id`) into the schema minimal_eval.py expects: `{ad_id, T, R_true}`
per row.

Run:
  python build_minimal_eval_gt.py \\
      --in_jsonl  /vol/data/splits/test.jsonl \\
      --out_jsonl /vol/data/splits/test_minimal_eval.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.in_jsonl) as f_in, open(args.out_jsonl, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            m = row["_meta"]
            f_out.write(json.dumps({
                "ad_id":  str(m["ad_id"]),
                "T":      int(m["duration_s"]),
                "R_true": [float(x) for x in m["retention_curve"]],
            }) + "\n")
            n += 1
    print(f"wrote {args.out_jsonl}: {n} rows")


if __name__ == "__main__":
    main()

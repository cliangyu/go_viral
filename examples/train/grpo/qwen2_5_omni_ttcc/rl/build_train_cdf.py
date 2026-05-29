#!/usr/bin/env python
"""Build the per-second train-population CDF used by the cross-ad ranking reward.
For each second t in [0, T_MAX], store the SORTED array of all TRAIN ads' R(t).
This is the 'grading curve' that turns a per-ad predicted retention into a
population percentile -> a per-ad reward that encodes the GLOBAL cross-ad ranking.

Train-ONLY (leak-safe). Optionally checks ad_id disjointness vs a val jsonl.

Usage: python build_train_cdf.py <train.jsonl> <out.npz> [val.jsonl]
"""
import json
import sys

import numpy as np

T_MAX = 60
train_path, out_path = sys.argv[1], sys.argv[2]
val_path = sys.argv[3] if len(sys.argv) > 3 else None


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        R = r.get('R') or r.get('R_true')
        if R:
            rows.append({'ad': str(r.get('ad_id')), 'R': [float(x) for x in R]})
    return rows


tr = load(train_path)
print(f'[cdf] train curves: {len(tr)}')
cdf = {}
for t in range(T_MAX + 1):
    vals = [r['R'][t] for r in tr if len(r['R']) > t]
    cdf[t] = np.sort(np.asarray(vals, dtype=np.float64))
np.savez_compressed(out_path, **{f't{t}': cdf[t] for t in range(T_MAX + 1)})
print(f'[cdf] saved {out_path}; per-t counts: {{t:len(cdf[t]) for t in [1,15,30,45,60]}}=',
      {t: len(cdf[t]) for t in [1, 15, 30, 45, 60]})

tr_ids = {r['ad'] for r in tr}
print(f'[cdf] unique train ad_ids: {len(tr_ids)}')
if val_path:
    va = load(val_path)
    va_ids = {r['ad'] for r in va}
    ov = tr_ids & va_ids
    print(f'[cdf] R4 LEAK GUARD: train/val ad_id overlap = {len(ov)} '
          f'({"CLEAN — train-only, no val leak" if not ov else "LEAK!!"})')
    assert not ov, f'train/val ad_id overlap = {len(ov)} -> CDF would leak val info'
print('[cdf] OK')

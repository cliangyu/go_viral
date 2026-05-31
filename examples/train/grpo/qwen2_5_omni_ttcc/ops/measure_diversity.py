#!/usr/bin/env python
"""Localize the diversity bottleneck for CoT-GRPO (evidence before tuning, per the run plan):
is the within-group spread thin because the G CoTs are too SIMILAR (sampling -> raise temperature)
or because diverse CoTs map to the SAME head curve (readout saturated -> mean-pool)?

Reads <out>/v*/logging.jsonl for the within-group reward/curve std trajectory, and (if present)
completions.jsonl for CoT-text diversity. Run on the box:  python ops/measure_diversity.py [OUT_DIR]
"""
import glob
import json
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else '/opt/dlami/nvme/ssm-out/rl_cot_grpo_v1'
dirs = sorted(glob.glob(f'{OUT}/v*/'), reverse=True)
if not dirs:
    print('no run dir'); sys.exit(0)
D = dirs[0]

rows = []
try:
    rows = [json.loads(l) for l in open(f'{D}/logging.jsonl')]
except Exception:
    pass
keys = ['reward_std', 'rewards/TTCCHeadPlaceholder/std', 'cot_curve_std']
print(f'=== within-group spread trajectory ({len(rows)} steps) ===')
for k in keys:
    vals = [r[k] for r in rows if k in r]
    if vals:
        print(f'  {k}: last={vals[-1]:.4f} mean={sum(vals)/len(vals):.4f} '
              f'min={min(vals):.4f} max={max(vals):.4f}')
print('  read: curve_std/reward_std ~0 sustained -> weak gradient. If raising temperature opens')
print('        curve_std -> it was sampling; if CoTs diversify but curve_std stays ~0 -> readout (mean-pool).')

# CoT-text diversity proxy (needs ad_id in completions to group within-ad; swift logs a sample).
try:
    comps = [json.loads(l) for l in open(f'{D}/completions.jsonl')]
    texts = [(c.get('completion') if isinstance(c.get('completion'), str)
              else (c.get('completion', [''])[0] if c.get('completion') else '')) for c in comps]
    uniq = len(set(texts))
    print(f'=== completions.jsonl sample: {len(texts)} logged, {uniq} unique ===')
    print('  NOTE: swift logs only a small completion sample, mostly cross-ad. For a real within-ad')
    print('        diversity number, log all G per ad with ad_id (TODO) or inspect a manual rollout.')
except Exception:
    print('=== no completions.jsonl yet ===')

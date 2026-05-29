#!/usr/bin/env python
"""Build the RL bypass TRAIN dataset: assistant span -> '<cot></cot>' (so the head
reads the same last-token context as the leak-free eval), keeping video + R. Run on
the cluster against the V8 train jsonl. Verifies count + that videos resolve + R present.

Usage: python build_bypass_dataset.py <src_train.jsonl> <dst_bypass.jsonl>
"""
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(src) as f, open(dst, 'w') as o:
    for line in f:
        r = json.loads(line)
        m = r.get('messages', [])
        if m and m[-1].get('role') == 'assistant':
            m[-1]['content'] = '<cot></cot>'
        else:
            m.append({'role': 'assistant', 'content': '<cot></cot>'})
        o.write(json.dumps(r) + '\n')
        n += 1
print(f'[bypass] wrote {n} rows -> {dst}')

# verify
with open(dst) as f:
    r = json.loads(f.readline())
v = r.get('videos') or r.get('video')
v = v[0] if isinstance(v, list) else v
R = r.get('R') or r.get('R_true')
print(f'[bypass] first row: video={v} {"EXISTS" if v and os.path.exists(v) else "MISSING"}; '
      f'R_len={len(R) if R else 0}; assistant={r["messages"][-1]["content"]!r}')
assert v and os.path.exists(v), 'first video missing — check video mount/path'
assert R and len(R) >= 2, 'R curve missing'
print('[bypass] OK')

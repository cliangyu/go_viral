#!/usr/bin/env bash
# Snapshot the CoT-GRPO run's key metrics for diff-based monitoring (run on the box).
# Prints one compact line of the latest-step metrics + checkpoints + a recent CoT sample.
# Usage: bash monitor_cot_grpo.sh [OUT_DIR]   (default rl_cot_grpo_v1)
set -uo pipefail
OUT="${1:-/opt/dlami/nvme/ssm-out/rl_cot_grpo_v1}"
D="$(ls -dt ${OUT}/v*/ 2>/dev/null | head -1)"
echo "=== $(date -u +%H:%M:%SZ)  run=${OUT}  alive=$(pgrep -f train_cot_grpo | wc -l) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -8 | tr '\n' ' '; echo
echo "--- latest step metrics (watch: fail_frac=0, reward_std>0.03, kl stable, grad_norm bounded) ---"
python3 - "${D}/logging.jsonl" <<'PY'
import json, sys
try: rows = [json.loads(l) for l in open(sys.argv[1])]
except Exception: rows = []
if rows:
    r = rows[-1]
    keys = ['global_step/max_steps','reward','reward_std','rewards/TTCCHeadPlaceholder/mean',
            'rewards/TTCCHeadPlaceholder/std','cot_curve_std','cot_head_fail_frac','kl','grad_norm',
            'completions/clipped_ratio','frac_reward_zero_std']
    print('  ' + '  '.join(f"{k.split('/')[-1]}={r[k]}" for k in keys if k in r))
    print(f'  (logged {len(rows)} steps)')
else:
    print('  (no logging.jsonl yet)')
PY
echo "--- checkpoints ---"
ls -d ${D}checkpoint-* 2>/dev/null | sed 's#.*/##' | tr '\n' ' '; echo
echo "--- error/abort scan (tail) ---"
grep -iE "error|traceback|abort|out of memory|STEP-0 WARNING" "${OUT}.log" 2>/dev/null | tail -3 || true
echo "--- 1 recent CoT (coherence eyeball) ---"
python3 -c "
import json
try:
    rows=[json.loads(l) for l in open('${D}/completions.jsonl')]
    c=rows[-1].get('completion','');  c=c[0] if isinstance(c,list) else c
    print('  '+str(c)[:200])
except Exception: print('  (no completions yet)')
" 2>/dev/null

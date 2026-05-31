#!/usr/bin/env python
"""Analyze CoT-GRPO training dynamics from logging.jsonl (= the same metrics W&B shows) and
emit a verdict + evidence-gated recommendation (per COT_GRPO_RUN_PLAN.md). Run on the box each
monitoring cycle:  python ops/analyze_dynamics.py [OUT_DIR]

Reads windowed trends (not just the last step) so the heartbeat sees TRAJECTORY, not a snapshot.
Flags: dead head (fail_frac), thin diversity (reward/curve std -> raise temp), KL runaway,
instability (grad_norm), Goodhart risk (reward up but held-out SRCC must be checked), reward trend.
"""
import glob
import json
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else '/opt/dlami/nvme/ssm-out/rl_cot_grpo_2node_v1'
dirs = sorted(glob.glob(f'{OUT}/v*/'), reverse=True)
if not dirs:
    print('NO RUN DIR'); sys.exit(0)
D = dirs[0]
try:
    rows = [json.loads(l) for l in open(f'{D}/logging.jsonl')]
except Exception as e:
    print(f'no logging.jsonl ({e})'); sys.exit(0)
if not rows:
    print('logging.jsonl empty'); sys.exit(0)


def col(k):
    return [r[k] for r in rows if k in r and isinstance(r[k], (int, float))]


def tail_mean(k, n=10):
    v = col(k)[-n:]
    return sum(v) / len(v) if v else None


def trend(k, n=20):
    """slope sign over last n: + rising, - falling, ~ flat (by first-vs-last-third mean)."""
    v = col(k)[-n:]
    if len(v) < 6:
        return '?'
    a = sum(v[:len(v)//3]) / (len(v)//3); b = sum(v[-len(v)//3:]) / (len(v)//3)
    d = b - a
    scale = (abs(a) + abs(b)) / 2 + 1e-9
    if d > 0.05 * scale:
        return 'UP'
    if d < -0.05 * scale:
        return 'DOWN'
    return 'flat'


step = rows[-1].get('global_step/max_steps', '?')
n = len(rows)
print(f'=== dynamics @ {step}  ({n} logged steps, run {D.split("/")[-2]}) ===')
m = {k: tail_mean(k) for k in ['reward', 'reward_std', 'rewards/TTCCHeadPlaceholder/std',
                               'cot_curve_std', 'cot_head_fail_frac', 'kl', 'grad_norm',
                               'completions/clipped_ratio', 'frac_reward_zero_std']}
print(f"  reward={m['reward']:.4f} ({trend('reward')})   "
      f"within-grp reward_std={m['reward_std']:.4f}   head_std={m['rewards/TTCCHeadPlaceholder/std']:.4f}")
cs = m['cot_curve_std']
print(f"  curve_std={'%.5f' % cs if cs is not None else 'n/a'}   "
      f"fail_frac={m['cot_head_fail_frac']:.3f}   kl={m['kl']:.3f} ({trend('kl')})   "
      f"grad_norm={m['grad_norm']:.3f}   clipped={m['completions/clipped_ratio']:.2f}")

# --- evidence-gated verdict ---
flags = []
if (m['cot_head_fail_frac'] or 0) > 0.3:
    flags.append('!! HEAD-FORWARD FAILING (fail_frac>0.3) -> r_pred not attaching; STOP+investigate')
spread = m['reward_std'] or 0
if spread < 0.03:
    flags.append('THIN DIVERSITY (reward_std<0.03) -> raise temperature 0.4->0.8 (text-channel artifact); if curves still flat -> mean-pool')
elif spread < 0.06:
    flags.append('borderline diversity (reward_std<0.06) -> watch; temp bump candidate')
if trend('kl') == 'UP' and (m['kl'] or 0) > 3.0:
    flags.append('KL RISING + high -> drift/hacking; raise beta or lower lr')
if (m['grad_norm'] or 0) > 8:
    flags.append('grad_norm high (>8) -> instability; lower lr / tighten clip')
if trend('reward') == 'DOWN':
    flags.append('reward trending DOWN -> check stability')
if (m['frac_reward_zero_std'] or 0) > 0.3:
    flags.append('many zero-std groups -> weak gradient on >30% of prompts')

print('  VERDICT: ' + ('HEALTHY, continue' if not flags else 'ACTION:'))
for f in flags:
    print('    - ' + f)
print('  REMINDER: in-loop reward is NOT the metric -> run held-out cross-ad SRCC on checkpoints (Goodhart).')

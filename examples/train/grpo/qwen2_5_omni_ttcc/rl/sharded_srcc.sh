#!/usr/bin/env bash
# sharded_srcc.sh <ckpt_dir> <label> <gpu_csv>
# Shards the val set across the given GPUs (one srcc_eval per GPU on a slice),
# then merges the per-shard npz dumps and recomputes the 200-ad cross-ad SRCC
# (identical algo to srcc_eval.py). Turns a ~60min single-GPU eval into ~N-way.
set -uo pipefail
source /etc/profile.d/leon-env.sh
PY=/opt/dlami/nvme/work/swift_venv/bin/python
VAL=${VAL_OVERRIDE:-/home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl}
REG=/opt/dlami/nvme/go_viral/examples/custom/qwen2_5_omni_retention/register.py
RL=/opt/dlami/nvme/go_viral/examples/train/grpo/qwen2_5_omni_ttcc/rl
ED=/opt/dlami/nvme/v8_eval/lora; mkdir -p "$ED/shards"
CKPT=$1; LBL=$2; GPUS=$3
IFS=',' read -ra GA <<< "$GPUS"; N=${#GA[@]}
cd "$RL"
split -n l/$N --numeric-suffixes=0 -a 2 "$VAL" "$ED/shards/${LBL}_"
pids=()
for i in $(seq 0 $((N-1))); do
  si=$(printf "%02d" "$i"); g=${GA[$i]}
  CUDA_VISIBLE_DEVICES=$g $PY srcc_eval.py --checkpoint "$CKPT" \
    --val-jsonl "$ED/shards/${LBL}_$si" --plugin "$REG" --attn-impl flash_attn \
    --head-type hazard --max-length 32768 \
    --dump-npz "$ED/dump_${LBL}_$si.npz" --output "$ED/srcc_${LBL}_$si.json" \
    > "$ED/eval_${LBL}_$si.out" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
$PY - "$ED" "$LBL" "$RL" <<'PYEOF'
import glob, sys
ed, lbl, rl = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, rl)
from retention_metrics import load_dump, cross_ad_srcc
by = {}
for f in sorted(glob.glob(f"{ed}/dump_{lbl}_*.npz")):
    by.update(load_dump(f))
ads = sorted(by)
srcc, per_t = cross_ad_srcc(by, ads, t_lo=1, t_hi=30)
print(f"[SHARDED-RESULT] {lbl}: n_ads={len(ads)} CROSS_AD_SRCC_t1_30={srcc:.4f}")
print("  per-t:", ", ".join(f"{t}:{per_t[t][0]:.3f}" for t in (1,5,10,15,20,25,30) if t in per_t))
PYEOF

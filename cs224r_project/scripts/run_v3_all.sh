#!/usr/bin/env bash
# v3 launch script: 3 seeds × {SFT-MSE, SFT-Hazard+CoT}, detached on Modal.
#
# v3 hparams (aligned to Liangyu's sft_v2cot_full.sh @ 664f297e, dated 2026-05-23):
#   max_pixels        = 200704
#   nframes           = 60     (was 32 in v2)
#   grad_accum        = 8      (was 16 in v2; effective batch 8)
#   epochs            = 10
#   lr                = 1e-5
#   tuner             = LoRA r=8 (we retain LoRA vs Liangyu's full FT — user choice)
#
# Output dirs on volume:
#   /vol/runs/sft_mse/seed{42,43,44}_v3/
#   /vol/runs/sft_hazard_cot/seed{42,43,44}_a0.1_v3/
#
# Usage:
#   conda activate cs224r-hw3
#   bash cs224r_project/scripts/run_v3_all.sh
set -euo pipefail

export MODAL_HEARTBEAT_TIMEOUT=120

NFRAMES=60
MAX_PIXELS=200704
GRAD_ACCUM=8
EPOCHS=10.0
ALPHA=0.1

for SEED in 42 43 44; do
    echo "===== [v3] SFT-Hazard+CoT seed=${SEED} alpha=${ALPHA} ====="
    modal run --detach cs224r_project/modal/modal_train_sft_hazard_cot.py \
        --seed "${SEED}" --alpha "${ALPHA}" \
        --nframes "${NFRAMES}" --max-pixels "${MAX_PIXELS}" \
        --gradient-accumulation-steps "${GRAD_ACCUM}" \
        --num-train-epochs "${EPOCHS}" \
        --out-subdir "seed${SEED}_a${ALPHA}_v3"
done

for SEED in 42 43 44; do
    echo "===== [v3] SFT-MSE seed=${SEED} ====="
    modal run --detach cs224r_project/modal/modal_train_sft_mse.py \
        --seed "${SEED}" \
        --nframes "${NFRAMES}" --max-pixels "${MAX_PIXELS}" \
        --gradient-accumulation-steps "${GRAD_ACCUM}" \
        --num-train-epochs "${EPOCHS}" \
        --out-subdir "seed${SEED}_v3"
done

echo "All 6 v3 runs launched (detached). Track in Modal dashboard."

#!/usr/bin/env bash
# Unified inference launcher for TTCC retention-curve prediction.
#
# Usage:
#   bash infer.sh <ckpt_path> <method_tag> <out_parquet> [--baseline-parquet PATH]
#
# What's new vs the retired infer_v2cot_full.sh:
#   * USE_AUDIO_IN_VIDEO=true exported (was silently False; train-test mismatch).
#   * Guided JSON decoding via swift's vllm engine — schema enforces R as
#     a list of T_i+1 floats in [0, 1] at the decoder, not post-hoc regex.
#   * Optional B1 paired-comparison via --baseline-parquet flag (was hardcoded).
#   * LoRA / full-FT auto-detect from adapter_config.json (unchanged).
#   * Visual config matches training (FPS=1, FPS_MAX_FRAMES=60, MAX_PIXELS=200704).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_common.sh"

CKPT="${1:?ckpt path required}"
METHOD="${2:?method tag required}"
OUT_PARQ="${3:?out parquet path required}"
shift 3 || true

BASELINE_PARQ=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline-parquet) BASELINE_PARQ="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

TEST_DATA="${TEST_DATA:-${WORK}/data/ttcc_swift/ttcc_test.jsonl}"
if [[ ! -f "${TEST_DATA}" ]]; then
    echo "test JSONL missing: ${TEST_DATA}" >&2
    echo "build it with: python scripts/data/build_ttcc_jsonl.py --split test --out ${TEST_DATA}" >&2
    exit 1
fi

# Auto-detect LoRA adapter vs full-FT checkpoint.
if [[ -f "${CKPT}/adapter_config.json" ]]; then
    MODEL_FLAG=(--model "${MODEL}" --adapters "${CKPT}")
else
    MODEL_FLAG=(--model "${CKPT}")
fi

TMP_JSONL="$(mktemp /tmp/ttcc_infer_XXXX.jsonl)"

# Critical: enable timeline-aligned audio-in-video processing (matches
# Qwen-Omni's intended multimodal mode). Without this, video frames and
# audio stream are processed independently, which is silent train-test
# drift if training uses the same setting (and ours should — _common.sh
# also exports this).
export USE_AUDIO_IN_VIDEO=true

# Talker is not needed for inference; save ~1.5 GB / device.
export ENABLE_AUDIO_OUTPUT=False

# Visual config matches v2cot training recipe.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
FPS_MAX_FRAMES=60 FPS=1.0 \
MAX_PIXELS=200704 VIDEO_MAX_PIXELS=200704 \
VIDEO_MAX_TOKEN_NUM=8192 \
"${VENV}/bin/swift" infer \
    "${MODEL_FLAG[@]}" \
    --infer_backend vllm \
    --val_dataset "${TEST_DATA}" \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --top_p 1.0 \
    --result_path "${TMP_JSONL}" \
    --max_pixels 200704

# Parse generations into predictions parquet. Strict JSON-first parser:
# fails loudly (NaN R_hat) on malformed output so ttcc-eval reports the
# bad rows as N/A rather than letting a permissive regex fabricate a
# clean-looking curve.
TEST_DATA="${TEST_DATA}" TMP_JSONL="${TMP_JSONL}" \
OUT_PARQ="${OUT_PARQ}" METHOD="${METHOD}" \
"${VENV}/bin/python" - <<'PY'
import json
import os
import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TEST_DATA = os.environ['TEST_DATA']
TMP_JSONL = os.environ['TMP_JSONL']
OUT_PARQ  = os.environ['OUT_PARQ']
METHOD    = os.environ['METHOD']


def parse_curve_json_strict(text: str, T: int) -> list | None:
    """JSON-first strict parser. Returns the curve or None on failure.

    Accepts a balanced JSON object containing key "R" mapping to a list
    of numbers. No regex fallback — guided JSON should make malformed
    output rare; when it does happen, we want N/A in the eval rather
    than a silently-padded fake curve.
    """
    cleaned = text.replace('```json', '').replace('```', '')
    start = cleaned.find('{')
    while start != -1:
        depth = 0
        for end in range(start, len(cleaned)):
            ch = cleaned[end]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    blob = cleaned[start:end + 1]
                    try:
                        obj = json.loads(blob)
                    except Exception:
                        break
                    if isinstance(obj, dict) and 'R' in obj and isinstance(obj['R'], list):
                        try:
                            nums = [float(x) for x in obj['R']]
                        except Exception:
                            return None
                        # Truncate / pad to T+1; enforce R(0)=1, monotone, [0,1].
                        if len(nums) < T + 1:
                            nums = nums + [nums[-1] if nums else 0.0] * (T + 1 - len(nums))
                        elif len(nums) > T + 1:
                            nums = nums[:T + 1]
                        nums[0] = 1.0
                        for i in range(1, len(nums)):
                            nums[i] = min(max(nums[i], 0.0), 1.0)
                            if nums[i] > nums[i - 1]:
                                nums[i] = nums[i - 1]
                        return nums
                    break
        start = cleaned.find('{', start + 1)
    return None


test_rows = [json.loads(l) for l in open(TEST_DATA) if l.strip()]
infer_rows = [json.loads(l) for l in open(TMP_JSONL) if l.strip()]
if len(test_rows) != len(infer_rows):
    print(f'WARN: row count mismatch test={len(test_rows)} infer={len(infer_rows)}')

n_pairs = min(len(test_rows), len(infer_rows))
ad_ids, R_hats = [], []
n_no_response = 0
n_bad_parse = 0
for tr, ir in zip(test_rows[:n_pairs], infer_rows[:n_pairs]):
    T = int(tr['T'])
    ad_id = str(tr['ad_id'])
    resp = ir.get('response') or ir.get('completion') or ir.get('output')
    if not resp:
        n_no_response += 1
        ad_ids.append(ad_id)
        R_hats.append([math.nan] * (T + 1))      # explicit N/A — eval will skip
        continue
    R = parse_curve_json_strict(resp, T)
    if R is None:
        n_bad_parse += 1
        ad_ids.append(ad_id)
        R_hats.append([math.nan] * (T + 1))
        continue
    ad_ids.append(ad_id)
    R_hats.append(R)

table = pa.table({
    'ad_id':  pa.array(ad_ids, type=pa.string()),
    'R_hat':  pa.array(R_hats, type=pa.list_(pa.float64())),
    'method': pa.array([METHOD] * len(ad_ids), type=pa.string()),
    'seed':   pa.array([0] * len(ad_ids), type=pa.int64()),
})
pq.write_table(table, OUT_PARQ)
print(f'wrote {len(ad_ids)} predictions to {OUT_PARQ}')
print(f'  no_response={n_no_response}  bad_parse={n_bad_parse}')
PY
rm -f "${TMP_JSONL}"

# ttcc-eval scoring.
HF_HOME="${HF_HOME:-${WORK}/hf-cache}" OUT_PARQ="${OUT_PARQ}" \
WORK="${WORK}" BASELINE_PARQ="${BASELINE_PARQ}" \
"${VENV}/bin/python" - <<'PY'
import json
import os
import sys
os.environ.setdefault('HF_HOME', f'{os.environ["WORK"]}/hf-cache')
sys.path.insert(0, '/home/ssm-user/work/ttcc-eval/src')
from ttcc_eval.eval import evaluate, paired_compare

OUT_PARQ = os.environ['OUT_PARQ']
BASELINE = os.environ.get('BASELINE_PARQ', '')

rep = evaluate(OUT_PARQ, B=10000, seed=0)
report_path = OUT_PARQ.replace('.parquet', '_report.json')
open(report_path, 'w').write(json.dumps(rep.to_dict(), indent=2, default=float))
print(f'report -> {report_path}')
m = rep.metrics
def fmt(d): return f"{d['point']:+.4f}  [{d['lo']:+.4f},{d['hi']:+.4f}]"
print(f'  IBS   = {fmt(m["ibs"])}')
print(f'  slope = {fmt(m["calibration_slope"])}')
print(f'  AUC   = {fmt(m["auc_spearman"])}')

if BASELINE and os.path.exists(BASELINE):
    cmp = paired_compare(BASELINE, OUT_PARQ, B=10000, seed=0)
    d = cmp['ibs']['diff']
    verdict = 'BEATS baseline' if d['point'] < 0 and d['hi'] < 0 else 'no win'
    print(f"  dIBS vs baseline = {d['point']:+.4f}  [{d['lo']:+.4f},{d['hi']:+.4f}]  ({verdict})")
PY

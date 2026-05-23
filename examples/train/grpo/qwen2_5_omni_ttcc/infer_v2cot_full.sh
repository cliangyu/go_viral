#!/usr/bin/env bash
# Eval a checkpoint on the TTCC test split. Visual config matches v2 full-FT
# training: FPS_MAX_FRAMES=60, max_pixels=200704. Auto-detects LoRA vs full-FT.
#
# Usage: bash infer_v2cot_full.sh <ckpt_path> <method_tag> <out_parquet>
set -euo pipefail
CKPT="${1:?ckpt path required}"
METHOD="${2:?method tag required}"
OUT_PARQ="${3:?out parquet path required}"

WORK="${WORK:-/home/ssm-user/work}"
VENV="${VENV:-/opt/dlami/nvme/work/swift_venv}"
TEST_DATA="${TEST_DATA:-${WORK}/data/ttcc_swift/ttcc_test.jsonl}"

if [[ -f "${CKPT}/adapter_config.json" ]]; then
    MODEL_FLAG=(--model /home/ssm-user/work/hf-cache/Qwen2.5-Omni-3B --adapters "${CKPT}")
else
    MODEL_FLAG=(--model "${CKPT}")
fi

TMP_JSONL="$(mktemp /tmp/ttcc_infer_XXXX.jsonl)"

# Disable Talker (speech generation) — saves ~1.5 GB GPU memory per device.
export ENABLE_AUDIO_OUTPUT="False"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
FPS_MAX_FRAMES=60 FPS=1.0 \
MAX_PIXELS=200704 VIDEO_MAX_PIXELS=200704 \
VIDEO_MAX_TOKEN_NUM=8192 \
"${VENV}/bin/python" -m swift.cli.main infer \
    "${MODEL_FLAG[@]}" \
    --infer_backend vllm \
    --val_dataset "${TEST_DATA}" \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --top_p 1.0 \
    --result_path "${TMP_JSONL}" \
    --max_pixels 200704

TEST_DATA="${TEST_DATA}" TMP_JSONL="${TMP_JSONL}" OUT_PARQ="${OUT_PARQ}" METHOD="${METHOD}" \
"${VENV}/bin/python" - <<'PY'
import os, json, re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TEST_DATA = os.environ["TEST_DATA"]
TMP_JSONL = os.environ["TMP_JSONL"]
OUT_PARQ  = os.environ["OUT_PARQ"]
METHOD    = os.environ["METHOD"]

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def parse_curve(text, T):
    cleaned = text.replace("```json", "").replace("```", "")
    nums = None
    start = cleaned.find("{")
    while start != -1 and nums is None:
        depth = 0
        for end in range(start, len(cleaned)):
            ch = cleaned[end]
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = cleaned[start:end+1]
                    try:
                        obj = json.loads(blob)
                    except Exception:
                        break
                    if isinstance(obj, dict) and "R" in obj:
                        try: nums = [float(x) for x in obj["R"]]
                        except Exception: pass
                    break
        start = cleaned.find("{", start+1)
    if nums is None:
        m = re.search(r'(?:"R(?:\(0\))?"|\bR)\s*[:=]\s*\[', cleaned)
        if m:
            tail = cleaned[m.end():]
            end_bracket = tail.find("]")
            body = tail if end_bracket == -1 else tail[:end_bracket]
            extracted = [float(s) for s in NUM_RE.findall(body)]
            if extracted: nums = extracted
    if nums is None: return None
    if len(nums) < T+1: nums = nums + [nums[-1]] * (T+1 - len(nums))
    elif len(nums) > T+1: nums = nums[:T+1]
    nums[0] = 1.0
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]: nums[i] = nums[i-1]
        nums[i] = max(0.0, min(1.0, nums[i]))
    return nums

# Read test rows for ad_id, T, R_true (source of truth)
test_rows = []
with open(TEST_DATA) as f:
    for line in f:
        if line.strip():
            test_rows.append(json.loads(line))

# Read infer outputs (swift preserves test-set order)
infer_rows = []
with open(TMP_JSONL) as f:
    for line in f:
        if line.strip():
            infer_rows.append(json.loads(line))

if len(test_rows) != len(infer_rows):
    print(f"WARN: row count mismatch: test={len(test_rows)} infer={len(infer_rows)}")

ad_ids, R_hats = [], []
n_pairs = min(len(test_rows), len(infer_rows))
n_no_response = 0
n_bad_parse = 0
for tr, ir in zip(test_rows[:n_pairs], infer_rows[:n_pairs]):
    T = int(tr["T"])
    ad_id = str(tr["ad_id"])
    resp = ir.get("response") or ir.get("completion") or ir.get("output")
    if not resp:
        n_no_response += 1; continue
    R = parse_curve(resp, T)
    if R is None:
        n_bad_parse += 1; continue
    ad_ids.append(ad_id)
    R_hats.append(R)

table = pa.table({
    "ad_id":  pa.array(ad_ids, type=pa.string()),
    "R_hat":  pa.array(R_hats, type=pa.list_(pa.float64())),
    "method": pa.array([METHOD]*len(ad_ids), type=pa.string()),
    "seed":   pa.array([0]*len(ad_ids), type=pa.int64()),
})
pq.write_table(table, OUT_PARQ)
print(f"wrote {len(ad_ids)}/{n_pairs} predictions to {OUT_PARQ}")
print(f"  (no_response={n_no_response}, bad_parse={n_bad_parse})")
PY
rm -f "${TMP_JSONL}"

HF_HOME="${HF_HOME:-${WORK}/hf-cache}" OUT_PARQ="${OUT_PARQ}" WORK="${WORK}" \
"${VENV}/bin/python" - <<'PY'
import os, sys, json
os.environ.setdefault("HF_HOME", f"{os.environ['WORK']}/hf-cache")
sys.path.insert(0, "/home/ssm-user/work/ttcc-eval/src")
from ttcc_eval.eval import evaluate, paired_compare

OUT_PARQ = os.environ["OUT_PARQ"]
WORK     = os.environ["WORK"]
rep = evaluate(OUT_PARQ, B=10000, seed=0)
report_path = OUT_PARQ.replace(".parquet", "_report.json")
open(report_path, "w").write(json.dumps(rep.to_dict(), indent=2, default=float))
print(f"report -> {report_path}")
m = rep.metrics
def fmt(d): return f"{d['point']:+.4f}  [{d['lo']:+.4f},{d['hi']:+.4f}]"
print(f"  IBS   = {fmt(m['ibs'])}")
print(f"  slope = {fmt(m['calibration_slope'])}")
print(f"  AUC   = {fmt(m['auc_spearman'])}")
b1 = f"{WORK}/work-out/B1_train_mean.parquet"
if os.path.exists(b1):
    cmp = paired_compare(b1, OUT_PARQ, B=10000, seed=0)
    d = cmp["ibs"]["diff"]
    verdict = 'BEATS B1' if d['point']<0 and d['hi']<0 else 'no win'
    print(f"  dIBS vs B1 = {d['point']:+.4f}  [{d['lo']:+.4f},{d['hi']:+.4f}]  ({verdict})")
PY

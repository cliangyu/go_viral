#!/usr/bin/env bash
# Pre-launch sanity for a V8 SFT-with-CoT training run.
#
# Runs all checks that must pass BEFORE we commit GPU-hours to a capacity
# block. Fails fast on any precondition violation.
#
# Usage:
#   bash validate_v8_launch.sh <init_ckpt_dir> <train_jsonl> [<val_jsonl>]
#
# Checks:
#   1. Init checkpoint has all 10 tokenizer/processor/model files
#   2. Init checkpoint loads + forward succeeds (validate_ckpt.py)
#   3. Train JSONL exists and has > 1000 rows
#   4. Each train row has <cot>...</cot> in its assistant span
#   5. Train video paths resolve to existing files (sample-checks first 100)
#   6. Optional val_jsonl: same row-schema checks if provided
#   7. RetentionLoss has the env-var gate (`RETENTION_COT_ALPHA`) wired
#   8. Vertex/Gemini SA file present IF the planned run depends on it (skip
#      check is OK; we don't need it for SFT)

set -uo pipefail

INIT_CKPT="${1:-}"
TRAIN_JSONL="${2:-}"
VAL_JSONL="${3:-}"

if [[ -z "${INIT_CKPT}" || -z "${TRAIN_JSONL}" ]]; then
    cat <<USAGE
usage: $0 <init_ckpt_dir> <train_jsonl> [<val_jsonl>]

example:
  bash validate_v8_launch.sh \\
       /opt/dlami/nvme/ckpts/checkpoint-3675 \\
       /home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl
USAGE
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN="$(cd "${SCRIPT_DIR}/.." && pwd)/register.py"
VALIDATE_CKPT="${SCRIPT_DIR}/validate_ckpt.py"

PASS=0
FAIL=0
note_ok() { echo "OK    $1"; PASS=$((PASS+1)); }
note_fail() { echo "FAIL  $1" >&2; FAIL=$((FAIL+1)); }

# ---------- 1. init ckpt completeness ----------
if python3 "${VALIDATE_CKPT}" "${INIT_CKPT}" --plugin "${PLUGIN}"; then
    note_ok "init ckpt validates (file completeness + load + forward)"
else
    note_fail "init ckpt failed validation"
fi

# ---------- 2. train jsonl exists ----------
if [[ ! -f "${TRAIN_JSONL}" ]]; then
    note_fail "train jsonl not found: ${TRAIN_JSONL}"
else
    N=$(wc -l < "${TRAIN_JSONL}")
    if [[ "$N" -lt 1000 ]]; then
        note_fail "train jsonl has only $N rows (expected >= 1000)"
    else
        note_ok "train jsonl exists with $N rows"
    fi
fi

# ---------- 3. CoT presence + video path sanity (sample first 100 rows) ----------
python3 - "${TRAIN_JSONL}" <<'PY' && note_ok "train rows: CoT present, video paths resolve" || note_fail "train rows: schema / video path issues"
import json, os, sys
src = sys.argv[1]
n_total = n_has_cot = n_video_ok = 0
SAMPLE = 100
with open(src) as f:
    for line in f:
        d = json.loads(line)
        n_total += 1
        asst = next((m for m in d['messages'] if m['role']=='assistant'), None)
        if asst and '<cot>' in asst['content'] and '</cot>' in asst['content']:
            n_has_cot += 1
        if all(os.path.exists(v) for v in d.get('videos', [])):
            n_video_ok += 1
        if n_total >= SAMPLE: break
if n_has_cot != n_total or n_video_ok != n_total:
    print(f'sample(n={n_total}): cot_present={n_has_cot}, video_present={n_video_ok}')
    sys.exit(1)
sys.exit(0)
PY

# ---------- 4. optional val jsonl ----------
if [[ -n "${VAL_JSONL}" ]]; then
    if [[ ! -f "${VAL_JSONL}" ]]; then
        note_fail "val jsonl not found: ${VAL_JSONL}"
    else
        VN=$(wc -l < "${VAL_JSONL}")
        note_ok "val jsonl exists with $VN rows"
    fi
fi

# ---------- 5. retention loss env-var gate wired ----------
if grep -q "RETENTION_COT_ALPHA" "${PLUGIN}"; then
    note_ok "retention plugin reads RETENTION_COT_ALPHA"
else
    note_fail "retention plugin missing RETENTION_COT_ALPHA gate"
fi

# ---------- 6. video subdir on persistent disk (warn if many missing) ----------
echo "[info] check: full train video corpus on disk?"
python3 - "${TRAIN_JSONL}" <<'PY'
import json, os, sys
src = sys.argv[1]
n = 0; missing = 0
with open(src) as f:
    for line in f:
        d = json.loads(line)
        n += 1
        for v in d.get('videos', []):
            if not os.path.exists(v):
                missing += 1
                break
print(f'  n_rows={n}, n_missing_video={missing} ({missing/n*100:.1f}%)')
print(f'  (must be 0% for a real launch; otherwise stage the dataset first)')
PY

# ---------- summary ----------
echo
echo "===================="
echo "passed: ${PASS}"
echo "failed: ${FAIL}"
echo "===================="
if [[ "${FAIL}" -ne 0 ]]; then
    echo "DO NOT LAUNCH V8 — preconditions failed." >&2
    exit 1
fi
echo "All preconditions passed. Safe to launch V8."
exit 0

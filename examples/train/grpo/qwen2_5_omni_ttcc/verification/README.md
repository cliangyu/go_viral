# Verification Scripts

Reproducible verification for the math claims in [PROPOSAL.md](../PROPOSAL.md) and [VERIFICATION_WALKTHROUGH.md](../VERIFICATION_WALKTHROUGH.md).

## What each script does

| Script | Verifies | Real-data needed? | Runtime |
|---|---|---|---|
| `verify_math_v3.py` | A1 (within-ad Spearman across 200 ads), B1 (register.py `_masked_mse` ≡ IBS), B3' (GRPO reward returns Python floats, head W frozen by absence during GRPO) | YES (val_200_no_cot.jsonl) | ~5s |
| `verify_crossing.py` | Theorem: two strict-monotone-decreasing sequences with no ties → Spearman = 1 regardless of value-space crossing (Round 3) | Partial (Tests 1-3 synthetic, Tests 4-5 need val data) | ~3s |
| `verify_gap_f.py` | Cross-ad signal distribution per second t in val_200 (gap F) | YES | ~2s |

## Data dependency

The val data file `val_200_no_cot.jsonl` (~200 ads × ~60s retention curves, ~200 KB) is **NOT** checked into the repo. It lives on the 2-card AWS box at `/opt/dlami/nvme/v8_eval/data/val_200_no_cot.jsonl`.

To reproduce locally, set the env var:

```bash
export TTCC_VAL_PATH=/path/to/val_200_no_cot.jsonl
```

Or download from the team S3 bucket (path TBD; ask Leon or check `8CARD_WORK_LOG.md` for the bucket URI).

If `TTCC_VAL_PATH` is unset and the default remote path doesn't exist:
- `verify_math_v3.py` and `verify_gap_f.py` exit with a clear error.
- `verify_crossing.py` runs the synthetic tests (1-3) and skips the real-data tests (4-5) with a warning.

## Python environment

These scripts need `torch`, `scipy`, `numpy`. Tested versions (2-card box):

```
torch 2.11.0+cu128
scipy 1.17.1
numpy 2.4.4
```

Lighter versions of torch/scipy/numpy should also work — none of the operations are version-specific.

## How to run on the 2-card box (canonical environment)

```bash
B64=$(base64 -i verify_math_v3.py)
aws ssm send-command \
  --instance-ids i-0821d01c4168eff62 \
  --document-name "AWS-RunShellScript" \
  --parameters "commands=[\"echo '$B64' | base64 -d > /tmp/v3.py && /opt/dlami/nvme/meta_qwen_retention/venv/bin/python /tmp/v3.py 2>&1\"]" \
  --region us-east-1 --profile gpu-box
```

Same pattern for the other two scripts.

## How to run locally

```bash
# Install minimal deps
python3 -m venv .venv
source .venv/bin/activate
pip install torch scipy numpy

# Get the val data
export TTCC_VAL_PATH=/path/to/val_200_no_cot.jsonl

# Run
python verify_math_v3.py | tee verify_math_v3.out
python verify_crossing.py | tee verify_crossing.out
python verify_gap_f.py | tee verify_gap_f.out
```

## Expected outputs

The `.out` files in this directory are the canonical reference outputs from the 2-card box (last run 2026-05-28). When you re-run, your output should match modulo:
- Float printing differences across torch/numpy versions (usually identical to 10 decimal places)
- Test 5 in `verify_crossing.py` is sensitive to the random seed of monotone-prediction generation — the 37/50 PASS count may shift by ±5 with different prediction-RNG state, but the high-level finding (float64 underflow causes prediction-side ties in some strict-truth ads) is robust.

All "key" assertions (Spearman std = 0 across 200 ads, `_masked_mse` ≡ IBS bit-identical, GRPO reward is Python float not Tensor, cross-ad gradient cos sim −0.0116) are deterministic and should reproduce exactly.

## What to do if a verification fails on your machine

1. Check that `TTCC_VAL_PATH` points at a valid `val_200_no_cot.jsonl` (200 lines, each a JSON dict with `R_true: list[float]` and `T: int`).
2. Check torch/numpy versions are not wildly different from the tested set.
3. Compare the failing test's output against the canonical `.out` file in this directory.
4. If the failure is in a "key" assertion (above list), open an issue / ping Leon. Something has changed in `register.py` or `ttcc_ibs_plugin.py` that breaks an invariant.

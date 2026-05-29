"""
Gap F: cross-ad truth tie distribution per second t.
For each t in 0..T_max, characterize:
  - std(R_i(t) across i=1..N) -- spread
  - number of unique R values -- discreteness
  - fraction of ads with R_i(t) == R_i(t-1) -- tail saturation
Determines which t-range has discriminative cross-ad signal for SRCC.
"""
import os, sys, json
import numpy as np

VAL_PATH = os.environ.get(
    "TTCC_VAL_PATH",
    "/opt/dlami/nvme/v8_eval/data/val_200_no_cot.jsonl",
)
if not os.path.exists(VAL_PATH):
    sys.exit(
        f"ERROR: val data not found at {VAL_PATH}\n"
        "Set TTCC_VAL_PATH env var to point at val_200_no_cot.jsonl, or run on the 2-card box "
        "where it lives at /opt/dlami/nvme/v8_eval/data/. See verification/README.md."
    )
rows = []
with open(VAL_PATH) as f:
    for line in f:
        r = json.loads(line)
        R = r.get("R_true") or r.get("R") or r.get("retention")
        T_ad = r.get("T") or (len(R) - 1 if R else None)
        rows.append({"R": np.asarray(R, dtype=np.float64), "T": int(T_ad)})

T_max = max(r["T"] for r in rows)
N = len(rows)
print(f"N ads = {N}, T_max = {T_max}")

# Pad with NaN; at each t, R_i(t) is valid only if t <= T_i
padded = np.full((N, T_max + 1), np.nan)
for i, r in enumerate(rows):
    padded[i, :r["T"] + 1] = r["R"]

print("\nPer-t cross-ad statistics (N ads with valid R(t) | std | n_unique | min | p25 | p50 | p75 | max)")
print("-" * 100)
print(f"{'t':>3} {'n_valid':>8} {'std':>8} {'n_uniq':>7} {'min':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'max':>8}")
for t in range(T_max + 1):
    col = padded[:, t]
    valid = col[~np.isnan(col)]
    if len(valid) < 5:
        continue
    std = valid.std()
    n_uniq = len(np.unique(valid))
    q = np.quantile(valid, [0.25, 0.5, 0.75])
    if t <= 5 or t % 5 == 0 or t == T_max:
        print(f"{t:>3} {len(valid):>8} {std:>8.4f} {n_uniq:>7} {valid.min():>8.4f} {q[0]:>8.4f} {q[1]:>8.4f} {q[2]:>8.4f} {valid.max():>8.4f}")

# Discriminative t-range
print("\n=== Discriminative t-range analysis ===")
print(f"{'t':>3} {'std':>8} {'effective_n_ranks':>17} {'tail_saturation_frac':>20}")
for t in range(0, T_max + 1, 5):
    col = padded[:, t]
    valid = col[~np.isnan(col)]
    if len(valid) < 5:
        continue
    # effective n_ranks = unique values among valid
    n_uniq = len(np.unique(valid))
    # tail saturation: fraction of ads where R_i(t) ≈ R_i(t-1) (within 0.001)
    if t > 0:
        prev = padded[:, t-1]
        both_valid = ~np.isnan(col) & ~np.isnan(prev)
        if both_valid.sum() > 0:
            sat_frac = np.mean(np.abs(col[both_valid] - prev[both_valid]) < 0.001)
        else:
            sat_frac = np.nan
    else:
        sat_frac = 0.0
    print(f"{t:>3} {valid.std():>8.4f} {n_uniq:>17} {sat_frac:>20.4f}")

# Recommendation
print("\n=== Recommendation ===")
print("Useful t-range for cross-ad SRCC reward = positions where std(R) > 0.05 AND effective_n_ranks > 20")
useful_ts = []
for t in range(0, T_max + 1):
    col = padded[:, t]
    valid = col[~np.isnan(col)]
    if len(valid) < 20:
        continue
    if valid.std() > 0.05 and len(np.unique(valid)) > 20:
        useful_ts.append(t)
print(f"Useful t-range: {useful_ts[:5]}...{useful_ts[-5:]}  (total {len(useful_ts)} seconds)")
print(f"Useful range t in [{min(useful_ts) if useful_ts else 'N/A'}, {max(useful_ts) if useful_ts else 'N/A'}]")

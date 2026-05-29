"""
Validate: two strictly monotone-decreasing sequences with no ties
=> Spearman = 1 exactly, even when their value curves cross.

Plus: characterize the boundary (ties, non-monotone) where this fails.
"""
import os, sys
import numpy as np
from scipy.stats import spearmanr
import json

np.random.seed(0)
T = 60

VAL_PATH = os.environ.get(
    "TTCC_VAL_PATH",
    "/opt/dlami/nvme/v8_eval/data/val_200_no_cot.jsonl",
)
RUN_REAL_DATA_TESTS = os.path.exists(VAL_PATH)
if not RUN_REAL_DATA_TESTS:
    print(f"WARN: val data not found at {VAL_PATH} — Tests 4 & 5 (real-data) will be skipped.")
    print("      Synthetic tests 1, 2, 3 will still run. Set TTCC_VAL_PATH to enable real-data tests.")

# ====== Test 1: explicit crossing example ======
print("="*70)
print("TEST 1: Explicit crossing of two strictly-decreasing curves")
print("="*70)
# Linear decay vs exponential decay -- they CROSS in value space
A = np.linspace(1.0, 0.0, T+1)             # linear from 1 down to 0
B = np.exp(-np.linspace(0, 5, T+1))        # exp from 1 down to ~0.007
print(f"  A: linear 1 -> 0           |  A[0]={A[0]:.4f}, A[T/2]={A[T//2]:.4f}, A[T]={A[T]:.4f}")
print(f"  B: exp(-5x), 1 -> 0.007    |  B[0]={B[0]:.4f}, B[T/2]={B[T//2]:.4f}, B[T]={B[T]:.4f}")
sign_changes = int(np.sum(np.diff(np.sign(A - B)) != 0))
print(f"  Sign changes in (A-B)      = {sign_changes}  (>=1 confirms they cross)")
print(f"  A strictly decreasing      = {bool((np.diff(A) < 0).all())}")
print(f"  B strictly decreasing      = {bool((np.diff(B) < 0).all())}")
rho, _ = spearmanr(A, B)
print(f"  Spearman(A, B)             = {rho}")
print(f"  RESULT                     = {'PASS (=1.0)' if abs(rho - 1.0) < 1e-12 else 'FAIL'}")

# ====== Test 2: 1000 random pairs of strict-monotone curves, all should cross various ways ======
print()
print("="*70)
print("TEST 2: 1000 random pairs of strictly-decreasing curves")
print("="*70)
n_trials = 1000
n_strict = 0
n_crossing = 0
rhos = []
for k in range(n_trials):
    lam_A = np.random.exponential(0.05, T) + 1e-6
    lam_B = np.random.exponential(0.15, T) + 1e-6
    A = np.concatenate([[1.0], np.exp(-np.cumsum(lam_A))])
    B = np.concatenate([[1.0], np.exp(-np.cumsum(lam_B))])
    if not ((np.diff(A) < 0).all() and (np.diff(B) < 0).all()):
        continue
    n_strict += 1
    # crossing? -- sign changes in (A-B)
    sc = int(np.sum(np.diff(np.sign(A - B)) != 0))
    if sc >= 1:
        n_crossing += 1
    rho, _ = spearmanr(A, B)
    rhos.append(rho)
rhos = np.array(rhos)
print(f"  Strict-monotone pairs      = {n_strict} / {n_trials}")
print(f"  Pairs that cross           = {n_crossing} / {n_strict} ({100*n_crossing/max(n_strict,1):.1f}%)")
print(f"  Spearman(A, B) statistics  : mean={rhos.mean():.15f}, std={rhos.std():.2e}, min={rhos.min()}, max={rhos.max()}")
print(f"  RESULT                     = {'PASS (all =1.0 exactly)' if (rhos == 1.0).all() else 'FAIL'}")

# ====== Test 3: ties break it (the boundary) ======
print()
print("="*70)
print("TEST 3: Boundary case -- ties break the identity")
print("="*70)
# A has plateau (tie) at positions 1,2,3
A_tie = np.array([1.0, 0.5, 0.5, 0.5, 0.2, 0.1])
B_strict = np.array([1.0, 0.7, 0.4, 0.3, 0.2, 0.05])
print(f"  A (with tied plateau)      = {A_tie}")
print(f"  B (strict)                 = {B_strict}")
print(f"  A strict-monotone?         = {(np.diff(A_tie) < 0).all()}  ('<' fails on the tie)")
print(f"  A monotone non-increasing? = {(np.diff(A_tie) <= 0).all()}")
rho_tie, _ = spearmanr(A_tie, B_strict)
print(f"  Spearman(A_tie, B_strict)  = {rho_tie:.6f}")
print(f"  RESULT                     = {'< 1.0 as predicted' if rho_tie < 1.0 else 'unexpectedly = 1'}")

# Compare with A and B both having tie at SAME positions
A_tie_same = np.array([1.0, 0.5, 0.5, 0.5, 0.2, 0.1])
B_tie_same = np.array([1.0, 0.8, 0.8, 0.8, 0.2, 0.05])
print()
print(f"  A (tied at 1,2,3)          = {A_tie_same}")
print(f"  B (tied at 1,2,3)          = {B_tie_same}")
rho_match, _ = spearmanr(A_tie_same, B_tie_same)
print(f"  Spearman(A, B)             = {rho_match:.6f}")
print(f"  RESULT                     = {'= 1.0' if abs(rho_match - 1.0) < 1e-9 else 'NOT 1.0'} (ties coincide -> ranks coincide -> rho=1)")

# ====== Test 4: real val data -- which val ads have strictly monotone-decreasing truth? ======
print()
print("="*70)
print("TEST 4: val_200 -- which ads have STRICTLY-decreasing truth?")
print("="*70)
if not RUN_REAL_DATA_TESTS:
    print("\nSkipping Tests 4 & 5 (real data unavailable). Run with TTCC_VAL_PATH set to enable.")
    sys.exit(0)
rows = []
with open(VAL_PATH) as f:
    for line in f:
        r = json.loads(line)
        R = r.get("R_true") or r.get("R") or r.get("retention")
        T_ad = r.get("T") or (len(R) - 1 if R else None)
        rows.append({"R": np.asarray(R, dtype=np.float64), "T": int(T_ad)})

n_strict_truth = 0
n_with_ties = 0
n_unique_distribution = []
for r in rows:
    truth = r["R"][:r["T"]+1]
    if (np.diff(truth) < 0).all():
        n_strict_truth += 1
    elif (np.diff(truth) <= 0).all():
        n_with_ties += 1
    n_unique_distribution.append(len(np.unique(truth)))
n_unique_distribution = np.array(n_unique_distribution)
print(f"  Total val ads              = {len(rows)}")
print(f"  Strictly-decreasing truth  = {n_strict_truth} / {len(rows)} ({100*n_strict_truth/len(rows):.1f}%)")
print(f"  Monotone non-inc w/ ties   = {n_with_ties} / {len(rows)} ({100*n_with_ties/len(rows):.1f}%)")
print(f"  Other (has increase)       = {len(rows) - n_strict_truth - n_with_ties}")
print(f"  n_unique values per ad     : mean={n_unique_distribution.mean():.1f}, median={np.median(n_unique_distribution):.0f}, min={n_unique_distribution.min()}, max={n_unique_distribution.max()}")
print()
print("  Histogram of n_unique values:")
bins = [1, 2, 5, 10, 20, 30, 50, 100]
for i in range(len(bins)-1):
    n = np.sum((n_unique_distribution >= bins[i]) & (n_unique_distribution < bins[i+1]))
    print(f"    {bins[i]:3d} <= n_unique < {bins[i+1]:3d}  : {'#'*int(60*n/len(rows))} {n}")
n = np.sum(n_unique_distribution >= bins[-1])
print(f"    n_unique >= {bins[-1]:3d}            : {'#'*int(60*n/len(rows))} {n}")

# For strictly-decreasing-truth ads only: confirm Spearman is exactly 1 for all monotone preds
print()
print("="*70)
print("TEST 5: On strictly-decreasing-truth val ads, ALL monotone preds give Spearman = 1?")
print("="*70)
strict_ads = [r for r in rows if (np.diff(r["R"][:r["T"]+1]) < 0).all()]
print(f"  Strict-truth ads available = {len(strict_ads)}")
N_PREDS = 100
results_per_ad = []
for ad in strict_ads[:50]:  # test 50 strict-truth ads
    truth = ad["R"][:ad["T"]+1]
    rhos = []
    for k in range(N_PREDS):
        lam = np.random.exponential(0.1, len(truth)-1) + 1e-6
        pred = np.concatenate([[1.0], np.exp(-np.cumsum(lam))])
        rho, _ = spearmanr(pred, truth)
        rhos.append(rho)
    rhos = np.array(rhos)
    results_per_ad.append({"all_one": bool((rhos == 1.0).all()),
                           "mean": rhos.mean(), "std": rhos.std()})
all_one_count = sum(1 for r in results_per_ad if r["all_one"])
print(f"  Tested ads with strict truth = {len(results_per_ad)}")
print(f"  ALL N={N_PREDS} preds give rho=1.0 exactly : {all_one_count} / {len(results_per_ad)} ({100*all_one_count/max(len(results_per_ad),1):.1f}%)")
print(f"  RESULT = {'PASS' if all_one_count == len(results_per_ad) else 'FAIL'}")

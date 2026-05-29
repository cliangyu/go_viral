"""Unit tests for xad_reward.py — Gate 1 (no GPU). Validates reward LOGIC,
especially the make-or-break property B': R_rank's within-group spread tracks
RANK-dimension variation, not magnitude. Run: python test_xad_reward.py
"""
import json
import numpy as np
import cross_ad_reward as X

rng = np.random.default_rng(0)
T_MAX = 60

# ---- build a synthetic population CDF (50 diverse ads) ----
def make_curve(decay, T=60):
    lam = np.full(T, decay)
    R = np.concatenate([[1.0], np.exp(-np.cumsum(lam))])
    return R.tolist()

pop = [make_curve(d) for d in rng.uniform(0.02, 0.5, 50)]
cdf = X.build_cdf(pop, T_MAX)

def curve_text(R):
    return '{"R": ' + json.dumps([round(float(x), 4) for x in R]) + '}'

fails = []
def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond: fails.append(name)

print("=== T1 parsing ===")
T = 10
truth = make_curve(0.2, T)
check("json parse", X.parse_curve(curve_text(truth), T) is not None)
check("R=[...] parse", X.parse_curve("Final R = [1.0, 0.8, 0.6]", T) is not None)
check("parse fail -> None", X.parse_curve("no curve here", T) is None)
check("r_fmt parse-fail = 0", X.r_fmt("garbage", T) == 0.0)
check("r_fmt ok = 1", X.r_fmt(curve_text(truth), T) == 1.0)

print("=== T2 perfect vs wrong ===")
ad = make_curve(0.15, 30)
rk_perfect = X.r_rank(ad, ad, cdf)
ac_perfect = X.r_acc(ad, ad, 30)
check("perfect r_rank ~1", rk_perfect > 0.98, f"(={rk_perfect:.3f})")
check("perfect r_acc ~1", ac_perfect > 0.999, f"(={ac_perfect:.3f})")
# a wrong prediction: predict a very different decay -> wrong percentile placement
wrong = make_curve(0.45, 30)
rk_wrong = X.r_rank(wrong, ad, cdf)
check("wrong r_rank < perfect", rk_wrong < rk_perfect, f"(wrong={rk_wrong:.3f} < perfect={rk_perfect:.3f})")

print("=== T3 lazy-median does NOT score high on r_rank ===")
# constant population-mean-ish curve for an ad that is actually an outlier (very slow decay)
outlier = make_curve(0.03, 30)       # slow decay -> high retention -> high percentile
median_pred = make_curve(0.25, 30)   # predict the population-typical curve
rk_lazy = X.r_rank(median_pred, outlier, cdf)
rk_true = X.r_rank(outlier, outlier, cdf)
check("lazy-median r_rank << true for an outlier", rk_lazy < rk_true - 0.1,
      f"(lazy={rk_lazy:.3f} vs true={rk_true:.3f})")

print("=== T4 (CORE / B') within-group spread is in the RANK dimension ===")
ad = make_curve(0.12, 30)
# (a) RANK-diverse rollouts: predict a range of decays -> place ad at different percentiles
rank_diverse = [make_curve(d, 30) for d in np.linspace(0.03, 0.45, 10)]
rr_rank_div = np.array([X.r_rank(p, ad, cdf) for p in rank_diverse])
# (b) RANK-consistent, magnitude-only-different: tiny multiplicative noise that keeps ordering
base = np.array(ad)
mag_only = []
for k in range(10):
    p = np.clip(base * (1.0 + rng.normal(0, 0.01, len(base))), 0, 1)
    p[0] = 1.0
    for i in range(1, len(p)): p[i] = min(p[i], p[i-1])
    mag_only.append(p.tolist())
rr_mag_only = np.array([X.r_rank(p, ad, cdf) for p in mag_only])
print(f"    rank-diverse  r_rank: mean={rr_rank_div.mean():.3f} std={rr_rank_div.std():.4f}")
print(f"    magnitude-only r_rank: mean={rr_mag_only.mean():.3f} std={rr_mag_only.std():.4f}")
check("rank-diverse rollouts -> r_rank has real spread (signal exists)", rr_rank_div.std() > 0.05,
      f"(std={rr_rank_div.std():.4f})")
check("rank-consistent rollouts -> r_rank spread ~0 (r_rank is rank-specific, not magnitude)",
      rr_mag_only.std() < rr_rank_div.std() / 3,
      f"(mag std={rr_mag_only.std():.4f} << rank std={rr_rank_div.std():.4f})")

print("=== T5 composite handles parse-fail gracefully ===")
truth30 = make_curve(0.15, 30)
c_ok = X.composite(curve_text(truth30), truth30, cdf, beta=1.0, gamma=0.1)
c_fail = X.composite("totally malformed", truth30, cdf, beta=1.0, gamma=0.1)
check("composite ok > composite parse-fail", c_ok > c_fail, f"(ok={c_ok:.3f} fail={c_fail:.3f})")
check("composite parse-fail == 0", abs(c_fail) < 1e-9, f"(={c_fail:.3f})")

print()
if fails:
    print(f"==== {len(fails)} TEST(S) FAILED: {fails} ====")
    raise SystemExit(1)
print("==== ALL TESTS PASSED ====")

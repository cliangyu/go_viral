"""
Gap-closing verification (standalone — no swift import).
A1: within-ad Spearman across all val ads
B1: RetentionHead + _masked_mse from register.py (inlined verbatim)
B3': TTCCIBSReward returns Python floats (inlined verbatim from plugin)
"""
import os, sys, json, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

torch.manual_seed(0)
np.random.seed(0)
T_MAX = 90  # match register.py default

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

# ===== INLINED from register.py:_masked_mse (lines 558-564) =====
def _masked_mse(r_pred, r_true, r_mask):
    diff = (r_pred - torch.nan_to_num(r_true, nan=0.0)) ** 2
    denom = r_mask.float().sum(dim=1).clamp(min=1.0)
    per_ad = (diff * r_mask.float()).sum(dim=1) / denom
    return per_ad.mean()

# ===== INLINED from register.py:RetentionHead (lines 95-132) =====
class RetentionHead(nn.Module):
    def __init__(self, hidden_size, head_type='hazard', t_max=T_MAX):
        super().__init__()
        self.head_type = head_type
        self.t_max = t_max
        self.linear = nn.Linear(hidden_size, t_max, dtype=torch.float32)
        if head_type == 'hazard':
            nn.init.constant_(self.linear.bias, -3.0)
    def forward(self, h):
        w_dtype = self.linear.weight.dtype
        z = self.linear(h.to(w_dtype)).float()
        if self.head_type == 'hazard':
            lam = F.softplus(z)
            return torch.exp(-torch.cumsum(lam, dim=-1))
        return torch.sigmoid(z)

# ===== INLINED from ttcc_ibs_plugin.py:_parse_curve + TTCCIBSReward =====
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def _parse_curve(text, T):
    cleaned = text
    for marker in ("```json", "```"):
        cleaned = cleaned.replace(marker, "")
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
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "R" in obj and isinstance(obj["R"], list):
                        try: nums = [float(x) for x in obj["R"]]
                        except: pass
                    break
        start = cleaned.find("{", start+1)
    if nums is None:
        m = re.search(r'(?:"R(?:\(0\))?"|\bR)\s*[:=]\s*\[', cleaned)
        if m is not None:
            tail = cleaned[m.end():]
            end_bracket = tail.find("]")
            body = tail if end_bracket == -1 else tail[:end_bracket]
            extracted = [float(s) for s in _NUM_RE.findall(body)]
            if extracted: nums = extracted
    if nums is None: return None
    if len(nums) < T+1:
        nums = nums + [nums[-1]] * (T+1 - len(nums))
    elif len(nums) > T+1:
        nums = nums[:T+1]
    nums[0] = 1.0
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]: nums[i] = nums[i-1]
        nums[i] = max(0.0, min(1.0, nums[i]))
    return nums

def ttcc_ibs_reward(completions, R_true_batch, T_batch):
    rewards = []
    for i, completion in enumerate(completions):
        R_true = R_true_batch[i]
        T = int(T_batch[i])
        R_hat = _parse_curve(completion, T)
        if R_hat is None:
            rewards.append(0.0); continue
        r_true = np.asarray(R_true, dtype=np.float64)
        r_hat = np.asarray(R_hat, dtype=np.float64)
        L = min(len(r_true), len(r_hat))
        ibs = float(np.mean((r_hat[:L] - r_true[:L]) ** 2))
        rewards.append(max(0.0, 1.0 - ibs))
    return rewards

# -------- load val data --------
rows = []
with open(VAL_PATH) as f:
    for line in f:
        r = json.loads(line)
        R = r.get("R_true") or r.get("R") or r.get("retention")
        T = r.get("T") or (len(R) - 1 if R else None)
        if R is None or T is None: continue
        rows.append({"R": np.asarray(R, dtype=np.float64), "T": int(T),
                     "ad_id": r.get("ad_id") or r.get("id")})
print(f"[load] {len(rows)} val rows")
if rows:
    r = rows[0]
    print(f"[load] ad0: T={r['T']}, R[:5]={r['R'][:5]}, R[-5:]={r['R'][-5:]}, n_unique={len(np.unique(r['R'][:r['T']+1]))}")

# ========== A1: within-ad Spearman across ALL val ads ==========
print("\n========== A1: within-ad Spearman across all val ads ==========")
N_PREDS = 200
results = []
for ad_i, r in enumerate(rows):
    truth = r["R"][:r["T"]+1]
    if len(truth) < 3: continue
    rhos = []
    for k in range(N_PREDS):
        lam = np.random.exponential(0.05, size=len(truth)-1)
        pred = np.concatenate([[1.0], np.exp(-np.cumsum(lam))])
        rho, _ = spearmanr(pred, truth)
        if np.isnan(rho): rho = 1.0
        rhos.append(rho)
    rhos = np.array(rhos)
    results.append({"ad_idx": ad_i, "T": r["T"],
                    "n_unique_truth": len(np.unique(truth)),
                    "mean_rho": rhos.mean(),
                    "std_rho": rhos.std(),
                    "min_rho": rhos.min(),
                    "max_rho": rhos.max()})

stds = np.array([x["std_rho"] for x in results])
means = np.array([x["mean_rho"] for x in results])
mins = np.array([x["min_rho"] for x in results])
print(f"  n_ads sampled        = {len(results)} (preds per ad = {N_PREDS})")
print(f"  std(rho) over preds : mean={stds.mean():.6f}  max={stds.max():.6f}  p95={np.percentile(stds, 95):.6f}")
print(f"  mean(rho)           : mean={means.mean():.6f}  min={means.min():.6f}  p05={np.percentile(means, 5):.6f}")
print(f"  min(rho) per ad      : mean={mins.mean():.6f}  min over all ads={mins.min():.6f}")
trivial = (stds < 1e-6).sum()
near_trivial = (stds < 1e-3).sum()
nontrivial = (stds >= 1e-3).sum()
print(f"  n_ads std == 0       = {trivial} / {len(results)} ({100*trivial/len(results):.1f}%)")
print(f"  n_ads std < 1e-3     = {near_trivial} / {len(results)} ({100*near_trivial/len(results):.1f}%)")
print(f"  n_ads std >= 1e-3    = {nontrivial} / {len(results)} ({100*nontrivial/len(results):.1f}%)")
print("  Top 5 ads by std (where Spearman discriminates):")
results_sorted = sorted(results, key=lambda x: -x["std_rho"])[:5]
for r in results_sorted:
    print(f"    ad_idx={r['ad_idx']:4d} T={r['T']} uniq={r['n_unique_truth']:3d} mean={r['mean_rho']:.4f} std={r['std_rho']:.6f} min={r['min_rho']:.4f}")
print("  Top 5 ads by lowest mean_rho:")
results_low = sorted(results, key=lambda x: x["mean_rho"])[:5]
for r in results_low:
    print(f"    ad_idx={r['ad_idx']:4d} T={r['T']} uniq={r['n_unique_truth']:3d} mean={r['mean_rho']:.4f} std={r['std_rho']:.6f} min={r['min_rho']:.4f}")

# ========== B1: register.py _masked_mse formula confirmation ==========
print("\n========== B1: register.py _masked_mse formula ==========")
T = rows[0]["T"]
H = 64
head = RetentionHead(hidden_size=H, head_type="hazard", t_max=T_MAX).double()
h = torch.randn(1, H, dtype=torch.float64, requires_grad=True)
r_true = torch.tensor(rows[0]["R"][:T+1].reshape(1, -1), dtype=torch.float64)
r_pred_full = head(h)
print(f"  head output shape = {r_pred_full.shape}")
r_pred = r_pred_full[:, :T+1]
r_mask = torch.ones_like(r_pred)
loss = _masked_mse(r_pred, r_true, r_mask)
manual_mean = ((r_pred - r_true)**2).mean()
manual_sum_over_T = ((r_pred - r_true)**2).sum() / (T+1)
print(f"  T = {T}")
print(f"  _masked_mse                = {loss.item():.12f}")
print(f"  manual MSE (mean over T+1) = {manual_mean.item():.12f}")
print(f"  manual sum/(T+1)           = {manual_sum_over_T.item():.12f}")
print(f"  matches mean-mode? {abs(loss.item() - manual_mean.item()) < 1e-12}")
ibs_manual = float(((r_pred[0].detach() - r_true[0]).numpy()**2).mean())
print(f"  numpy mean (sklearn-style IBS) = {ibs_manual:.12f}")
print(f"  PASS: _masked_mse equals IBS metric (1/(T+1))·Σ(R̂-R)² per ad" if abs(loss.item() - ibs_manual) < 1e-10 else "  FAIL")

# ========== B3': reward returns Python floats ==========
print("\n========== B3': ttcc_ibs_reward returns Python floats (detached) ==========")
truth_R = rows[0]["R"][:T+1].tolist()
c_perfect = '{"R": ' + json.dumps([float(x) for x in truth_R]) + '}'
c_const = '{"R": ' + json.dumps([1.0] + [0.5]*T) + '}'
c_parse_fail = "no curve here"
rewards = ttcc_ibs_reward([c_perfect, c_const, c_parse_fail], [truth_R]*3, [T]*3)
print(f"  rewards = {rewards}")
print(f"  type   : list of {type(rewards[0]).__name__}")
print(f"  any has grad_fn? {any(hasattr(x, 'grad_fn') for x in rewards)}")
print(f"  perfect reward ≈ 1.0?    {abs(rewards[0] - 1.0) < 1e-9} (got {rewards[0]:.6f})")
print(f"  const-0.5 reward < 1.0?  {rewards[1] < 1.0} (got {rewards[1]:.6f})")
print(f"  parse fail reward == 0?  {rewards[2] == 0.0}")
print("")
print("  Implication: ttcc_ibs_reward emits scalar Python floats.")
print("  GRPO policy gradient = ∇log π(text) · r")
print("  RetentionHead.W is NOT differentiated by reward → frozen by absence in GRPO.")
print("  Body update direction ≠ SFT MSE update direction (different parameter target).")

# ========== Sanity check: does completion need exact text format? ==========
print("\n========== Sensitivity: completion format vs reward ==========")
test_completions = [
    ('JSON object', '{"R": [1.0, 0.9, 0.8, 0.7, 0.6]}'),
    ('R = [...]', 'Final answer: R = [1.0, 0.9, 0.8, 0.7, 0.6]'),
    ('R: [...]', 'R: [1.0, 0.9, 0.8, 0.7, 0.6]'),
    ('with prose', 'After analysis the curve is {"R": [1.0, 0.9, 0.8, 0.7, 0.6]} and that is final.'),
    ('numbers only', '1.0 0.9 0.8 0.7 0.6'),  # likely parses fail (no R: marker)
    ('non-monotone', '{"R": [1.0, 0.5, 0.9, 0.3, 0.6]}'),  # auto-monotone'd
]
short_truth = [1.0, 0.95, 0.9, 0.85, 0.8]
T_short = 4
for label, c in test_completions:
    r_arr = ttcc_ibs_reward([c], [short_truth], [T_short])
    parsed = _parse_curve(c, T_short)
    print(f"  {label:14s}: reward={r_arr[0]:.4f}  parsed={parsed}")
print("")
print("Done.")

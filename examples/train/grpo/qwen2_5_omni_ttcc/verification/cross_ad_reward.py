"""Cross-ad ranking reward for V8 GRPO (design per RL_DESIGN.md).

Reward per rollout i of ad a:
    r = beta*R_rank + alpha*R_acc + gamma*R_fmt    (KL handled by the trainer)

- R_fmt : parse success of the generated curve  (anti parse-cliff, cond C)
- R_rank: percentile match vs a FIXED train-population CDF      (cond E, main)
          1 - mean_t |F_t(Rhat(t)) - F_t(Rtrue(t))|
          Its WITHIN-GROUP spread = the ranking signal (cond B').
- R_acc : 1 - IBS  (dense within-group anchor, cond B; add only if needed)

The percentile floor cancels in GRPO's group-mean-subtracted advantage, so only
within-group spread matters. This module is pure numpy + the parser; no model.
"""
from __future__ import annotations
import json, re
import numpy as np

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_curve(text: str, T: int):
    """Extract an R-curve from a free-text completion. Returns list[float] len T+1 or None.
    Mirrors ttcc_ibs_plugin._parse_curve: JSON {"R":[...]} or bare R=[...]/R:[...],
    coerce to T+1, R(0)=1, monotone non-increasing, clip [0,1]."""
    cleaned = text
    for marker in ("```json", "```"):
        cleaned = cleaned.replace(marker, "")
    nums = None
    start = cleaned.find("{")
    while start != -1 and nums is None:
        depth = 0
        for end in range(start, len(cleaned)):
            ch = cleaned[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "R" in obj and isinstance(obj["R"], list):
                        try:
                            nums = [float(x) for x in obj["R"]]
                        except (TypeError, ValueError):
                            pass
                    break
        start = cleaned.find("{", start + 1)
    if nums is None:
        m = re.search(r'(?:"R(?:\(0\))?"|\bR)\s*[:=]\s*\[', cleaned)
        if m is not None:
            tail = cleaned[m.end():]
            eb = tail.find("]")
            body = tail if eb == -1 else tail[:eb]
            ext = [float(s) for s in _NUM_RE.findall(body)]
            if ext:
                nums = ext
    if nums is None:
        return None
    if len(nums) < T + 1:
        nums = nums + [nums[-1]] * (T + 1 - len(nums))
    elif len(nums) > T + 1:
        nums = nums[:T + 1]
    nums[0] = 1.0
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            nums[i] = nums[i - 1]
        nums[i] = max(0.0, min(1.0, nums[i]))
    return nums


def build_cdf(train_curves, T_max=60):
    """Per-second sorted value arrays from TRAIN curves (train-only; cond E)."""
    cdf = {}
    for t in range(T_max + 1):
        vals = [c[t] for c in train_curves if len(c) > t]
        cdf[t] = np.sort(np.asarray(vals, dtype=np.float64)) if vals else np.array([])
    return cdf


def percentile(cdf_t, x):
    if len(cdf_t) == 0:
        return 0.5
    return float(np.searchsorted(cdf_t, x, side="right")) / len(cdf_t)


def r_rank(R_hat, R_true, cdf, t_lo=1, t_hi=30):
    """1 - mean_t |F_t(Rhat)-F_t(Rtrue)| over the discriminative band."""
    T = min(len(R_hat), len(R_true)) - 1
    hi = min(t_hi, T)
    diffs = []
    for t in range(t_lo, hi + 1):
        diffs.append(abs(percentile(cdf[t], R_hat[t]) - percentile(cdf[t], R_true[t])))
    return 1.0 - float(np.mean(diffs)) if diffs else 0.0


def r_acc(R_hat, R_true, T):
    h = np.asarray(R_hat[:T + 1], float); tr = np.asarray(R_true[:T + 1], float)
    return float(max(0.0, 1.0 - ((h - tr) ** 2).mean()))


def r_fmt(text, T):
    return 1.0 if parse_curve(text, T) is not None else 0.0


def composite(text, R_true, cdf, *, beta=1.0, alpha=0.0, gamma=0.1):
    """Full per-rollout reward from a raw completion string."""
    T = len(R_true) - 1
    fmt = r_fmt(text, T)
    if fmt == 0.0:
        return gamma * 0.0  # parse fail -> only format term (0)
    R_hat = parse_curve(text, T)
    rk = r_rank(R_hat, R_true, cdf)
    ac = r_acc(R_hat, R_true, T)
    return beta * rk + alpha * ac + gamma * fmt

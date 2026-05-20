"""TTCC reward plugin for ms-swift GRPO.

Registers `ttcc_ibs_reward`: parses the model's completion into a per-second
retention curve R_hat, compares to the dataset's ground-truth R_true,
returns `1 - IBS` as the per-rollout reward in [0, 1].

Citation chain (see ttcc-eval/docs/07):
  - Brier (1950) — original strictly proper quadratic score
  - Graf et al. (1999) — Brier adapted to survival functions (= IBS)
  - Gneiting & Raftery (2007) JASA — use proper scoring rules both as
    training objectives AND eval metrics
  - Sonabend et al. (2024) — Brier (RSBS) as training loss for survival
    networks; this is the direct precedent
  - Hartman et al. (2023) — critique of rank-only metrics, motivating
    the move away from Spearman-based rewards

Usage:
    swift rlhf --rlhf_type grpo \
        --external_plugins examples/train/grpo/plugin/ttcc_ibs_plugin.py \
        --reward_funcs ttcc_ibs_reward \
        ...
"""
from __future__ import annotations
import json
import re
from typing import Any, List

import numpy as np

from swift.rewards import ORM, orms

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse_curve(text: str, T: int) -> list[float] | None:
    """Extract an R-curve from free-text completion.

    Accepts:
      - {"R": [1.0, ..., R(T)]}
      - Curve: R = [1.0, ..., R(T)]
      - R: [1.0, ...] / R = [1.0, ...] anywhere in text.
    Coerces length to T+1 (pad-last or truncate), enforces R(0) = 1.0,
    enforces monotone non-increasing via running min, clips to [0, 1].
    Returns None if no R-list can be extracted.
    """
    cleaned = text
    for marker in ("```json", "```"):
        cleaned = cleaned.replace(marker, "")

    nums: list[float] | None = None

    # Pass 1: balanced JSON object {"R": [...]}.
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
                    blob = cleaned[start : end + 1]
                    try:
                        obj = json.loads(blob)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "R" in obj and isinstance(obj["R"], list):
                        try:
                            nums = [float(x) for x in obj["R"]]
                        except (TypeError, ValueError):
                            pass
                    break
        start = cleaned.find("{", start + 1)

    # Pass 2: bare 'R = [...]' / 'R: [...]' / '"R": [...]'.
    if nums is None:
        m = re.search(r'(?:"R(?:\(0\))?"|\bR)\s*[:=]\s*\[', cleaned)
        if m is not None:
            tail = cleaned[m.end():]
            end_bracket = tail.find("]")
            body = tail if end_bracket == -1 else tail[:end_bracket]
            extracted = [float(s) for s in _NUM_RE.findall(body)]
            if extracted:
                nums = extracted

    if nums is None:
        return None

    # Coerce to T + 1.
    if len(nums) < T + 1:
        nums = nums + [nums[-1]] * (T + 1 - len(nums))
    elif len(nums) > T + 1:
        nums = nums[: T + 1]
    nums[0] = 1.0
    # Monotone non-increasing + clip to [0, 1].
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            nums[i] = nums[i - 1]
        nums[i] = max(0.0, min(1.0, nums[i]))
    return nums


class TTCCIBSReward(ORM):
    """`r_i = 1 - IBS_i` per rollout, where IBS is the integrated Brier score
    over the prediction window [0, T_i]. Strictly proper (no censoring →
    Brier 1950 form), bounded in [0, 1], max-at-truth → cannot be saturated
    by the within-ad-Spearman tautology documented in ttcc-eval/docs/07.

    Dataset columns expected per row:
      - 'R_true': list[float] of length T+1 (GT curve, peak-normalized)
      - 'T':      int, the eval horizon (could also be inferred from len-1)
    These flow in as kwargs (one per completion) because ms-swift broadcasts
    dataset columns over the batch.
    """

    name = "ttcc_ibs_reward"

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        R_true_batch = kwargs.get("R_true")
        T_batch = kwargs.get("T")
        rewards: list[float] = []
        for i, completion in enumerate(completions):
            R_true = R_true_batch[i] if R_true_batch is not None else None
            T = int(T_batch[i]) if T_batch is not None else (len(R_true) - 1 if R_true else None)
            if R_true is None or T is None:
                rewards.append(0.0)
                continue
            R_hat = _parse_curve(completion, T)
            if R_hat is None:
                rewards.append(0.0)  # parse failure -> zero reward
                continue
            r_true_arr = np.asarray(R_true, dtype=np.float64)
            r_hat_arr = np.asarray(R_hat, dtype=np.float64)
            L = min(len(r_true_arr), len(r_hat_arr))
            ibs = float(np.mean((r_hat_arr[:L] - r_true_arr[:L]) ** 2))
            rewards.append(max(0.0, 1.0 - ibs))  # bounded [0, 1]
        return rewards


orms["ttcc_ibs_reward"] = TTCCIBSReward

"""Placeholder ORM that reserves the HEAD-reward column for CoT-GRPO.

The head reward must forward the policy through the retention head on each
generated CoT -- a pure-text ORM can't do that (it only sees the completion
string). So we register this trivial placeholder (returns 0.0) purely to give
swift's GRPOTrainer a reward COLUMN named `ttcc_head` with a weight; the real
value is filled in by `CoTGRPOTrainer._compute_rewards_per_func` (which has the
model + the retention head + the train CDF). Keeping the column bookkeeping in
swift's hands (instead of hand-managing tensor shapes/weights) is the low-risk path.
"""
from __future__ import annotations

from swift.rewards import ORM, orms


class TTCCHeadPlaceholder(ORM):
    name = "ttcc_head"

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        # Overwritten by CoTGRPOTrainer; 0.0 here so a missing override is obvious.
        return [0.0 for _ in completions]


orms["ttcc_head"] = TTCCHeadPlaceholder

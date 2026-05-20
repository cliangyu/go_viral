"""Companion format-reward plugin for TTCC GRPO.

Gives a small reward (0.5) when the completion contains a parseable
'R = [...]' or '{"R": [...]}' construct, regardless of correctness.
Encourages the model to keep emitting parseable curves during early
GRPO when the IBS reward is still noisy. Pair with --reward_weights 1 0.5
(IBS-weight 1, format-weight 0.5) per the Qwen2.5-Omni grpo.sh example.
"""
from __future__ import annotations
import re
from typing import List

from swift.rewards import ORM, orms


class TTCCFormatReward(ORM):
    name = "ttcc_format"

    _PAT = re.compile(r'(?:"R(?:\(0\))?"|\bR)\s*[:=]\s*\[')

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return [1.0 if self._PAT.search(c) else 0.0 for c in completions]


orms["ttcc_format"] = TTCCFormatReward

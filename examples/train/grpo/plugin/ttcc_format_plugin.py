"""Companion format-reward plugin for TTCC GRPO.

Returns 1.0 when the completion contains a parseable ``R = [...]`` or
``{"R": [...]}`` construct, else 0.0. The training script applies the
mixing weight (we use ``--reward_weights 1.0 0.2``, so format contributes
up to +0.2 on top of the IBS reward).

Purpose: keep early GRPO from drifting away from a parseable format
while the IBS reward is still noisy. Once the IBS reward is reliable, the
format reward becomes a constant (always parseable from SFT init) and
contributes no gradient — but acts as cheap insurance against drift.
"""
from __future__ import annotations

import re

from swift.rewards import ORM, orms


class TTCCFormatReward(ORM):
    name = "ttcc_format"

    _PAT = re.compile(r'(?:"R(?:\(0\))?"|\bR)\s*[:=]\s*\[')

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        return [1.0 if self._PAT.search(c) else 0.0 for c in completions]


orms["ttcc_format"] = TTCCFormatReward

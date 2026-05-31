"""CoT-structure format reward for CoT-GRPO (head channel).

Unlike `ttcc_format` (which checks for a parseable ``R=[...]`` curve in the TEXT
-- the abandoned text channel), CoT-GRPO's completion is pure reasoning and the
curve comes from the HEAD reading the generated CoT. So the format reward here
guards the *structure* of the reasoning, not a parseable curve:

  1.0  iff the completion contains a single well-formed ``<cot> ... </cot>`` with
       a non-empty interior whose length is within [MIN_CHARS, MAX_CHARS];
  0.0  otherwise (empty `<cot></cot>`, missing tags, runaway, or multiple blocks).

Purpose: keep early GRPO from drifting to degenerate / empty / runaway CoT while
the rank reward is still noisy. Applied with a small weight (e.g. 0.1-0.2). Once
the policy reliably emits well-formed CoT it becomes ~constant and contributes no
gradient -- cheap insurance against structural drift. This stays a pure-text ORM
(it only inspects the completion string), so it needs no model access.
"""
from __future__ import annotations

import re

from swift.rewards import ORM, orms


class TTCCCoTFormatReward(ORM):
    name = "ttcc_cot_format"

    # single non-greedy <cot>...</cot> block; DOTALL so the interior may span lines
    _BLOCK = re.compile(r"<cot>(.*?)</cot>", re.DOTALL)
    MIN_CHARS = 8      # non-trivial interior (a few words)
    MAX_CHARS = 4000   # ~ p99 CoT chars; guards runaway generations

    def _score(self, c: str) -> float:
        blocks = self._BLOCK.findall(c)
        if len(blocks) != 1:           # missing, or multiple cot blocks
            return 0.0
        interior = blocks[0].strip()
        n = len(interior)
        if n < self.MIN_CHARS or n > self.MAX_CHARS:
            return 0.0                 # empty / too short / runaway
        return 1.0

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        return [self._score(c) for c in completions]


orms["ttcc_cot_format"] = TTCCCoTFormatReward

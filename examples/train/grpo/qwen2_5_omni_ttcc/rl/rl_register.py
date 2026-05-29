#!/usr/bin/env python
"""ms-swift external_plugin: point the TrainerFactory at our HeadPGTrainer for
the `causal_lm` task, WITHOUT editing swift core. Loaded via the yaml
`external_plugins:` list (alongside register.py), so it runs during arg setup —
before SwiftSft.run() calls TrainerFactory.get_trainer_cls(). See NORTH_STAR §10.

Why monkeypatch the mapping (vs subclassing SwiftSft.run): the factory selects the
trainer by a string path it importlib-imports; swapping that one entry is the
minimal, shared-fork-safe injection (no swift-core diff, no Ray-decorator copy)."""
import os
import sys

# make head_pg_trainer importable by the factory's importlib.import_module(...)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from swift.trainers.trainer_factory import TrainerFactory
from swift.utils import get_logger

# copy-then-mutate so we don't alias the class-level dict in a surprising way
TrainerFactory.TRAINER_MAPPING = dict(TrainerFactory.TRAINER_MAPPING)
TrainerFactory.TRAINER_MAPPING['causal_lm'] = 'head_pg_trainer.HeadPGTrainer'
get_logger().info('[rl_register] TrainerFactory: causal_lm -> head_pg_trainer.HeadPGTrainer (head-PG RL)')

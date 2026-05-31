#!/usr/bin/env python
"""Entry for CoT-GRPO. **RL via the OFFICIAL swift GRPO pipeline** (`SwiftRLHF`) with
ONE swap: `trainer_cls = CoTGRPOTrainer`. We keep the entire GRPO machinery (multimodal
vLLM rollouts, ref model + KL, group-relative advantage, per-token logps, multi-node) and
only change how the reward is computed (the head reads each generated CoT).

Mirrors rl/train_head_pg.py (the HeadPG entry pattern), but bases on SwiftRLHF instead of
SwiftSft because CoT-GRPO needs generation + ref model (head-PG did not). See COT_RL_PLAN.md.

Launch:
  1-GPU smoke : python train_cot_grpo.py <config.yaml> [--override k=v ...]
  multi-node  : RL_ENTRY=rl/train_cot_grpo.py NNODES=2 NODE_RANK=R MASTER_ADDR=A bash rl.sh <config.yaml>

Env: TRAIN_CDF (required, train-only CDF npz); COT_BETA/COT_ALPHA/COT_T_LO/COT_T_HI (head
reward weights, read by the trainer); COT_GUARD_HOLDOUT (optional held-out SRCC guard jsonl).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # cot_grpo_trainer + deps

import numpy as np

from swift.pipelines.train.rlhf import SwiftRLHF
from swift.utils import get_logger, get_model_parameter_info

from cot_grpo_trainer import CoTGRPOTrainer

logger = get_logger()


def _load_cdf(path):
    z = np.load(path)
    return {int(k[1:]): z[k] for k in z.files}  # 't12' -> 12 -> sorted train values at t=12


class CoTGRPOPipeline(SwiftRLHF):
    """Official GRPO pipeline mechanics + the CoT-GRPO trainer. Mirrors SwiftSft.run()
    except trainer_cls = CoTGRPOTrainer (no TrainerFactory dispatch), plus injecting the
    train CDF that the head reward grades against."""

    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()
        args.save_args()
        self.model = self.prepare_model(self.args, self.model, template=self.template,
                                        train_dataset=train_dataset)
        logger.info(f'model_parameter_info: {get_model_parameter_info(self.model)}')

        trainer = CoTGRPOTrainer(                      # <-- the only swap vs SwiftSft.run()
            model=self.model,
            args=self.args.training_args,
            template=self.template,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            **self._get_trainer_kwargs(),               # reward_funcs, ref_model, vllm_client, ...
        )

        cdf_path = os.environ['TRAIN_CDF']              # KeyError if unset -> fail fast
        trainer._cot_cdf = _load_cdf(cdf_path)
        logger.info(f'[cot-grpo] injected train CDF: {cdf_path} ({len(trainer._cot_cdf)} t-bins)')

        guard = os.environ.get('COT_GUARD_HOLDOUT')     # optional observe-only held-out SRCC
        if guard:
            try:
                from srcc_guard import HeldoutSRCCGuard
                trainer.add_callback(HeldoutSRCCGuard(
                    guard, self.template, guard_steps=int(os.environ.get('COT_GUARD_STEPS', '25'))))
                logger.info(f'[cot-grpo] held-out SRCC guard ON: {guard}')
            except Exception as e:  # noqa: BLE001 -- guard is optional, never block the run
                logger.warning(f'[cot-grpo] held-out guard not attached: {e}')

        return self.train(trainer)


def main():
    # swift's own yaml loader: reads <config.yaml>, exports the ENV: block, expands the
    # rest into --key value argv in place (same path `swift rlhf` uses).
    from swift.cli.main import parse_yaml_args
    argv = sys.argv[1:]
    parse_yaml_args(argv)
    return CoTGRPOPipeline(argv).main()


if __name__ == '__main__':
    main()

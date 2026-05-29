#!/usr/bin/env python
"""Dedicated entry for head-PG RL. **This is RL, not SFT** — do NOT launch with
`swift sft`. We reuse the SFT *pipeline mechanics* on purpose: head-PG needs only
ONE forward + a custom loss, and NO text generation / reference model / reward
model / critic — so swift's RLHF pipeline (`SwiftRLHF(SwiftSft)`, which builds
ref/reward/value models + vllm rollouts for DPO/PPO/GRPO) does not apply and would
only get in the way. So we subclass `SwiftSft` and swap ONLY the trainer class to
`HeadPGTrainer`. See NORTH_STAR §10.

Launch:
  1-GPU smoke : python train_head_pg.py <config.yaml> [--override k=v ...]
  multi-GPU   : torchrun --nnodes N --node_rank R --nproc_per_node 8 \
                  --master_addr A --master_port P train_head_pg.py <config.yaml> ...
  (see rl.sh — the launcher that mirrors sft.sh's multi-node env.)
RayHelper.function is a passthrough without Ray (ray_utils/base.py:167), so the
plain run() override below is faithful to SwiftSft.run() under torchrun.
"""
import os
import sys

# make head_pg_trainer importable (it pulls cross_ad_reward/reinforce_core from ../verification)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swift.pipelines.train.sft import SwiftSft
from swift.utils import get_logger, get_model_parameter_info

from head_pg_trainer import HeadPGTrainer

logger = get_logger()


class HeadPGSft(SwiftSft):
    """SFT pipeline mechanics + the head-PG RL trainer. Mirrors SwiftSft.run()
    exactly except `trainer_cls = HeadPGTrainer` (no TrainerFactory dispatch)."""

    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()
        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        args.save_args()
        self.model = self.prepare_model(self.args, self.model, template=self.template,
                                        train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer = HeadPGTrainer(                       # <-- the only change vs SwiftSft.run()
            model=self.model,
            args=self.args.training_args,
            template=self.template,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)


def main():
    # Reuse swift's own yaml loader (the `swift sft` CLI path): it reads <config.yaml>,
    # exports the ENV: block (RETENTION_HEAD_TYPE, HPG_*), and expands the rest into
    # --key value argv IN PLACE. This is why HeadPGTrainer reads HPG_* in __init__, not
    # at import: ENV is exported here, at runtime.
    from swift.cli.main import parse_yaml_args
    argv = sys.argv[1:]                     # [<config.yaml>, --override ...]
    parse_yaml_args(argv)                   # mutates argv: yaml -> --args; os.environ <- ENV
    return HeadPGSft(argv).main()


if __name__ == '__main__':
    main()

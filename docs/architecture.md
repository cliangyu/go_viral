# Architecture & file-placement guide

This doc captures the systematic analysis of ms-swift's codebase so that
future "where does this go?" decisions are evidence-based, not pattern-matched.

## ms-swift's layered architecture (C4-style summary)

### L1 — System context

ms-swift is a training framework that integrates:

- **Hub providers**: ModelScope Hub, HuggingFace Hub (downloads, uploads).
- **Trainer backends**: HF transformers `Trainer`, DeepSpeed, FSDP, Megatron-LM (mcore), Ray.
- **Inference engines**: HF generate, vLLM, SGLang, LMDeploy.
- **Loggers**: TensorBoard, W&B, SwanLab.

The user surface is the `swift` CLI: `swift sft|rlhf|infer|export|eval|deploy`.

### L2 — Subsystems inside `swift/`

Every extension subsystem follows the **same pattern**: a `mapping.py` (or
`register.py`) with both `register_X()` and a global `X_MAPPING` dict.
Built-in modules populate it at import time; user files populate it at
runtime via `--external_plugins path/to/user.py`.

| Subsystem | Mapping file | Registers |
|---|---|---|
| `swift/model/` | `register.py` | model_type → ModelMeta + ModelLoader |
| `swift/template/` | mapping inside templates | template_name → TemplateMeta + Template class |
| `swift/dataset/` | dataset_info.json + register | dataset_name → preprocessor |
| `swift/loss/` | `mapping.py` | loss_type → BaseLoss subclass |
| `swift/rewards/` | `orm.py` | reward_name → ORM (used by GRPO) |
| `swift/tuner_plugin/` | `mapping.py` | tuner_type → Tuner |
| `swift/loss_scale/` | `mapping.py` | loss_scale_name → token weighting |
| `swift/metrics/` | `metric.py` | metric_name → eval metric |
| `swift/optimizers/` | `mapping.py` | optimizer_name → custom optimizer |
| `swift/callbacks/` | callbacks module | TrainerCallback subclasses |
| `swift/agent_template/` | mapping | agent format adapter |

This is a textbook Service Locator + Plugin Architecture (Fowler).

### L3 — Command flow (e.g., `swift sft <yaml>`)

```
swift/cli/main.py  (ROUTE_MAPPING['sft'] -> swift.cli.sft)
  -> swift/cli/sft.py  (parses YAML if first arg ends .yaml/.json)
    -> swift/pipelines/train/sft.py::sft_main
      1. SftArguments() from swift/arguments/...
      2. For each path in --external_plugins: importlib.import_module()
         (side-effect: registrations fire into the mappings)
      3. get_model_processor()  -> resolves --model_type via MODEL_MAPPING
      4. get_template()         -> resolves template via TEMPLATE_MAPPING
      5. load_dataset()         -> swift/dataset
      6. Build SFTTrainer       -> swift/trainers/sft_trainer.py
      7. Loss dispatch          -> loss_map[args.loss_type](args, self)
```

## Where do new things go?

This table is the deliverable. Memorize it; cite it on every placement.

| Artifact kind | Correct location | Evidence |
|---|---|---|
| Custom model class with custom head | `examples/custom/<model_name>/register.py` (or single `.py`) — loaded by `--external_plugins`. | `docs/source_en/Customization/Custom-model.md:11`; canonical example at `examples/custom/my_qwen2_5_omni/`. |
| Custom loss function | Same plugin file as the model, OR standalone `.py` under `examples/custom/` or `examples/train/<task>/plugin/`. Registers via `loss_map['name'] = MyLoss`. | `Customization/Architecture.md §Loss`. |
| Custom data collator / template | Same plugin file as the model; subclass `Template` and override `_encode` / `_data_collator`. | `Custom-model.md:33`. |
| Custom GRPO reward function | `examples/train/grpo/plugin/<name>.py`. Registers via `orms['name'] = MyReward`. | `Instruction/GRPO/DeveloperGuide/reward_function.md:44`. |
| Custom dataset preprocessor | Plugin file under `examples/custom/` with `register_dataset(DatasetMeta(...))`. | `Custom-dataset.md`. |
| YAML variant config | `examples/train/<task>/<project>/configs/<variant>.yaml`. | `examples/yaml/deepspeed/sft.yaml` (upstream demo). |
| Launcher shell script | `examples/train/<task>/<project>/<launcher>.sh`. | Convention across `examples/train/*/*.sh`. |
| Project-specific Docker / setup | `examples/train/<task>/<project>/{Dockerfile, host-setup.sh, requirements.lock}`. NOT repo root — ms-swift does not ship Dockerfiles upstream. | Convention; absence of repo-root Dockerfile. |
| Project-specific ops tool | Same `<project>/` dir (e.g. `watch_and_upload_ckpts.py`). | Locality of reference. |
| Project research log / handoff | Same `<project>/` dir OR `docs/handoffs/`. | Cosmetic. |
| One-off analysis script | NOT in the ms-swift fork. Sibling repo. | Inferred — no convention for it in ms-swift. |

## What this rules OUT

Anti-patterns that come up repeatedly:

- **Putting user-extension code inside `swift/`.** Never edit framework code. Always extend via `--external_plugins`.
- **Putting custom-model registrations under `examples/train/grpo/plugin/`.** That directory is reserved for GRPO **reward** functions; the docs are explicit.
- **Creating a parallel framework** (e.g. dropping a `cs224r_project/` dir at repo root with its own raw-HF-Trainer pipeline). Either port into ms-swift via a plugin or live in a sibling repo.
- **YAML inheritance / Hydra defaults_list patterns.** ms-swift's YAML loader is single-file; multiple-config merging is the launcher's job. Use CLI overrides for one-off changes.
- **Per-variant shell scripts** that copy-paste a `swift sft` invocation. One launcher per training stage (`sft.sh`, `dpo.sh`, `grpo.sh`, `rloo.sh`, `infer.sh`); the variant is a YAML arg.
- **Version-numbered filenames** (`v3.sh`, `sft_v2cot_full.sh`, `chain_resume2.sh`). Use semantic names. Configuration history lives in git.
- **Hardcoded LoRA in the model file.** Tuner is `--tuner_type`; the model definition must not know whether the trainer wraps it with PEFT or not.

## Reference: extension-point cheat sheet

```
# Custom loss
from swift.loss import BaseLoss, loss_map

class MyLoss(BaseLoss):
    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        ...
loss_map['my_loss'] = MyLoss

# Custom reward (GRPO)
from swift.rewards import ORM, orms

class MyReward(ORM):
    name = 'my_reward'
    def __call__(self, completions, **kwargs) -> list[float]:
        ...
orms['my_reward'] = MyReward

# Custom model + template
from swift.model import (ModelLoader, ModelMeta, Model, ModelGroup,
                         MultiModelKeys, register_model, register_model_arch)
from swift.template import Template, TemplateMeta, register_template

class MyLoader(ModelLoader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        model = super().get_model(model_dir, config, processor, model_kwargs)
        # patch / wrap as needed
        return model

register_model_arch(MultiModelKeys('my_model', language_model=[...], ...))
register_model(ModelMeta('my_model', [ModelGroup([...])], MyLoader, ...))

class MyTemplate(Template):
    def _encode(self, inputs): ...
    def _data_collator(self, batch, *, padding_to=None): ...
register_template(TemplateMeta('my_template', ..., template_cls=MyTemplate))
```

## TTCC-specific layout (this fork)

| Path | Purpose |
|---|---|
| `examples/custom/qwen2_5_omni_retention/` | Retention-curve head plugin (model + template + loss). |
| `examples/train/grpo/qwen2_5_omni_ttcc/` | TTCC launcher directory: `sft.sh`, `dpo.sh`, `grpo.sh`, `rloo.sh`, `infer.sh`, `_common.sh`, `_chain_lib.sh`, `chain.sh`, `Dockerfile`, `host-setup.sh`, `prepare_dataset.py`, `watch_and_upload_ckpts.py`, `README_8card.md`, `requirements.lock`, `8CARD_WORK_LOG.md`, `HEAD_COMPARISON.md`, `configs/`. |
| `examples/train/grpo/qwen2_5_omni_ttcc/configs/` | One YAML per variant; see `configs/README.md`. |
| `examples/train/grpo/plugin/ttcc_ibs_plugin.py` | GRPO reward: 1 - IBS against ground-truth retention curve. |
| `examples/train/grpo/plugin/ttcc_format_plugin.py` | GRPO format reward: 1.0 if R-list parseable from output. |
| `scripts/launch_distributed.sh` | Multi-node wrapper around any launcher in the TTCC dir. |
| `scripts/data/build_ttcc_jsonl.py` | HF-stream preprocessor → ms-swift JSONL. |
| `docs/architecture.md` | This document. |

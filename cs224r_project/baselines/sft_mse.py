"""
SFT-MSE per-second baseline (§5 of the experiment plan).

Architecture:
  Qwen2.5-Omni-3B Thinker trunk (Talker disabled) + LoRA (rank 8, all-linear)
  + nn.Linear(hidden_dim, 60) + sigmoid head that emits R_hat(t) directly.

Loss:
  Masked MSE between R_hat[:T_i] and R_true[1:T_i+1] (skip R(0)=1).

Run:
  python cs224r_project/baselines/sft_mse.py \
      --train_jsonl cs224r_project/data/splits/train.jsonl \
      --val_jsonl   cs224r_project/data/splits/val.jsonl \
      --output_dir  cs224r_project/runs/sft_mse/seed42 \
      --seed 42
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5OmniForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)


T_MAX = 60
USER_PROMPT = (
    "Predict the per-second retention curve. "
    "Reason about hook, pacing, and audio cues."
)


class RetentionHeadModel(nn.Module):
    """Qwen2.5-Omni-3B trunk + per-second sigmoid head over the last hidden state."""

    def __init__(self, model_id: str):
        super().__init__()
        self.trunk = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        )
        if hasattr(self.trunk, "disable_talker"):
            self.trunk.disable_talker()
        # In Qwen2.5-Omni the LM hidden size lives at thinker.text_config.hidden_size
        # (the thinker_config itself doesn't carry it directly).
        thinker = getattr(self.trunk, "thinker", self.trunk)
        thinker_cfg = thinker.config
        d = getattr(getattr(thinker_cfg, "text_config", thinker_cfg), "hidden_size",
                    getattr(thinker_cfg, "hidden_size", None))
        if d is None:
            raise RuntimeError(f"could not locate hidden_size on {type(thinker_cfg).__name__}")
        # fp32 head for numerical stability of the sigmoid output.
        self.head = nn.Linear(d, T_MAX, dtype=torch.float32)

    def forward(self, mm_inputs: dict, attention_mask: torch.Tensor):
        # The outer Qwen2_5OmniForConditionalGeneration has no implemented
        # forward; it only routes generation through thinker/talker. We hit
        # the thinker (which is the LM-with-MM-encoders) directly.
        out = self.trunk.thinker(**mm_inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]                            # (B, L, d)
        # Pool over the last non-pad token of each sequence.
        lengths = attention_mask.sum(dim=1) - 1              # (B,)
        idx = lengths.clamp(min=0).long()
        h_last = h[torch.arange(h.size(0), device=h.device), idx]   # (B, d)
        r_hat = torch.sigmoid(self.head(h_last.float()))     # (B, 60), fp32
        return r_hat


class TTCCDataset(Dataset):
    """Reads ms-swift conversation JSONL; yields raw fields. Collator does MM tokenization."""

    def __init__(self, jsonl_path: str):
        self.rows = []
        with open(jsonl_path) as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


@dataclass
class TTCCCollator:
    processor: Any
    nframes: int = 32
    max_pixels: int = 200704

    def __call__(self, batch: list[dict]) -> dict:
        from qwen_omni_utils import process_mm_info

        conversations = []
        meta = []
        for row in batch:
            video_path = row["videos"][0]
            audio_path = row["audios"][0]
            conv = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path,
                     "max_pixels": self.max_pixels, "nframes": self.nframes},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text",  "text":  USER_PROMPT},
                ],
            }]
            conversations.append(conv)
            meta.append(row["_meta"])

        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False,
        )
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        # §4 silent-skip trap: assert MM tensors actually populated.
        assert inputs["input_ids"].size(0) == len(batch), (
            f"processor dropped rows: got {inputs['input_ids'].size(0)} != {len(batch)}"
        )
        assert "pixel_values_videos" in inputs and inputs["pixel_values_videos"].numel() > 0, \
            "video tensors are empty — processor likely dropped MM input"

        durations = torch.tensor([m["duration_s"] for m in meta], dtype=torch.long)
        r_true = torch.zeros(len(batch), T_MAX, dtype=torch.float32)
        mask = torch.zeros(len(batch), T_MAX, dtype=torch.float32)
        for i, m in enumerate(meta):
            T_i = int(m["duration_s"])
            # retention_curve has length T_i + 1, with curve[0] = 1.
            curve = m["retention_curve"][1:T_i + 1]
            r_true[i, :T_i] = torch.tensor(curve, dtype=torch.float32)
            mask[i, :T_i] = 1.0

        attention_mask = inputs["attention_mask"]
        return {
            "mm_inputs": dict(inputs),
            "attention_mask": attention_mask,
            "r_true": r_true,
            "mask": mask,
            "durations": durations,
        }


class SFTMSETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mm = inputs["mm_inputs"]
        attn = inputs["attention_mask"]
        r_hat = model(mm, attn)                                  # (B, 60)
        r_true = inputs["r_true"].to(r_hat.device).to(r_hat.dtype)
        mask = inputs["mask"].to(r_hat.device).to(r_hat.dtype)
        sq_err = (r_hat - r_true) ** 2 * mask
        loss = sq_err.sum() / mask.sum().clamp(min=1.0)
        return (loss, {"r_hat": r_hat}) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--nframes", type=int, default=32,
                    help="frames per video (v2=32, v3=60)")
    ap.add_argument("--max_pixels", type=int, default=200704,
                    help="max pixels per frame")
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--smoke_only", action="store_true",
                    help="Load first batch, print shapes, exit before training.")
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = RetentionHeadModel(args.model_id)
    # Gradient checkpointing on the thinker (the LM-with-encoders) saves
    # ~30-50% activation memory at ~20% throughput cost. Necessary for the
    # 30k-token MM sequences we see at batch_size > 1 on a 80GB H100.
    if hasattr(model.trunk.thinker, "gradient_checkpointing_enable"):
        model.trunk.thinker.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 4,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # IMPORTANT: wrap the *thinker*, not the outer Omni wrapper. The outer
    # Qwen2_5OmniForConditionalGeneration has no implemented .forward()
    # (it delegates to thinker/talker via its own dispatch), so PEFT's
    # base_model.forward(input_ids=...) call would hit _forward_unimplemented.
    model.trunk.thinker = get_peft_model(model.trunk.thinker, lora_cfg)

    # Freeze vision + audio encoders AFTER LoRA wrap. PEFT injects adapters
    # on all linears including audio_tower.* and visual.*; we explicitly
    # zero their requires_grad to comply with §5's
    # freeze_vision_encoder=True, freeze_audio_encoder=True.
    n_frozen = 0
    for n, p in model.named_parameters():
        if p.requires_grad and (
            "visual" in n or "audio_tower" in n or "vision" in n
            or ".merger." in n
        ):
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"post-LoRA freeze: zero'd {n_frozen:,} params in vision/audio")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    train_ds = TTCCDataset(args.train_jsonl)
    val_ds = TTCCDataset(args.val_jsonl)
    collator = TTCCCollator(processor=processor, nframes=args.nframes,
                            max_pixels=args.max_pixels)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        # Disable eval: HF Trainer calls model(**inputs) with `r_true` etc
        # that our custom forward doesn't accept. We don't track val loss
        # during training; final eval happens via make_test_preds.py.
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=(["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"]),
        run_name=f"sft_mse_seed{args.seed}",
        warmup_ratio=0.05,
    )

    trainer = SFTMSETrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
    )

    # §4 smoke verification: load batch 0 and print shapes before kicking off
    # training. Catches silent MM-row drops and shape mismatches up front.
    print("[smoke] loading first batch...")
    smoke_batch = collator([train_ds[i] for i in range(min(2, len(train_ds)))])
    print(f"[smoke] input_ids: {smoke_batch['mm_inputs']['input_ids'].shape}")
    if "pixel_values_videos" in smoke_batch["mm_inputs"]:
        print(f"[smoke] pixel_values_videos: {smoke_batch['mm_inputs']['pixel_values_videos'].shape}")
    if "input_features" in smoke_batch["mm_inputs"]:
        print(f"[smoke] input_features (audio): {smoke_batch['mm_inputs']['input_features'].shape}")
    print(f"[smoke] r_true: {smoke_batch['r_true'].shape}, "
          f"mask sum per row: {smoke_batch['mask'].sum(1).tolist()}, "
          f"durations: {smoke_batch['durations'].tolist()}")
    if args.smoke_only:
        print("[smoke] --smoke_only set, exiting before training.")
        return

    trainer.train()

    # Save the LoRA adapter (on the thinker) and the sigmoid head separately.
    model.trunk.thinker.save_pretrained(args.output_dir)
    torch.save(
        {"head": model.head.state_dict()},
        os.path.join(args.output_dir, "retention_head.pt"),
    )
    print(f"saved LoRA adapter + retention head to {args.output_dir}")


if __name__ == "__main__":
    main()

"""
SFT-Hazard+CoT baseline (§6 of the experiment plan).

Joint loss:
    L = L_hazard + alpha * L_cot,
where L_hazard is the masked log-hazard MSE (§3) and L_cot is next-token
cross-entropy on the distilled CoT span only.

The retention curve is reconstructed from the softplus hazard head reading
the hidden state at the </cot> position (see retention_vlm.py).

Run:
  python cs224r_project/baselines/sft_hazard_cot.py \
      --train_jsonl cs224r_project/data/splits/train_with_cot.jsonl \
      --val_jsonl   cs224r_project/data/splits/val.jsonl \
      --output_dir  cs224r_project/runs/sft_hazard_cot/seed42 \
      --seed 42
"""

import argparse
import json
import os
import sys
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
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retention_vlm import (  # noqa: E402
    RetentionVLM, T_MAX, hazards_from_retention, masked_hazard_log_mse,
)


USER_PROMPT = (
    "Predict the per-second retention curve. "
    "Reason about hook, pacing, and audio cues. "
    "Wrap reasoning in <cot>...</cot>."
)


class TTCCDataset(Dataset):
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
class TTCCCollatorCoT:
    processor: Any
    tokenizer: Any
    nframes: int = 32
    max_pixels: int = 200704
    _assistant_marker_ids: list[int] | None = None
    _im_end_id: int | None = None
    _pad_id: int | None = None

    def _init_markers(self):
        if self._assistant_marker_ids is not None:
            return
        # The Qwen chat template emits "<|im_start|>assistant\n" before each
        # assistant turn and "<|im_end|>" after. We anchor the label mask to
        # these markers so it is robust to MM-token expansion (which inserts
        # ids only inside the user turn, never inside the assistant turn).
        self._assistant_marker_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False,
        )
        im_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(im_end) == 1, f"expected single <|im_end|> token, got {im_end}"
        self._im_end_id = im_end[0]
        self._pad_id = self.tokenizer.pad_token_id

    def _build_messages(self, video_path: str, audio_path: str, cot_text: str):
        user = {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path,
                 "max_pixels": self.max_pixels, "nframes": self.nframes},
                {"type": "audio", "audio": audio_path},
                {"type": "text",  "text":  USER_PROMPT},
            ],
        }
        assistant = {"role": "assistant", "content": f"<cot>{cot_text}</cot>"}
        return [user, assistant]

    def __call__(self, batch: list[dict]) -> dict:
        from qwen_omni_utils import process_mm_info

        self._init_markers()

        convs_full, meta = [], []
        for row in batch:
            v, a = row["videos"][0], row["audios"][0]
            cot_text = row["_meta"].get("cot", "")
            convs_full.append(self._build_messages(v, a, cot_text))
            meta.append(row["_meta"])

        text_full = self.processor.apply_chat_template(
            convs_full, add_generation_prompt=False, tokenize=False,
        )
        audios, images, videos = process_mm_info(convs_full, use_audio_in_video=False)
        inputs = self.processor(
            text=text_full,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        # §4 silent-skip trap.
        assert inputs["input_ids"].size(0) == len(batch), (
            f"processor dropped rows: got {inputs['input_ids'].size(0)} != {len(batch)}"
        )

        # Build CoT label mask anchored to the assistant marker tokens.
        # labels[t] = input_ids[t] for tokens strictly inside the LAST
        # assistant span (between '<|im_start|>assistant\n' and '<|im_end|>'),
        # otherwise -100.
        input_ids = inputs["input_ids"]
        labels = torch.full_like(input_ids, -100)
        marker = torch.tensor(self._assistant_marker_ids, dtype=input_ids.dtype)
        k = marker.numel()
        L = input_ids.size(1)
        n_with_cot = 0
        if L >= k:
            windows = input_ids.unfold(dimension=1, size=k, step=1)
            match = (windows == marker).all(dim=-1)                 # (B, L-k+1)
            for b in range(input_ids.size(0)):
                idxs = match[b].nonzero(as_tuple=False).flatten()
                if idxs.numel() == 0:
                    continue
                # Last occurrence; assistant content begins right after the marker.
                start = int(idxs[-1].item()) + k
                # Find the trailing <|im_end|> after start.
                tail = input_ids[b, start:]
                end_rel = (tail == self._im_end_id).nonzero(as_tuple=False).flatten()
                if end_rel.numel() == 0:
                    end = input_ids.size(1)
                else:
                    end = start + int(end_rel[0].item())
                if end > start:
                    labels[b, start:end] = input_ids[b, start:end]
                    n_with_cot += 1
        # Diagnostic: if zero rows have any CoT tokens, the mask is broken.
        if n_with_cot == 0:
            print("[CoT collator] WARNING: no assistant span located in any row "
                  "of this batch; L_cot will be zero.")

        durations = torch.tensor([m["duration_s"] for m in meta], dtype=torch.long)
        lam_true = torch.stack([
            hazards_from_retention(torch.tensor(m["retention_curve"]), int(m["duration_s"]))
            for m in meta
        ], dim=0)                                                   # (B, 60)

        return {
            "mm_inputs": dict(inputs),
            "labels": labels,
            "durations": durations,
            "lam_true": lam_true,
        }


class SFTHazardCoTTrainer(Trainer):
    def __init__(self, *args, alpha: float = 0.1, strict_spec: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.strict_spec = strict_spec

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mm = inputs["mm_inputs"]
        logits, lam_hat = model(mm)                                 # (B, L, V), (B, 60)

        # CoT cross-entropy on assistant span only.
        labels = inputs["labels"].to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        n_valid = (shift_labels != -100).sum()
        if n_valid > 0:
            L_cot = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            # Batch has no assistant tokens (e.g. val_with_cot with empty CoT).
            # Skip CoT gradient rather than emit NaN.
            L_cot = shift_logits.new_zeros(())

        # Hazard MSE in log space.
        T_i = inputs["durations"].to(lam_hat.device)
        lam_true = inputs["lam_true"].to(lam_hat.device).to(lam_hat.dtype)
        L_hazard = masked_hazard_log_mse(
            lam_hat.float(), lam_true.float(), T_i,
            strict_spec=self.strict_spec,
        )

        loss = L_hazard + self.alpha * L_cot
        self.log({"L_hazard": float(L_hazard.detach()),
                  "L_cot": float(L_cot.detach())})
        return (loss, {"logits": logits, "lam_hat": lam_hat}) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--strict_spec", action="store_true", default=True,
                    help="Use spec §3 hazard mask (duration-only, no "
                         "informative-segment filter). Default True for "
                         "spec compliance; --no_strict_spec to disable.")
    ap.add_argument("--no_strict_spec", dest="strict_spec",
                    action="store_false")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = RetentionVLM(args.model_id, tokenizer=tokenizer)

    # Gradient checkpointing on the thinker. Without this the 30k-token MM
    # sequence + CoT activations exceed 80GB even at batch_size 1.
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
    # Wrap the thinker, not the outer wrapper (see sft_mse.py comment).
    model.trunk.thinker = get_peft_model(model.trunk.thinker, lora_cfg)

    # Freeze vision + audio encoders AFTER LoRA wrap (§6 = §5: same flags).
    # PEFT injects adapters on all linears; we zero requires_grad on the
    # vision/audio adapters.
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
    collator = TTCCCollatorCoT(processor=processor, tokenizer=tokenizer,
                               nframes=args.nframes, max_pixels=args.max_pixels)

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
        # Disable eval: HF Trainer's eval loop calls model(**inputs) with
        # collator keys our forward doesn't accept. Final eval is via
        # make_test_preds_hazard.py + ttcc-eval.
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=(["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"]),
        run_name=f"sft_hazard_cot_a{args.alpha}_seed{args.seed}",
        warmup_ratio=0.05,
    )

    trainer = SFTHazardCoTTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
        alpha=args.alpha,
        strict_spec=args.strict_spec,
    )

    # §4 smoke verification.
    print("[smoke] loading first batch...")
    smoke_batch = collator([train_ds[i] for i in range(min(2, len(train_ds)))])
    print(f"[smoke] input_ids: {smoke_batch['mm_inputs']['input_ids'].shape}")
    if "pixel_values_videos" in smoke_batch["mm_inputs"]:
        print(f"[smoke] pixel_values_videos: {smoke_batch['mm_inputs']['pixel_values_videos'].shape}")
    if "input_features" in smoke_batch["mm_inputs"]:
        print(f"[smoke] input_features (audio): {smoke_batch['mm_inputs']['input_features'].shape}")
    labels = smoke_batch["labels"]
    n_cot = (labels != -100).sum(dim=1).tolist()
    print(f"[smoke] labels non-ignored per row: {n_cot}")
    print(f"[smoke] durations: {smoke_batch['durations'].tolist()}, "
          f"lam_true non-zero per row: {(smoke_batch['lam_true'] > 0).sum(1).tolist()}")
    if args.smoke_only:
        print("[smoke] --smoke_only set, exiting before training.")
        return

    trainer.train()

    # §6 requires saving the LoRA-merged trunk plus the hazard head separately,
    # so the GRPO stage can load a clean HF-format model. Since LoRA wraps
    # the thinker, merge happens at that level; the outer Omni wrapper is
    # then saved normally.
    merged_dir = os.path.join(args.output_dir, "merged_trunk")
    print(f"merging LoRA into thinker and saving full model to {merged_dir}")
    model.trunk.thinker = model.trunk.thinker.merge_and_unload()
    model.trunk.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)

    torch.save(
        {"hazard_head": model.hazard_head.state_dict()},
        os.path.join(args.output_dir, "hazard_head.pt"),
    )
    # The thinker is now merged in-place; the merged_trunk dir already has
    # the full HF-format model. No separate adapter-only save needed since
    # we merged before reaching this point.
    print(f"saved merged trunk + hazard head to {args.output_dir}")


if __name__ == "__main__":
    main()

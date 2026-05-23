"""
Produce test_preds.parquet for SFT-MSE checkpoint (§9 schema).

Columns: ad_id, duration, R_true, R_hat, hat_at_3, true_at_3, hat_at_T,
true_at_T, spearman_within_ad.

Run:
  python cs224r_project/eval/make_test_preds.py \
      --checkpoint cs224r_project/runs/sft_mse/seed42 \
      --test_jsonl cs224r_project/data/splits/test.jsonl \
      --out_parquet cs224r_project/runs/sft_mse/seed42/test_preds.parquet
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.sft_mse import (  # noqa: E402
    RetentionHeadModel, TTCCCollator, TTCCDataset, T_MAX,
)


def load_checkpoint(model_id: str, ckpt_dir: str) -> RetentionHeadModel:
    model = RetentionHeadModel(model_id)
    # LoRA adapter is on the thinker (see sft_mse.py).
    model.trunk.thinker = PeftModel.from_pretrained(model.trunk.thinker, ckpt_dir)
    head_state = torch.load(os.path.join(ckpt_dir, "retention_head.pt"), map_location="cpu")
    model.head.load_state_dict(head_state["head"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = load_checkpoint(args.model_id, args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ds = TTCCDataset(args.test_jsonl)
    collator = TTCCCollator(processor=processor)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator)

    rows = []
    with torch.no_grad():
        ad_iter = iter(ds)
        for batch in tqdm(loader, desc="infer"):
            mm = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in batch["mm_inputs"].items()}
            attn = batch["attention_mask"].to(device)
            r_hat = model(mm, attn).float().cpu().numpy()       # (B, 60)
            durations = batch["durations"].cpu().numpy()
            for b in range(r_hat.shape[0]):
                row = next(ad_iter)
                T_i = int(durations[b])
                ad_id = row["_meta"]["ad_id"]
                R_true_full = np.asarray(row["_meta"]["retention_curve"], dtype=np.float32)
                # Model predicts R_hat(t) for t = 1..60; prepend R_hat(0)=1.
                R_hat_tail = r_hat[b, :T_i]
                R_hat_full = np.concatenate([[1.0], R_hat_tail]).astype(np.float32)

                hat3 = float(R_hat_full[3]) if T_i >= 3 else float("nan")
                true3 = float(R_true_full[3]) if T_i >= 3 else float("nan")
                hatT = float(R_hat_full[T_i])
                trueT = float(R_true_full[T_i])
                # Within-ad Spearman over t = 1..T_i only (exclude the forced
                # R(0) = 1.0 anchor, which would inflate the correlation).
                rho, _ = spearmanr(R_hat_full[1:T_i + 1], R_true_full[1:T_i + 1])

                rows.append({
                    "ad_id": ad_id,
                    "duration": T_i,
                    "R_true": R_true_full.tolist(),
                    "R_hat": R_hat_full.tolist(),
                    "hat_at_3": hat3,
                    "true_at_3": true3,
                    "hat_at_T": hatT,
                    "true_at_T": trueT,
                    "spearman_within_ad": float(rho) if rho == rho else float("nan"),
                })

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {len(rows)} rows")


if __name__ == "__main__":
    main()

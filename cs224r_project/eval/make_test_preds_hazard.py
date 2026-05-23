"""
Produce test_preds.parquet for a SFT-Hazard+CoT checkpoint.

Schema matches make_test_preds.py (the SFT-MSE version):
  ad_id, duration, R_true, R_hat, hat_at_3, true_at_3, hat_at_T, true_at_T,
  spearman_within_ad.

NOTE: At inference we pass an *empty* `<cot></cot>` rather than letting the
model autoregressively generate CoT. The hazard head only depends on the
hidden state at `</cot>`; with empty content that hidden state still aggregates
the multimodal context. Generated-CoT inference (where each rollout samples
a CoT then reads the hazard at the closing tag, optionally averaging K
rollouts as in §9) is a follow-up.

Run:
  python cs224r_project/eval/make_test_preds_hazard.py \\
      --checkpoint /vol/runs/sft_hazard_cot/seed42_a0.1 \\
      --test_jsonl /vol/data/splits/test.jsonl \\
      --out_parquet /vol/runs/sft_hazard_cot/seed42_a0.1/test_preds.parquet
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.retention_vlm import (  # noqa: E402
    RetentionVLM, recover_curve, T_MAX,
)
from baselines.sft_hazard_cot import (  # noqa: E402
    TTCCCollatorCoT, TTCCDataset,
)


def load_checkpoint(model_id: str, ckpt_dir: str) -> RetentionVLM:
    """Stage 2 saves the LoRA-merged trunk WITHOUT the talker (the trainer
    called disable_talker before saving). To rebuild a loadable model we
    start from the public Qwen-Omni-3B base (which has the talker) and
    overwrite its thinker weights with the merged-trunk thinker weights.
    The hazard head loads from its own file.
    """
    print(f"Loading base {model_id} (has talker pieces)")
    backbone = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    )

    merged_dir = os.path.join(ckpt_dir, "merged_trunk")
    if os.path.isdir(merged_dir):
        # Directly load the merged thinker safetensors and stuff its weights
        # into the base backbone. Avoids the OmniForConditionalGeneration
        # loader checking for talker / spk_dict files we never saved.
        from safetensors.torch import load_file as load_safetensors
        from pathlib import Path
        import glob
        print(f"Overwriting thinker weights from {merged_dir}")
        sd = {}
        shards = sorted(glob.glob(os.path.join(merged_dir, "model*.safetensors")))
        for shard in shards:
            sd.update(load_safetensors(shard))
        # Strip the leading "thinker." prefix and load into the existing thinker.
        thinker_sd = {
            k[len("thinker."):]: v for k, v in sd.items() if k.startswith("thinker.")
        }
        if not thinker_sd:
            raise RuntimeError(
                f"No thinker.* weights found in {merged_dir}; check the merged checkpoint."
            )
        missing, unexpected = backbone.thinker.load_state_dict(thinker_sd, strict=False)
        print(f"  loaded {len(thinker_sd)} thinker weights; "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        # Fallback: original Qwen-Omni-3B with PEFT adapter on top.
        from peft import PeftModel
        backbone.thinker = PeftModel.from_pretrained(backbone.thinker, ckpt_dir)

    if hasattr(backbone, "disable_talker"):
        backbone.disable_talker()
    model = RetentionVLM(backbone)
    head_state = torch.load(
        os.path.join(ckpt_dir, "hazard_head.pt"), map_location="cpu",
    )
    model.hazard_head.load_state_dict(head_state["hazard_head"])
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
    tokenizer = processor.tokenizer
    model = load_checkpoint(args.model_id, args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ds = TTCCDataset(args.test_jsonl)
    # Reuse the training collator. The test split has empty CoT (post-merge),
    # so the assistant content is `<cot></cot>` — empty CoT span but the marker
    # is present, so the hazard head still reads the correct hidden state.
    collator = TTCCCollatorCoT(processor=processor, tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator)

    rows = []
    with torch.no_grad():
        ad_iter = iter(ds)
        for batch in tqdm(loader, desc="infer"):
            mm = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in batch["mm_inputs"].items()}
            _, lam_hat = model(mm)                              # (B, 60), fp32
            lam_hat = lam_hat.float().cpu()
            durations = batch["durations"].cpu().numpy()
            for b in range(lam_hat.size(0)):
                row = next(ad_iter)
                T_i = int(durations[b])
                ad_id = row["_meta"]["ad_id"]
                R_true_full = np.asarray(
                    row["_meta"]["retention_curve"], dtype=np.float32,
                )
                R_hat_full = recover_curve(lam_hat[b], T_i).numpy().astype(np.float32)

                hat3 = float(R_hat_full[3]) if T_i >= 3 else float("nan")
                true3 = float(R_true_full[3]) if T_i >= 3 else float("nan")
                hatT = float(R_hat_full[T_i])
                trueT = float(R_true_full[T_i])
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

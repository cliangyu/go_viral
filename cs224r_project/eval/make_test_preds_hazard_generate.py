"""
Stage 2 inference with autoregressive CoT generation + K-rollout averaging
(§9 protocol).

For each test ad:
  1. Build user-only conversation (no assistant turn).
  2. For each of K rollouts (with sampling at temperature `--temp`):
       a. model.thinker.generate(...) on prompt + "<cot>" → CoT text
          (stop at "</cot>" or max_new_tokens).
       b. Concatenate "<cot>{cot}</cot>" into the assistant turn.
       c. Forward pass to read hazard from `</cot>` position.
  3. Average lam_hat across K rollouts.
  4. Recover R_hat via exp(-cumsum).

Output schema identical to make_test_preds_hazard.py:
  ad_id, duration, R_true, R_hat, hat_at_3, true_at_3, hat_at_T, true_at_T,
  spearman_within_ad
plus diagnostic columns:
  n_rollouts, cot_texts (list of K strings).

Run:
  python make_test_preds_hazard_generate.py \\
      --checkpoint /vol/runs/sft_hazard_cot/seed42_a0.1_strict \\
      --test_jsonl /vol/data/splits/test.jsonl \\
      --out_parquet ... --rollouts 3 --temp 0.7 --top_p 0.9
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
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.retention_vlm import (  # noqa: E402
    RetentionVLM, recover_curve, T_MAX,
)


USER_PROMPT = (
    "<video><audio>Predict the per-second retention curve. "
    "Reason about hook, pacing, and audio cues. "
    "Wrap reasoning in <cot>...</cot>."
)


def load_checkpoint(model_id: str, ckpt_dir: str) -> RetentionVLM:
    """Same as the empty-CoT variant: stitch merged thinker into a fresh base."""
    print(f"Loading base {model_id}")
    backbone = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    )
    merged_dir = os.path.join(ckpt_dir, "merged_trunk")
    if os.path.isdir(merged_dir):
        from safetensors.torch import load_file as load_safetensors
        import glob
        print(f"Overwriting thinker weights from {merged_dir}")
        sd = {}
        for shard in sorted(glob.glob(os.path.join(merged_dir, "model*.safetensors"))):
            sd.update(load_safetensors(shard))
        thinker_sd = {
            k[len("thinker."):]: v for k, v in sd.items() if k.startswith("thinker.")
        }
        if not thinker_sd:
            raise RuntimeError(f"No thinker.* weights in {merged_dir}")
        missing, unexpected = backbone.thinker.load_state_dict(thinker_sd, strict=False)
        print(f"  loaded {len(thinker_sd)} thinker weights; "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
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


def build_prompt(video_path: str, audio_path: str, with_assistant_cot: str | None,
                 processor):
    """Build either a user-only prompt (for generation) or a full conv
    (for hazard forward pass).

    Returns: (text, audios, images, videos)
    """
    from qwen_omni_utils import process_mm_info
    user = {
        "role": "user",
        "content": [
            # Aligned to Liangyu's full-SFT config (sft_v2cot_full.sh).
            {"type": "video", "video": video_path,
             "max_pixels": 200704, "nframes": 32},
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": USER_PROMPT},
        ],
    }
    if with_assistant_cot is None:
        conv = [user]
        add_gen = True
    else:
        conv = [user, {"role": "assistant", "content": f"<cot>{with_assistant_cot}</cot>"}]
        add_gen = False
    text = processor.apply_chat_template([conv], add_generation_prompt=add_gen,
                                         tokenize=False)
    audios, images, videos = process_mm_info([conv], use_audio_in_video=False)
    return text, audios, images, videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--rollouts", type=int, default=3,
                    help="Number of CoT samples to average")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    model = load_checkpoint(args.model_id, args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Find </cot> token id(s) for stopping criteria and position lookup.
    close_cot_ids = tokenizer.encode("</cot>", add_special_tokens=False)
    print(f"</cot> tokenizes to {close_cot_ids}")

    # Load test rows.
    rows_data = []
    with open(args.test_jsonl) as f:
        for line in f:
            rows_data.append(json.loads(line))

    out_rows = []
    failed = 0
    for row in tqdm(rows_data, desc="ads"):
        ad_id = row["_meta"]["ad_id"]
        T_i = int(row["_meta"]["duration_s"])
        R_true_full = np.asarray(row["_meta"]["retention_curve"], dtype=np.float32)
        video_path = row["videos"][0]
        audio_path = row["audios"][0]

        # Step 1: K rollouts of CoT generation + hazard forward.
        cot_texts: list[str] = []
        lam_hats: list[torch.Tensor] = []

        try:
            for k in range(args.rollouts):
                # 1a. Build user-only prompt for generation.
                text, audios, images, videos = build_prompt(
                    video_path, audio_path, with_assistant_cot=None,
                    processor=processor,
                )
                inputs = processor(
                    text=text, audio=audios, images=images, videos=videos,
                    return_tensors="pt", padding=True, use_audio_in_video=False,
                )
                inputs = {k_: v.to(device) if torch.is_tensor(v) else v
                          for k_, v in inputs.items()}

                # 1b. Generate CoT tokens with the THINKER's lm_head.
                # We append "<cot>" manually after the assistant role marker
                # by constructing the assistant content directly. Easier:
                # generate until </cot> appears, decode, post-process.
                with torch.no_grad():
                    # Note: we set thinker_do_sample=True to get diverse rollouts
                    # via sampling. Use temperature + top_p for variety.
                    gen_out = model.trunk.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=(args.temp > 0),
                        temperature=args.temp if args.temp > 0 else 1.0,
                        top_p=args.top_p,
                        return_audio=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                # gen_out can be a tuple (text_ids, audio) — we already turned audio off
                if isinstance(gen_out, tuple):
                    text_ids = gen_out[0]
                else:
                    text_ids = gen_out
                # Extract new tokens beyond the prompt.
                new_tokens = text_ids[0, inputs["input_ids"].shape[1]:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                # 1c. Extract just the CoT body (between any <cot> the model
                # produced and </cot>). The model is trained to wrap reasoning
                # in <cot>...</cot> — if it skips the opening, treat the
                # whole thing up to </cot> as CoT.
                cot_text = generated_text
                if "</cot>" in cot_text:
                    cot_text = cot_text.split("</cot>")[0]
                if "<cot>" in cot_text:
                    cot_text = cot_text.split("<cot>", 1)[1]
                cot_text = cot_text.strip()
                cot_texts.append(cot_text)

                # 1d. Build a full conv with the generated CoT and run a
                # forward pass to read hazard from </cot>.
                text2, audios2, images2, videos2 = build_prompt(
                    video_path, audio_path,
                    with_assistant_cot=cot_text, processor=processor,
                )
                inputs2 = processor(
                    text=text2, audio=audios2, images=images2, videos=videos2,
                    return_tensors="pt", padding=True, use_audio_in_video=False,
                )
                inputs2 = {k_: v.to(device) if torch.is_tensor(v) else v
                           for k_, v in inputs2.items()}
                with torch.no_grad():
                    _, lam_hat = model(inputs2)
                lam_hats.append(lam_hat.float().cpu()[0])  # (60,)

            # 2. Average hazards across rollouts.
            lam_avg = torch.stack(lam_hats, dim=0).mean(dim=0)  # (60,)
            R_hat_full = recover_curve(lam_avg, T_i).numpy().astype(np.float32)
        except Exception as e:
            failed += 1
            print(f"ad {ad_id} failed: {e}")
            # Fallback: skip.
            continue

        hat3 = float(R_hat_full[3]) if T_i >= 3 else float("nan")
        true3 = float(R_true_full[3]) if T_i >= 3 else float("nan")
        hatT = float(R_hat_full[T_i])
        trueT = float(R_true_full[T_i])
        rho, _ = spearmanr(R_hat_full[1:T_i + 1], R_true_full[1:T_i + 1])

        out_rows.append({
            "ad_id": ad_id,
            "duration": T_i,
            "R_true": R_true_full.tolist(),
            "R_hat": R_hat_full.tolist(),
            "hat_at_3": hat3,
            "true_at_3": true3,
            "hat_at_T": hatT,
            "true_at_T": trueT,
            "spearman_within_ad": float(rho) if rho == rho else float("nan"),
            "n_rollouts": args.rollouts,
            "cot_texts": cot_texts,
        })

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {len(out_rows)} rows ({failed} failed)")


if __name__ == "__main__":
    main()

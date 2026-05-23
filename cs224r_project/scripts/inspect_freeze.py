"""
Verify that vision + audio encoders are actually frozen after the substring
freeze + LoRA wrap. Prints:
  - total trainable params
  - any trainable params whose name contains 'visual', 'vision', 'audio',
    'mm_', 'projector', or 'merger' (these should NOT be trainable per §5).

Run on Modal (we don't have qwen-omni locally):
  modal run cs224r_project/modal/modal_inspect_freeze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.sft_mse import RetentionHeadModel
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"


def main():
    print(f"Loading {MODEL_ID} ...")
    model = RetentionHeadModel(MODEL_ID)

    # Replicate the freeze + LoRA wrap from sft_mse.py main().
    for n, p in model.trunk.named_parameters():
        if "visual" in n or "audio_tower" in n or "vision" in n:
            p.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=32, target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model.trunk.thinker = get_peft_model(model.trunk.thinker, lora_cfg)

    # ---- Inspect ----
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")

    suspicious_keys = ("visual", "vision", "audio", "mm_", "projector", "merger")
    suspicious = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lower = n.lower()
        if any(k in lower for k in suspicious_keys):
            suspicious.append((n, p.numel()))

    print(f"\n=== Trainable params matching suspicious keys ===")
    print(f"({len(suspicious)} parameters)")
    sus_total = sum(c for _, c in suspicious)
    print(f"Total suspicious-trainable params: {sus_total:,} "
          f"({100*sus_total/trainable:.2f}% of trainable, "
          f"{100*sus_total/total:.4f}% of total)")
    print()
    # Print the unique parameter name *prefixes* (collapse layer index).
    import re
    prefixes = {}
    for n, c in suspicious:
        prefix = re.sub(r"\.\d+\.", ".<i>.", n)
        prefixes.setdefault(prefix, [0, 0])
        prefixes[prefix][0] += 1
        prefixes[prefix][1] += c
    for prefix in sorted(prefixes):
        n_occurrences, n_params = prefixes[prefix]
        print(f"  [{n_occurrences:3d}x, {n_params:>12,} params]  {prefix}")


if __name__ == "__main__":
    main()

# Incident Review: Audio Tower Positional Embedding Overflow (2026-05-28)

## Part 1 — What happened

**Timeline (UTC, 2026-05-28):**

- **05:00**: After resolving the SDPA-override incident, relaunched V8 training
  with `attn_impl: flash_attention_3` actually wired through.

- **05:02**: Both nodes' 16 H100s allocated, ZeRO-3 sharded, model loaded
  (4.7B params, 3.4B trainable). Training reached `compute_loss` → first
  forward pass.

- **05:03**: Training crashed on all 16 ranks with:
  ```
  torch.AcceleratorError: CUDA error: device-side assert triggered
  ```
  with kernel-side message:
  ```
  IndexKernelUtils.cu:16: vectorized_gather_kernel: block: [2310,0,0],
  thread: [64,0,0] Assertion `ind >=0 && ind < ind_dim_size` failed
  ```

- **05:10**: Stack trace localized the error to `audio_tower.forward` →
  encoder_layer → `_index_first_axis` (FA2's varlen `_upad_input` path).
  Not FA3-related; not OOM-related (`dmesg` clean, free RAM 1.1 TB).

- **05:15**: Wrote isolated single-GPU repro (`aud_repro.py`) that loads one
  V8 jsonl row, dumps audio_tower config, runs `get_audio_features` alone.

- **05:20**: Repro on ad_id 7631598650221608968 (T=45s) confirmed:
  ```
  positional_embedding.shape = (1500, 1280)
  max_source_positions       = 1500
  input_features.shape       = (1, 128, 30000)   # 30K mel time-steps
  feature_attention_mask sum = 4517              # 4517 frames after CNN
  4517 > 1500 → ROOT CAUSE
  ```

**Net cost**: ~30 min GPU compute (model load + abort cycles on 16 H100s) ≈
$160 wasted. Caught at first forward step → no checkpoint contamination.

## Part 2 — Why it happened (technical deep dive)

### Qwen2.5-Omni audio tower architecture

Qwen2.5-Omni-3B includes an **audio encoder** subblock (`model.thinker.audio_tower`)
that processes mel-spectrogram features into the LM's hidden space. Its
config (from the base model's `config.json`):

```python
audio_config = {
    "num_mel_bins":       128,    # 128-band log-mel spectrogram
    "d_model":            1280,   # encoder hidden size
    "n_window":           100,    # local attention window size
    "num_hidden_layers":  32,     # encoder depth
    "output_dim":         2048,   # projects to LM hidden size
    "max_source_positions": 1500, # ← CAPACITY OF positional_embedding
}
```

The `max_source_positions: 1500` corresponds to the size of the **learned
positional embedding** at `model.thinker.audio_tower.positional_embedding.positional_embedding`
(shape `[1500, 1280]`).

### Audio sample rate math

Qwen-Omni's audio frontend:
1. Raw audio → log-mel spectrogram at standard 30s windows
2. Mel features: `[batch, 128 mels, T_time_steps]` with `T ≈ 100 * seconds`
   (effective rate ~100 Hz after STFT and stacking)
3. Two strided Conv1D layers downsample: `T → T/4` approximately
4. Result: ~25 Hz "audio frames" fed to encoder

For a 60-second video, expect ~6000 mel time-steps → ~1500 frames after CNN.
For a 45-second video, ~4500 mel time-steps → ~1100 frames after CNN.

**But our repro showed 4517 frames after the CNN for a 45s ad** — meaning the
effective rate is closer to ~100 Hz per second, not 25 Hz. (Likely because
Qwen-Omni stacks mel features and the "frame" boundary depends on the
stride; the framework reports the pre-CNN count.)

Either way, max_source_positions=1500 is sized for ~15 seconds of audio.
TTCC ads range 5-60s. Anything > 15s overflows.

### Why this doesn't crash on text-only Qwen2.5-7B fine-tuning

The audio_tower is a Qwen2.5-Omni-specific subblock. Pure text or vision-only
training never instantiates it. We are the rare case that:
1. Trains Qwen2.5-Omni (multimodal)
2. Uses ads with audio > 15s
3. Doesn't pre-truncate audio at the data prep step

### Why the failure mode is silent until forward

The audio encoder forward (`modeling_qwen2_5_omni.py:807` in transformers
v4.56.2) does:
```python
padded_embed = padded_embed + self.positional_embedding.positional_embedding[
    : padded_embed.shape[1], :
].unsqueeze(0).to(padded_embed.dtype)
```

When `padded_embed.shape[1] > 1500`, the slice `[:padded_embed.shape[1]]`
on a 1500-row tensor **silently returns the full 1500 rows** (Python's
slice semantics, not an error). Then `padded_embed + pe[1500]` requires
broadcasting along dim-1, which fails dimensionally — BUT bf16 + GPU + the
specific shape can let the broadcast through with truncated semantics in
some cases. Subsequent `padded_mask_after_cnn` indexing and cu_seqlens
computation then produce indices > the actual hidden_state length →
out-of-bounds gather in the FA2 varlen `_index_first_axis` call.

The kernel-side assert fires asynchronously, so the Python stack trace
points to `_index_first_axis` but the **logical** error is upstream in the
PE-slice silent-truncate.

### Why it was masked during SDPA testing

The previous incident (`INCIDENT_2026-05-28_SDPA_OVERRIDE.md`) killed
training before any forward step completed (we caught the SDPA override
mid-warmup). So we never reached the audio tower forward and didn't see
this. The SDPA override → FA3 transition is unrelated to the underlying
audio length issue, which would have crashed regardless.

## Part 3 — What we change

### Immediate (this run)

**Drop audio from V8 training data.**

Rationale:
- `audio_tower` is frozen in our yaml (`freeze_aligner: true` + freeze list
  includes `thinker.audio_tower`). It contributes a constant (non-learning)
  multimodal signal anyway.
- TTCC retention prediction primarily uses video frames + text CoT, not
  audio prosody. Audio is a nice-to-have, not load-bearing.
- Other fixes (PE interpolation, audio chunking, dual-tower) require
  transformers source patches and re-running validation. Drop-audio is a
  one-line jsonl change.

**Implementation**: pass `audios: []` in every V8 jsonl row. Template will
skip `input_features` allocation, `get_audio_features` won't be called.

### What we lose by dropping audio (and how to recover later)

- Music/voice-over signal that correlates with retention (silent ads
  retain worse, energetic music keeps viewers)
- Spoken call-to-action timing
- Sound effect cues at transitions

If V8 IBS underperforms expectations, the next iteration should either:

1. **Audio truncation** at the template/preprocessor level: keep first 15
   seconds of audio (≤ max_source_positions / 100). Captures the hook
   period which is usually most retention-decisive.

2. **Audio chunking with windowed encoding**: process audio in 15s windows,
   concatenate encoder outputs. Requires patching `audio_tower.forward` to
   loop over chunks. ~50 LOC.

3. **PE interpolation**: linearly interpolate the [1500, 1280] PE up to
   [6000, 1280] at model load time. Standard ALiBi-style trick. Risk:
   audio encoder was trained at the 1500 capacity so interpolated PE may
   degrade quality. Cheap to try.

4. **Separate audio tower with longer context**: train a from-scratch audio
   encoder with `max_source_positions=6000`, swap it in. Most principled,
   most expensive.

### Process changes

- **Data-prep validation must include media-length checks**: add to
  `validate_v8_launch.sh` a pass that reads a sample of jsonl rows, computes
  audio frame count, asserts `<= max_source_positions`. We had ad-length T
  checks (5-60s) but not audio-frame checks.

- **Pre-launch single-row forward smoke must succeed**: the `aud_repro.py`
  pattern (one row, one GPU, isolated subblock) should run before any
  full-cluster launch. Add this to the V8 launch runbook as a mandatory
  step.

- **Multimodal data invariants need explicit documentation**: TTCC ads have
  length 5-60s. Any model we use must support:
  - Video frames ≥ 60 / fps (current fps=1.0 → 60 frames OK)
  - Audio frames ≥ 60 * audio_fps (currently fails for audio_fps≥25)
  - Text length ≥ 49152 tokens (current yaml max_length covers)
  Document these in `data/MULTIMODAL_BOUNDS.md` next to the V8 jsonl spec.

### Reproducing the diagnosis (for future incidents)

```bash
# 1. SSM into the affected node
aws ssm start-session --target <node-id> --region us-east-2

# 2. Push the repro script (gzip+base64 to avoid SSM line-length limits)
# (script content in commit c018b584's FLASH_ATTENTION_INSTALL.md or repo)

# 3. Run with CUDA_LAUNCH_BLOCKING=1 to surface the synchronous assert
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
    /opt/dlami/nvme/work/swift_venv/bin/python /tmp/aud_repro.py \
    flash_attention_3 <row-index> > /tmp/repro.log 2>&1

# 4. Key signals in output:
#    - pe.shape  vs  MAX_AUDIO_FRAMES
#    - stack trace location (audio tower vs LM vs head)
#    - exit code -9 (SIGKILL → OOM-likely) vs 1 (assert)
```

## Affected files

- `/home/ssm-user/work/data/ttcc_v8/ttcc_train_with_cot.jsonl` (39,375 rows
  with `audios: [<mp4>]` field — will be rewritten to `audios: []`)
- `/home/ssm-user/work/data/ttcc_holdout/val_200_no_cot.jsonl` (same)
- Both nodes need re-staged data; data lives only on NVMe (ephemeral).

## Related incidents

- [INCIDENT_2026-05-26_EVAL_LEAK.md](../../../../ttcc-eval/INCIDENT_2026-05-26_EVAL_LEAK.md) —
  V7 R(t) leak in assistant span.
- [INCIDENT_2026-05-28_SDPA_OVERRIDE.md](INCIDENT_2026-05-28_SDPA_OVERRIDE.md) —
  Silent SDPA override defeated FA3 install.

All three share the failure pattern: **a silent default value or silent
truncation produced wrong results without surface error**. The class is
"silent semantic divergence in a multimodal pipeline." Process change: any
shared infrastructure (data prep, launchers, model configs) must surface
its choices loudly in logs, not silently absorb them.

## Aphorisms

- "A silent slice on an undersized tensor is a future incident."
- "If your model has a multimodal subblock you're not actually training,
  it's a load-bearing dependency anyway — its forward pass must succeed."
- "max_source_positions is a hard contract with the data, not a hint."

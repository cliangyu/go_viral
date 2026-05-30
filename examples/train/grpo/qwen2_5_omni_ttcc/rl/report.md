# Baseline vs RL — cross-ad SRCC + IBS

Two metrics, same leak-free bypass eval, same val set (n=1332), fps=1.0.
- **SRCC** (cross-ad ranking, t∈[1,30], higher better)
- **IBS** (Integrated Brier Score = mean_t (R_pred−R_true)², lower better)

| ckpt | SRCC | ΔSRCC (95% CI) | IBS_full | ΔIBS (95% CI) | calibration |
|---|---|---|---|---|---|
| **baseline ckpt-225 (SFT)** | 0.4393 | — | 0.00672 | — | beats climatology (0.00814) |
| nodeA_reshape_sft | 0.4553 (null) | +0.0161 [-0.0151, +0.0465] | 0.01307 (WORSE) | +0.00635 [+0.00570, +0.00705] | worse than climatology |
| nodeB_proper_rl | 0.4265 (null) | -0.0128 [-0.0346, +0.0078] | 0.00888 (WORSE) | +0.00216 [+0.00153, +0.00280] | worse than climatology |

**Read:** SRCC null for both RL runs (CIs include 0); IBS significantly WORSE for both (CIs entirely positive). The SRCC-only view hid real calibration damage — which is why both metrics are reported. climatology B₁ IBS = 0.00814 (val-derived reference).

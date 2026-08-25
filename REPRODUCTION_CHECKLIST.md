# Reproduction Checklist

> **Target Paper:** "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"  
> Mathematics 2025, 13, 3942

Track the status of each paper requirement.

Legend: `[x]` Done — `[/]` Partial — `[ ]` Not started

---

## Architecture (Section 2.1)

- [x] 3D U-Net base architecture
- [x] 5 encoder levels
- [x] Channel progression: 16 → 32 → 64 → 128 → 256 (Figure 1)
- [x] 3×3×3 Conv3d + BatchNorm3d + ReLU
- [x] MaxPool3d (2×2×2) downsampling
- [x] ConvTranspose3d (2×2×2) upsampling
- [x] Attention Gates at every skip connection
- [x] 1×1×1 convolutions for channel alignment in AG
- [x] Layer Normalization in AG
- [x] Additive attention (θ(x) + φ(g))
- [x] Sigmoid gating → attention coefficients ∈ [0,1]
- [x] 1×1×1 final convolution for binary output

## Preprocessing (Section 2.2)

- [x] HU clipping to [-100, 240]
- [x] Min-max normalization to [0, 1]
- [x] Resampling to 1×1×1 mm³ isotropic
- [x] Linear interpolation for images
- [x] Nearest-neighbor interpolation for labels
- [x] Crop to 224×224×128
- [x] Center-crop (documented ambiguity — no GT guidance)

## Augmentation (Section 2.2)

- [x] Random rotation ±10° per axis
- [x] Random flip, probability 0.5
- [x] Random shift up to 10%
- [x] Identical transformations for image and mask
- [x] Correct interpolation (linear for image, nearest for mask)

## Loss Function (Section 2.3)

- [x] Dice Loss implementation
- [x] Focal Loss implementation
- [x] Combined Dice + Focal Loss
- [x] Focal γ = 2 (documented — paper omits value)
- [x] Standard Dice with 2× numerator (documented — paper omits factor)

## Optimizer & Scheduler (Section 2.3)

- [x] Adam optimizer
- [x] Learning rate = 6e-4 (documented typo — paper prints 6×10⁴)
- [x] ReduceLROnPlateau scheduler
- [x] Patience = 100 (paper-specified)
- [x] Weight decay = 0 (Adam default, paper does not specify)

## Training Configuration

- [x] Batch size = 1 (documented — paper does not specify)
- [x] AMP mixed precision (engineering optimization)
- [x] Checkpoint saving every epoch
- [x] Checkpoint resume after disconnect
- [x] Per-epoch CSV logging

## Dataset (Section 2.4)

- [x] NIH Pancreas-CT support (TCIA v2, 80 cases)
- [ ] MSD Pancreas support (281 cases — future work)
- [x] 5-fold cross-validation
- [x] Independent test set (16 cases)
- [x] Patient-level splitting (no data leakage)
- [x] Splits saved as reproducible JSON

## Evaluation (Section 2.4)

- [x] Volumetric Dice Similarity Coefficient (DSC)
- [x] Average Symmetric Surface Distance (ASSD) in mm
- [x] 95th Percentile Hausdorff Distance (HD95) in mm
- [x] Physical voxel spacing used for surface distances
- [x] Per-patient results CSV
- [x] Aggregate results (mean ± std) CSV
- [x] NIfTI prediction mask saving

## Infrastructure

- [x] Google Colab notebook
- [x] Google Drive integration
- [x] Lazy loading (no RAM preload of 9 GB dataset)
- [x] Comprehensive test suite (12 tests)
- [x] Requirements.txt
- [x] .gitignore (dataset excluded from Git)
- [x] Visualization utilities (volume slices, loss/dice curves)
- [x] Memory benchmark tool

## Documentation

- [x] README.md with architecture, pipeline, and quick start
- [x] PAPER_REPRODUCTION_NOTES.md with all 8 ambiguities documented
- [x] REPRODUCTION_CHECKLIST.md (this file)
- [ ] VALIDATION_REPORT.md (populated after training runs)

---

## Summary

| Category | Done | Total | % |
|---|---|---|---|
| Architecture | 12 | 12 | 100% |
| Preprocessing | 7 | 7 | 100% |
| Augmentation | 5 | 5 | 100% |
| Loss | 5 | 5 | 100% |
| Optimizer | 5 | 5 | 100% |
| Training | 5 | 5 | 100% |
| Dataset | 6 | 7 | 86% |
| Evaluation | 7 | 7 | 100% |
| Infrastructure | 8 | 8 | 100% |
| Documentation | 3 | 4 | 75% |
| **Total** | **63** | **65** | **97%** |

**Remaining:** MSD dataset support (future work), VALIDATION_REPORT (after training)

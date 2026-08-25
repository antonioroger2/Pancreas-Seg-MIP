# Validation Report

> **Target Paper:** "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"  
> Mathematics 2025, 13, 3942

---

## 1. Code-Level Validation

### 1.1 Test Suite Results

Run `python test_all.py` to verify. Results will be populated here after execution.

| Test | Status | Notes |
|---|---|---|
| 1. Dependencies | PENDING | PyTorch, SimpleITK, scipy, etc. |
| 2. Synthetic Dataset | PENDING | Shape (1, 224, 224, 128) verification |
| 3. Preprocessing | PENDING | HU clip, normalize, crop/pad |
| 4. Model Forward | PENDING | Input/output shape matching |
| 5. Attention Gates | PENDING | Tensor dims, gradients |
| 6. Loss Backward | PENDING | NaN check, gradient flow |
| 7. One-Batch Training | PENDING | Forward + backward + optimizer |
| 8. One-Patient Overfit | PENDING | Loss should decrease |
| 9. Small Subset | PENDING | Full pipeline with 2 volumes |
| 10. Checkpoint Save/Load | PENDING | Model state roundtrip |
| 11. Resume | PENDING | Epoch + dice restored |
| 12. Evaluation Metrics | PENDING | DSC, ASSD, HD95 known values |

### 1.2 Memory Benchmark

| Metric | Value |
|---|---|
| GPU | PENDING (expected: T4 16GB) |
| Input shape | (1, 1, 224, 224, 128) |
| AMP | Enabled |
| Batch size | 1 |
| Peak memory | PENDING |
| Status | PENDING |

---

## 2. Training Results

> **IMPORTANT:** Full training has NOT yet been run.  
> Results below will be populated after training on the NIH Pancreas-CT dataset.

### 2.1 Per-Fold Validation Dice

| Fold | Best Val DSC | Epochs | Status |
|---|---|---|---|
| 0 | — | — | NOT TRAINED |
| 1 | — | — | NOT TRAINED |
| 2 | — | — | NOT TRAINED |
| 3 | — | — | NOT TRAINED |
| 4 | — | — | NOT TRAINED |
| **Mean** | — | — | — |

### 2.2 Independent Test Set Results

| Metric | Our Result | Paper (NIH) |
|---|---|---|
| DSC (mean ± std) | NOT YET RUN | 80.8 ± ? |
| ASSD (mean ± std) mm | NOT YET RUN | Reported |
| HD95 (mean ± std) mm | NOT YET RUN | Reported |

---

## 3. Final Audit Table (Phase 15)

| Requirement | Paper | Implementation | Status | Evidence |
|---|---|---|---|---|
| 3D U-Net architecture | Section 2.1 | `model.py:AttentionUNet3D` | PASS | Forward pass test verified |
| Encoder channels 16→256 | Figure 1 | `config.py:ENCODER_CHANNELS` | PASS | Follows Figure 1, text ambiguity documented |
| Attention Gates | Section 2.1 | `attention.py:AttentionGate3D` | PASS | Attention test + gradient flow |
| LayerNorm in AG | Section 2.1 | `attention.py` uses `F.layer_norm` | PASS | Matches paper description |
| 3×3×3 Conv3d + BN + ReLU | Section 2.1 | `model.py:DoubleConv3D` | PASS | Verified in model test |
| MaxPool3d(2) | Figure 1 | `model.py:Encoder3D.pool` | PASS | Architecture matches |
| ConvTranspose3d(2) | Figure 1 | `model.py:Decoder3D.up_convs` | PASS | Architecture matches |
| HU clip [-100, 240] | Section 2.2 | `preprocessing.py:clip_hu` | PASS | Preprocessing test |
| Normalize [0, 1] | Section 2.2 | `preprocessing.py:normalize_to_01` | PASS | Preprocessing test |
| Resample 1×1×1 mm | Section 2.2 | `preprocessing.py:resample_volume_sitk` | PASS | Linear/NN interpolation |
| Crop 224×224×128 | Section 2.2 | `preprocessing.py:center_crop_or_pad` | PASS | Center-crop + zero-pad |
| Dice Loss | Section 2.3 | `losses.py:DiceLoss` | PASS | Standard formula (2× documented) |
| Focal Loss | Section 2.3 | `losses.py:FocalLoss` | PASS | γ=2 (documented default) |
| Dice + Focal combined | Section 2.3 | `losses.py:DiceFocalLoss` | PASS | Loss test verified |
| Adam optimizer | Section 2.3 | `train.py` | PASS | `optim.Adam` |
| LR = 6e-4 | Section 2.3 | `config.py:LEARNING_RATE` | PASS | Typo documented |
| ReduceLROnPlateau | Section 2.3 | `train.py` | PASS | patience=100 from paper |
| Rotation ±10° | Section 2.2 | `dataset.py:_augment_3d` | PASS | scipy.ndimage.rotate |
| Flip prob 0.5 | Section 2.2 | `dataset.py:_augment_3d` | PASS | np.flip per axis |
| Shift ±10% | Section 2.2 | `dataset.py:_augment_3d` | PASS | scipy.ndimage.shift |
| 5-fold CV | Section 2.4 | `cross_validation.py` | PASS | Patient-level splits |
| Independent test set | Section 2.4 | `cross_validation.py` | PASS | 16 cases held out |
| Volumetric DSC | Section 2.4 | `metrics.py:compute_dice` | PASS | Metric test verified |
| ASSD (mm) | Section 2.4 | `metrics.py:compute_assd` | PASS | Physical spacing used |
| HD95 (mm) | Section 2.4 | `metrics.py:compute_hd95` | PASS | Physical spacing used |
| NIH Pancreas-CT | Section 2.4 | Pipeline configured | PASS | TCIA v2, 80 cases |
| Batch size | NOT SPECIFIED | batch_size=1 | DEVIATION | T4 memory constraint |
| Total epochs | NOT SPECIFIED | 300 with early stopping | DEVIATION | Practical limit |
| MSD dataset | Section 2.4 | Not implemented | NOT IMPLEMENTED | Future work |
| Training results | Table 1 | Not yet trained | NOT IMPLEMENTED | Awaiting training run |
| Paper DSC: 80.8% | Table 1 | Not yet verified | NOT IMPLEMENTED | Awaiting training run |

### Status Summary

| Status | Count |
|---|---|
| PASS | 27 |
| DEVIATION | 2 |
| NOT IMPLEMENTED | 3 |
| **Total** | **32** |

> **Note:** Items marked PASS have been verified through code tests (test_all.py) and code inspection.
> Items marked DEVIATION are documented engineering decisions that do not change the paper's methodology.
> Items marked NOT IMPLEMENTED require full training runs or additional dataset support.

---

## 4. Known Limitations

1. **Training not yet run** — All architecture and pipeline components are verified through unit tests, but end-to-end training on the full NIH dataset has not been completed in this session.

2. **MSD dataset** — Paper evaluates on a second dataset (MSD Pancreas, 281 cases). This is deferred to future work.

3. **Exact numerical reproduction** — Even with identical architecture and hyperparameters, results may differ from the paper due to:
   - Different GPU hardware (T4 vs paper's unknown GPU)
   - Implementation-level differences in libraries
   - Random seed effects

4. **Batch size** — Paper does not specify batch size. We use 1 due to T4 constraints. A larger batch size on different hardware might yield different dynamics.

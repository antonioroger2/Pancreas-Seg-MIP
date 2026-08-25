# Paper Reproduction Notes

> **Target Paper:** "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"  
> Mathematics 2025, 13, 3942 — DOI: 10.3390/math13243942

This document records the exact distinction between:
- **A. Directly Stated by the Paper**
- **B. Clearly Implied by the Paper**
- **C. Ambiguous in the Paper**
- **D. Implementation Assumptions**
- **E. Hardware-Driven Deviations**
- **F. Dataset-Driven Deviations**

---

## 1. Classification of Architectural & Methodological Details

| Item | Category | Paper Specification | Reproduction Implementation | Rationale / Documentation |
|---|---|---|---|---|
| **Base Model** | **A. Directly Stated** | 3D Attention U-Net | `model.py:AttentionUNet3D` | Full 3D convolutional encoder-decoder network. |
| **Encoder Channels** | **C. Ambiguous** | Text: "2→16"; Figure 1: 16→32→64→128→256 | `config.py:ENCODER_CHANNELS = [16, 32, 64, 128, 256]` | Figure 1 is mathematically standard; text represents a typo. |
| **Attention Gate** | **A. Directly Stated** | Additive attention with LayerNorm and Sigmoid gating | `attention.py:AttentionGate3D` | $\sigma(\psi(\text{ReLU}(\theta(x) + \phi(g))))$, feature modulation $x \odot \alpha$. |
| **HU Windowing** | **A. Directly Stated** | Clipped to [-100, 240] HU | `preprocessing.py:clip_hu` | Exact soft-tissue pancreatic range. |
| **Normalization** | **A. Directly Stated** | Normalized to [0, 1] | `preprocessing.py:normalize_to_01` | Linear min-max scaling after clipping. |
| **Resampling** | **A. Directly Stated** | 1×1×1 mm³ isotropic | `preprocessing.py:resample_volume_sitk` | Linear interpolation for CT; Nearest-Neighbor for labels. |
| **Crop Dimensions** | **A. Directly Stated** | Fixed 224×224×128 | `preprocessing.py:center_crop_or_pad` | Spatial tensor dimensions $(H=224, W=224, D=128)$. |
| **Crop Origin** | **D. Implementation Assumption** | "Centered on abdominal region" (no algorithm given) | Geometric volume center-crop without GT guidance | No ground-truth mask is used to prevent label leakage. |
| **Loss Function** | **A. Directly Stated** | Combined Dice + Focal Loss | `losses.py:DiceFocalLoss` | Primary objective function ($L_{\text{total}} = L_{\text{Dice}} + L_{\text{Focal}}$). |
| **Focal Loss Parameters** | **D. Implementation Assumption** | Not numerically specified | $\gamma = 2.0, \alpha = 0.25$ | Lin et al. (2017) standard default values used and documented. |
| **Dice Loss Formula** | **C. Ambiguous** | Eq.(3) omits $2\times$ factor in numerator | Standard $2\times$ DSC formulation | Standard formulation implemented; constant absorbed by LR. |
| **Optimizer** | **A. Directly Stated** | Adam optimizer | `torch.optim.Adam` | Primary optimizer. |
| **Learning Rate** | **C. Ambiguous** | Literal print: "6 × 10⁴" ($6000$) | $6 \times 10^{-4}$ ($0.0006$) | Obvious typesetting typo (missing negative exponent). |
| **Scheduler** | **A. Directly Stated** | ReduceLROnPlateau (patience=100) | `torch.optim.lr_scheduler.ReduceLROnPlateau` | Monitors validation DSC. |
| **Total Epochs** | **D. Implementation Assumption** | Not specified in paper | 300 epochs (default, practical choice) | Implied by patience=100; deferred to runtime budget. |
| **Batch Size** | **E. Hardware Deviation** | Not specified in paper | `BATCH_SIZE = 1` | Dictated by Tesla T4 (14.5 GB VRAM) memory constraints. |
| **Mixed Precision** | **E. Hardware Deviation** | Not mentioned in paper | PyTorch AMP enabled | Engineering optimization for GPU feasibility. |
| **Dataset Case Count** | **F. Dataset Deviation** | Text implies 81/82 cases ($5 \times 13 + 16 = 81$) | 80 verified TCIA v2 cases | Cases #25 and #70 excluded by TCIA as duplicates. |
| **Augmentation** | **A. Directly Stated** | Rot $\pm10^\circ$, Flip $p=0.5$, Shift $\pm10\%$ | `dataset.py:_augment_3d` | Synchronized 3D geometric transformations. |
| **Cross-Validation** | **A. Directly Stated** | 5-Fold CV + 16 independent test cases | `cross_validation.py:create_patient_splits` | Patient-level splitting saved to `splits/patient_splits.json`. |
| **Evaluation Metrics** | **A. Directly Stated** | Volumetric DSC, ASSD (mm), HD95 (mm) | `metrics.py` & `evaluate.py` | Volumetric surface distances using physical voxel spacing. |

---

## 2. Source Data Policy & Output Isolation

1. **Original Raw Data Location:** `/content/drive/MyDrive/Pancreas-CT/pancreas_ct/` or raw paired NIfTIs.
2. **Untouched Existing Data:** `/content/drive/MyDrive/Pancreas-CT/Processed_data/` (2018 pipeline output) is **NEVER** modified or overwritten.
3. **2025 Preprocessed Target Directory:** `/content/drive/MyDrive/Pancreas-CT/2025_Processed_data/`
   - `2025_Processed_data/images/PANCREAS_XXXX.nii.gz`
   - `2025_Processed_data/labels/PANCREAS_XXXX.nii.gz`
4. **Single-Patient Protocol:** Patient `PANCREAS_0001` must be processed and verified first before executing batch preprocessing on the remaining 79 cases.

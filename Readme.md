# 3D Attention U-Net for Pancreas Segmentation in CT Scans

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20Ready-orange.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Complete Reproducibility & Execution Guide**  
> Target Paper: *"Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"*  
> *Mathematics* 2025, 13(24), 3942 — DOI: [10.3390/math13243942](https://doi.org/10.3390/math13243942)  
> Authors: Antonio Roger Tondji, Chiara Scapicchio, Francesca Lizzi, Maria Evelina Fantacci, Piergiorgio Oliva, Alessandra Retico.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Target Paper Specifications](#2-target-paper-specifications)
3. [Repository File Map](#3-repository-file-map)
4. [Dataset & Directory Layout on Google Drive](#4-dataset--directory-layout-on-google-drive)
5. [Quick Start: 3-Minute Setup](#5-quick-start-3-minute-setup)
6. [Cell-by-Cell Colab Execution Walkthrough](#6-cell-by-cell-colab-execution-walkthrough)
7. [2025 Preprocessing Pipeline Deep Dive](#7-2025-preprocessing-pipeline-deep-dive)
8. [3D Attention U-Net Architecture](#8-3d-attention-u-net-architecture)
9. [Loss Function & Training Dynamics](#9-loss-function--training-dynamics)
10. [Cross-Validation & Evaluation Protocol](#10-cross-validation--evaluation-protocol)
11. [Paper Ambiguities & Implementation Decisions](#11-paper-ambiguities--implementation-decisions)
12. [Critical Safety Rules: What NOT to Modify](#12-critical-safety-rules-what-not-to-modify)
13. [Troubleshooting & FAQ](#13-troubleshooting--faq)

---

## 1. Project Overview

This repository is an authoritative, end-to-end research paper reproduction of the 2025 *Mathematics* paper on automated 3D Pancreas Segmentation from computed tomography (CT) scans. 

The pancreas is an abdominal organ with high anatomical variability, irregular shape, and low contrast relative to surrounding soft tissues (duodenum, spleen, stomach). This project implements a **3D Attention U-Net** that leverages isotropic 1 mm resampling, soft-tissue HU windowing, additive attention gating, and combined Dice + Focal loss to segment the organ in full 3D volumes.

All code is optimized for execution on **Google Colab with a Tesla T4 GPU (~14.5 GB VRAM)**.

---

## 2. Target Paper Specifications

| Component | Paper Specification | Repository Implementation |
|---|---|---|
| **Base Architecture** | 3D U-Net with Attention Gates | `model.py:AttentionUNet3D` |
| **Encoder Channels** | Figure 1: 16 → 32 → 64 → 128 → 256 | `config.py:ENCODER_CHANNELS = [16, 32, 64, 128, 256]` |
| **Attention Gate** | Additive attention with LayerNorm & Sigmoid gating | `attention.py:AttentionGate3D` |
| **HU Clipping** | [-100, 240] HU | `preprocessing.py:clip_hu` |
| **Normalization** | [0, 1] linear min-max | `preprocessing.py:normalize_to_01` |
| **Voxel Resampling** | 1.0 × 1.0 × 1.0 mm³ isotropic | SimpleITK (Linear for CT, Nearest for label) |
| **Crop Size** | 224 × 224 × 128 (H, W, D) | Blind geometric volume center-crop |
| **Loss Function** | Combined Dice + Focal Loss | `losses.py:DiceFocalLoss` ($\gamma=2.0, \alpha=0.25$) |
| **Optimizer** | Adam, LR = $6 \times 10^{-4}$ | `torch.optim.Adam(lr=6e-4)` |
| **Scheduler** | ReduceLROnPlateau (patience=100) | `torch.optim.lr_scheduler.ReduceLROnPlateau` |
| **Batch Size** | 1 (hardware-constrained) | `BATCH_SIZE = 1` with PyTorch AMP enabled |
| **Dataset** | NIH Pancreas-CT (TCIA v2, 80 cases) | 16 test holdout + 5-fold CV (64 patients) |
| **Evaluation Metrics** | Volumetric DSC, ASSD (mm), HD95 (mm) | `metrics.py` (physical voxel spacing aware) |

---

## 3. Repository File Map

```text
Pancreas-Seg-MIP/
│
├── README.md                           # Master execution guide (this file)
├── requirements.txt                    # Python dependencies with pinned minimum versions
├── .gitignore                          # Excludes datasets, caches, and weight files
│
├── src/                                # Lightweight Python package (imports via src.*)
│   ├── __init__.py
│   ├── config.py                       # Single centralized configuration for all constants and paths
│   ├── losses.py                       # Combined DiceLoss + FocalLoss implementation
│   ├── metrics.py                      # Exact distance transform and surface distance algorithms
│   ├── utils.py                        # Reproducibility seed, publication plots, and GPU benchmark
│   ├── train.py                        # Resumable training loop engine with PyTorch AMP & logging
│   ├── cross_validation.py             # 5-fold patient-level cross-validation orchestrator
│   ├── evaluate.py                     # 3D volumetric evaluation (DSC, ASSD, HD95) & NIfTI export
│   ├── inference.py                    # Standalone inference on new preprocessed CT scans
│   ├── test_all.py                     # Comprehensive 12-stage automated test suite
│   ├── models/                         # 3D attention U-Net architecture modules
│   │   ├── __init__.py
│   │   ├── attention.py                # 3D Additive Attention Gate module with LayerNorm
│   │   └── model.py                    # Complete 3D Attention U-Net architecture
│   └── data/                           # Data loading & preprocessing modules
│       ├── __init__.py
│       ├── preprocessing.py            # HU clipping, normalization, SimpleITK resampling, crop/pad
│       ├── prepare_data.py             # 80-case batch preprocessing CLI & automated QC manifest generator
│       ├── qc_preprocessing.py         # Preprocessing QC visualization & per-case audits
│       └── dataset.py                  # 3D PyTorch Dataset with lazy NIfTI loading & data augmentation
│
├── notebooks/
│   └── pancreas_segmentation_colab.ipynb  # Interactive step-by-step Google Colab notebook
│
└── docs/
    ├── PAPER_REPRODUCTION_NOTES.md     # Full taxonomy of paper specs vs ambiguities vs assumptions
    ├── REPRODUCTION_CHECKLIST.md       # Stage-by-stage requirement verification checklist
    ├── SETUP.md                        # GPU setup & quick execution guide
    ├── VALIDATION_REPORT.md            # Quantitative results template for training runs
    └── REPORT.md                       # End-to-end paper reproduction results report
```

---

## 4. Dataset & Directory Layout on Google Drive

To keep code and large data cleanly isolated without duplication, your Google Drive must be organized as follows:

```text
/content/drive/MyDrive/
│
├── Pancreas-Seg-MIP/                   # <<< CODE DIRECTORY (Extracted from pancreas_seg_code.zip)
│   ├── config.py
│   ├── model.py
│   ├── train.py
│   └── ...
│
└── Pancreas-CT/                        # <<< DATA DIRECTORY (Configured as BASE)
    │
    ├── Processed_data/                 # [SOURCE DATA] 80 native NIfTI volumes (READ-ONLY)
    │   ├── images/
    │   │   ├── PANCREAS_0001.nii.gz
    │   │   └── ... (80 total files)
    │   └── labels/
    │       ├── PANCREAS_0001.nii.gz
    │       └── ... (80 total files)
    │
    ├── 2025_Processed_data/            # [OUTPUT] Generated by 2025 Preprocessing (Cell 8)
    │   ├── images/                     # 80 volumes (224x224x128, 1mm isotropic, [0,1] normalized)
    │   └── labels/                     # 80 binary masks (224x224x128, {0,1})
    │
    ├── splits/                         # [OUTPUT] Generated by Patient-Level CV (Cell 11)
    │   ├── patient_splits.json         # Master 5-fold + 16 test splits
    │   └── fold_0.json ... fold_4.json
    │
    ├── checkpoints/                    # [OUTPUT] Generated during Training (Cell 16)
    │   └── fold_0/
    │       ├── latest_checkpoint.pth   # Checkpoint saved every epoch for disconnect recovery
    │       ├── best_model.pth          # Best validation Dice model weights
    │       └── training_log.csv        # Per-epoch loss, validation Dice, learning rate
    │
    └── results/                        # [OUTPUT] Generated during Evaluation (Cell 17)
        ├── fold_0/
        │   ├── fold_0_per_patient_results.csv
        │   ├── fold_0_summary.csv
        │   └── predictions/            # Predicted binary 3D NIfTI masks
        └── test_set/
            ├── test_per_patient_results.csv
            └── test_summary.csv
```

---

## 5. Quick Start: 3-Minute Setup

1. **Upload Dataset:** Place the folder `Pancreas-CT` (containing `Processed_data/images/` and `Processed_data/labels/`) in your Google Drive root (`My Drive`).
2. **Upload Code:** Drag and drop `pancreas_seg_code.zip` into your Google Drive root (`My Drive`).
3. **Open Colab:** Open `notebooks/pancreas_segmentation_colab.ipynb` in [Google Colab](https://colab.research.google.com/).
4. **Set GPU:** In Google Colab, go to **Runtime → Change runtime type** → select **T4 GPU** → click **Save**.
5. **Run Cells 1 through 15:** Step through the verification and benchmark pipeline.
6. **Train (Cell 16):** Run Cell 16 to start training on Fold 0.

---

## 6. Cell-by-Cell Colab Execution Walkthrough

The notebook `pancreas_segmentation_colab.ipynb` is structured into 17 clear sections:

### Stage 1: Environment & Code Deployment
- **Cell 1 (Mount Google Drive):** Mounts `/content/drive`.
- **Cell 2 (Verify GPU):** Verifies Tesla T4 GPU and available VRAM (~14.75 GB).
- **Cell 3 (Install Dependencies):** Installs required libraries (`SimpleITK`, `nibabel`, `monai`, `scipy`, `pandas`).
- **Cell 4 (Deploy Code from Zip):** Automatically unpacks `pancreas_seg_code.zip` into `/content/drive/MyDrive/Pancreas-Seg-MIP/` and injects it into `sys.path`.

### Stage 2: Data Audit & Single-Case QC
- **Cell 5 (Configure Paths & Imports):** Sets `BASE = "/content/drive/MyDrive/Pancreas-CT"`. All sub-paths are derived automatically.
- **Cell 6 (Verify 80 Matched Pairs):** Scans `Processed_data` and confirms exactly 80 images and 80 labels match by patient ID.
- **Cell 7 (Inspect Single Volume):** Loads `PANCREAS_0001` and prints original matrix size `(512, 512, 240)`, spacing `(0.86, 0.86, 1.0) mm`, and raw HU intensity range `[-1024, +2421]`.

### Stage 3: Batch Preprocessing & Integrity Verification
- **Cell 8 (Run 2025 Batch Preprocessing):** Executes memory-safe sequential preprocessing across all 80 cases:
  ```bash
  python -m src.data.prepare_data \
      --preprocess_all \
      --raw_dir "/content/drive/MyDrive/Pancreas-CT/Processed_data" \
      --preprocessed_dir "/content/drive/MyDrive/Pancreas-CT/2025_Processed_data" \
      --manifest "/content/drive/MyDrive/Pancreas-CT/2025_PREPROCESSING_MANIFEST.csv" \
      --summary "/content/drive/MyDrive/Pancreas-CT/2025_PREPROCESSING_SUMMARY.md"
  ```
- **Cell 9 (Audit Manifest):** Inspects the generated CSV manifest to confirm `80/80 PASS`, 100% shape uniformity `(224, 224, 128)`, and 100% spacing uniformity `(1.0, 1.0, 1.0) mm`.
- **Cell 10 (Visualize Preprocessed Slices):** Displays 6 representative axial slices with ground-truth overlays.

### Stage 4: Splitting, Testing & Benchmarking
- **Cell 11 (Generate Patient-Level CV Splits):** Creates deterministic 5-fold CV splits (64 training/validation patients, 16 held-out test patients).
- **Cell 12 (Run Test Suite):** Executes `python -m src.test_all` (Must output: **`12/12 tests passed - PASS [OK]`**).
- **Cell 13 (Model Parameter Audit):** Instantiates the model and prints total parameters (`5,668,269` ~ 21.62 MB).
- **Cell 14 (T4 GPU Memory Benchmark):** Simulates a training step with input shape `(1, 1, 224, 224, 128)` and confirms peak VRAM is ~2.3 GB (well below the 14.5 GB limit).
- **Cell 15 (Real-Data One-Batch Test):** Passes an actual preprocessed 3D volume through forward pass, loss calculation, backward pass, and optimizer step.

### Stage 5: Training & Quantitative Evaluation
- **Cell 16 (Train Model):** Executes 3D Attention U-Net training on Fold 0 (or any specified fold).
- **Cell 17 (Evaluate Model):** Computes volumetric DSC, ASSD (mm), and HD95 (mm) on validation and test sets.

---

## 7. 2025 Preprocessing Pipeline Deep Dive

The 2025 paper defines a strict sequential preprocessing protocol:

```text
Raw CT Volume (e.g. 512x512x240, Spacing: 0.86x0.86x1.0 mm)
   │
   ├── 1. SimpleITK Isotropic Resampling (1.0 x 1.0 x 1.0 mm³)
   │      - CT Image: Linear / Trilinear interpolation
   │      - Label Mask: Nearest-Neighbor interpolation (preserves binary {0,1})
   │
   ├── 2. Soft-Tissue HU Windowing
   │      - Clipped strictly to [-100, 240] HU
   │
   ├── 3. Min-Max Normalization
   │      - Linearly scaled: (HU - (-100)) / (240 - (-100)) -> [0.0, 1.0]
   │
   ├── 4. Geometric Center-Crop / Pad
   │      - Fixed spatial dimensions: (224, 224, 128)
   │      - Strictly blind geometric crop (NO ground-truth mask guidance)
   │
   └── 5. Storage
          - Saved as compressed NIfTI (.nii.gz) in 2025_Processed_data/
```

---

## 8. 3D Attention U-Net Architecture

The model is built in `model.py` and `attention.py` following Figure 1 of the paper:

```text
Input (1, 224, 224, 128)
  │
  ├── Level 1: DoubleConv3D(1 -> 16)   ───────[Attention Gate 1]───────┐
  │     ↓ MaxPool3D(2)                                                 │
  ├── Level 2: DoubleConv3D(16 -> 32)  ───────[Attention Gate 2]─────┐ │
  │     ↓ MaxPool3D(2)                                               │ │
  ├── Level 3: DoubleConv3D(32 -> 64)  ───────[Attention Gate 3]───┐ │ │
  │     ↓ MaxPool3D(2)                                             │ │ │
  ├── Level 4: DoubleConv3D(64 -> 128) ───────[Attention Gate 4]─┐ │ │ │
  │     ↓ MaxPool3D(2)                                           │ │ │ │
  └── Bottleneck: DoubleConv3D(128 -> 256)                       │ │ │ │
        ↓ ConvTranspose3D(256 -> 128)                            │ │ │ │
      Decoder 4: Concat + DoubleConv3D(256 -> 128) ◄─────────────┘ │ │ │
        ↓ ConvTranspose3D(128 -> 64)                               │ │ │
      Decoder 3: Concat + DoubleConv3D(128 -> 64) ◄────────────────┘ │ │
        ↓ ConvTranspose3D(64 -> 32)                                  │ │
      Decoder 2: Concat + DoubleConv3D(64 -> 32) ◄───────────────────┘ │
        ↓ ConvTranspose3D(32 -> 16)                                    │
      Decoder 1: Concat + DoubleConv3D(32 -> 16) ◄─────────────────────┘
        ↓ 1x1x1 Conv3D(16 -> 1)
Output Binary Logits (1, 224, 224, 128)
```

### Additive Attention Gate Formula
$$\alpha = \sigma\left(\psi^T\left(\text{ReLU}\left(\text{LayerNorm}(\theta(x) + \phi(g))\right)\right)\right)$$
$$\hat{x} = x \odot \alpha$$
Where $x$ is the skip-connection feature map, $g$ is the gating signal from the coarser level, and $\alpha \in [0, 1]$ is the learned spatial attention coefficient.

---

## 9. Loss Function & Training Dynamics

### Combined Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{Focal}}$$

- **Soft Dice Loss:**
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}$$
- **Focal Loss ($\gamma = 2.0, \alpha = 0.25$):**
  $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

### Resumable Training & Disconnect Recovery
Colab sessions can disconnect during long runs. The training loop in `train.py` automatically writes:
- `latest_checkpoint.pth`: Contains model weights, optimizer state, scheduler state, GradScaler state, current epoch, and best Dice score.
- `best_model.pth`: Saved whenever validation Dice reaches a new maximum.
- `training_log.csv`: Appends epoch-by-epoch loss and Dice metrics.

If Colab disconnects, simply re-run Cell 16. It will automatically detect `latest_checkpoint.pth` and resume from the exact epoch where it stopped.

---

## 10. Cross-Validation & Evaluation Protocol

### Patient-Level Partitioning
- **Total Patients:** 80 verified cases.
- **Independent Test Set:** 16 cases held out permanently (never seen during training or validation).
- **5-Fold Cross-Validation:** Remaining 64 cases divided into 5 folds (~13 patients per validation split).
- **Splits File:** Saved to `splits/patient_splits.json` for deterministic reproducibility across runs.

### Volumetric 3D Evaluation Metrics
Evaluation in `evaluate.py` and `metrics.py` computes full 3D surface distances using physical voxel spacing $(1.0, 1.0, 1.0)\text{ mm}$:
1. **Dice Similarity Coefficient (DSC):** Volume overlap percentage.
2. **Average Symmetric Surface Distance (ASSD):** Average distance (in mm) between prediction surface and ground-truth surface.
3. **95th Percentile Hausdorff Distance (HD95):** 95th percentile maximum surface distance (in mm) to measure boundary outlier errors.

---

## 11. Paper Ambiguities & Implementation Decisions

All reproduction decisions are mathematically documented in [PAPER_REPRODUCTION_NOTES.md](PAPER_REPRODUCTION_NOTES.md):

1. **Channel Progression:** Section 2.1 text states "2→16", but Figure 1 explicitly illustrates $16 \to 32 \to 64 \to 128 \to 256$. We implement Figure 1.
2. **Learning Rate:** Paper prints $6 \times 10^4$ ($6000$), an obvious typography error (missing negative sign). Implemented as $6 \times 10^{-4}$ ($0.0006$).
3. **Focal Loss Parameters:** Paper does not specify numerical values for $\gamma$ and $\alpha$. Implemented standard defaults $\gamma = 2.0, \alpha = 0.25$.
4. **Crop Center:** Paper states "centered on abdominal region" without mathematical coordinates. We use blind volume geometric center-crop to strictly prevent ground-truth mask leakage.
5. **Batch Size:** Not stated in paper. Set to `1` due to Tesla T4 GPU 14.5 GB VRAM limits for 3D tensors.
6. **Case Count:** Paper text states 81/82 cases ($5 \times 13 + 16$). Standard verified TCIA v2 dataset has 80 cases.

---

## 12. Critical Safety Rules: What NOT to Modify

To maintain scientific integrity and prevent accidental data loss:
- ❌ **NEVER modify or overwrite `Processed_data/`:** This is your untouched raw baseline.
- ❌ **DO NOT change model architecture in `model.py`:** Must strictly match Figure 1 of the 2025 paper.
- ❌ **DO NOT change the loss function in `losses.py`:** Must remain combined Dice + Focal Loss.
- ❌ **DO NOT crop using ground-truth masks:** Blind cropping ensures valid clinical generalization.
- ❌ **DO NOT set `batch_size > 1` on Tesla T4:** 3D tensors `(1, 1, 224, 224, 128)` with feature maps will trigger CUDA Out-Of-Memory if batch size is increased.

---

## 13. Troubleshooting & FAQ

### Q1: Colab says `ModuleNotFoundError: No module named 'config'`
**Fix:** Run Cell 4. It unpacks `pancreas_seg_code.zip` to `/content/drive/MyDrive/Pancreas-Seg-MIP` and adds it to `sys.path`.

### Q2: CUDA Out-Of-Memory (OOM) during training
**Fix:** Ensure `BATCH_SIZE = 1` in `config.py` and PyTorch AMP is enabled (`USE_AMP = True`). If using a custom script, verify you did not set `batch_size > 1`.

### Q3: My Colab session disconnected after 3 hours. Do I have to restart from Epoch 1?
**Fix:** No. Simply re-run Cell 1 through Cell 5, then run Cell 16. It will automatically load `latest_checkpoint.pth` and continue training from the last saved epoch.

### Q4: How do I train a different fold?
**Fix:** In Cell 16, change `--fold 0` to `--fold 1` (or 2, 3, 4). Each fold saves its checkpoints in its own dedicated folder (`checkpoints/fold_1/`, etc.).

---

*For full reproduction records, refer to [PAPER_REPRODUCTION_NOTES.md](PAPER_REPRODUCTION_NOTES.md), [REPRODUCTION_CHECKLIST.md](REPRODUCTION_CHECKLIST.md), and [VALIDATION_REPORT.md](VALIDATION_REPORT.md).*

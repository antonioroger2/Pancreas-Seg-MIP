# U-Net for Pancreas Segmentation in Abdominal CT Scans

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.2+-5c2d91.svg)](https://monai.io/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20Ready-orange.svg)](https://colab.research.google.com/)

A lightweight, robust PyTorch implementation of a 2D U-Net designed to segment the pancreas from 3D abdominal contrast-enhanced CT scans (TCIA Pancreas-CT Dataset). Optimized for fast **~30-minute execution** on **Google Colab T4 GPUs** with automatic **Google Drive** checkpointing and paper-ready plot generation.

---

## Dataset Overview (TCIA Pancreas-CT)

* **Source:** [NIH Clinical Center Pancreas-CT Dataset](https://www.cancerimagingarchive.net/collection/pancreas-ct/) (The Cancer Imaging Archive).
* **Subjects:** 80 unique subjects (53 male, 27 female, ages 18–76).
* **Scans:** 3D abdominal contrast-enhanced CT scans (~70s portal-venous phase post IV contrast).
* **Resolution:** $512 \times 512$ matrix with $1.5\text{ mm} - 2.5\text{ mm}$ slice thickness.
* **Ground Truth:** Slice-by-slice manual pancreas segmentations verified by an experienced radiologist.
* **Split:** Standard 80/20 train/validation split.

---

## Key Technical Highlights

1. **Lightweight & Fast Architecture:** Features `[32, 64, 128, 256]` channel U-Net allowing full convergence in ~30 minutes on a single Colab T4 GPU.
2. **Hounsfield Unit (HU) Windowing:** Clips CT intensity values to `[-125, 225] HU` and normalizes to `[0.0, 1.0]`.
3. **Hybrid Loss (`DiceCELoss`):** Combines BCE with Soft Dice Loss for extreme class imbalance (pancreas $<1.5\%$ of voxels).
4. **Automatic Mixed Precision (AMP):** `torch.cuda.amp` for fast training with lower VRAM usage.
5. **DICOM + NIfTI Support:** `prepare_data.py` converts raw TCIA nested DICOM downloads to NIfTI; `dataset.py` auto-detects both formats.
6. **Paper-Ready Artifacts:** Saves `training_log.csv`, `loss_curve.png`, `dice_curve.png`, and `metrics_summary.png` (300 DPI).

---

## Repository Structure

```text
Pancreas-Seg-MIP/
├── pancreas_segmentation_colab.ipynb   # Google Colab Notebook
├── prepare_data.py                     # DICOM-to-NIfTI converter for TCIA downloads
├── dataset.py                          # Data loading (supports DICOM + NIfTI)
├── model.py                            # Lightweight U-Net & DiceCELoss
├── train.py                            # AMP training, CSV logging & plot export
├── utils.py                            # Dice metric & publication plot generator
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── .gitignore                          # Git ignore rules
```

---

## Google Colab Quickstart Guide

### Step 1: Clone & Setup
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
%cd /content/drive/MyDrive
!git clone https://github.com/antonioroger2/Pancreas-Seg-MIP.git
%cd /content/drive/MyDrive/Pancreas-Seg-MIP

# Install dependencies
!pip install -q -r requirements.txt
```

### Step 2: Data Preparation

**Option A — If you downloaded raw DICOM from TCIA** (nested folder structure):

Upload your TCIA download and annotations to Google Drive, then run `prepare_data.py`:
```bash
!python prepare_data.py \
  --dicom_dir /content/drive/MyDrive/Pancreas-CT/pancreas_ct \
  --labels_dir /content/drive/MyDrive/TCIA_Pancreas_Labels \
  --output_dir ./data
```

This converts the nested DICOM layout:
```text
pancreas_ct/
  PANCREAS_0003/
    02648/
      59468/
        0287466e-da98-4659-ad2c-e5663c21ccd2.dcm, ...

  PANCREAS_0002/
    ...
```

Into the flat NIfTI structure expected by the training script:
```text
data/
├── images/
│   ├── PANCREAS_0001.nii.gz
│   ├── PANCREAS_0002.nii.gz
│   └── ...
└── labels/
    ├── PANCREAS_0001.nii.gz
    ├── PANCREAS_0002.nii.gz
    └── ...
```

**Option B — If you already have NIfTI files:**

Place them directly under `./data/images/` and `./data/labels/`.

### Step 3: Train (~30 min on T4 GPU)
```bash
!python train.py \
  --data_dir ./data \
  --output_dir ./checkpoints \
  --save_drive_path "/content/drive/MyDrive/Pancreas_Checkpoints" \
  --epochs 25 \
  --batch_size 8 \
  --lr 1e-4 \
  --light
```

---

## Generated Artifacts After Training

| File | Location | Purpose |
|------|----------|---------|
| `best_model_YYYYMMDD_HHMMSS.pth` | `checkpoints/` + Google Drive | Timestamped best model checkpoint |
| `best_model.pth` | `checkpoints/` + Google Drive | Static pointer to latest best model |
| `training_log.csv` | `checkpoints/` + Google Drive | Per-epoch train_loss, val_loss, val_dice |
| `loss_curve.png` | `checkpoints/plots/` | Training vs Validation loss curve (300 DPI) |
| `dice_curve.png` | `checkpoints/plots/` | Validation Dice score over epochs (300 DPI) |
| `metrics_summary.png` | `checkpoints/plots/` | Combined 2-panel figure for research papers (300 DPI) |

---

## Evaluation Metrics

The segmentation accuracy is measured using the **Dice Similarity Coefficient (DSC)**:

$$\text{DSC} = \frac{2 |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}$$

During training, `train.py` logs validation Dice scores per epoch and automatically saves the best performing weights. Visualize predictions overlaid on raw CT slices directly inside the Colab notebook.

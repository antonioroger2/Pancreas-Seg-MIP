# U-Net for Pancreas Segmentation in Abdominal CT Scans

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.2+-5c2d91.svg)](https://monai.io/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-orange.svg)](https://colab.research.google.com/)

A lightweight, production-grade PyTorch implementation of a 2D U-Net designed to segment the pancreas from 3D abdominal contrast-enhanced CT scans (TCIA Pancreas-CT Dataset). Optimized for fast execution on **Google Colab** with GPU acceleration and automatic **Google Drive** checkpoint synchronization.

---

## 📊 Dataset Overview (TCIA Pancreas-CT)

* **Source:** [NIH Clinical Center Pancreas-CT Dataset](https://www.cancerimagingarchive.net/collection/pancreas-ct/) (The Cancer Imaging Archive).
* **Subjects:** 80 unique subjects (53 male, 27 female, ages 18–76).
* **Scans:** 3D abdominal contrast-enhanced CT scans (~70s portal-venous phase post IV contrast).
* **Resolution:** $512 \times 512$ matrix with $1.5\text{ mm} - 2.5\text{ mm}$ slice thickness.
* **Ground Truth:** Slice-by-slice manual pancreas segmentations performed by a medical student and verified by an experienced radiologist.

---

## ⚡ Key Technical Highlights

1. **Hounsfield Unit (HU) Windowing:** Clips raw CT intensity values to the pancreatic soft-tissue range (`[-125, 225] HU`) and normalizes to `[0.0, 1.0]`.
2. **Hybrid Loss Function (`DiceCELoss`):** Combines Binary Cross-Entropy (BCE) with Soft Dice Loss to address extreme class imbalance (the pancreas comprises $<1.5\%$ of total abdominal voxels).
3. **Automatic Mixed Precision (AMP):** Utilizes `torch.cuda.amp` for faster training iterations and lower GPU VRAM consumption.
4. **Drive Checkpointing with Timestamps:** Automatically saves timestamped model weights (`best_model_YYYYMMDD_HHMMSS.pth`) directly to Google Drive so training sessions never overwrite each other.

---

## 📂 Repository Structure

```text
Pancreas-Seg-MIP/
├── pancreas_segmentation_colab.ipynb   
├── dataset.py   # 📦 DICOM/NIfTI loading, HU windowing & MONAI transforms
├── model.py  # 🧠 2D U-Net Architecture & DiceCELoss implementation
├── train.py  # 🏋️ Training engine with AMP & Drive checkpointing
├── utils.py  # 📊 Dice metric, seed setter & slice visualization overlays
├── requirements.txt # 📋 Python dependencies
├── README.md    # 📖 Project documentation
└── .gitignore   # 🙈 Git ignore rules for datasets & checkpoints
```

---

## Google Colab Quickstart Guide

Paste this into a Google Colab VM terminal to clone the entire project into the Colab VM and start working:

```bash
git clone https://github.com/antonioroger2/Pancreas-Seg-MIP.git
```

### Step 1: Clone Repository & Mount Google Drive
Run this cell in Google Colab:
```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repository into Google Drive (if not already)
%cd /content/drive/MyDrive
!git clone https://github.com/antonioroger2/Pancreas-Seg-MIP.git
%cd /content/drive/MyDrive/Pancreas-Seg-MIP
```

### Step 2: Install Dependencies
```bash
%cd /content/drive/MyDrive/Pancreas-Seg-MIP
!pip install -r requirements.txt
```

### Step 3: Dataset Preparation
Place your NIfTI image and mask files under `./data`:
```text
data/
├── images/   (e.g., PANCREAS_0001.nii.gz)
└── labels/   (e.g., label0001.nii.gz)
```

### Step 4: Run Training Script
```bash
!python train.py \
  --data_dir ./data \
  --output_dir ./checkpoints \
  --save_drive_path "/content/drive/MyDrive/Pancreas_Checkpoints" \
  --epochs 25 \
  --batch_size 8 \
  --lr 1e-4
```

---

## 📈 Evaluation & Metrics

The segmentation accuracy is measured using the **Dice Similarity Coefficient (DSC)**:

$$\text{DSC} = \frac{2 |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}$$

During training, `train.py` logs validation Dice scores per epoch and automatically saves the best performing weights. You can visualize predictions overlaid on raw CT slices directly inside the Colab notebook.

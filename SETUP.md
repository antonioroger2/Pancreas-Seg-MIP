# GPU Setup & Quick Execution Guide

This document provides exact instructions to quickly clone, set up, and train the **3D Attention U-Net Pancreas Segmentation** pipeline on any GPU machine (Google Colab, local Linux GPU server, AWS EC2, Lambda Labs, etc.).

---

## 1. Fast Setup (Under 2 Minutes)

### Step A: Clone the Repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd Pancreas-Seg-MIP
```

### Step B: Create & Activate Virtual Environment (Optional for Colab)
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# .\venv\Scripts\activate   # On Windows
```

### Step C: Install Requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Directory Layout & Dataset Placement

Place your dataset in a folder (for example, `/content/drive/MyDrive/Pancreas-CT` or `/home/user/data/Pancreas-CT`):

```text
Pancreas-CT/
└── Processed_data/             # <<< RAW DATA (Never modified)
    ├── images/
    │   ├── PANCREAS_0001.nii.gz
    │   └── ... (80 files)
    └── labels/
        ├── PANCREAS_0001.nii.gz
        └── ... (80 files)
```

---

## 3. Execution Pipeline (Sequential Commands)

### Step 1: Run Integration Test Suite (Verify Everything Works)
```bash
python test_all.py
```
*Expected output:* **`12/12 tests passed - PASS [OK]`**

---

### Step 2: Run 2025 Paper Batch Preprocessing (80 Cases)
```bash
python prepare_data.py \
    --preprocess_all \
    --raw_dir "/content/drive/MyDrive/Pancreas-CT/Processed_data" \
    --preprocessed_dir "/content/drive/MyDrive/Pancreas-CT/2025_Processed_data" \
    --manifest "/content/drive/MyDrive/Pancreas-CT/2025_PREPROCESSING_MANIFEST.csv" \
    --summary "/content/drive/MyDrive/Pancreas-CT/2025_PREPROCESSING_SUMMARY.md"
```
*What this does:*
- Resamples CT & labels to `1.0 x 1.0 x 1.0 mm³` isotropic spacing
- Clips intensity to `[-100, 240] HU` & normalizes to `[0, 1]`
- Crops volume to `(224, 224, 128)` geometrically (no label leakage)
- Saves `.nii.gz` to `2025_Processed_data/`
- Validates 80/80 cases and outputs a CSV manifest and Markdown summary.

---

### Step 3: Train 3D Attention U-Net (Fold 0)
```bash
python cross_validation.py \
    --data_dir "/content/drive/MyDrive/Pancreas-CT/2025_Processed_data" \
    --checkpoint_dir "/content/drive/MyDrive/Pancreas-CT/checkpoints" \
    --drive_checkpoint_dir "/content/drive/MyDrive/Pancreas-CT/checkpoints" \
    --splits_dir "/content/drive/MyDrive/Pancreas-CT/splits" \
    --fold 0 \
    --epochs 300 \
    --batch_size 1 \
    --lr 6e-4
```
*Features:*
- Uses PyTorch AMP (Mixed Precision) for memory efficiency
- Automatically saves `latest_checkpoint.pth` (per-epoch) and `best_model.pth`
- **Disconnect-Safe:** If the run stops or Colab disconnects, simply re-run this exact command to automatically resume training from the last saved epoch!

*(To train other folds, change `--fold 0` to `--fold 1`, `2`, `3`, or `4`)*

---

### Step 4: Evaluate Trained Checkpoints
```bash
python evaluate.py \
    --data_dir "/content/drive/MyDrive/Pancreas-CT/2025_Processed_data" \
    --checkpoint_dir "/content/drive/MyDrive/Pancreas-CT/checkpoints" \
    --results_dir "/content/drive/MyDrive/Pancreas-CT/results" \
    --splits_dir "/content/drive/MyDrive/Pancreas-CT/splits" \
    --fold 0
```
*Computes exact 3D volumetric metrics:*
- **DSC:** Dice Similarity Coefficient
- **ASSD (mm):** Average Symmetric Surface Distance
- **HD95 (mm):** 95th Percentile Hausdorff Distance
- Saves per-patient CSV, aggregate summary CSV, and predicted binary NIfTI masks.

---

## 4. Hardware Requirements

- **GPU:** NVIDIA GPU with $\ge 8$ GB VRAM (Tested on Google Colab Tesla T4 14.5 GB).
- **Peak Training VRAM:** ~2.3 GB (with `batch_size=1` & AMP).
- **Disk Space:** ~2.5 GB for preprocessed volumes.

---

## 5. Summary of Key Files

| File | Purpose |
|---|---|
| `model.py` | 3D Attention U-Net architecture (`16 -> 32 -> 64 -> 128 -> 256`) |
| `attention.py` | Additive Attention Gate with LayerNorm |
| `losses.py` | Combined Dice + Focal Loss ($\gamma=2.0, \alpha=0.25$) |
| `config.py` | All paper constants and paths in one central file |
| `pancreas_segmentation_colab.ipynb` | Google Colab interactive notebook with step-by-step UI |
| `README.md` | Full paper reproduction notes, theory, and background |

# GPU Setup & Quick Execution Guide

> **HISTORICAL (Review-1, Aug 2026) - read with these corrections (Oct 2026, see CHANGELOG.md):**
> (1) the paper reports Dice **0.808 +- 0.021** (NIH, Table 1), not ~0.86; (2) runs 2-3 used a **ground-truth-centroid
> crop** (`center_crop_on_label`, now removed): the label position leaked into the input at validation/test time, so their
> scores are **invalid** as a baseline; (3) the source data is aligned (T1, `results/alignment/`), but the Review-1
> preprocessed arrays were not: its overlay figure shows GT on heart/lung slices for test patients 0073/0030, so
> "Not alignment" was wrong; (4) validation Dice **peaked (~0.3, epochs 40-80) then fell** - overfitting, not a plateau;
> (5) Review-1 runs used 2x RTX 6000 Ada; the paper used a single **T4 16 GB**, so hardware does not explain the gap;
> (6) `src/cross_validation.py` and the epoch trainer were replaced by the step-based pipeline (README.md, REPRODUCE.md).


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
python -m src.test_all
```
*Expected output:* **`12/12 tests passed - PASS [OK]`**

---

### Step 2: Run 2025 Paper Batch Preprocessing (80 Cases)
```bash
python -m src.data.prepare_data \
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
python -m src.cross_validation \
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
python -m src.evaluate \
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
- **Peak Training VRAM:** measured 6.55 GB for R0 (224x224x128, batch 1, AMP) and 6.32 GB for R3 (2x128x128x96) - the old ~2.3 GB claim was wrong.
- **Disk Space:** ~2.5 GB for preprocessed volumes.

---

## 5. Summary of Key Files

| File | Purpose |
|---|---|
| `src/model.py` | 3D Attention U-Net architecture (`16 -> 32 -> 64 -> 128 -> 256`) |
| `src/attention.py` | Additive Attention Gate with LayerNorm |
| `src/losses.py` | Combined Dice + Focal Loss ($\gamma=2.0, \alpha=0.25$) |
| `src/config.py` | All paper constants and paths in one central file |
| `notebooks/pancreas_segmentation_colab.ipynb` | Google Colab interactive notebook with step-by-step UI |
| `README.md` | Full paper reproduction notes, theory, and background |

"""
config.py - Centralized Configuration for 3D Attention U-Net Pancreas Segmentation.

All constants derived from:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942 — DOI: 10.3390/math13243942

Paper ambiguities and implementation assumptions are documented explicitly.
"""

import os

# ==============================================================================
# PREPROCESSING (Section 2.2 of paper)
# ==============================================================================

# HU clipping range — Paper Section 2.2: "clipped to [-100, 240] HU"
HU_MIN = -100.0
HU_MAX = 240.0

# Normalization — Paper Section 2.2: "normalized to [0, 1]"
NORM_MIN = 0.0
NORM_MAX = 1.0

# Voxel resampling — Paper Section 2.2: "resampled to 1×1×1 mm³ isotropic"
TARGET_SPACING = (1.0, 1.0, 1.0)  # mm

# Crop size — Paper Section 2.2: "cropped to 224×224×128"
# IMPLEMENTATION ASSUMPTION:
#   Paper states "centered on abdominal region" without specifying mathematical coordinates.
#   We use geometric volume center crop (no ground-truth mask guidance) to avoid leakage.
CROP_SIZE = (224, 224, 128)  # (H, W, D)

# ==============================================================================
# ARCHITECTURE (Section 2.1, Figure 1)
# ==============================================================================

# PAPER AMBIGUITY #1 — Channel Count:
#   Text (Section 2.1): "starting from 2 and progressing to 16"
#   Figure 1: clearly shows 16 → 32 → 64 → 128 → 256
#
#   RESOLUTION: Follow Figure 1. Figure 1 is mathematically consistent and standard.
ENCODER_CHANNELS = [16, 32, 64, 128, 256]

IN_CHANNELS = 1   # Single-channel CT
OUT_CHANNELS = 1  # Binary segmentation (pancreas vs background)

# ==============================================================================
# LOSS FUNCTION (Section 2.3)
# ==============================================================================

# Paper Section 2.3: "Dice + Focal Loss"
# L_total = L_Dice + L_Focal

# IMPLEMENTATION ASSUMPTION (Focal loss parameters):
#   Paper does NOT numerically specify gamma or alpha for Focal Loss.
#   Standard default from Lin et al. (2017) is used: gamma = 2.0, alpha = 0.25.
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# PAPER AMBIGUITY — Dice loss formula:
#   Paper Eq.(3) omits factor of 2 in numerator.
#   Standard 2x DSC formulation implemented.
DICE_SMOOTH = 1e-5

# ==============================================================================
# OPTIMIZER & SCHEDULER (Section 2.3)
# ==============================================================================

# Paper Section 2.3: "Adam optimizer"
OPTIMIZER = "Adam"

# PAPER AMBIGUITY #2 — Learning Rate:
#   Paper literally prints "6 × 10^4" which equals 6000.
#   Typographical error (missing minus sign). Implemented as 6 × 10^(-4) = 0.0006.
LEARNING_RATE = 6e-4
PAPER_LITERAL_LR = "6 × 10^4"
IMPLEMENTED_LR_NOTE = "Interpreted as 6e-4 (typo: missing negative exponent)"

# Paper Section 2.3: "ReduceLROnPlateau with patience of 100 epochs"
SCHEDULER = "ReduceLROnPlateau"
SCHEDULER_PATIENCE = 100  # Paper's stated value
SCHEDULER_FACTOR = 0.5    # Standard default (not specified by paper)
SCHEDULER_MIN_LR = 1e-7   # Floor LR

# IMPLEMENTATION ASSUMPTION (Epoch Count):
#   Paper does NOT explicitly specify total epochs or early stopping criteria.
#   Patience=100 implies a training run of several hundred epochs.
#   Default set to 300 (practical choice deferred to user/Colab runtime).
DEFAULT_EPOCHS = 300

# Weight decay — Paper does not specify. Adam default = 0.
WEIGHT_DECAY = 0.0

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

# HARDWARE DEVIATION (Batch Size):
#   Paper does NOT specify batch size.
#   Colab Tesla T4 (14.5 GB VRAM) requires batch_size=1 for 224x224x128 3D volumes.
BATCH_SIZE = 1

# Mixed precision — AMP used for hardware feasibility on T4.
USE_AMP = True

# Random seed for reproducibility
SEED = 42

# ==============================================================================
# DATA SPLITTING (Section 2.4)
# ==============================================================================

# Paper: "5-fold cross-validation" + "16 cases held out for independent testing"
# DATASET-AVAILABILITY DEVIATION:
#   Paper states 5x13 + 16 = 81 (text mentions 82). Verified TCIA v2 has 80 cases.
#   Configured for 80 cases: ~64 for 5-fold CV + 16 for test.
N_FOLDS = 5
N_TEST_CASES = 16

# ==============================================================================
# AUGMENTATION (Section 2.2)
# ==============================================================================

# Paper Section 2.2: "random rotations ±10°, flips prob 0.5, shifts up to 10%"
AUG_ROTATION_RANGE = 10.0  # degrees per axis
AUG_FLIP_PROB = 0.5
AUG_SHIFT_RANGE = 0.1  # fraction of spatial dimension

# ==============================================================================
# EVALUATION METRICS (Section 2.4)
# ==============================================================================

# Paper reports: volumetric DSC, ASSD (mm), HD95 (mm)
EVAL_THRESHOLD = 0.5

# ==============================================================================
# PATHS (Colab Drive structure)
# ==============================================================================

# Default local fallback
DEFAULT_DATA_DIR = "./data"
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_RESULTS_DIR = "./results"
DEFAULT_SPLITS_DIR = "./splits"

# Google Drive Paths (Configured for Pancreas-CT)
DRIVE_ROOT = "/content/drive/MyDrive/Pancreas-CT"
DRIVE_RAW_DIR = "/content/drive/MyDrive/Pancreas-CT/pancreas_ct"
DRIVE_ORIGINAL_PROCESSED_DIR = "/content/drive/MyDrive/Pancreas-CT/Processed_data"
DRIVE_2025_PROCESSED_DIR = "/content/drive/MyDrive/Pancreas-CT/2025_Processed_data"
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/Pancreas-CT/checkpoints"
DRIVE_RESULTS_DIR = "/content/drive/MyDrive/Pancreas-CT/results"
DRIVE_SPLITS_DIR = "/content/drive/MyDrive/Pancreas-CT/splits"

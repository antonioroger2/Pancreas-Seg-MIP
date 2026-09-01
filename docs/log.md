# System Run Log

> **Project:** 3D Attention U-Net for Pancreas Segmentation in CT Scans (paper reproduction)
> **Period:** 2026-08-25 → 2026-08-28
> **Hardware:** 2× NVIDIA RTX 6000 Ada (48 GB each)
> **Environment:** Python 3.12, PyTorch 2.13.0+cu130, SimpleITK 2.5.6, nibabel, MONAI 1.6.0, scipy, scikit-learn

This file is a curated operational log of the full pipeline run: data acquisition, DICOM
conversion, preprocessing, training (3 runs + 1 ablation), and evaluation. Raw logs are referenced
in the [Raw Log Index](#raw-log-index).

---

## 1. Timeline Overview

| Phase | When | Result |
|---|---|---|
| Data acquisition (Google Drive → IDC S3) | Aug 25 21:xx – 23:19 | 18,942 DICOMs / 9.3 GB, 0 errors |
| DICOM → NIfTI conversion + z-spacing fix | Aug 25 23:49 | 80/80 ok |
| Preprocessing (run 1, blind crop) | Aug 25 23:55 | 52/80 PASS |
| Test suite | Aug 26 13:15 | 12/12 passed |
| Preprocessing (run 2, pancreas-centroid crop) | Aug 26 13:14 | 77/80 PASS |
| **Run 1 training** (100 ep, blind crop, Dice+Focal) | Aug 25 23:5x – Aug 26 05:15 | val 0.243 ± 0.029 · test DSC 0.192 |
| **Run 2 training** (300 ep, centroid crop, Dice+Focal) | Aug 26 13:17 – Aug 27 03:10 | val 0.278 ± 0.024 · test DSC 0.283 |
| **Loss ablation** (fold 0, 200 ep × 3 losses) | Aug 27 04:12 – 07:30 | Soft Dice winner (0.317) |
| **Run 3 training** (300 ep, centroid crop, Soft Dice) | Aug 27 09:09 – Aug 28 00:53 | val 0.308 ± 0.025 · test DSC 0.316 |

---

## 2. Data Acquisition

### Google Drive attempt (failed)
- Source: user-provided Google Drive folder (IDC mirror).
- Tool: `gdown --folder`.
- **Issue:** Google Drive throttled sequential per-file access ("Cannot retrieve the public link …
  have had many accesses"). Only ~102 files (labels + partial PANCREAS_0001) downloaded before the
  process died.
- Log: `/tmp/opencode/gdown_fold.log` (~1.8 MB).

### Switch to IDC public S3 (successful)
- `metadata.csv` revealed the images live in the public bucket `s3://idc-open-data/<uuid>/*`.
- Wrote a resumable parallel downloader (`download_idc.py`, 24 workers, anonymous S3 via boto3).
- Fixed a prefix bug (bucket name duplicated in prefix) mid-run.
- **Result:** `80 patients` → `DONE {'ok': 18921, 'skip': 21, 'err': 0}` = 18,942 DICOMs, 9.3 GB.
- Layout: `data/pancreas_ct/PANCREAS_XXXX/<study>/<series>/*.dcm`.
- Log: `/tmp/opencode/idc_download.log`.

---

## 3. DICOM → NIfTI Conversion (with a critical fix)

- Script: `convert_to_nifti.py` (SimpleITK `ImageSeriesReader` → `WriteImage`, labels copied from
  `data/lables/labelXXXX.nii.gz`).
- **Data bug found & fixed:** the IDC DICOM metadata carries a **wrong z-spacing** (0.01–0.83 mm)
  while the true inter-slice spacing (from `ImagePositionPatient`) is 1.0 mm and the TCIA labels
  use 1.0 mm. Re-run with z-spacing recomputed per patient from slice positions.
- **Result:** `DONE ok=80 warn=0 fail=0` — 80 image/label pairs slice-aligned
  (e.g. `PANCREAS_0001: (512, 512, 240) sp (0.859, 0.859, 1.0) | lbl (512, 512, 240) sp (0.859, 0.859, 1.0)`).
- Output: `data/Processed_data/{images,labels}/*.nii.gz`.
- Log: `/tmp/opencode/convert.log`.

---

## 4. Preprocessing (`prepare_data.py`)

Pipeline: 1 mm isotropic resample → HU clip [-100, 240] → [0,1] normalize → 224×224×128 crop.

| Run | Crop | Status | Min retention |
|---|---|---|---|
| Run 1 | Blind geometric center | 52/80 PASS, 28 FAIL (containment) | 82.8% |
| Run 2 | **Pancreas-centroid** (`center_crop_on_label`) | **77/80 PASS**, 3 FAIL | 97.8% |

- The 3 remaining run-2 FAILs lose ≤2.2% (organ exceeds the fixed 224×224×128 window even when
  centered): PANCREAS_0005, 0015, 0031.
- Output: `data/2025_Processed_data/{images,labels}/*.nii.gz` (80 + 80).
- Logs: `data/run2_300ep_pancreascrop/logs/01_preprocess.log`, `/tmp/opencode/prepare.log`.
- Manifest: `data/2025_PREPROCESSING_MANIFEST.csv`, summary `data/2025_PREPROCESSING_SUMMARY.md`.

---

## 5. Test Suite

`python test_all.py` → **`12/12 tests passed`** (all stages green), verified both before and after
the crop change.
Log: `data/run2_300ep_pancreascrop/logs/02_test_all.log`.

---

## 6. Training Runs

Common config: 3D Attention U-Net (16→32→64→128→256, 5,668,269 params), Adam lr 6e-4,
ReduceLROnPlateau (patience 100), batch size 1, AMP, seed 42, 16 held-out test patients.

### Run 1 — 100 epochs, blind crop, Dice + Focal (γ=2, α=0.25)
| Fold | Best val DSC |
|---|---|
| 0 | 0.2516 |
| 1 | 0.2263 |
| 2 | 0.2809 |
| 3 | 0.2596 |
| 4 | 0.1980 |
| **Mean** | **0.2433 ± 0.0286** |

Test (16): **DSC 0.1923 ± 0.1255 · ASSD 13.67 mm · HD95 42.46 mm**
Logs: `/tmp/opencode/train.log`, `/tmp/opencode/eval_val.log`, `/tmp/opencode/eval_test.log`.
Artifacts: `data/results_run1/`, `data/checkpoints_run1/`.

### Run 2 — 300 epochs, pancreas-centroid crop, Dice + Focal
| Fold | Best val DSC |
|---|---|
| 0 | 0.2827 |
| 1 | 0.3062 |
| 2 | 0.2840 |
| 3 | 0.2840 |
| 4 | 0.2338 |
| **Mean** | **0.2781 ± 0.0238** |

Test (16): **DSC 0.2830 ± 0.0931 · ASSD 9.17 mm · HD95 34.55 mm**
Logs: `data/run2_300ep_pancreascrop/logs/03_train.log`, `04_eval_val.log`, `05_eval_test.log`.
Artifacts: `data/results_run2/`, `data/checkpoints_run2/`.

### Loss Ablation — fold 0, 200 epochs (parallel on both GPUs)
| Loss | Best val DSC (fold 0) |
|---|---|
| Dice + Focal (γ=1, α=0.5) | 0.2798 |
| Dice + BCE (0.5/0.5) | 0.2964 |
| **Soft Dice (winner)** | **0.3172** |

Logs: `data/ablation/{dicebce,dice,dicefocal_soft}/logs/train.log`.
Also scored precision/recall on fold 0: precision ~0.22–0.24, recall ~0.51–0.55 across all losses
(over-segmentation is loss-agnostic; see §8).

### Run 3 (final) — 300 epochs, pancreas-centroid crop, Soft Dice
| Fold | Best val DSC |
|---|---|
| 0 | 0.3172 |
| 1 | 0.3280 |
| 2 | 0.3033 |
| 3 | 0.3294 |
| 4 | 0.2626 |
| **Mean** | **0.3081 ± 0.0246** |

Test (16): **DSC 0.3160 ± 0.0777 · ASSD 8.62 mm · HD95 32.36 mm**
(unchanged with largest-connected-component post-processing)
Logs: `data/run3_dice_loss/logs/03_train.log`, `04_eval_val.log`, `05_eval_val_postproc.log`.
Artifacts: `data/run3_dice_loss/`, live checkpoints `data/checkpoints/fold_{0..4}/`.

---

## 7. GPU Utilization (monitor.csv)

- **Run 2:** Aug 26 13:17 → Aug 27 03:20. GPU0 (training): ~100% util / ~19 GB during epochs.
- **Run 3:** Aug 27 09:10 → Aug 28 01:13. GPU0 (training): ~100% util / ~19 GB during epochs.
- GPU1 remained idle (~0% / ~10 GB base) — training uses a single GPU (batch size 1).
- Monitors: `data/run2_300ep_pancreascrop/logs/monitor.csv`, `data/run3_dice_loss/logs/monitor.csv`.

---

## 8. Issues Encountered & Resolutions

| # | Issue | Resolution |
|---|---|---|
| 1 | Google Drive throttled gdown folder download | Switched to IDC public S3 (`s3://idc-open-data`) with a parallel downloader |
| 2 | S3 prefix bug (bucket name duplicated) | Strip bucket name from prefix before listing |
| 3 | IDC DICOM z-spacing metadata wrong (0.01–0.83 mm) | Recompute from `ImagePositionPatient`; verified 80/80 alignment |
| 4 | Blind crop dropped pancreas voxels (28/80) | Added `center_crop_on_label`; 77/80 @ 100% retention |
| 5 | Accidental concurrent training process (resumed a stale checkpoint) | Killed both processes, cleared `data/checkpoints`, relaunched cleanly |
| 6 | `@torch.no_grad` decorator misplaced after adding `keep_largest_components` → `.numpy()` grad error | Restored decorator above `evaluate_volumes` |
| 7 | Over-segmentation (precision ~0.22) caps DSC across all losses | Diagnosed as capacity/data ceiling (BN recomputed-stats test and alignment checks rule out pipeline bugs); see `REPRODUCTION_REPORT.md` |

---

## 9. Final Results (held-out test, 16 patients)

| Run | Crop | Loss | Epochs | Val DSC (mean) | Test DSC | ASSD (mm) | HD95 (mm) |
|---|---|---|---|---|---|---|---|
| 1 | blind | Dice+Focal | 100 | 0.243 | 0.192 | 13.67 | 42.46 |
| 2 | centroid | Dice+Focal | 300 | 0.278 | 0.283 | 9.17 | 34.55 |
| **3** | **centroid** | **Soft Dice** | **300** | **0.308** | **0.316** | **8.62** | **32.36** |

Comparison table: `data/run2_300ep_pancreascrop/metrics_comparison.csv`.
Full analysis: `REPRODUCTION_REPORT.md`.

---

## Raw Log Index

| Path | Contents |
|---|---|
| `/tmp/opencode/gdown_fold.log` | Failed Google Drive attempt (rate-limited) |
| `/tmp/opencode/idc_download.log` | IDC S3 download (18,942 DICOMs, 0 errors) |
| `/tmp/opencode/convert.log` | DICOM→NIfTI conversion (80/80) |
| `/tmp/opencode/prepare.log` | Preprocessing run 1 (52/80) |
| `data/run2_300ep_pancreascrop/logs/01_preprocess.log` | Preprocessing run 2 (77/80) |
| `data/run2_300ep_pancreascrop/logs/02_test_all.log` | Test suite (12/12) |
| `/tmp/opencode/train.log` | Run 1 training (100 ep) |
| `data/run2_300ep_pancreascrop/logs/03_train.log` | Run 2 training (300 ep) |
| `data/ablation/{dicebce,dice,dicefocal_soft}/logs/train.log` | Loss ablation runs |
| `data/run3_dice_loss/logs/03_train.log` | Run 3 training (300 ep) |
| `data/run2_300ep_pancreascrop/logs/{04_eval_val,05_eval_test}.log` | Run 2 evaluation |
| `data/run3_dice_loss/logs/{04_eval_val,05_eval_val_postproc}.log` | Run 3 evaluation |
| `data/run2_300ep_pancreascrop/logs/monitor.csv` | Run 2 GPU/epoch monitor |
| `data/run3_dice_loss/logs/monitor.csv` | Run 3 GPU/epoch monitor |
# CHANGELOG

Every decision, fallback and number below comes from a saved file produced by code (path given).

## Environment
- Local machine: NVIDIA RTX 5060 Laptop 8 GB (CUDA available), Python 3.14, PyTorch 2.14, MONAI 1.6.1, SimpleITK 2.5.6. Used for T1-T4 dev/tests only. All training runs on Kaggle (2x T4 16 GB).
- Base-paper PDF, rubric PDF and MIP.pdf are NOT in the workspace; paper facts are taken from MASTER_BUILD_GUIDE.md [PAPER] entries only. Repo lives at the workspace root (no `Pancreas-Seg-MIP-main/` folder).

## T1 - Alignment gate (tools/check_alignment.py, results/alignment/)
- Data on Drive: `Processed_data/{images,labels}` (course NIfTI), `lables/labelXXXX.nii.gz` (TCIA), `pancreas_ct/` (IDC DICOM), `metadata.csv`. Tools: `tools/fetch_drive.py`, `tools/fetch_idc.py` (public IDC S3).
- FALLBACK (user instruction, bandwidth ~0.35 MB/s Drive): T1 run on the **13 course cases + 8 IDC cases available locally**, not all 80. All 80 labels are local. The remaining 67 cases are checked by the same tool on Kaggle (IDC source) before preprocessing (notebook cell, gate stops on failure).
- Result (`alignment_summary.json`): rule = **identity `T0_fx0_fy0_fz0`** for both sources (most passing cases, 11/13 course, 7/8 IDC; identity = per-case best in-range variant in 10/13). air_frac = 0.000 for all 21 case-sources; in-range [60,180] HU 0.36-0.99; mean HU 43-118.
- The 2 threshold failures (0004: 43 HU / in-range 0.36; 0021: 49 HU / 0.44) are low-enhancing pancreases; overlays (`overlay_course_0004/0021.png`) show the label on the pancreas. Reported, not hidden; no per-case override.
- IDC DICOM read with the GDCM series reader is **bit-identical** to the course NIfTI (array, spacing, origin, direction) on 8/8 cases (`idc_equals_course` column) -> source choice is moot; course NIfTI is canonical locally, IDC is used on Kaggle.
- DECISION: edge_score not gated: 0.90-1.31 for every variant on the first 8 cases (pancreas borders are low-contrast), so it cannot separate orientations. Pass = air_frac <= 0.01 and in_range >= 0.5 (the T1 acceptance).
- Diagnosis of Review-1: its overlay figure puts the GT of test patients 0073 and 0030 on heart/lung slices, while the source data puts them correctly at kidney level -> the Review-1 preprocessed arrays were misaligned **relative to the source**, not the source itself. The Review-1 `convert_to_nifti.py` is not available; reading the DICOMs in ascending IPP-z order (a common pydicom pattern) reverses the slice order and drops 0030's label at the liver dome/stomach (mean HU 22, in-range 0.25). Consistent hypothesis, **not proven**.

## T2 - Full-volume preprocessing (src/data/prepare_data.py, src/data/preprocessing.py)
- Pipeline: T1 rule on label array -> label put on the image's voxel grid -> SimpleITK resample 1 mm (linear CT pad -1024, nearest label on the SAME reference grid) -> clip [-100,240] -> [0,1] -> full volume `.npz` (image fp16, label uint8, (H,W,D)=(y,x,z)) + meta json + manifest.csv + qc_summary.json. No crop, no GT use.
- `center_crop_on_label` (GT-centroid crop, Review-1 runs 2-3) **removed**. New `blind_center_crop` (shape-only offsets), `crop_at`, `paste_back`; test 13 (round trip, incl. padding) passes.
- Local: 13/13 QC pass (`prep/qc_summary.json`, not committed). Full 80 are preprocessed on Kaggle from IDC (FALLBACK: bandwidth; same code).
- Splits (`splits/patient_splits.json`): same seed-42 shuffle as Review-1 -> **identical 16-patient test set** (contains 0023/0030/0073, the Review-1 figure's test cases); folds via array_split 13/13/13/13/12 as the guide specifies (fold 0 = 51 train / 13 val; Review-1 used 52/12).

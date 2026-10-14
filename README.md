# 3D Attention U-Net for Pancreas Segmentation in CT: replication + ablation (22AIE437)

Replication of *Tsewalo Tondji et al., "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in
CT Scans", Mathematics 2025, 13(24), 3942* (doi 10.3390/math13243942) on NIH Pancreas-CT, plus three modifications
(M1 foreground-biased patches + sliding window, M2 GroupNorm, M3 deep supervision) and two clinical extensions
(C1 volumetry, C2 TTA-uncertainty review flag).
Team: Baranidharan (CB.SC.U4AIE23015), Pranav (...054), Antonio Roger (...104), Naren B S (...152).

**Status / honesty:** every number in `results/` is produced by code from saved outputs; the decision log is
[CHANGELOG.md](CHANGELOG.md). Review-1 results (`docs/`, `results/*.png`) are historical: runs 2-3 used a ground-truth
-centroid crop (label leakage) and the Review-1 preprocessed arrays were misaligned (see CHANGELOG T1).

## Pipeline

| Step | Code | Output |
|---|---|---|
| T1 alignment gate (16 label orientation variants) | `tools/check_alignment.py` | `results/alignment/` |
| T2 full-volume preprocessing (1 mm, HU [-100, 240] -> [0, 1], no crop, no GT) + seed-42 splits | `src/data/prepare_data.py` | `prep/`, `splits/patient_splits.json` |
| Training R0-R4 (step-based, AMP, resumable) | `src/train.py` | `results/R*/train_log.csv, best.pth, last.pth, config.json, curves.png` |
| Full-volume test evaluation | `src/evaluate.py` | `results/R*/test_per_patient.csv, summary.json` |
| C1 volumetry, C2 TTA uncertainty, case cards | `src/clinical.py`, `src/tta.py` | `results/clinical/` |
| All rubric figures/tables | `tools/make_figures.py` | `results/figures/` + `captions.md` |

Runs (identical split, seed 42, Adam 6e-4, Dice + Focal (gamma 2, no alpha), GPU augmentation, AMP, threshold 0.5,
the same number of optimizer steps N, full 1 mm volume evaluation):

| Run | Flags | Input | Norm | Deep sup. | Attention |
|---|---|---|---|---|---|
| R0 paper-faithful baseline | (defaults) | blind-centre 224x224x128 crop, batch 1 | BN | - | yes |
| R1 (+M1) | `--patch` | 2 x 128x128x96 patches (pos:neg 1:2), sliding window | BN | - | yes |
| R2 (+M2) | `--patch --norm group` | patches | GN(8) | - | yes |
| R3 (+M3, final) | `--patch --norm group --deepsup` | patches | GN | yes | yes |
| R4 (optional) | R3 + `--no_attention` | patches | GN | yes | no |

## Setup

```bash
pip install -r requirements.txt          # Python >= 3.10, PyTorch >= 2.0, MONAI >= 1.2
python -m src.test_all                   # 18 tests, CPU or GPU
```
Data: course NIfTI pairs (`tools/fetch_drive.py`) or the identical IDC DICOM (`tools/fetch_idc.py --meta data/metadata.csv`)
+ TCIA labels in `data/raw_labels/`. Exact commands, including Kaggle (2x T4), are in [REPRODUCE.md](REPRODUCE.md).

## One-command reproduction (2-GPU Linux / Kaggle)

```bash
bash scripts/run_ablation.sh /tmp/prep results
```
(tests -> benchmark -> R0-R3 on two GPUs -> test evaluation -> clinical -> figures).

## Repository layout
`src/` (config, data, models, losses, augment, train, evaluate, metrics, inference, tta, clinical, test_all) ·
`tools/` (check_alignment, fetch_drive, fetch_idc, unpack_npz, benchmark_steps, make_figures) ·
`scripts/` (kaggle_launch, run_ablation.sh, package) · `notebooks/kaggle_ablation.ipynb` ·
`docs/` (Review-1 documents, corrected; FINAL_RESULTS.md) · `MASTER_BUILD_GUIDE.md` (plan).

# HANDOFF - agent context for continuing this project

Branch: `t1-t10-replication-pipeline` (pushed to `origin` = github.com/antonioroger2/Pancreas-Seg-MIP). Local `main` = `origin/main`
= `07fe4e4 build guide` (untouched). PR NOT opened yet (no `gh` CLI on the dev machine, browser not signed in); prefilled
compare URL + description were saved in `dist/PR_DESCRIPTION.md` / `dist/PR_PREFILLED_URL.txt` (dist/ is gitignored, local
only). Last commit at handoff: `3f1081b`. Read in this order: this file -> `CHANGELOG.md` (every decision, number,
fallback) -> `MASTER_BUILD_GUIDE.md` (plan, rules D1-D10, sec. 15 hard rules) -> `README.md` / `REPRODUCE.md`.

## 1. Project in one paragraph
22AIE437 final project (team: Baranidharan CB.SC.U4AIE23015, Pranav ...054, Antonio Roger ...104, Naren B S ...152).
Replicate Tsewalo Tondji et al., Mathematics 2025, 13(24), 3942 (3D Attention U-Net, NIH Pancreas-CT; paper Dice
80.8 +- 2.1 %, HD95 8.1 +- 1.6 mm, ASSD 2.1 +- 0.3 mm, Jaccard not reported, trained on a T4 16 GB) + M1 patches/sliding
window, M2 GroupNorm, M3 deep supervision + C1 volumetry, C2 TTA-uncertainty flag. Graded on correctness, understanding,
honesty - not score. Review-1 best test Dice 0.316 is HISTORICAL/invalid (GT-centroid crop leakage + misaligned arrays).

## 2. Status per task (T1-T10)
| Task | Status | Key facts |
|---|---|---|
| T1 alignment | DONE on 13 local cases (+8 IDC); all-80 check runs on the cloud box before training | rule = identity `T0_fx0_fy0_fz0`; air_frac 0.000 everywhere; 0004 (43 HU) and 0021 (49 HU) fail in-range>=0.5 but overlays show correct placement (low-enhancing pancreas); edge_score not gated (0.90-1.31 for every variant); IDC DICOM bit-identical to course NIfTI 8/8; course labels == TCIA raw labels 80/80. Review-1 figure put GT of 0073/0030 on heart/lung -> Review-1 *preprocessed* arrays were wrong (z-reversed read = consistent hypothesis, unproven; old convert script unavailable). Outputs `results/alignment/`. |
| T2 preprocessing | DONE (13/13 QC locally); all 80 produced on the cloud box | full 1 mm volume, no crop, no GT; label put on image grid (CopyInformation) then nearest-resampled on the image's reference grid; npz fp16 image + uint8 label, (H,W,D)=(y,x,z); `center_crop_on_label` deleted. Splits `splits/patient_splits.json`: same seed-42 shuffle as Review-1 -> identical 16 test IDs (0001,0005,0011,0013,0019,0023,0030,0032,0033,0035,0037,0047,0051,0069,0071,0073); folds np.array_split 13/13/13/13/12; fold 0 = 51 train / 13 val. |
| T3 metrics/eval | DONE, tested | Dice, Jaccard, precision, recall, HD95, ASSD (surface-to-surface, union bbox, ASSD = mean of directed means), volumes, V ratio; empty pred -> inf, counted (`n_inf_*`, `n_empty_pred`), Dice kept. `src/evaluate.py` = only script reading test IDs; saves `test_per_patient.csv`, `summary.json`, `preds_test/*.npz` (bit-packed). |
| T4 model/trainer | DONE, tested | R0 params 5,668,269 exact; GN same; +deepsup 5,668,496; R4 5,646,612; `--ag paper` 5,722,669. Focal gamma 2 alpha None (D1). Sched ReduceLROnPlateau on val Dice, patience = ceil(N/K)//3 checks (D3). Own pos:neg 1:2 sampler with cached fg indices instead of MONAI RandCropByPosNegLabeld (logged). Step-based, AMP, best.pth by val Dice, last.pth every 300 steps, auto-resume, train_log.csv logs CURRENT lr. Local peak VRAM: R0 6.55 GB, R3 6.32 GB. |
| T5 benchmark | CODE DONE; N, K NOT CHOSEN | `tools/benchmark_steps.py --wall_hours H [--n_gpus G]` -> `results/benchmark.json` (N multiple of 100, K for ~15 % validation cost, longest-first schedule per GPU, warns if N < 3000 + hours needed for 3000). Laptop dry run (NOT a decision): R0 1.83 s/step, R1 0.83, R2 0.96, R3 1.01, R4 0.79 -> R0 is the bottleneck for a shared N. |
| T6 sanity | PENDING (user runs) | `kaggle_launch.py --sanity_steps S --K k`: R0 + R3, validation only; report val-Dice trend to the user BEFORE the full set. |
| T7 ablation launcher | CODE DONE | `scripts/kaggle_launch.py`: one thread-driven queue per GPU (CUDA_VISIBLE_DEVICES, REQUIRE_CUDA=1 -> no silent CPU), train -> evaluate; schedule from benchmark.json; `--with_r4`, `--patch_size`/`--batch_size` OOM ladder (112 112 80 -> 96 96 64 -> batch 1, all patch runs identically). Smoke-tested on 1 GPU (sanity mode). |
| T8 clinical | CODE DONE (smoke-tested only) | `src/clinical.py`: C1 all runs; C2 for R3 with 4-flip TTA overlap 0.25; flag threshold = val P75 uncertainty, "poor" = Dice < val P25 (both fixed on VALIDATION before results); case cards best/median/worst. Outputs `results/clinical/`. |
| T9 figures | CODE DONE (smoke-tested only) | `tools/make_figures.py` -> `results/figures/` (a)-(f) + `captions.md` + `final_results_auto.md` (verdict rule fixed in advance: delta mean Dice >= +0.02 improved, <= -0.02 worse, else no clear difference). Axial slices shown anterior-up. Palette R0 #2a78d6, R1 #eb6834, R2 #1baf7a, R3 #eda100, R4 #e87ba4. |
| T10 docs | DONE except results | README.md, REPRODUCE.md, CHANGELOG.md; Review-1 docs carry a HISTORICAL correction banner; `docs/FINAL_RESULTS.md` intentionally has no numbers yet. |

Tests: `python -m src.test_all` = 18/18 (1-12 original intent, 2/8/9/11 ported; 13 crop round trip, 14 toy metrics,
15 model flags, 16 augmentation, 17 one-step overfit, 18 sampler + R0 paste-back). Last full run before the
portability commit; re-run first.

## 3. What to do next (in order)
1. Re-run `python -m src.test_all` (18/18 expected).
2. On the GPU box: `bash scripts/prepare_cloud.sh /scratch/work` (Kaggle: notebook cells 1-3) - stops unless T1 rule
   == identity, max air_frac <= 0.01, QC 80/80, splits identical. Report the all-80 T1 numbers into CHANGELOG T1.
3. Benchmark on THAT machine; record N, K, schedule, s/step, VRAM in CHANGELOG T5. If N < 3000 tell the user.
4. T6 sanity; send val-Dice trend to the user; wait for go.
5. Full launch -> clinical -> figures (commands in REPRODUCE.md / the user summary below).
6. Copy `results/figures/final_results_auto.md` tables into `docs/FINAL_RESULTS.md` verbatim; fill verdicts with caveats
   (n = 16, single fold, single seed). Update `results/vis_results.md` / slides content only from generated files.
7. Open the PR (`gh pr create --base main --head t1-t10-replication-pipeline`, not draft) or via the saved URL.

Commands for a non-Kaggle Linux GPU box (from the unzipped `dist/pancreas-code.zip` or a clone + `data/raw_labels`):
```
pip install -r requirements.txt
bash scripts/prepare_cloud.sh /scratch/work
python tools/benchmark_steps.py --data /scratch/work/prep_npy --wall_hours 3 --out results/benchmark.json
python scripts/kaggle_launch.py --data /scratch/work/prep_npy --sanity_steps 900 --K 150 --out sanity --logs sanity/logs
python scripts/kaggle_launch.py --data /scratch/work/prep_npy --out results            # add --with_r4 if time
python -m src.clinical --results results --data /scratch/work/prep_npy
python tools/make_figures.py --results results --data /scratch/work/prep_npy
```
Kaggle: upload `dist/pancreas-code.zip` (built by `python scripts/package.py`) as dataset `pancreas-code`; notebook
`notebooks/kaggle_ablation.ipynb`, GPU T4 x2, Internet On; flags RUN_SANITY / RUN_FULL / WITH_R4 / WALL_HOURS in cell 1.

## 4. Tiny details that matter
- **Data is NOT in git.** `/data/` and `/prep/` are gitignored. `data/raw_labels/*.nii.gz` (80 TCIA labels, 5.8 MB) and
  `data/metadata.csv` (IDC S3 prefixes) exist only on the dev machine and inside `dist/pancreas-code.zip`. A fresh clone
  must get them: `python tools/fetch_drive.py --out data --subset raw_labels` (labels from the shared Drive folder
  119d5ldajERVPFwZj0C4K0Io1d3x6A6kH) and metadata.csv from Drive `metadata/` (id 1V3xXbIw51ie9bY-5Ofb7GVg5OOFTFrxs, via gdown).
- Drive is slow (~0.35 MB/s/stream, throttles); IDC S3 (`tools/fetch_idc.py`, anonymous HTTPS, retries) is the bulk
  source. DICOM read must use the GDCM series reader (`sitk.ImageSeriesReader`) - that order equals the course NIfTI.
  Never sort slices by ascending IPP-z (reverses z vs labels).
- IDC/course geometry: origin 0, direction diag(1,-1,-1): array index 0 = most superior slice, row 0 = posterior. Some
  TCIA labels (0007, 0027, 0032, 0045) print harmless "unexpected scales in sform" ITK warnings.
- In-plane spacing seen: 0.66-0.96 mm (13 local cases); z = 1.0 mm.
- `prepare_data.py` reads the rule from `results/alignment/alignment_summary.json` (key "course" else "idc"); writes
  splits only when all 80 labels are present; a failing case is written as FAIL meta, never skipped.
- Sample i of the training samplers is a pure function of (seed, i) -> exact resume; CropSampler zero-pads volumes
  smaller than 224x224x128; inference R0 pastes the crop back with background logit -20.
- LayerNorm in the attention gates is over (C,H,W,D) -> ROI must equal the training patch (both 128x128x96).
- Validation: full volume, overlap 0.25, all 13 val cases (`--val_max_cases` exists, default 0 = all - keep identical
  across runs). Test: overlap 0.5.
- Windows dev box: Python 3.14 via `py`; `bash` there resolves to WSL (no distro) - launcher no longer needs bash.
  `.gitattributes` forces LF for .py/.sh/.ipynb. Kaggle Python 3.10/3.11: avoid 3.12-only f-string quote nesting.
- `src/cross_validation.py`, `src/data/qc_preprocessing.py`, `PancreasVolumeDataset`, epoch trainer: deleted on purpose.
- Docs D9: the Review-1 slide claim "authors had bigger GPUs" must be removed (teammate edits Beamer .tex; MIP.pdf,
  paper PDF and rubric PDF are NOT in the workspace).

## 5. Hard rules (do not break)
No GT at validation/test time (only training targets, training-patch location, evaluation). Test IDs only in
`src/evaluate.py` and `src/clinical.py` (C2 thresholds from validation). Same N, split, seed, optimizer, scheduler,
augmentation, AMP, threshold, full-volume evaluation for R0-R4. Do not "improve" R0. Every number from a saved CSV/JSON;
never hand-edit or round; report failures plainly; log every decision/fallback in CHANGELOG.md; commit after each task.

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

## T3 - Metrics + full-volume evaluation (src/metrics.py, src/evaluate.py, src/inference.py)
- Metrics on the full 1 mm volume: Dice, Jaccard, precision, recall, HD95, ASSD (spacing-aware), V_pred/V_GT, volumes (mL). Empty prediction -> HD95/ASSD = inf, `empty_pred` flag, counted as `n_inf_*`/`n_empty_pred` in summary.json; Dice keeps the case (never dropped).
- Surface distances are surface-to-surface (standard; Review-1 measured to the nearest GT voxel) on the union bounding box (exact, faster). ASSD = mean of the two directed means (Review-1/medpy definition).
- `predict_logits`: R0 = blind 224x224x128 crop -> model -> paste back, outside = background (logit -20); R1-R4 = MONAI sliding window (ROI = training patch, sw_batch 4, gaussian, overlap 0.5 test / 0.25 validation).
- `evaluate.py` is the only script that loads test patients; writes `test_per_patient.csv` + `summary.json` (mean, SD ddof=1, median, min, max).
- Tests 14 (toy masks: shifted cube Dice 0.8, Jaccard 0.667, HD95 4.0 mm at 2 mm spacing; over-segmentation precision/V-ratio; empty -> inf counted) and 18 (R0 paste-back) pass; end-to-end evaluate on a dummy checkpoint ran (test 9 + manual run on 2 local cases).

## T4 - Flags, GPU augmentation, step-based trainer (src/models/*, src/losses.py, src/augment.py, src/data/dataset.py, src/train.py)
- Model flags `--norm {batch,group} --deepsup --no_attention --ag {simple,paper}`; R0 defaults reproduce **5,668,269** params exactly (test 15). GN: same count; +deepsup 5,668,496; R4 (no attention) 5,646,612; `--ag paper` 5,722,669.
- D1: focal gamma 2, **alpha None** (config `FOCAL_ALPHA=None`, `--focal_alpha` flag). Losses computed in fp32 under AMP. Deep supervision w = [1, .5, .25, .125]/1.875, aux heads full-size.
- D2: `--ag paper` implemented (C projections + LN, C/2 reduction + LN, final 1x1x1 conv + LN, parameter-free LN as in the repo gate); the ablation uses `--ag simple`.
- D3: ReduceLROnPlateau on val Dice (mode max, factor 0.5, min 1e-7), patience = ceil(N/K)//3 validation checks; `--sched_metric loss` available (val loss = Dice+Focal on the full-volume logits).
- D7: one `gpu_augment` (affine_grid/grid_sample) for all runs; test 16 (identity, forced flip == torch.flip, binary label, image in [0,1]).
- DECISION (M1 sampler): MONAI RandCropByPosNegLabeld builds full fg/bg index arrays (~320 MB) on every call over a full volume; replaced by the same rule (pos:neg 1:2, centre on random fg / bg voxel, clamped) with cached fg indices. Test 18: fg-centred fraction ~1/3.
- Trainer: step-based, AMP, `best.pth` by val Dice, `last.pth` every 300 steps + automatic resume (test 11), `train_log.csv` with the CURRENT lr per row (fixes the Review-1 single-lr bug), seconds, peak VRAM; npy mmap via `tools/unpack_npz.py`; sample i is a pure function of (seed, i) so resume is exact.
- Local smoke (RTX 5060 8 GB, toy split of local non-test cases): R0 peak VRAM 6.55 GB (224x224x128, batch 1, AMP); R3 6.32 GB (2x128x128x96). Full-volume validation ~15 s/case locally.
- Removed dead modules `src/cross_validation.py` (imported the old epoch trainer) and `src/data/qc_preprocessing.py` (QC now in prepare_data). Tests 1-12 adapted to the new APIs (2, 8, 9, 11 rewritten, same intent): **18/18 pass** (`python -m src.test_all`).

## T5 - Benchmark (tools/benchmark_steps.py) - N and K are NOT chosen yet
- Measures s/step (20 iters after 5 warm-up, cudnn.benchmark as in training, real data loader), peak VRAM, one validation case (overlap 0.25) and one test case (overlap 0.5) for R0-R4; picks ONE N (multiple of 100) for all runs = largest N whose longest-first 2-GPU schedule of R0-R3 fits `--wall_hours` (default 3 h), K so validation is ~15 % of the slowest run; warns if N < 3000 and prints the hours needed for N = 3000. Writes results/benchmark.json.
- Local dry run (RTX 5060 laptop, toy split, NOT the decision): R0 1.83 s/step (6.5 GB), R1 0.83, R2 0.96, R3 1.01 (4.5 GB), R4 0.79; GPU-only R0 1.60 s/step; data loading 0.04 s/sample -> R0 (one 224x224x128 crop per step) is the bottleneck for a shared N.
- OOM ladder available as `--patch_size` in the launcher (applied to every patch run identically).

## T7 - Kaggle kit (scripts/kaggle_launch.py, scripts/run_ablation.sh, scripts/package.py, notebooks/kaggle_ablation.ipynb)
- FALLBACK (bandwidth): data is fetched and preprocessed ON KAGGLE from IDC S3 (`tools/fetch_idc.py`) + bundled TCIA labels (`data/raw_labels`, 80/80 identical to the course labels: array, spacing, direction). The notebook gates on T1 for all 80 (identity rule, air <= 0.01, no shape mismatch), T2 QC 80/80 and splits == committed file before training.
- One process per GPU (CUDA_VISIBLE_DEVICES), runs chained train -> evaluate; schedule and N, K from benchmark.json; resumable; T6 sanity = short R0 (GPU0) + R3 (GPU1), validation only.
- `python scripts/package.py` -> dist/pancreas-code.zip (1.1 MB: code, splits, T1 rule, TCIA labels, IDC metadata).

## T8 - Clinical extensions (src/tta.py, src/clinical.py) - code done, results pending Kaggle
- C1: volumes from each run's non-TTA test prediction (`test_per_patient.csv`): Pearson r, MAE, mean abs % error, bias, 95 % LoA, V ratio -> `results/clinical/volumes.csv`, `volumetry_summary.json`, `volumetry.png` (scatter + Bland-Altman).
- C2 (R3): 4-flip TTA (identity + one flip per axis, D10) at overlap 0.25, entropy of the mean probability, case score = mean entropy in mask. DECISIONS fixed before results, both on VALIDATION only: flag threshold = P75 of validation scores; "poor case" = Dice < P25 of validation TTA Dice. Test: Spearman rho, flag precision/recall, TTA vs single Dice (Wilcoxon). Case cards best/median/worst (slices chosen at GT z-percentiles 25/50/75 for display only).
- `evaluate.py` now also saves bit-packed test masks (`preds_test/*.npz`) for figures.
- Smoke-tested end to end on untrained toy runs (degenerate: empty masks -> NaN statistics, as expected).

## T9 - Figures (tools/make_figures.py) - code done, results pending Kaggle
- (a) paper vs R0 vs R3 table (.csv/.md/.tex, paper Jaccard "n/r", D4 footnote), (b) best/median/worst Input|GT|R0|R3, (c) loss + val Dice curves, (d) ablation table + bars with per-patient dots, (e) per-patient Dice, volume scatter, precision-vs-recall (over-segmentation), (f) paired Wilcoxon (R0 vs R3 + consecutive pairs; HD95 pairs with inf dropped and n reported), `captions.md`, and `final_results_auto.md` with the verdict per addition (rule fixed in advance: |delta Dice| < 0.02 = no clear difference). Only the paper's Table 1 values are constants (quoted with source).
- Colours: reference categorical order (R0 blue, R1 orange, R2 aqua, R3 yellow, R4 magenta); scatter plots limited to R0 vs R3 (<= 3 series).

## T10 - Docs
- New README.md (renamed from Readme.md), REPRODUCE.md, docs/FINAL_RESULTS.md (skeleton, no numbers until Kaggle results).
- Review-1 docs (docs/*.md, results/vis_results.md) get a HISTORICAL banner with the corrections: 0.808 not ~0.86; GT-centroid crop invalid (leakage); "Not alignment" corrected; "plateau" -> peak then decline (overfitting); real hardware (Review-1: RTX 6000 Ada; paper: T4 16 GB, D9); old 2.3 GB VRAM claim replaced by measured 6.55 GB (R0) / 6.32 GB (R3). Historical numbers themselves are left as logged.
- D9: the Review-1 slide claim that the authors had bigger GPUs is wrong (paper: T4 16 GB) and must be deleted from the slides (MIP.pdf is not in this workspace; teammate B to edit the .tex).

## Portability fix (any Linux/cloud GPU runner, e.g. RTX 6000 Ada)
- Launcher and benchmark no longer assume 2 GPUs: `--n_gpus` (default = visible GPUs) drives the longest-first schedule; the launcher runs one queue per GPU, so a single GPU runs its queue sequentially and sanity mode runs R0 then R3. Queues are Python threads (no bash dependency); launched runs set REQUIRE_CUDA=1, so a missing GPU fails loudly instead of silently training on CPU. Default output is `results/` (Kaggle paths are passed explicitly by the notebook). `--batch_size` passthrough completes the OOM ladder.
- New `scripts/prepare_cloud.sh`: IDC download -> T1 on all 80 -> T2 -> npy, with the same gates as the notebook.
- Verified locally only as a smoke test: sanity mode on 1 GPU (R0 then R3, 4 steps each, exit 0). Full-schedule launch was not re-run locally (user asked to stop local runs).

## B1 - Best-model run: nnU-Net v2 ResEnc-M, 240-min budget (src/nnunet_trainer.py, tools/nnunet_*.py, scripts/nnunet_run.sh)
- Goal (user): one run, test Dice >= 0.80 (paper 0.808), training finished within 240 min on one RTX 6000 Ada (GPU box, GPU 1). Separate from the R0-R4 ablation (those rules are unchanged).
- DECISION architecture: nnU-Net v2 3d_fullres with the residual-encoder M preset (nnUNetPlannerResEncM; Isensee et al., MICCAI 2024: ResEnc presets beat the original nnU-Net and transformer models). ResEnc-L rejected: ~3x the compute of M, does not fit 240 min. Same split as R0-R4: fold 0 = 51 train / 13 val, the 16 test IDs only in imagesTs/labelsTs.
- DECISION budget: `nnUNetTrainerBudget` re-plans the epoch count after every epoch from the median epoch time so epochs end within NNUNET_BUDGET_MIN (210) min; the poly LR end point moves with it, so the LR still decays to ~0 at the last epoch. `timeout 240m` on the train command as a hard stop. Resumable (`--c`).
- Post-processing (largest component or not) chosen by nnU-Net on the 13 VALIDATION cases only, then applied to test. Test scored with src/metrics.py at native spacing (tools/nnunet_eval.py), mirroring TTA (nnU-Net default).
- DATA: `data/Processed_data` on the GPU box is NOT the canonical course NIfTI: T1 identity there gives air_frac 0.05 and mean HU 41 on 0001 (canonical: 0.000 / 103) and a different image origin - it is the misaligned Review-1 conversion. The dataset is built from the TCIA DICOM (`data/pancreas_ct`, read with the T1 loader `load_idc`) + raw TCIA labels (`data/lables`); `nnunet_prepare.py` refuses to run unless the all-80 T1 check (`results/alignment_all80/`) gives the identity rule with no shape mismatch.
- T1 all 80 (`results/alignment_all80/alignment_summary.json`, IDC DICOM + TCIA labels): rule identity, 80 cases, 0 shape mismatches, 73/80 pass; max air_frac 0.0007. The 7 below in-range 0.5 (0004, 0014, 0018, 0021, 0026, 0044, 0058) have air 0.000 and mean HU 37-56: low-enhancing pancreas, not misalignment (no flip variant fixes them).
- nnU-Net plan (ResEnc-M): spacing 1.0 x 0.879 x 0.879 mm, patch 80 x 192 x 160, batch 2, 6 stages 32-320 features.
- GPU box environment: nnunetv2 2.8.1 in the project `.venv` (system torch 2.13.0+cu130 reused; torch-dependent packages installed with --no-deps because pip otherwise pulls a second torch); torchvision 0.28.0+cu130 in the venv (the shared 0.20.1 breaks nnU-Net's trainer discovery: `torchvision::nms does not exist`). FALLBACK: torch.compile off (`nnUNet_compile=f`): the triton cache fails on the NFS home (Errno 61) and caches must stay in the project folder.
- Smoke run (`nnUNetTrainerBudget_smoke`, 4 epochs) end to end: 59 s/epoch + ~3 s overhead, startup ~2 min, final validation of 13 cases 7.8 min, test stage (post-processing + 16 predictions with TTA + metrics) ~14 min; 4-epoch test Dice 0.4704 (pipeline check only). Budget bug found by the self-check (`python -m src.nnunet_trainer`) and fixed before the real run: an exhausted budget kept granting one more epoch.
- Real run: NNUNET_BUDGET_MIN=222 (2 min startup + 222 min epochs + 8 min validation = ~232 min < 240 min hard stop).

# Architecture: from the paper's 3D Attention U-Net to nnU-Net ResEnc-M

This document explains the model we started from (the base paper's 3D Attention U-Net, as implemented in
`src/models/`), why it scored poorly in Review-1, and the architecture and training pipeline we moved to
(nnU-Net v2 with a residual encoder, "ResEnc-M") that reached **test Dice 0.859** against the paper's **0.808**.

Every number below comes from a saved file: the paper's own table, `docs/REPORT.md` (Review-1),
`results/nnunet/nnUNetTrainerBudget/` (our run) or `CHANGELOG.md`. Parameter counts come from instantiating the
networks.

| | Model | Test Dice (16 patients) | HD95 (mm) | ASSD (mm) |
|---|---|---|---|---|
| Base paper (NIH, Table 1) | 3D Attention U-Net | 0.808 ± 0.021 | 8.1 ± 1.6 | 2.1 ± 0.3 |
| Review-1 (ours, historical, invalid) | same, run 3 | 0.316 ± 0.078 | 32.36 | 8.62 |
| **B1 (ours, final)** | **nnU-Net v2 ResEnc-M** | **0.859 ± 0.044** | **4.46 ± 2.42** | **1.10 ± 0.32** |

---

## 1. The architecture before: 3D Attention U-Net (the paper)

Source: Tsewalo Tondji et al., *Mathematics* 2025, 13(24), 3942. Implemented in `src/models/model.py` and
`src/models/attention.py`, with hyper-parameters in `src/config.py`.

### 1.1 Network

A 5-level 3D U-Net with channels **16-32-64-128-256** (paper Fig. 1) and an additive attention gate on every skip
connection.

```
input 1 x 224 x 224 x 128   (one blind-centre crop of the 1 mm volume, HU [-100, 240] -> [0, 1])
│
├─ Enc 1  DoubleConv(1 -> 16)      16 @ 224 x 224 x 128 ───────────────── skip 1 ─┐
│  MaxPool 2                                                                     │
├─ Enc 2  DoubleConv(16 -> 32)     32 @ 112 x 112 x 64  ──────────── skip 2 ─┐   │
│  MaxPool 2                                                                 │   │
├─ Enc 3  DoubleConv(32 -> 64)     64 @  56 x  56 x 32  ─────── skip 3 ─┐    │   │
│  MaxPool 2                                                            │    │   │
├─ Enc 4  DoubleConv(64 -> 128)   128 @  28 x  28 x 16  ── skip 4 ─┐    │    │   │
│  MaxPool 2                                                       │    │    │   │
└─ Bottleneck DoubleConv(128 -> 256)  256 @ 14 x 14 x 8            │    │    │   │
                                                                   │    │    │   │
   Up 1: ConvT 256->128, AttentionGate(skip 4, g) , concat, DoubleConv(256 -> 128)
   Up 2: ConvT 128->64,  AttentionGate(skip 3, g) , concat, DoubleConv(128 -> 64)
   Up 3: ConvT 64->32,   AttentionGate(skip 2, g) , concat, DoubleConv(64 -> 32)
   Up 4: ConvT 32->16,   AttentionGate(skip 1, g) , concat, DoubleConv(32 -> 16)
   Head: Conv 1x1x1 (16 -> 1) -> sigmoid -> threshold 0.5
```

- **DoubleConv** = (Conv3d 3×3×3, no bias → BatchNorm3d → ReLU) × 2.
- **Downsampling**: max-pool 2×2×2 (4 times). **Upsampling**: transposed conv 2×2×2, stride 2.
- **Attention gate** (`AttentionGate3D`, the Review-1 gate used in the ablation):
  `θ = LN(Conv1×1(x))`, `φ = LN(Conv1×1(g))`, `f = ReLU(θ + φ)`, `α = σ(Conv1×1(f))`, `out = x ⊙ α`, with
  `F_int = C/2`. Here `x` is the encoder skip feature and `g` is the decoder feature (the gating signal). The gate
  learns a 0–1 voxel map that suppresses skip features outside the pancreas region. LN is a parameter-free
  LayerNorm over (C, H, W, D). The paper's fuller gate (§3.2.3: projection to C, a C/2 reduction layer, and a final
  1×1×1 conv + LN) exists as `--ag paper` (deviation D2).
- **Output**: one channel with a sigmoid (binary pancreas vs background).
- **Parameters**: **5,668,269** (R0; `--ag paper` 5,722,669).

### 1.2 Training recipe (paper + our documented interpretations)

| Item | Value |
|---|---|
| Pre-processing | resample to 1 × 1 × 1 mm, clip HU [-100, 240], scale to [0, 1] |
| Input | one 224 × 224 × 128 crop per step, centred on the volume (paper: "centred on the abdominal region") |
| Batch | 1 (T4 16 GB) |
| Loss | Dice + Focal (γ = 2, α = None, D1) on a single sigmoid output |
| Optimizer | Adam, lr 6 × 10⁻⁴ (the paper prints "6 × 10⁴", a missing minus sign) |
| Schedule | ReduceLROnPlateau, patience 100 epochs, factor 0.5 |
| Augmentation | rotation ±10°, flips p 0.5, shifts up to 10 % |
| Inference | the same single centre crop, thresholded at 0.5 |

### 1.3 What went wrong in Review-1 (test Dice 0.192 → 0.283 → 0.316)

The Review-1 runs (`docs/REPORT.md`, `docs/log.md`) are kept as historical; their numbers are not valid
measurements of the architecture. The causes, as diagnosed in `CHANGELOG.md` (T1–T3):

1. **Misaligned data.** The Review-1 pre-processed arrays did not match the labels. The overlay figure placed the
   ground truth of test patients 0073 and 0030 on heart and lung slices, and the model learned that wrong position.
   No architecture can fix this. It is the main reason for the 0.3 ceiling.
2. **Label leakage.** Runs 2 and 3 cropped around the *ground-truth* centroid, which is unavailable at test time, so
   their gain over run 1 was partly a position prior.
3. **Severe over-segmentation.** Precision was about 0.22 and the median predicted/GT volume was 2.4×. A single 1 mm
   crop and a sigmoid at threshold 0.5 gave the network no way to separate the pancreas from neighbouring
   bowel/duodenum of similar HU.
4. **Unstable training.** BatchNorm with batch size 1 estimates statistics from one sample per step. Validation Dice
   peaked (about 0.3, epochs 40–80) and then fell to about 0.13: overfitting, not a plateau.
5. **Weak model for the task.** The network has 5.7 M parameters, learns from scratch on one crop per step, and sees
   far fewer distinct training patches than a patch-based sampler would give it.

### 1.4 The intermediate step: R0–R4 ablation (branch `t1-t10-replication-pipeline`, T1–T10)

Before switching architecture, the pipeline was rebuilt so the paper model could be tested fairly:

- **T1** alignment gate on all 80 cases: identity orientation rule, max air fraction 0.0007, 0 shape mismatches.
- **T2** full-volume pre-processing with no GT crop.
- **T3** full-volume metrics.

Three modifications to the paper network were added as flags:

| Run | Change | What it targets |
|---|---|---|
| R0 | paper-faithful (above) | baseline |
| R1 (+M1) | 2 × 128×128×96 patches, pos:neg 1:2 sampling, sliding-window inference | more, foreground-biased training samples; whole-volume inference |
| R2 (+M2) | GroupNorm(8) instead of BatchNorm | batch-size-independent normalisation |
| R3 (+M3) | deep supervision (aux heads at 1/2, 1/4, 1/8, weights 1, ½, ¼, ⅛) | gradient to the deep decoder levels |
| R4 | R3 without attention gates | does attention help? |

The R0–R4 code is complete and tested (`python -m src.test_all`, 18/18). Their full training runs are not part of
the results reported here: the step budget N was never fixed (T5) and the sanity check (T6) is pending. **Every
improvement in section 3 is measured against the paper's published number, not against our own R0.**

---

## 2. The architecture now: nnU-Net v2 ResEnc-M (B1)

Code: `scripts/nnunet_run.sh`, `tools/nnunet_prepare.py`, `src/nnunet_trainer.py`, `tools/nnunet_eval.py`.
References:
- Isensee et al., *nnU-Net*, Nature Methods 2021.
- Isensee et al., *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation*, MICCAI 2024.
  It introduced the residual-encoder presets (ResEnc M/L/XL) and showed that they beat the original nnU-Net and
  recent transformer/Mamba models on several benchmarks.

nnU-Net is not a single network. It is a *self-configuring pipeline*: it reads a "fingerprint" of the dataset
(spacings, shapes, intensities) and derives the pre-processing, patch size, network topology and training schedule
from it. "ResEnc-M" is the planner preset that builds a residual-encoder U-Net sized for about 9–11 GB of VRAM.
Our run used 9.9 GB.

### 2.1 Plan derived for this dataset (`nnUNetResEncUNetMPlans`, 3d_fullres)

These values were derived from our 64 train+val cases and logged in the training log:

| Item | Value |
|---|---|
| Target spacing (z, y, x) | 1.0 × 0.879 × 0.879 mm (the dataset median; in-plane is **finer** than the paper's 1 mm) |
| Median image size | 218 × 512 × 512 voxels |
| Patch size (z, y, x) | **80 × 192 × 160** |
| Batch size | 2 |
| Normalisation | CTNormalization: clip to the foreground 0.5–99.5 percentiles (**-71 to 171 HU**), then z-score with the foreground mean 84.3 HU and SD 40.4 HU |
| Resampling | image: 3rd-order spline in-plane; label: linear on one-hot |
| Network | `ResidualEncoderUNet`, 6 stages |
| Batch Dice | yes (Dice computed over the whole batch as one volume) |

### 2.2 Network

```
input 1 x 80 x 192 x 160  (patch, z-scored HU)
│
├─ Stem     Conv 3x3x3 (1 -> 32) + IN + LeakyReLU
├─ Stage 0   1 residual block,   32 @ 80 x 192 x 160  ─────────────────── skip 0 ─┐
├─ Stage 1   3 residual blocks,  64 @ 40 x  96 x  80  (stride 2,2,2) ─── skip 1 ─┐│
├─ Stage 2   4 residual blocks, 128 @ 20 x  48 x  40  (stride 2,2,2) ── skip 2 ─┐││
├─ Stage 3   6 residual blocks, 256 @ 10 x  24 x  20  (stride 2,2,2) ─ skip 3 ─┐│││
├─ Stage 4   6 residual blocks, 320 @  5 x  12 x  10  (stride 2,2,2)  skip 4 ─┐││││
└─ Stage 5   6 residual blocks, 320 @  5 x   6 x   5  (stride 1,2,2)  bottleneck
                                                                            │││││
   Dec 4: ConvT (1,2,2) 320->320, concat skip 4, ConvBlock(640 -> 320) ─ head -> 2 @ 1/16  (DS, weight 0)
   Dec 3: ConvT  2      320->256, concat skip 3, ConvBlock(512 -> 256) ─ head -> 2 @ 1/8   (DS, 0.125)
   Dec 2: ConvT  2      256->128, concat skip 2, ConvBlock(256 -> 128) ─ head -> 2 @ 1/4   (DS, 0.25)
   Dec 1: ConvT  2      128->64,  concat skip 1, ConvBlock(128 -> 64)  ─ head -> 2 @ 1/2   (DS, 0.5)
   Dec 0: ConvT  2       64->32,  concat skip 0, ConvBlock(64 -> 32)   ─ head -> 2 @ full  (main, 1.0)
   Output: 2-class softmax (background, pancreas) -> argmax
```

- **Residual block** (`BasicBlockD`): Conv 3×3×3 → InstanceNorm → LeakyReLU → Conv 3×3×3 → InstanceNorm, plus an
  identity shortcut, then LeakyReLU. In the first block of a strided stage, the first conv does the downsampling,
  and the shortcut is AvgPool + 1×1×1 conv (the "ResNet-D" shortcut).
- **Encoder depth**: 1-3-4-6-6-6 residual blocks, 26 blocks / 52 conv layers in total. The paper's encoder has
  2 conv layers per level, 10 in total. Most capacity sits at low resolution, where the network sees the most
  context.
- **Anisotropic last stage**: the deepest stride is (1, 2, 2). z is already 5 voxels at stage 4, and halving it again
  would go below nnU-Net's minimum feature-map size of 4, so the planner stops pooling z while still pooling
  in-plane.
- **Decoder**: one conv block (Conv 3×3×3 + IN + LeakyReLU) per level. The decoder is deliberately light, and the
  capacity is in the encoder.
- **Normalisation**: InstanceNorm (affine) — per sample, per channel, so it is independent of the batch size.
- **Deep supervision**: a 1×1×1 segmentation head on every decoder level, each compared with a downsampled label.
  The weights are ½ⁱ, normalised; the lowest (1/16) level gets weight 0.
- **No attention gates.** The paper's central component is gone. The decoder relies on plain skip
  concatenation; the larger receptive field and the deeper encoder supply the context the gates were meant to
  provide.
- **Parameters**: **101,940,170** (about 18× the paper model).

### 2.3 Training

| Item | Value |
|---|---|
| Split | fold 0 of `splits/patient_splits.json`: 51 train / 13 validation; the 16 test IDs only in `imagesTs`, never read during training |
| Sampling | random 80×192×160 patches; one of the two patches per batch is forced to contain pancreas (nnU-Net's 33 % oversampling rounds to 1 of 2 at batch 2) |
| Loss | Dice (batch Dice) + cross-entropy on 2-class softmax, summed over deep-supervision levels |
| Optimizer | SGD, Nesterov, momentum 0.99, weight decay 3 × 10⁻⁵, initial lr 0.01 |
| Schedule | poly: `lr = 0.01 · (1 − epoch/E)^0.9`, decaying to about 0 at the last epoch (8 × 10⁻⁵ logged at epoch 213) |
| Iterations | 250 per epoch × **214 epochs** = 53,500 steps (107,000 patches) |
| Augmentation (nnU-Net v2 defaults) | rotation ±30° and scaling 0.7–1.4 (p 0.2 each), Gaussian noise, Gaussian blur, brightness, contrast, simulated low resolution, gamma (incl. inverted), mirroring on all 3 axes |
| Precision | AMP (mixed precision), torch.compile off (triton cache fails on the NFS home) |
| Hardware | one RTX 6000 Ada 48 GB, 9.9 GB used, 73–77 °C, no throttling |

**Time budget (`src/nnunet_trainer.py`, our addition).** Default nnU-Net trains for 1000 epochs, about 17 h here.
The requirement was "finish training in 240 min". `nnUNetTrainerBudget` re-plans the epoch count after every epoch:

```
t_epoch    = median start-to-start epoch time (last 20 epochs)
left       = budget − wall-clock used
num_epochs = min(1000, done + left // t_epoch)
lr_scheduler.max_steps = num_epochs
```

The poly schedule is stretched or compressed to the new end point, so the LR still reaches about 0 at the final
epoch. Annealing is a big part of nnU-Net's final accuracy, and simply stopping epoch 1000-of-1000 training after
214 epochs would freeze a high LR. If epochs slow down, the run ends earlier instead of overrunning. An
exhausted-budget bug (one extra epoch was always granted) was caught by the built-in self-check
(`python -m src.nnunet_trainer`) before the real run. Result: 214 epochs in 221.4 of 222 budgeted minutes.

### 2.4 Inference and post-processing

1. **Sliding window** over the full CT volume. Window = training patch, 50 % overlap, Gaussian importance
   weighting, so voxels near a window's centre count more than voxels near its edge.
2. **Test-time augmentation**: predictions averaged over all 8 mirror combinations of the 3 axes.
3. **Post-processing chosen on validation only**: nnU-Net's `determine_postprocessing` compared "keep the largest
   connected component" against no post-processing on the 13 validation cases. It chose largest-component
   (0.8426 → 0.8426), and that choice was then applied unchanged to the 16 test cases.
4. **Evaluation** with our own `src/metrics.py` (the same code as for R0–R4) at each case's native spacing, on the
   full volume.

---

## 3. Side-by-side and why the score improved

| Aspect | Before (paper 3D Attention U-Net) | Now (nnU-Net ResEnc-M) |
|---|---|---|
| Data | Review-1 arrays (misaligned) | rebuilt from TCIA DICOM + TCIA labels, gated by the all-80 T1 check |
| Spacing | 1 × 1 × 1 mm | 1.0 × 0.879 × 0.879 mm (dataset median) |
| Intensity | clip [-100, 240] HU → [0, 1] | clip [-71, 171] HU (foreground percentiles) → z-score |
| Input | one 224×224×128 centre crop | random 80×192×160 patches, pancreas-oversampled |
| Batch / norm | 1 / BatchNorm | 2 / InstanceNorm |
| Encoder | 5 levels, 2 convs each | 6 stages, 1-3-4-6-6-6 residual blocks |
| Skips | attention-gated | plain concatenation |
| Parameters | 5.67 M | 101.9 M |
| Output / loss | 1 sigmoid, Dice + Focal | 2-class softmax, Dice + CE, deep supervision ×5 |
| Optimizer / LR | Adam 6e-4, ReduceLROnPlateau | SGD Nesterov 0.01, poly decay to 0 (budget-compressed) |
| Augmentation | rot ±10°, flips, shifts | rot ±30°, scale, noise, blur, intensity, low-res, gamma, mirroring |
| Inference | single crop, threshold 0.5 | Gaussian sliding window, 8× mirror TTA |
| Post-processing | none | largest component (chosen on validation) |
| Test Dice | 0.808 (paper) / 0.316 (Review-1) | **0.859** |

The changes were made together, in one run, so **we cannot attribute the gain to individual factors**: there is no
per-factor ablation for B1. The reasons below are the expected mechanisms, each backed by what we observed.

1. **Correct data first.** Rebuilding the dataset from DICOM with the alignment gate removed the error that capped
   Review-1 at about 0.3. This is a fix, not an improvement. `data/Processed_data` on the GPU box is the misaligned
   Review-1 conversion (air fraction 0.054 on case 0001) and was not used.
2. **Finer resolution and whole-volume coverage.** The pancreas is thin, often 1–2 cm across, with a
   low-contrast border. Keeping in-plane resolution at 0.879 mm instead of 1 mm keeps more boundary voxels.
   Sliding-window inference covers the whole scan instead of hoping the pancreas lies in a fixed centre crop.
3. **Patch sampling with foreground oversampling.** At least half of all training patches contain pancreas, at random
   positions, so the network gets 107,000 varied, informative samples. The old pipeline saw one centred crop per
   step, mostly background.
4. **Capacity where it matters.** The residual encoder has 18× the parameters, concentrated at 1/8–1/32
   resolution, so its receptive field covers the pancreas together with its surroundings (duodenum, splenic vein,
   stomach). Telling the pancreas from bowel of similar HU needs that context. Residual shortcuts keep a 26-block
   encoder trainable from scratch. This targets Review-1's over-segmentation directly: our precision is now 0.884
   (Review-1 about 0.22), and the volume ratio is 0.951 instead of 2.4×.
5. **Batch-independent normalisation.** InstanceNorm does not depend on the batch size, unlike BatchNorm at batch 1.
   This removes a source of train/test mismatch and noisy updates.
6. **Better-posed loss.** Batch Dice handles the extreme class imbalance (the pancreas is well under 1 % of the
   volume). CE adds stable per-voxel gradients. Deep supervision at 5 scales trains the deep decoder levels
   directly. A 2-class softmax with argmax avoids tuning a sigmoid threshold.
7. **Schedule that converges.** High-momentum SGD with a poly decay to about 0 ends training on a low-LR annealing
   phase. Our budget trainer keeps that annealing inside 214 epochs. Pseudo-Dice (EMA) rose steadily to 0.8458
   without the peak-then-collapse seen in Review-1 (`results/nnunet/figures/fig4_training_progress.png`).
8. **Stronger, dataset-tuned augmentation and intensity handling.** Wider rotations, scaling and intensity
   transforms reduce overfitting with only 51 training patients. The z-score window is set from the dataset's own
   pancreas intensities, not a fixed HU range.
9. **Ensembled inference.** Gaussian-weighted overlapping windows plus 8-way mirror TTA average 8+ predictions per
   voxel, which smooths boundary noise. Largest-component post-processing removes stray islands; it was selected
   on validation, so it does not touch the test set.

### What the attention gates' removal means for the paper's claim

The paper reports that attention improves a plain 3D U-Net from 77.6 % to 80.8 %. B1 does not test that claim,
because it changes everything at once; R3 vs R4 in the ablation is the experiment designed for it. B1 does show
that a well-configured CNN *without* attention exceeds the paper's attention model on this split. This is
consistent with the nnU-Net Revisited finding that the training pipeline (pre-processing, sampling, schedule,
augmentation, inference) matters more than architectural add-ons.

---

## 4. Results of the new model

From `results/nnunet/nnUNetTrainerBudget/test_summary.json`; the full tables and figures are in
`results/nnunet/results.md`.

| Metric (16 test patients) | Mean ± SD | Median | Min | Max |
|---|---|---|---|---|
| Dice | 0.859 ± 0.044 | 0.868 | 0.711 | 0.902 |
| Jaccard | 0.755 ± 0.062 | 0.766 | 0.551 | 0.821 |
| Precision | 0.884 ± 0.024 | 0.881 | 0.834 | 0.931 |
| Recall | 0.840 ± 0.074 | 0.859 | 0.594 | 0.911 |
| HD95 (mm) | 4.46 ± 2.42 | 4.03 | 2.16 | 12.73 |
| ASSD (mm) | 1.10 ± 0.32 | 1.10 | 0.67 | 2.06 |
| V_pred / V_GT | 0.951 ± 0.090 | 0.976 | 0.672 | 1.026 |

- Validation (13 cases, nnU-Net full volume): Dice 0.8426.
- On test, 15 of 16 patients reach Dice ≥ 0.80, with 0 empty predictions.
- The remaining error is now **under**-segmentation, the opposite of Review-1: recall 0.840 < precision 0.884. The
  worst case, PANCREAS_0033 (Dice 0.711), misses 40 % of the gland (40.4 vs 60.0 mL).

**Caveats.** Single fold, single seed, n = 16. The margin over the paper (+0.051 Dice) holds for this split and is
not a cross-validated estimate. The paper used 5-fold CV on its own split.

## 5. Reproduce

```
sh scripts/nnunet_run.sh check && sh scripts/nnunet_run.sh prepare
NNUNET_BUDGET_MIN=222 sh scripts/nnunet_run.sh train && sh scripts/nnunet_run.sh test
python tools/nnunet_figures.py
```

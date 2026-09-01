# Visual Results — 3D Attention U-Net Pancreas Segmentation

> This document explains the five figures in the `results/` folder. They summarize the final
> reproduction run (run 3: pancreas-centroid crop, soft-Dice loss, 300 epochs/fold) on the
> NIH Pancreas-CT dataset (80 patients; 16 held-out test).
>
> Key numbers: **test DSC 0.316 ± 0.078 · ASSD 8.62 mm · HD95 32.36 mm · 5-fold val 0.308 ± 0.025**

---

## 1. Segmentation Overlays — `1_segmentation_overlays.png`

![overlays](results/1_segmentation_overlays.png)

Three rows, one per held-out test patient chosen by DSC: **best**, **median**, **worst**. Each row
shows the same axial slice (at the pancreas's largest extent) three ways:

- **CT** — the raw preprocessed image (224×224, [0,1] normalized window).
- **GT overlay** — ground-truth manual segmentation (red).
- **Prediction overlay** — model output at threshold 0.5 (blue).

Reading the figure:
- **Best patient** — high overlap; blue closely tracks red.
- **Median patient** — typical case: the model covers much of the pancreas (decent recall) but also
  predicts beyond the boundary (over-segmentation).
- **Worst patient** — blue and red overlap only partially; this is the low-DSC tail of the
  distribution and is driven by the same over-segmentation behavior plus low contrast.

This figure is the qualitative companion to the numbers in `3_test_dice_per_patient.png`.

---

## 2. Training Curves — `2_training_curves.png`

![curves](results/2_training_curves.png)

Left: **train loss** per epoch. Right: **validation Dice** per epoch, all five folds overlaid.

Reading the figure:
- Train loss falls smoothly from ~1.0 to ~0.4 — the model is learning.
- Validation Dice climbs quickly in the first ~60 epochs, then plateaus with high epoch-to-epoch
  variance (the validation set is only ~12 patients/fold, so it is noisy).
- All folds converge to the same plateau (0.26–0.33), i.e. consistent behavior across splits.

The plateau is the "capacity ceiling" discussed in the report — not a training-instability artifact.

---

## 3. Test DSC per Patient — `3_test_dice_per_patient.png`

![per-patient](results/3_test_dice_per_patient.png)

Bar chart of DSC for all 16 held-out patients, sorted and color-coded (green = high, red = low),
with the **mean (0.316)** marked.

Reading the figure:
- Scores span roughly 0.13–0.45; most patients sit in the 0.25–0.38 band.
- No patient reaches strong segmentation (≥0.7), consistent with the over-segmentation diagnosis.
- This is the honest, patient-level view behind the aggregate test DSC.

---

## 4. Run Comparison — `4_run_comparison.png`

![comparison](results/4_run_comparison.png)

Side-by-side test-set metrics for the three reproduction runs:

| Metric | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| DSC | 0.192 | 0.283 | **0.316** |
| ASSD (mm) | 13.67 | 9.17 | **8.62** |
| HD95 (mm) | 42.46 | 34.55 | **32.36** |

Reading the figure:
- **Run 1 → 2** (+0.09 DSC): pancreas-centroid crop fixed the blind crop that was cutting off the
  organ in 28/80 cases.
- **Run 2 → 3** (+0.03 DSC): soft-Dice loss (ablation winner) tightened predictions and cut
  surface-distance errors ~6%.
- Cumulative: test DSC nearly doubled, and boundary errors (ASSD/HD95) dropped ~35–40%.

---

## 5. Over-Segmentation Diagnostic — `5_over_segmentation.png`

![over-seg](results/5_over_segmentation.png)

Scatter of **predicted vs ground-truth foreground volume** for the 16 test patients. Each point is
one patient, colored by its DSC. The dashed identity line marks perfect volume agreement.

Reading the figure:
- Every point sits **above** the identity line → the model always predicts *more* foreground than
  the true pancreas.
- **Median ratio ≈ 2.4×** — the model over-segments by ~2.4× in volume.
- Low-DSC patients (red) are the most over-segmented; high-DSC ones are closer to the line.

This single figure explains the whole result: the model localizes the pancreas well (recall is
high) but lacks boundary precision (precision ~0.22). This is why threshold tuning, post-processing,
and loss swaps only moved DSC to ~0.32 — the over-segmentation is intrinsic to a from-scratch
batch-1 3D U-Net on 64 training patients. See `REPRODUCTION_REPORT.md` §6 for the full diagnostic.

---

## Summary

| Figure | Purpose |
|---|---|
| `1_segmentation_overlays.png` | Qualitative: best / median / worst segmentation examples |
| `2_training_curves.png` | Convergence and per-fold consistency |
| `3_test_dice_per_patient.png` | Distribution of per-patient test DSC |
| `4_run_comparison.png` | Improvement across the three reproduction runs |
| `5_over_segmentation.png` | Root-cause diagnostic (over-segmentation) |
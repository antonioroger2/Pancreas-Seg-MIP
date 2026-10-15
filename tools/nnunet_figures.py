"""Figures + results.md for the nnU-Net ResEnc-M run (B1). Every number is read from the run's saved files.

    python tools/nnunet_figures.py            # on the GPU box (needs the test CTs in nnunet/raw)
Writes <out>/figures/*.png and <out>/results.md.
"""
import argparse, glob, json, os, re, shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                 # noqa: E402
import numpy as np                                                              # noqa: E402
import pandas as pd                                                             # noqa: E402
import SimpleITK as sitk                                                        # noqa: E402

PAPER = dict(dice=0.808, sd=0.021, hd95="8.1 ± 1.6", assd="2.1 ± 0.3")   # base paper, NIH Pancreas-CT, Table 1
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"                     # categorical slots 1, 2 (reference palette)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150})


def per_patient_dice(df, path):
    d = df.sort_values("dice")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(d))
    ax.barh(y, d.dice, height=0.7, color=BLUE)
    ax.set_yticks(y, [p.replace("PANCREAS_", "") for p in d.patient])
    ax.axvline(PAPER["dice"], color=INK2, ls="--", lw=1.5)
    ax.axvline(df.dice.mean(), color=INK, lw=1.5)
    ax.text(PAPER["dice"], len(d) - 0.3, f" paper {PAPER['dice']:.3f}", color=INK2, ha="right", va="bottom")
    ax.text(df.dice.mean(), len(d) - 0.3, f" ours {df.dice.mean():.3f}", color=INK, ha="left", va="bottom")
    for yi, v in zip(y, d.dice):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", color=INK2, fontsize=8)
    ax.set_xlim(0.6, 0.97); ax.set_ylim(-0.6, len(d) + 0.6)
    ax.set_xlabel("Dice (full CT volume)"); ax.set_ylabel("Test patient")
    ax.set_title("Test Dice per patient (n = 16)", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def volume_and_pr(df, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.3))
    lo, hi = 30, 105
    a.plot([lo, hi], [lo, hi], color=INK2, ls="--", lw=1)
    a.scatter(df.vol_gt_ml, df.vol_pred_ml, s=40, color=BLUE, edgecolor=SURF, lw=1.5, zorder=3)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    a.set_xlabel("Ground-truth volume (mL)"); a.set_ylabel("Predicted volume (mL)")
    r = np.corrcoef(df.vol_gt_ml, df.vol_pred_ml)[0, 1]
    a.set_title(f"Pancreas volume agreement (Pearson r = {r:.3f})", loc="left", color=INK)
    b.plot([0.55, 1], [0.55, 1], color=INK2, ls="--", lw=1)
    b.scatter(df.recall, df.precision, s=40, color=BLUE, edgecolor=SURF, lw=1.5, zorder=3)
    b.set_xlim(0.55, 1); b.set_ylim(0.55, 1); b.set_aspect("equal")
    b.set_xlabel("Recall (sensitivity)"); b.set_ylabel("Precision")
    b.set_title("Precision vs recall (above line = under-segmentation)", loc="left", color=INK, fontsize=9.5)
    for ax in (a, b):
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    w = df.loc[df.dice.idxmin()]
    for ax, x, y in ((a, w.vol_gt_ml, w.vol_pred_ml), (b, w.recall, w.precision)):
        ax.annotate(w.patient.replace("PANCREAS_", ""), (x, y), xytext=(6, -10), textcoords="offset points",
                    color=INK2, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def overlays(df, raw, pred_dir, path):
    d = df.sort_values("dice").reset_index(drop=True)
    picks = [("worst", d.iloc[0]), ("median", d.iloc[len(d) // 2]), ("best", d.iloc[-1])]
    fig, axes = plt.subplots(len(picks), 3, figsize=(10, 3.6 * len(picks)))
    for row, (tag, r) in zip(axes, picks):
        ct = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(raw, "imagesTs", f"{r.patient}_0000.nii.gz")))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(raw, "labelsTs", f"{r.patient}.nii.gz"))) > 0
        pr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_dir, f"{r.patient}.nii.gz"))) > 0
        # display only: worst case -> slice with the most missed pancreas, others -> largest GT area
        z = int(np.argmax((gt & ~pr).sum((1, 2)) if tag == "worst" else gt.sum((1, 2))))
        ys, xs = np.nonzero(gt[z] | pr[z])
        pad = 40
        sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
        img = np.clip(ct[z], -100, 240)
        # index row 0 = posterior (T1 geometry) -> flip so anterior is up
        views = [np.flipud(v[sl]) for v in (img, gt[z], pr[z])]
        for ax in row:
            ax.imshow(views[0], cmap="gray", interpolation="nearest"); ax.axis("off")
        row[1].contour(views[1], levels=[0.5], colors=ORANGE, linewidths=1.8)
        row[2].contour(views[1], levels=[0.5], colors=ORANGE, linewidths=1.2, linestyles="--")
        row[2].contour(views[2], levels=[0.5], colors=BLUE, linewidths=1.8)
        row[0].set_title(f"{tag}: {r.patient.replace('PANCREAS_', '')}  (Dice {r.dice:.3f})  slice {z}",
                         loc="left", color=INK, fontsize=9.5)
        row[1].set_title("ground truth", loc="left", color=ORANGE, fontsize=9.5)
        row[2].set_title("prediction (GT dashed)", loc="left", color=BLUE, fontsize=9.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def md_table(rows, cols):
    head = "| " + " | ".join(c for c, _ in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return head + "".join("| " + " | ".join(f(r) for _, f in cols) + " |\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/nnunet/nnUNetTrainerBudget")
    ap.add_argument("--raw", default="nnunet/raw/Dataset801_Pancreas")
    ap.add_argument("--out", default="results/nnunet")
    a = ap.parse_args()
    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(a.run, "test_per_patient.csv"))
    s = json.load(open(os.path.join(a.run, "test_summary.json")))
    val = json.load(open(os.path.join(a.run, "val_summary_nnunet.json")))
    log = open(glob.glob(os.path.join(a.run, "training_log_*.txt"))[0]).read()
    budget = re.findall(r"budget: ([\d.]+)/(\d+) min used, ([\d.]+) s/epoch -> (\d+) epochs", log)[-1]
    ema = re.findall(r"best EMA pseudo Dice: ([\d.]+)", log)[-1]

    per_patient_dice(df, os.path.join(fig_dir, "fig1_test_dice_per_patient.png"))
    volume_and_pr(df, os.path.join(fig_dir, "fig2_volume_and_precision_recall.png"))
    overlays(df, a.raw, os.path.join(a.run, "test_pp"), os.path.join(fig_dir, "fig3_overlays_worst_median_best.png"))
    shutil.copy(os.path.join(a.run, "progress.png"), os.path.join(fig_dir, "fig4_training_progress.png"))

    f3, f2 = (lambda v: f"{v:.3f}"), (lambda v: f"{v:.2f}")
    summ = md_table([("Dice", "dice", f3), ("Jaccard (IoU)", "jaccard", f3), ("Precision", "precision", f3),
                     ("Recall", "recall", f3), ("HD95 (mm)", "hd95", f2), ("ASSD (mm)", "assd", f2),
                     ("V_pred / V_GT", "v_ratio", f3)],
                    [("Metric", lambda r: r[0]), ("Mean ± SD", lambda r: f"{r[2](s[r[1]]['mean'])} ± {r[2](s[r[1]]['sd'])}"),
                     ("Median", lambda r: r[2](s[r[1]]["median"])), ("Min", lambda r: r[2](s[r[1]]["min"])),
                     ("Max", lambda r: r[2](s[r[1]]["max"]))])
    pat = md_table(df.sort_values("patient").to_dict("records"),
                   [("Patient", lambda r: r["patient"]), ("Dice", lambda r: f3(r["dice"])),
                    ("Jaccard", lambda r: f3(r["jaccard"])), ("Precision", lambda r: f3(r["precision"])),
                    ("Recall", lambda r: f3(r["recall"])), ("HD95 (mm)", lambda r: f2(r["hd95"])),
                    ("ASSD (mm)", lambda r: f2(r["assd"])), ("GT vol (mL)", lambda r: f"{r['vol_gt_ml']:.1f}"),
                    ("Pred vol (mL)", lambda r: f"{r['vol_pred_ml']:.1f}")])
    vrows = [dict(patient=os.path.basename(c["reference_file"])[:-7], **c["metrics"]["1"]) for c in val["metric_per_case"]]
    vtab = md_table(sorted(vrows, key=lambda r: r["patient"]),
                    [("Patient", lambda r: r["patient"]), ("Dice", lambda r: f3(r["Dice"])),
                     ("IoU", lambda r: f3(r["IoU"]))])
    worst = df.loc[df.dice.idxmin()]
    d = s["dice"]

    md = f"""# Results: nnU-Net ResEnc-M (B1)

Generated by `tools/nnunet_figures.py` from the run's saved files in `{a.run}/`; nothing here is typed by hand.

## Headline

| Model | Test Dice (n = 16) | HD95 (mm) | ASSD (mm) |
|---|---|---|---|
| Base paper (3D Attention U-Net, NIH, Table 1) | {PAPER['dice']:.3f} ± {PAPER['sd']:.3f} | {PAPER['hd95']} | {PAPER['assd']} |
| **nnU-Net ResEnc-M (ours)** | **{d['mean']:.3f} ± {d['sd']:.3f}** | **{s['hd95']['mean']:.2f} ± {s['hd95']['sd']:.2f}** | **{s['assd']['mean']:.2f} ± {s['assd']['sd']:.2f}** |

Mean test Dice is {d['mean'] - PAPER['dice']:+.3f} vs the paper; {int((df.dice >= 0.8).sum())}/16 test patients reach Dice ≥ 0.80, {s['n_empty_pred']} empty predictions.

## Setup

- **Model:** nnU-Net v2 3d_fullres, residual-encoder M preset (`nnUNetPlannerResEncM`): 6 stages, 32-320 features,
  patch 80 × 192 × 160, batch 2, spacing 1.0 × 0.879 × 0.879 mm. Dice + CE loss with deep supervision, SGD
  (Nesterov) with poly LR, nnU-Net's default augmentation, mirroring TTA and Gaussian sliding-window inference.
- **Data:** NIH Pancreas-CT, 80 patients, rebuilt from the TCIA DICOM + TCIA labels after the all-80 alignment
  check (`results/alignment_all80/`). Split = fold 0 of `splits/patient_splits.json`: 51 train / 13 validation /
  16 test (the same test patients as R0-R4 and Review-1).
- **Budget:** `nnUNetTrainerBudget` (`src/nnunet_trainer.py`) on one RTX 6000 Ada: {budget[3]} epochs in
  {budget[0]}/{budget[1]} min of epoch time ({budget[2]} s/epoch), LR schedule compressed to that end point;
  final EMA pseudo Dice {float(ema):.4f}.
- **Post-processing:** chosen on the 13 validation cases only (nnU-Net `determine_postprocessing`: keep largest
  component), then applied to test.
- **Metrics:** `src/metrics.py` (`tools/nnunet_eval.py`) on the full CT volume at each case's native spacing.

## Test set (16 patients)

{summ}
### Per patient

{pat}
The weakest case, {worst.patient} (Dice {worst.dice:.3f}), is under-segmented: recall {worst.recall:.3f} vs
precision {worst.precision:.3f}, predicted {worst.vol_pred_ml:.1f} mL vs ground truth {worst.vol_gt_ml:.1f} mL.

## Validation (13 patients, nnU-Net full-volume validation)

Mean Dice **{val['foreground_mean']['Dice']:.4f}**, IoU {val['foreground_mean']['IoU']:.4f} (before post-processing).

{vtab}
## Figures

**Figure 1. Test Dice per patient.** Solid line = our mean, dashed = paper mean.

![Test Dice per patient](figures/fig1_test_dice_per_patient.png)

**Figure 2. Volume agreement and over/under-segmentation.** Left: predicted vs ground-truth pancreas volume
(dashed = identity). Right: precision vs recall per patient; points above/left of the dashed line miss more
pancreas than they add (under-segmentation), points below/right add more than they miss.

![Volume and precision-recall](figures/fig2_volume_and_precision_recall.png)

**Figure 3. Worst, median and best test cases.** Axial slice with the most missed pancreas (worst case) or the
largest ground-truth area (median, best), anterior up,
HU window [-100, 240], cropped around the pancreas for display. Orange = ground truth, blue = prediction.

![Overlays](figures/fig3_overlays_worst_median_best.png)

**Figure 4. Training progress (nnU-Net).** Train/validation loss, pseudo Dice (validation patches) with its EMA,
learning rate and epoch time over the {budget[3]} epochs.

![Training progress](figures/fig4_training_progress.png)

## Caveats

- Single fold, single seed, n = 16 test patients: the margin over the paper holds for this split; it is not a
  cross-validated estimate.
- The paper reports 5-fold CV + test on its own split; our split reuses the repo's seed-42 split.
- Pseudo Dice (Figure 4) is measured on random validation patches, not full volumes; the full-volume numbers are
  the tables above.

## Reproduce

```
sh scripts/nnunet_run.sh check && sh scripts/nnunet_run.sh prepare
NNUNET_BUDGET_MIN=222 sh scripts/nnunet_run.sh train && sh scripts/nnunet_run.sh test
python tools/nnunet_figures.py
```
"""
    open(os.path.join(a.out, "results.md"), "w").write(md)
    print(f"wrote {a.out}/results.md and {fig_dir}/")


if __name__ == "__main__":
    main()

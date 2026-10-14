"""
clinical.py - T8 clinical extensions (inference only, from saved checkpoints/outputs).

C1 volumetry (every run with test_per_patient.csv; main non-TTA prediction): pancreas volume mL = voxels x 1 mm^3 / 1000,
   Pearson r, MAE (mL), mean absolute % error, bias and 95 % limits of agreement (Bland-Altman), scatter.
C2 reliability (one run, default R3_final): 4-flip TTA -> voxel entropy -> case score = mean entropy inside the mask.
   Chosen on the VALIDATION fold only: flag threshold = 75th percentile of validation scores; "poor case" = Dice below the
   25th percentile of validation TTA Dice. Reported on TEST: Spearman rho(score, Dice), flag precision/recall, and
   whether TTA changes Dice vs the single prediction (paired Wilcoxon). n = 16 -> suggestive, not proof.
   NIH Pancreas-CT = healthy pancreas: measurement accuracy only, no disease claim.
Case cards (best / median / worst test case of the C2 run): 3 axial slices Input | GT | TTA prediction | uncertainty.
  python -m src.clinical --results results --data /tmp/prep
"""
import argparse, glob, json, os

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import load_case
from src.evaluate import load_run
from src.metrics import overlap_metrics
from src.tta import tta_predict

NOTE = "NIH Pancreas-CT (healthy pancreas): volume measurement accuracy only, no disease claim; n = 16 test cases."


def volume_agreement(vp, vg):
    vp, vg = np.asarray(vp, float), np.asarray(vg, float); d = vp - vg
    sd = d.std(ddof=1)
    return dict(n=len(vp), pearson_r=float(np.corrcoef(vp, vg)[0, 1]), mae_ml=float(np.abs(d).mean()),
                mape_pct=float((np.abs(d) / vg).mean() * 100), bias_ml=float(d.mean()),
                loa_low_ml=float(d.mean() - 1.96 * sd), loa_high_ml=float(d.mean() + 1.96 * sd),
                mean_v_ratio=float((vp / vg).mean()))


def c1_volumetry(results, out):
    runs = sorted(glob.glob(os.path.join(results, "R*", "test_per_patient.csv")))
    vol = pd.concat([pd.read_csv(f).assign(run=os.path.basename(os.path.dirname(f))) for f in runs])
    vol[["run", "patient", "vol_pred_ml", "vol_gt_ml", "v_ratio", "dice"]].to_csv(os.path.join(out, "volumes.csv"), index=False)
    summ = {r: volume_agreement(d.vol_pred_ml, d.vol_gt_ml) for r, d in vol.groupby("run")}
    json.dump(dict(note=NOTE, runs=summ), open(os.path.join(out, "volumetry_summary.json"), "w"), indent=1)
    n = len(summ)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    lim = [0, max(vol.vol_pred_ml.max(), vol.vol_gt_ml.max()) * 1.05]
    for i, (r, d) in enumerate(vol.groupby("run")):
        ax[0].scatter(d.vol_gt_ml, d.vol_pred_ml, s=22, label=f"{r} (r={summ[r]['pearson_r']:.2f}, MAE {summ[r]['mae_ml']:.1f} mL)")
        m, df = (d.vol_pred_ml + d.vol_gt_ml) / 2, d.vol_pred_ml - d.vol_gt_ml
        ax[1].scatter(m, df, s=22, color=f"C{i}")
        ax[1].axhline(summ[r]["bias_ml"], color=f"C{i}", lw=1)
        for k in ("loa_low_ml", "loa_high_ml"):
            ax[1].axhline(summ[r][k], color=f"C{i}", lw=0.8, ls="--")
    ax[0].plot(lim, lim, "k--", lw=1, label="identity"); ax[0].set(xlim=lim, ylim=lim, xlabel="GT volume (mL)",
                                                                   ylabel="predicted volume (mL)", title="C1 volume agreement (test)")
    ax[1].axhline(0, c="k", lw=0.6)
    ax[1].set(xlabel="mean of predicted and GT volume (mL)", ylabel="predicted - GT (mL)",
              title="Bland-Altman (solid = bias, dashed = 95 % LoA)")
    ax[0].legend(fontsize=7)
    for a_ in ax:
        a_.grid(alpha=.3)
    fig.text(0.5, 0.005, NOTE, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(out, "volumetry.png"), dpi=200); plt.close(fig)
    print(f"C1: {n} runs -> volumes.csv, volumetry_summary.json, volumetry.png")
    return summ


def case_card(img, gt, pred, ent, title, path):
    zs = np.where(gt.any((0, 1)))[0]
    zz = [int(np.percentile(zs, q)) for q in (25, 50, 75)]          # GT used only to choose display slices
    fig, ax = plt.subplots(3, 4, figsize=(13, 10))
    for r, z in enumerate(zz):
        im = np.asarray(img[:, :, z], np.float32)
        for c, (lab, m, cm) in enumerate(((None, None, None), ("GT", gt, "autumn"), ("Prediction (TTA)", pred, "winter"))):
            ax[r, c].imshow(im, cmap="gray", vmin=0, vmax=1)
            if m is not None:
                ax[r, c].imshow(np.ma.masked_where(~m[:, :, z], m[:, :, z]), cmap=cm, alpha=0.5, vmin=0, vmax=1)
            ax[r, c].set_title(f"{'Input' if lab is None else lab}  (z={z})", fontsize=9)
        h = ax[r, 3].imshow(np.asarray(ent[:, :, z], np.float32), cmap="magma", vmin=0, vmax=np.log(2))
        ax[r, 3].set_title("voxel entropy (TTA)", fontsize=9)
    for a_ in ax.ravel():
        a_.axis("off")
    fig.colorbar(h, ax=ax[:, 3], fraction=0.04)
    fig.suptitle(title, fontsize=10); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def c2_reliability(results, data, splits, run, out, overlap, device):
    model, cfg, _ = load_run(os.path.join(results, run), device)
    sp = json.load(open(splits))
    single = pd.read_csv(os.path.join(results, run, "test_per_patient.csv")).set_index("patient")
    order = single.dice.sort_values()
    cards = {"worst": order.index[0], "median": order.index[len(order) // 2], "best": order.index[-1]}
    rows = []
    for split, ids in (("val", sp["folds"][cfg.get("fold", 0)]["val"]), ("test", sp["test"])):
        for pid in ids:
            img, lbl = load_case(data, pid)
            gt = np.asarray(lbl) > 0
            mask, ent, score = tta_predict(model, img, cfg, overlap, device)
            d = overlap_metrics(mask, gt)["dice"]
            rows.append(dict(split=split, patient=pid, uncertainty=score, dice_tta=d,
                             vol_tta_ml=mask.sum() / 1000.0, vol_gt_ml=gt.sum() / 1000.0,
                             dice_single=single.dice.get(pid, np.nan) if split == "test" else np.nan))
            print(split, pid, f"unc {score:.4f} Dice(TTA) {d:.4f}", flush=True)
            if split == "test" and pid in cards.values():
                tag = [k for k, v in cards.items() if v == pid][0]
                cards[tag] = (pid, np.asarray(img), gt, mask, ent, score, d)
    df = pd.DataFrame(rows)
    val, test = df[df.split == "val"], df[df.split == "test"].copy()
    thr = float(np.nanpercentile(val.uncertainty, 75))                 # chosen on validation only
    poor_thr = float(np.percentile(val.dice_tta, 25))
    test["flag"] = test.uncertainty.fillna(np.inf) > thr                # empty mask -> flagged
    test["poor"] = test.dice_tta < poor_thr
    tp = int((test.flag & test.poor).sum())
    rho_t = stats.spearmanr(test.uncertainty, test.dice_tta, nan_policy="omit")
    rho_v = stats.spearmanr(val.uncertainty, val.dice_tta, nan_policy="omit")
    diff = test.dice_tta - test.dice_single
    w = stats.wilcoxon(test.dice_tta, test.dice_single) if (diff != 0).any() else None
    summ = dict(run=run, flips=4, overlap=overlap, note=NOTE, flag_threshold_val_p75=thr, poor_dice_threshold_val_p25=poor_thr,
                spearman_rho_test=float(rho_t.statistic), spearman_p_test=float(rho_t.pvalue),
                spearman_rho_val=float(rho_v.statistic), spearman_p_val=float(rho_v.pvalue),
                n_test=len(test), n_flagged=int(test.flag.sum()), n_poor=int(test.poor.sum()),
                flag_precision=tp / test.flag.sum() if test.flag.sum() else float("nan"),
                flag_recall=tp / test.poor.sum() if test.poor.sum() else float("nan"),
                dice_single_mean=float(test.dice_single.mean()), dice_tta_mean=float(test.dice_tta.mean()),
                dice_tta_minus_single_mean=float(diff.mean()),
                wilcoxon_tta_vs_single_p=float(w.pvalue) if w is not None else float("nan"))
    pd.concat([val, test]).to_csv(os.path.join(out, "uncertainty_cases.csv"), index=False)
    json.dump(summ, open(os.path.join(out, "c2_summary.json"), "w"), indent=1)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(val.uncertainty, val.dice_tta, marker="s", c="gray", label=f"validation (rho={rho_v.statistic:.2f})")
    ax.scatter(test.uncertainty, test.dice_tta, c=np.where(test.flag, "C3", "C0"),
               label=f"test (rho={rho_t.statistic:.2f}); red = flagged")
    ax.axvline(thr, ls="--", c="C3", lw=1, label="flag threshold (val P75)")
    ax.axhline(poor_thr, ls=":", c="k", lw=1, label="'poor' Dice (val P25)")
    ax.set(xlabel="mean voxel entropy inside predicted mask", ylabel="Dice (4-flip TTA)", title=f"C2 reliability - {run}")
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "uncertainty_vs_dice.png"), dpi=200); plt.close(fig)
    for tag, (pid, img, gt, mask, ent, score, d) in cards.items():
        t = test.set_index("patient").loc[pid]
        case_card(img, gt, mask, ent, f"{tag.upper()} test case {pid} ({run}): Dice(TTA) {d:.3f} | volume pred "
                  f"{mask.sum() / 1000:.1f} mL vs GT {gt.sum() / 1000:.1f} mL | uncertainty {score:.4f} | "
                  f"{'FLAGGED for review' if t.flag else 'not flagged'}\n{NOTE}", os.path.join(out, f"case_card_{tag}.png"))
    print("C2:", json.dumps({k: v for k, v in summ.items() if k != "note"}, indent=1))
    return summ


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results"); p.add_argument("--data", required=True)
    p.add_argument("--splits", default="splits/patient_splits.json"); p.add_argument("--c2_run", default="R3_final")
    p.add_argument("--overlap", type=float, default=0.25); p.add_argument("--skip_c2", action="store_true")
    a = p.parse_args(argv)
    out = os.path.join(a.results, "clinical"); os.makedirs(out, exist_ok=True)
    c1_volumetry(a.results, out)
    if not a.skip_c2:
        c2_reliability(a.results, a.data, a.splits, a.c2_run, out, a.overlap, "cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()

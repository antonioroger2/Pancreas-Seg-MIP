"""T9: regenerate every rubric figure/table from SAVED outputs only (no number is typed by hand except the paper's
published Table 1 values below, quoted with their source).

  python tools/make_figures.py --results results [--data /tmp/prep]     (--data only for figure b, the CT slices)
Writes <results>/figures/: a_paper_vs_ours.{csv,md,tex}, b_samples.png, c_curves.png, d_ablation.{csv,md,png},
e_per_patient_dice.png, e_volume_scatter.png, e_overseg.png, f_wilcoxon.csv, captions.md
"""
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import load_pred                       # noqa: E402

PAPER = dict(source="Tsewalo Tondji et al., Mathematics 2025, 13(24), 3942, Table 1 (NIH)",
             dice=(80.8, 2.1), hd95=(8.1, 1.6), assd=(2.1, 0.3), jaccard=None)
ORDER = ["R0_baseline", "R1_patch", "R2_gn", "R3_final", "R4_noattn"]
LABEL = {"R0_baseline": "R0 baseline", "R1_patch": "R1 +patches (M1)", "R2_gn": "R2 +GroupNorm (M2)",
         "R3_final": "R3 +deep sup. (M3)", "R4_noattn": "R4 no attention"}
COLOR = {"R0_baseline": "#2a78d6", "R1_patch": "#eb6834", "R2_gn": "#1baf7a", "R3_final": "#eda100", "R4_noattn": "#e87ba4"}
INK, MUTED = "#0b0b0b", "#52514e"
METRICS = ["dice", "jaccard", "precision", "recall", "hd95", "assd", "v_ratio"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e4e3df"})


def to_md(df):
    rows = [list(map(str, df.columns))] + df.astype(str).values.tolist()
    rows = rows[:1] + [["---"] * len(df.columns)] + rows[1:]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def to_tex(df):
    def esc(t):
        return str(t).replace("%", r"\%").replace("±", r"$\pm$").replace("†", r"$^\dagger$").replace("_", r"\_")
    lines = [" & ".join(esc(c) for c in r) + r" \\" for r in [list(df.columns)] + df.values.tolist()]
    return "\n".join([r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\hline", lines[0], r"\hline", *lines[1:],
                      r"\hline", r"\end{tabular}"]) + "\n"



def ms(s, k, scale=1.0, nd=1):
    d = s[k]
    return f"{d['mean'] * scale:.{nd}f} ± {d['sd'] * scale:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--data")
    ap.add_argument("--splits", default="splits/patient_splits.json")
    a = ap.parse_args()
    out = os.path.join(a.results, "figures"); os.makedirs(out, exist_ok=True)
    runs = [r for r in ORDER if os.path.exists(os.path.join(a.results, r, "summary.json"))]
    S = {r: json.load(open(os.path.join(a.results, r, "summary.json"))) for r in runs}
    P = {r: pd.read_csv(os.path.join(a.results, r, "test_per_patient.csv")).set_index("patient") for r in runs}
    C = {r: json.load(open(os.path.join(a.results, r, "config.json"))) for r in runs}
    final = "R3_final" if "R3_final" in runs else runs[-1]
    n_test = S[final]["n_cases"]
    cap = []

    # (a) paper vs R0 vs final
    rows = [dict(method="Paper (TF, 5-fold CV)", dice=f"{PAPER['dice'][0]} ± {PAPER['dice'][1]}†", jaccard="n/r",
                 hd95=f"{PAPER['hd95'][0]} ± {PAPER['hd95'][1]}", assd=f"{PAPER['assd'][0]} ± {PAPER['assd'][1]}")]
    for r in dict.fromkeys(["R0_baseline", final]):
        if r in S:
            s = S[r]
            rows.append(dict(method=f"Ours {LABEL[r]}", dice=ms(s, "dice", 100), jaccard=ms(s, "jaccard", 100),
                             hd95=ms(s, "hd95", nd=2) + (f" (inf: {s['n_inf_hd95']})" if s["n_inf_hd95"] else ""),
                             assd=ms(s, "assd", nd=2) + (f" (inf: {s['n_inf_assd']})" if s["n_inf_assd"] else "")))
    ta = pd.DataFrame(rows).rename(columns=dict(dice="Dice (%)", jaccard="Jaccard (%)", hd95="HD95 (mm)", assd="ASSD (mm)"))
    ta.to_csv(os.path.join(out, "a_paper_vs_ours.csv"), index=False)
    foot = ("† Paper SD/range (78.6-84.1) is most likely across its 5 folds (not stated); ours is the SD across the "
            f"held-out test patients (n = {n_test}; seed 42, fold-0 model, single seed) - not like-for-like. n/r = not reported. "
            f"Paper values: {PAPER['source']}.")
    open(os.path.join(out, "a_paper_vs_ours.md"), "w", encoding="utf-8").write(to_md(ta) + "\n\n" + foot + "\n")
    open(os.path.join(out, "a_paper_vs_ours.tex"), "w", encoding="utf-8").write(
        to_tex(ta) + "% " + foot + "\n")
    cap.append(("a_paper_vs_ours.md/.tex", "Paper vs replicated (R0) vs final (R3) on the NIH test set. " + foot))

    # (b) best / median / worst test case of the final run: Input | GT | R0 | final
    if a.data:
        from src.data.dataset import load_case
        order = P[final].dice.sort_values()
        cases = [("Best", order.index[-1]), ("Median", order.index[len(order) // 2]), ("Worst", order.index[0])]
        show = [r for r in dict.fromkeys(["R0_baseline", final]) if r in runs]
        fig, ax = plt.subplots(3, 2 + len(show), figsize=(3.3 * (2 + len(show)), 10))
        for i, (tag, pid) in enumerate(cases):
            img, lbl = load_case(a.data, pid)
            gt = np.asarray(lbl) > 0
            z = int(np.round(np.argwhere(gt)[:, 2].mean()))           # GT centroid slice (display only)
            panels = [("Input CT", None), ("Ground truth", gt)] + [(LABEL[r], load_pred(os.path.join(a.results, r), pid))
                                                                    for r in show]
            for j, (t, m) in enumerate(panels):
                ax[i, j].imshow(np.asarray(img[::-1, :, z], np.float32), cmap="gray", vmin=0, vmax=1)  # rows flipped: anterior up
                if m is not None:
                    ax[i, j].imshow(np.ma.masked_where(~m[::-1, :, z], m[::-1, :, z]), cmap="autumn", alpha=0.55)
                d = "" if j < 2 else f"\nDice {P[show[j - 2]].dice[pid]:.3f}"
                ax[i, j].set_title(f"{tag}: {pid[-4:]} - {t}{d}" if j == 0 else t + d, fontsize=9)
                ax[i, j].axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(out, "b_samples.png"), dpi=200); plt.close(fig)
        cap.append(("b_samples.png", f"Best / median / worst held-out test case (ranked by {LABEL[final]} Dice); axial slice "
                    "through the ground-truth centroid (1 mm, HU window [-100, 240]); red = mask."))

    # (c) curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in runs:
        d = pd.read_csv(os.path.join(a.results, r, "train_log.csv")); v = d.dropna(subset=["val_dice"])
        ax[0].plot(d.step, d.train_loss.rolling(4, min_periods=1).mean(), color=COLOR[r], lw=1.5, label=LABEL[r])
        ax[1].plot(v.step, v.val_dice, "-o", color=COLOR[r], lw=2, ms=4, label=LABEL[r])
    ax[0].set(xlabel="optimizer step", ylabel="train loss (Dice+Focal; moving avg of 4 logs)", title="Training loss")
    ax[1].set(xlabel="optimizer step", ylabel="validation Dice (13 full volumes)", title="Validation Dice (fold 0)", ylim=(0, 1))
    for x in ax:
        x.grid(alpha=.6); x.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "c_curves.png"), dpi=200); plt.close(fig)
    N = {C[r]["steps"] for r in runs}
    cap.append(("c_curves.png", f"Training loss and full-volume validation Dice; all runs N = {', '.join(map(str, N))} "
                f"steps, Adam 6e-4, AMP, hardware: {', '.join(sorted({C[r]['gpu'] for r in runs}))}. R0 sees one 224x224x128 "
                "crop per step, R1-R4 two 128x128x96 patches per step."))

    # (d) ablation table + bars with per-patient points
    tab = []
    for r in runs:
        s = S[r]
        tab.append(dict(run=LABEL[r], **{m: s[m]["mean"] for m in METRICS}, **{f"{m}_sd": s[m]["sd"] for m in METRICS},
                        n_empty=s["n_empty_pred"], n_inf_hd95=s["n_inf_hd95"], best_step=s["best_step"],
                        best_val_dice=s["best_val_dice"], params=C[r]["params"]))
    td = pd.DataFrame(tab); td.to_csv(os.path.join(out, "d_ablation.csv"), index=False)
    md = pd.DataFrame([{"run": LABEL[r], **{m: ms(S[r], m, 100 if m in ("dice", "jaccard", "precision", "recall") else 1,
                                               1 if m in ("dice", "jaccard", "precision", "recall") else 2) for m in METRICS}}
                       for r in runs])
    open(os.path.join(out, "d_ablation.md"), "w", encoding="utf-8").write(
        to_md(md) + "\n\nDice/Jaccard/precision/recall in %, HD95/ASSD in mm, v_ratio = V_pred/V_GT; "
        f"mean ± SD over {n_test} test patients (inf distances excluded and counted in d_ablation.csv).\n")
    fig, ax = plt.subplots(figsize=(1.6 * len(runs) + 2, 4.5))
    for i, r in enumerate(runs):
        ax.bar(i, S[r]["dice"]["mean"], 0.6, color=COLOR[r], zorder=2)
        ax.errorbar(i, S[r]["dice"]["mean"], S[r]["dice"]["sd"], color=INK, capsize=4, lw=1, zorder=3)
        ax.scatter(i + np.random.default_rng(0).uniform(-.15, .15, len(P[r])), P[r].dice, s=12, color=INK, alpha=.5, zorder=4)
        ax.text(i, S[r]["dice"]["mean"] + S[r]["dice"]["sd"] + 0.02, f"{S[r]['dice']['mean']:.3f}", ha="center", color=INK, fontsize=9)
    ax.axhline(PAPER["dice"][0] / 100, ls="--", color=MUTED, lw=1)
    ax.text(len(runs) - 0.5, PAPER["dice"][0] / 100 + 0.01, "paper 0.808", ha="right", color=MUTED, fontsize=8)
    ax.set_xticks(range(len(runs)), [LABEL[r] for r in runs], fontsize=8)
    ax.set(ylabel=f"test Dice ({n_test} patients)", ylim=(0, 1), title="Cumulative ablation (mean ± SD, dots = patients)")
    ax.grid(axis="y", alpha=.6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "d_ablation.png"), dpi=200); plt.close(fig)
    cap.append(("d_ablation.md/.png", "Cumulative ablation on identical settings (split, seed, N, optimizer, augmentation, "
                f"AMP, threshold, full-volume evaluation). Single fold, single seed, n = {n_test}: differences < ~0.02 Dice are noise."))

    # (e) per-patient Dice, volume scatter, over-segmentation
    pts = list(P[runs[0]].index)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for r in runs:
        ax.plot(range(len(pts)), P[r].dice.reindex(pts), "-o", color=COLOR[r], lw=1.5, ms=5, label=LABEL[r])
    ax.set_xticks(range(len(pts)), [p[-4:] for p in pts], rotation=45)
    ax.set(xlabel="test patient", ylabel="Dice", ylim=(0, 1), title="Per-patient test Dice")
    ax.grid(alpha=.6); ax.legend(fontsize=8, frameon=False, ncol=len(runs))
    fig.tight_layout(); fig.savefig(os.path.join(out, "e_per_patient_dice.png"), dpi=200); plt.close(fig)
    two = [r for r in dict.fromkeys(["R0_baseline", final]) if r in runs]          # scatter: <= 3 series
    for name, (xk, yk, xl, yl, ident) in {
            "e_volume_scatter.png": ("vol_gt_ml", "vol_pred_ml", "GT volume (mL)", "predicted volume (mL)", True),
            "e_overseg.png": ("recall", "precision", "recall", "precision", False)}.items():
        fig, ax = plt.subplots(figsize=(5.2, 5))
        for r in two:
            ax.scatter(P[r][xk], P[r][yk], s=40, color=COLOR[r], edgecolor="white", lw=1.5, label=LABEL[r], zorder=3)
        if ident:
            lim = [0, max(P[r][[xk, yk]].max().max() for r in two) * 1.05]
            ax.plot(lim, lim, "--", color=MUTED, lw=1, label="identity"); ax.set(xlim=lim, ylim=lim)
        else:
            ax.set(xlim=(0, 1.02), ylim=(0, 1.02)); ax.plot([0, 1], [0, 1], ":", color=MUTED, lw=1)
        ax.set(xlabel=xl, ylabel=yl); ax.grid(alpha=.6); ax.legend(fontsize=8, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(out, name), dpi=200); plt.close(fig)
    cap.append(("e_per_patient_dice.png", f"Per-patient Dice on the {n_test} held-out test patients for every run."))
    cap.append(("e_volume_scatter.png", "Predicted vs ground-truth pancreas volume (mL, 1 mm voxels); above the identity line "
                "= over-segmentation. Healthy pancreas: measurement accuracy only."))
    cap.append(("e_overseg.png", "Precision vs recall per test patient: precision << recall indicates over-segmentation."))

    # (f) paired Wilcoxon
    pairs = [("R0_baseline", final)] + [(x, y) for x, y in zip(runs, runs[1:])]
    wr = []
    for x, y in dict.fromkeys(pairs):
        if x in P and y in P and x != y:
            for m in ("dice", "hd95"):
                d = pd.concat([P[x][m], P[y][m]], axis=1, keys=["x", "y"]).replace([np.inf], np.nan).dropna()
                diff = d.y - d.x
                t = stats.wilcoxon(d.x, d.y) if (diff != 0).any() and len(d) > 0 else None
                wr.append(dict(a=x, b=y, metric=m, n=len(d), median_diff_b_minus_a=float(diff.median()),
                               mean_diff_b_minus_a=float(diff.mean()), statistic=t.statistic if t else np.nan,
                               p=t.pvalue if t else np.nan, b_better_in=int((diff > 0).sum() if m == "dice" else (diff < 0).sum())))
    pd.DataFrame(wr).to_csv(os.path.join(out, "f_wilcoxon.csv"), index=False)
    cap.append(("f_wilcoxon.csv", "Two-sided paired Wilcoxon signed-rank tests on per-patient test metrics (HD95: pairs with "
                f"an infinite value dropped, n reported). n = {n_test}, single fold, single seed, no multiple-comparison correction."))

    # verdicts per addition (rule fixed in advance: |delta mean Dice| < 0.02 = "no clear difference"; guide sec. 16.4)
    W = pd.DataFrame(wr)
    lines = ["# Final results - auto-generated by tools/make_figures.py (do not edit numbers by hand)", "",
             "Verdict rule (fixed before results): delta mean test Dice >= +0.02 -> improved, <= -0.02 -> worse, otherwise "
             f"no clear difference; Wilcoxon p reported, n = {n_test} test patients, single fold, single seed.", "",
             "| Addition | Comparison | mean Dice a -> b | delta | Wilcoxon p (Dice) | patients improved | verdict |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, x, y in (("M1 patches + sliding window", "R0_baseline", "R1_patch"), ("M2 GroupNorm", "R1_patch", "R2_gn"),
                       ("M3 deep supervision", "R2_gn", "R3_final"), ("Attention gates (R4 removes them)", "R4_noattn", "R3_final"),
                       ("All of M1-M3", "R0_baseline", "R3_final")):
        if x in S and y in S:
            dlt = S[y]["dice"]["mean"] - S[x]["dice"]["mean"]
            w = W[(W.a == x) & (W.b == y) & (W.metric == "dice")] if len(W) else W
            if not len(w) and len(W):
                w = W[(W.a == y) & (W.b == x) & (W.metric == "dice")]
            pv = f"{w.p.iloc[0]:.3g}" if len(w) else "n/a"
            better = int((P[y].dice > P[x].dice.reindex(P[y].index)).sum())
            v = "improved" if dlt >= 0.02 else "worse" if dlt <= -0.02 else "no clear difference"
            lines.append(f"| {name} | {x} -> {y} | {S[x]['dice']['mean']:.4f} -> {S[y]['dice']['mean']:.4f} | {dlt:+.4f} | "
                         f"{pv} | {better}/{len(P[y])} | {v} |")
    cdir = os.path.join(a.results, "clinical")
    if os.path.exists(os.path.join(cdir, "volumetry_summary.json")):
        vs = json.load(open(os.path.join(cdir, "volumetry_summary.json")))["runs"]
        lines += ["", "## C1 volumetry (test, main prediction)", "", "| Run | Pearson r | MAE (mL) | mean abs % error | bias (mL) "
                  "| 95 % LoA (mL) | mean V_pred/V_GT |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for r, v in vs.items():
            lines.append(f"| {r} | {v['pearson_r']:.3f} | {v['mae_ml']:.1f} | {v['mape_pct']:.1f} | {v['bias_ml']:+.1f} | "
                         f"[{v['loa_low_ml']:.1f}, {v['loa_high_ml']:.1f}] | {v['mean_v_ratio']:.2f} |")
    if os.path.exists(os.path.join(cdir, "c2_summary.json")):
        c = json.load(open(os.path.join(cdir, "c2_summary.json")))
        lines += ["", f"## C2 reliability ({c['run']}, {c['flips']}-flip TTA)", "",
                  f"- Spearman rho(uncertainty, Dice): test {c['spearman_rho_test']:.3f} (p {c['spearman_p_test']:.3g}), "
                  f"validation {c['spearman_rho_val']:.3f} (p {c['spearman_p_val']:.3g})",
                  f"- Flag threshold (validation P75) {c['flag_threshold_val_p75']:.4f}; 'poor' = Dice < {c['poor_dice_threshold_val_p25']:.3f} "
                  f"(validation P25): {c['n_flagged']} flagged, {c['n_poor']} poor of {c['n_test']} -> flag precision "
                  f"{c['flag_precision']:.2f}, recall {c['flag_recall']:.2f}",
                  f"- TTA vs single prediction: mean Dice {c['dice_single_mean']:.4f} -> {c['dice_tta_mean']:.4f} "
                  f"(delta {c['dice_tta_minus_single_mean']:+.4f}, Wilcoxon p {c['wilcoxon_tta_vs_single_p']:.3g})"]
    open(os.path.join(out, "final_results_auto.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    with open(os.path.join(out, "captions.md"), "w", encoding="utf-8") as f:
        f.write("# Figure / table captions (auto-generated by tools/make_figures.py)\n\n")
        for k, v in cap:
            f.write(f"**{k}** - {v}\n\n")
    print("written:", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()

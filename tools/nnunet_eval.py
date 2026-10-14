"""Score nnU-Net predictions with the repo's metrics (src/metrics.py) at each case's native CT spacing.

Every GT case must have a prediction (a missing one is an error, never skipped).
Writes <out>/<name>_per_patient.csv and <out>/<name>_summary.json.
"""
import argparse, glob, json, os, sys

import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import aggregate_metrics, compute_all_metrics                  # noqa: E402

PAPER_DICE = dict(mean=0.808, sd=0.021, source="base paper, NIH Pancreas-CT, Table 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--name", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for g_path in sorted(glob.glob(os.path.join(a.gt, "*.nii.gz"))):
        pid = os.path.basename(g_path)[:-7]
        g, p = sitk.ReadImage(g_path), sitk.ReadImage(os.path.join(a.pred, f"{pid}.nii.gz"))
        if g.GetSize() != p.GetSize():
            raise SystemExit(f"{pid}: prediction size {p.GetSize()} != GT size {g.GetSize()}")
        m = compute_all_metrics(sitk.GetArrayFromImage(p) > 0, sitk.GetArrayFromImage(g) > 0,
                                spacing=g.GetSpacing()[::-1])                 # arrays are (z, y, x)
        rows.append(dict(patient=pid, **m))
        print(f"{pid} dice={m['dice']:.4f} hd95={m['hd95']:.2f} assd={m['assd']:.2f}", flush=True)
    if not rows:
        raise SystemExit(f"no GT cases in {a.gt}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{a.name}_per_patient.csv"), index=False)
    summary = dict(aggregate_metrics(rows), paper_dice=PAPER_DICE)
    json.dump(summary, open(os.path.join(a.out, f"{a.name}_summary.json"), "w"), indent=2)
    d = summary["dice"]
    print(f"{a.name}: n={summary['n_cases']} Dice {d['mean']:.4f} +- {d['sd']:.4f} (paper {PAPER_DICE['mean']}), "
          f"HD95 {summary['hd95']['mean']:.2f} mm, ASSD {summary['assd']['mean']:.2f} mm")


if __name__ == "__main__":
    main()

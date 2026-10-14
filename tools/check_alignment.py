"""T1 alignment gate: test 16 orientation variants of the label array against the CT array.

Variants: optional x<->y transpose x 8 flip combinations of (x, y, z), applied to the label's voxel array
(the image array is the reference). Metrics per variant (all inside the label):
  air_frac      fraction of label voxels with HU < -200 (must be ~0)
  mean_hu       mean HU (contrast-enhanced pancreas ~60-130)
  in_range_frac fraction of label voxels with HU in [60, 180]
  edge_score    mean |grad(HU)| on the label surface / same for the label shifted 10 mm in-plane
                (mean of +-x, +-y shifts); > 1 means the label follows real edges.

Sources: 'course' = Processed_data NIfTI pair; 'idc' = DICOM series (SimpleITK series reader, z-spacing from
ImagePositionPatient, as the Review-1 conversion did) + raw TCIA label copied unchanged.

Usage:
  python tools/check_alignment.py --images data/images --labels data/labels --out results/alignment \
         [--dicom data/dicom --raw_labels data/raw_labels] [--ids 1,2,3] [--overlays 6]
"""
import argparse, glob, itertools, json, os, re
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AIR_HU, RANGE = -200.0, (60.0, 180.0)
# Pass = the T1 acceptance (air ~0, in-range high). edge_score is reported but not gated: on the first 8 cases it
# ranged 0.90-1.31 for EVERY variant (pancreas borders are low-contrast), so it cannot separate orientations.
PASS = dict(air_frac_max=0.01, in_range_min=0.5)
VARIANTS = [(t, fx, fy, fz) for t in (0, 1) for fx, fy, fz in itertools.product((0, 1), repeat=3)]


def vname(v):
    return "T{}_fx{}_fy{}_fz{}".format(*v)


def apply_variant(lbl, v):
    """lbl: (z, y, x) array. Transpose x<->y first, then flip."""
    t, fx, fy, fz = v
    out = np.swapaxes(lbl, 1, 2) if t else lbl
    axes = [a for a, f in ((2, fx), (1, fy), (0, fz)) if f]
    return np.flip(out, axes) if axes else out


def load_course(img_path, lbl_path):
    img, lbl = sitk.ReadImage(img_path), sitk.ReadImage(lbl_path)
    return img, sitk.GetArrayFromImage(img).astype(np.float32), sitk.GetArrayFromImage(lbl) > 0, lbl


def load_idc(dicom_dir, raw_label_path):
    r = sitk.ImageSeriesReader()
    files = r.GetGDCMSeriesFileNames(dicom_dir)
    r.SetFileNames(files)
    img = r.Execute()
    # recompute z-spacing from slice positions (IDC metadata spacing is wrong, see docs/log.md section 3)
    zs = np.array([float(x) for x in _slice_positions(files)])
    sp = list(img.GetSpacing()); sp[2] = float(np.median(np.abs(np.diff(np.sort(zs))))); img.SetSpacing(sp)
    lbl = sitk.ReadImage(raw_label_path)
    return img, sitk.GetArrayFromImage(img).astype(np.float32), sitk.GetArrayFromImage(lbl) > 0, lbl


def _slice_positions(files):
    rd = sitk.ImageFileReader()
    out = []
    for f in files:
        rd.SetFileName(f); rd.ReadImageInformation()
        out.append(rd.GetMetaData("0020|0032").split("\\")[2])
    return out


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def metrics(hu, lbl, spacing_zyx):
    """hu, lbl: (z,y,x). Returns dict of the four alignment metrics."""
    n = int(lbl.sum())
    if n == 0:
        return dict(n_vox=0, air_frac=np.nan, mean_hu=np.nan, in_range_frac=np.nan, edge_score=np.nan)
    v = hu[lbl]
    sh = [int(round(10.0 / spacing_zyx[a])) for a in (1, 2)]               # 10 mm in y and x (voxels)
    zz, yy, xx = np.where(lbl)
    m = max(sh) + 2
    sl = tuple(slice(max(c.min() - m, 0), min(c.max() + m + 1, hu.shape[i])) for i, c in enumerate((zz, yy, xx)))
    h = np.clip(hu[sl], -100, 240)
    g = np.sqrt(sum(gi ** 2 for gi in np.gradient(h, *spacing_zyx)))
    lb = lbl[sl]
    on = g[surface(lb)].mean()
    shifted = []
    for ax, s in ((1, sh[0]), (2, sh[1])):
        for d in (s, -s):
            shifted.append(g[surface(np.roll(lb, d, axis=ax))].mean())
    return dict(n_vox=n, air_frac=float((v < AIR_HU).mean()), mean_hu=float(v.mean()),
                in_range_frac=float(((v >= RANGE[0]) & (v <= RANGE[1])).mean()),
                edge_score=float(on / np.mean(shifted)))


def passes(m):
    return m["air_frac"] <= PASS["air_frac_max"] and m["in_range_frac"] >= PASS["in_range_min"]


def views(hu, lbl, z_superior_first):
    """Coronal, sagittal, axial slices through the label centroid, oriented superior-up / anterior-up."""
    c = np.round(np.argwhere(lbl).mean(0)).astype(int) if lbl.any() else np.array(hu.shape) // 2
    up = (lambda a: a) if z_superior_first else (lambda a: a[::-1])
    return [("coronal", up(hu[:, c[1], :]), up(lbl[:, c[1], :])), ("sagittal", up(hu[:, :, c[2]]), up(lbl[:, :, c[2]])),
            ("axial", hu[c[0]], lbl[c[0]])]


def overlay(hu, lbl, z_sup, title, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, (name, im, lb) in zip(ax, views(hu, lbl, z_sup)):
        a.imshow(np.clip(im, -160, 240), cmap="gray", aspect="auto")
        a.imshow(np.ma.masked_where(~lb, lb), cmap="autumn", alpha=0.5, vmin=0, vmax=1, aspect="auto")
        a.set_title(name); a.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def pid(path):
    return int(re.match(r"(?:PANCREAS_|label)(\d+)\.nii", os.path.basename(path)).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=""); ap.add_argument("--labels", default="")
    ap.add_argument("--dicom"); ap.add_argument("--raw_labels")
    ap.add_argument("--out", default="results/alignment"); ap.add_argument("--ids", default="")
    ap.add_argument("--overlays", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids = {int(i) for i in a.ids.split(",") if i}
    imgs = {pid(p): p for p in glob.glob(os.path.join(a.images, "*.nii.gz"))} if a.images else {}
    lbls = {pid(p): p for p in glob.glob(os.path.join(a.labels, "*.nii.gz"))} if a.labels else {}
    cases = sorted(k for k in imgs if k in lbls and (not ids or k in ids))

    jobs = [("course", k) for k in cases]
    if a.dicom and a.raw_labels:
        dcm = {int(os.path.basename(d)[-4:]): d for d in glob.glob(os.path.join(a.dicom, "PANCREAS_*"))}
        raw = {pid(p): p for p in glob.glob(os.path.join(a.raw_labels, "*.nii.gz"))}
        jobs += [("idc", k) for k in sorted(dcm) if k in raw and (not ids or k in ids)]

    rows, slices = [], {}
    for src, k in jobs:
        img, hu, lbl, lbl_img = (load_course(imgs[k], lbls[k]) if src == "course"
                                 else load_idc(dcm[k], raw[k]))
        sp = img.GetSpacing()[::-1]                                   # (z, y, x)
        z_sup = img.GetDirection()[8] < 0                             # index 0 = most superior slice
        hdr = dict(img_dir=np.round(img.GetDirection(), 3).tolist(), lbl_dir=np.round(lbl_img.GetDirection(), 3).tolist(),
                   img_origin=np.round(img.GetOrigin(), 1).tolist(), lbl_origin=np.round(lbl_img.GetOrigin(), 1).tolist())
        if src == "idc" and k in imgs:                                # is the DICOM read identical to the course NIfTI?
            hdr["idc_equals_course"] = bool(np.array_equal(hu, sitk.GetArrayFromImage(sitk.ReadImage(imgs[k]))))
        case = []
        for v in VARIANTS:
            lv = np.ascontiguousarray(apply_variant(lbl, v))
            if lv.shape != hu.shape:
                continue
            case.append(dict(source=src, patient=f"PANCREAS_{k:04d}", variant=vname(v), shape=str(hu.shape),
                             spacing_zyx=str(tuple(round(s, 3) for s in sp)), **metrics(hu, lv, sp),
                             **{k2: str(v2) for k2, v2 in hdr.items()}))
            slices[(src, k, vname(v))] = [(n, im[::2, ::2], lb[::2, ::2]) for n, im, lb in views(hu, lv, z_sup)]
        best = min(case, key=lambda r: (round(r["air_frac"], 2), -r["in_range_frac"], -r["edge_score"]))
        for r in case:
            r["case_best"] = best["variant"]
        rows += case
        i0 = case[0]
        print(f"{src} {k:04d} identity air={i0['air_frac']:.3f} hu={i0['mean_hu']:.0f} in={i0['in_range_frac']:.2f} "
              f"edge={i0['edge_score']:.2f} | case-best {best['variant']} in={best['in_range_frac']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "alignment_all_variants.csv"), index=False)

    # ONE fixed rule per source: the variant with the most passing cases (ties: lower median air, higher median in-range)
    summary, report = {}, []
    for src, d in df.groupby("source"):
        d = d.assign(passed=d.apply(passes, axis=1))
        agg = d.groupby("variant").agg(n_pass=("passed", "sum"), n_case_best=("case_best", lambda s: 0),
                                       med_air=("air_frac", "median"), med_in=("in_range_frac", "median"),
                                       med_edge=("edge_score", "median"), med_hu=("mean_hu", "median")).reset_index()
        cb = d.drop_duplicates("patient").case_best.value_counts()
        agg["n_case_best"] = agg.variant.map(cb).fillna(0).astype(int)
        agg = agg.sort_values(["n_pass", "med_air", "med_in"], ascending=[False, True, False])
        rule = agg.iloc[0]["variant"]
        agg.to_csv(os.path.join(a.out, f"variant_summary_{src}.csv"), index=False)
        before = d[d.variant == vname(VARIANTS[0])].set_index("patient")
        after = d[d.variant == rule].set_index("patient")
        m = ("air_frac", "mean_hu", "in_range_frac", "edge_score")
        for p in before.index:
            report.append(dict(source=src, patient=p, rule=rule, case_best=before.loc[p, "case_best"],
                               **{f"before_{c}": before.loc[p, c] for c in m}, **{f"after_{c}": after.loc[p, c] for c in m},
                               after_pass=bool(after.loc[p, "passed"]),
                               idc_equals_course=before.loc[p].get("idc_equals_course", "")))
        fails = sorted(after.index[~after.passed])
        summary[src] = dict(rule=rule, n_cases=int(len(before)), n_pass_after=int(after.passed.sum()),
                            n_pass_identity=int(before.passed.sum()), failing_after=fails,
                            n_case_best_is_rule=int((before.case_best == rule).sum()), thresholds=PASS)
        # overlays: first N cases + every failing case; montage of all cases (coronal, rule applied)
        pids = list(before.index)
        for p in sorted(set(pids[:a.overlays]) | set(fails)):
            k = int(p[-4:])
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for axx, (n, im, lb) in zip(ax, slices[(src, k, rule)]):
                axx.imshow(np.clip(im, -160, 240), cmap="gray", aspect="auto")
                axx.imshow(np.ma.masked_where(~lb, lb), cmap="autumn", alpha=0.5, vmin=0, vmax=1, aspect="auto")
                axx.set_title(n); axx.axis("off")
            r = after.loc[p]
            fig.suptitle(f"{src} {p} rule {rule}: air {r.air_frac:.3f}  mean HU {r.mean_hu:.0f}  "
                         f"in[60,180] {r.in_range_frac:.2f}  edge {r.edge_score:.2f}  {'PASS' if r.passed else 'FAIL'}")
            fig.tight_layout(); fig.savefig(os.path.join(a.out, f"overlay_{src}_{k:04d}.png"), dpi=100); plt.close(fig)
        nc = 10; nr = int(np.ceil(len(pids) / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(2.2 * nc, 2.4 * nr), squeeze=False)
        for axx in ax.ravel():
            axx.axis("off")
        for axx, p in zip(ax.ravel(), pids):
            n, im, lb = slices[(src, int(p[-4:]), rule)][0]
            axx.imshow(np.clip(im, -160, 240), cmap="gray", aspect="auto")
            axx.imshow(np.ma.masked_where(~lb, lb), cmap="autumn", alpha=0.6, vmin=0, vmax=1, aspect="auto")
            axx.set_title(p[-4:] + ("" if after.loc[p, "passed"] else " FAIL"), fontsize=8,
                          color="k" if after.loc[p, "passed"] else "r")
        fig.suptitle(f"{src}: coronal slice through label centroid, rule {rule} (superior up)"); fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"montage_coronal_{src}.png"), dpi=90); plt.close(fig)
    pd.DataFrame(report).to_csv(os.path.join(a.out, "alignment_report.csv"), index=False)
    json.dump(summary, open(os.path.join(a.out, "alignment_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

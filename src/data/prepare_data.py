"""
prepare_data.py - T2 full-volume preprocessing + QC + seed-42 patient splits.

Input  : course NIfTI pairs (--images/--labels, PANCREAS_XXXX.nii.gz) or IDC DICOM (--dicom) + TCIA labels
         (bit-identical to the course NIfTI on every case checked in T1, see results/alignment/).
Output : <out>/cases/PANCREAS_XXXX.npz  image float16 (H,W,D) in [0,1], label uint8 (H,W,D)
         <out>/meta/PANCREAS_XXXX.json  original/resampled geometry, rule applied, QC values
         <out>/manifest.csv, <out>/qc_summary.json, <out>/patient_splits.json (+ copy in splits/)
No cropping, no GT-guided anything. Usage:
  python -m src.data.prepare_data --images data/images --labels data/labels --out prep --workers 4
"""
import argparse, csv, glob, hashlib, json, os, re
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk

from src.config import N_FOLDS, N_TEST_CASES, SEED, TARGET_SPACING
from src.data.preprocessing import preprocess_case

IDENTITY = "T0_fx0_fy0_fz0"


def read_dicom(d):
    r = sitk.ImageSeriesReader()
    r.SetFileNames(r.GetGDCMSeriesFileNames(d))
    return r.Execute()


def process(job):
    try:
        return _process(job)
    except Exception as e:                       # a broken case is reported as FAIL, never silently skipped
        pid, img_src, lbl_path, out, rule = job
        meta = dict(patient=pid, source=img_src, label_source=lbl_path, rule=rule, error=repr(e),
                    qc=dict(error=False), qc_pass=False)
        json.dump(meta, open(os.path.join(out, "meta", f"{pid}.json"), "w"), indent=1)
        print(pid, "FAIL", repr(e), flush=True)
        return meta


def _process(job):
    pid, img_src, lbl_path, out, rule = job
    img = read_dicom(img_src) if os.path.isdir(img_src) else sitk.ReadImage(img_src)
    lbl = sitk.ReadImage(lbl_path)
    raw_lbl = sitk.GetArrayFromImage(lbl) > 0
    raw_ml = float(raw_lbl.sum() * np.prod(img.GetSpacing()) / 1000.0)
    image, label, grid = preprocess_case(img, lbl, rule)
    exp_shape = tuple(int(round(n * s)) for n, s in zip(np.array(img.GetSize())[[1, 0, 2]],
                                                        np.array(img.GetSpacing())[[1, 0, 2]]))
    res_ml = float(label.sum()) / 1000.0
    qc = dict(shape_ok=image.shape == exp_shape == label.shape,
              spacing_ok=bool(np.allclose(grid.GetSpacing(), TARGET_SPACING)),
              range_ok=bool(np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1),
              label_binary=bool(set(np.unique(label).tolist()) <= {0, 1}),
              label_nonempty=bool(label.sum() > 0),
              volume_ok=bool(abs(res_ml - raw_ml) / raw_ml < 0.05))
    np.savez_compressed(os.path.join(out, "cases", f"{pid}.npz"), image=image.astype(np.float16), label=label)
    meta = dict(patient=pid, source=img_src, label_source=lbl_path, rule=rule,
                orig_size_xyz=list(img.GetSize()), orig_spacing_xyz=list(img.GetSpacing()),
                orig_origin=list(img.GetOrigin()), orig_direction=list(img.GetDirection()),
                shape_hwd=list(image.shape), spacing_mm=list(TARGET_SPACING),
                raw_image_sha1=hashlib.sha1(sitk.GetArrayViewFromImage(img).tobytes()).hexdigest(),
                label_ml_raw=raw_ml, label_ml_resampled=res_ml, label_voxels=int(label.sum()),
                image_min=float(image.min()), image_max=float(image.max()),
                qc=qc, qc_pass=all(qc.values()))
    json.dump(meta, open(os.path.join(out, "meta", f"{pid}.json"), "w"), indent=1)
    print(pid, image.shape, f"{res_ml:.1f} mL", "PASS" if meta["qc_pass"] else f"FAIL {qc}", flush=True)
    return meta


def make_splits(ids, seed=SEED, n_test=N_TEST_CASES, n_folds=N_FOLDS):
    """Same shuffle as the Review-1 create_patient_splits (=> identical 16-patient test set);
    folds via array_split (13/13/13/13/12) as the build guide specifies (fold 0 = 51 train / 13 val)."""
    rng = np.random.RandomState(seed)          # == np.random.seed(seed); np.random.shuffle(...)
    idx = list(range(len(ids)))
    rng.shuffle(idx)
    test, cv = sorted(idx[:n_test]), idx[n_test:]
    rng.shuffle(cv)
    folds = []
    for f, val in enumerate(np.array_split(np.array(cv), n_folds)):
        val = sorted(val.tolist())
        folds.append(dict(fold=f, train=[ids[i] for i in sorted(set(cv) - set(val))], val=[ids[i] for i in val]))
    return dict(seed=seed, n_patients=len(ids), test=[ids[i] for i in test], folds=folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images"); ap.add_argument("--dicom"); ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="prep"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--alignment", default="results/alignment/alignment_summary.json",
                    help="T1 summary; its rule is applied to every label")
    ap.add_argument("--ids", default="", help="optional subset (comma list of numbers)")
    a = ap.parse_args()
    if not os.path.exists(a.alignment):
        raise SystemExit(f"T1 rule file {a.alignment} not found: run tools/check_alignment.py first")
    t1 = json.load(open(a.alignment))
    rule = t1["course" if "course" in t1 else "idc"]["rule"]          # course and IDC are bit-identical (T1)
    for d in ("cases", "meta"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    num = lambda p: int(re.search(r"(\d+)\.nii", os.path.basename(p)).group(1))
    lbls = {num(p): p for p in glob.glob(os.path.join(a.labels, "*.nii.gz"))}
    if a.dicom:
        srcs = {int(os.path.basename(d)[-4:]): d for d in glob.glob(os.path.join(a.dicom, "PANCREAS_*"))}
    else:
        srcs = {num(p): p for p in glob.glob(os.path.join(a.images, "*.nii.gz"))}
    sel = {int(i) for i in a.ids.split(",") if i}
    keys = sorted(k for k in srcs if k in lbls and (not sel or k in sel))
    jobs = [(f"PANCREAS_{k:04d}", srcs[k], lbls[k], a.out, rule) for k in keys
            if not os.path.exists(os.path.join(a.out, "meta", f"PANCREAS_{k:04d}.json"))]
    print(f"rule {rule}: {len(keys)} cases, {len(jobs)} to process")
    with Pool(a.workers) as pool:
        pool.map(process, jobs, chunksize=1)

    metas = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.out, "meta", "*.json")))]
    cols = ["patient", "rule", "shape_hwd", "label_voxels", "label_ml_raw", "label_ml_resampled",
            "image_min", "image_max", "raw_image_sha1", "qc_pass"]
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + list(metas[0]["qc"]), extrasaction="ignore")
        w.writeheader()
        for m in metas:
            w.writerow({**m, **m["qc"], "shape_hwd": "x".join(map(str, m["shape_hwd"]))})
    summary = dict(n_cases=len(metas), n_pass=sum(m["qc_pass"] for m in metas), rule=rule,
                   failing=[m["patient"] for m in metas if not m["qc_pass"]])
    json.dump(summary, open(os.path.join(a.out, "qc_summary.json"), "w"), indent=2)
    if len(lbls) == 80:                           # splits are defined on all 80 label IDs, not on what was processed
        splits = make_splits([f"PANCREAS_{k:04d}" for k in sorted(lbls)])
        for p in (os.path.join(a.out, "patient_splits.json"), "splits/patient_splits.json"):
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            json.dump(splits, open(p, "w"), indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

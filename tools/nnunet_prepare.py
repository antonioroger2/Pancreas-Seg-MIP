"""Convert the TCIA DICOM + raw labels to an nnU-Net v2 raw dataset with the repo's fixed split.

Fold 0 of splits/patient_splits.json (51 train + 13 val) -> imagesTr/labelsTr, splits_final.json = that fold.
The 16 test patients -> imagesTs; their labels -> labelsTs, read only by tools/nnunet_eval.py (never by training).
DICOM is read exactly as in the T1 check (tools/check_alignment.load_idc: GDCM series order, z-spacing from slice
positions); labels get the T1 rule of the all-80 check and are put on the image's voxel grid (as in T2).
(data/Processed_data on the GPU box is NOT the canonical course NIfTI: it fails T1, see CHANGELOG.)
"""
import argparse, json, os, sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import apply_label_rule                             # noqa: E402
from tools.check_alignment import load_idc                                      # noqa: E402


def write_case(pid, a, rule, img_dir, lbl_dir):
    img, hu, lbl, _ = load_idc(os.path.join(a.dicom, pid), os.path.join(a.raw_labels, f"label{pid[-4:]}.nii.gz"))
    arr = apply_label_rule(lbl, rule).astype(np.uint8)
    if arr.shape != hu.shape:
        raise SystemExit(f"{pid}: label shape {arr.shape} != image shape {hu.shape} after rule {rule}")
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)                       # label voxels are defined on the image's voxel grid (T1)
    sitk.WriteImage(img, os.path.join(img_dir, f"{pid}_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(out, os.path.join(lbl_dir, f"{pid}.nii.gz"), useCompression=True)
    return int(arr.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom", default="nnunet/dicom", help="one folder of .dcm files per PANCREAS_XXXX")
    ap.add_argument("--raw_labels", default="data/lables")
    ap.add_argument("--alignment", default="results/alignment_all80/alignment_summary.json")
    ap.add_argument("--splits", default="splits/patient_splits.json")
    ap.add_argument("--root", default="nnunet")
    ap.add_argument("--dataset", default="Dataset801_Pancreas")
    a = ap.parse_args()

    t1 = json.load(open(a.alignment))["idc"]
    if t1["rule"] != "T0_fx0_fy0_fz0" or t1.get("shape_mismatch") or t1["n_cases"] != 80:
        raise SystemExit(f"T1 gate failed: {t1}")
    sp = json.load(open(a.splits))
    fold = sp["folds"][0]
    raw = os.path.join(a.root, "raw", a.dataset)
    dirs = {k: os.path.join(raw, k) for k in ("imagesTr", "labelsTr", "imagesTs", "labelsTs")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    empty = [pid for pid in fold["train"] + fold["val"]
             if write_case(pid, a, t1["rule"], dirs["imagesTr"], dirs["labelsTr"]) == 0]
    empty += [pid for pid in sp["test"] if write_case(pid, a, t1["rule"], dirs["imagesTs"], dirs["labelsTs"]) == 0]
    if empty:
        raise SystemExit(f"empty labels: {empty}")

    n_tr = len(fold["train"]) + len(fold["val"])
    json.dump(dict(channel_names={"0": "CT"}, labels={"background": 0, "pancreas": 1},
                   numTraining=n_tr, file_ending=".nii.gz"), open(os.path.join(raw, "dataset.json"), "w"), indent=1)
    pre = os.path.join(a.root, "preprocessed", a.dataset)
    os.makedirs(pre, exist_ok=True)
    json.dump([dict(train=fold["train"], val=fold["val"])], open(os.path.join(pre, "splits_final.json"), "w"), indent=1)
    print(f"{a.dataset}: {n_tr} train+val ({len(fold['train'])}/{len(fold['val'])}), {len(sp['test'])} test, rule {t1['rule']}")


if __name__ == "__main__":
    main()

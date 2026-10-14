"""Unpack T2 .npz volumes to uncompressed .npy for memory-mapped training (run once per Kaggle session).
Usage: python tools/unpack_npz.py --src /kaggle/input/pancreas-prep --dst /tmp/prep"""
import argparse, glob, os, shutil
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
a = ap.parse_args()
os.makedirs(os.path.join(a.dst, "npy"), exist_ok=True)
files = sorted(glob.glob(os.path.join(a.src, "**", "cases", "*.npz"), recursive=True))
for f in files:
    pid = os.path.basename(f)[:-4]
    out = os.path.join(a.dst, "npy", f"{pid}_image.npy")
    if not os.path.exists(out.replace("_image", "_label")):
        z = np.load(f)
        np.save(out, z["image"]); np.save(out.replace("_image", "_label"), z["label"])
for name in ("manifest.csv", "qc_summary.json", "patient_splits.json"):
    hits = glob.glob(os.path.join(a.src, "**", name), recursive=True)
    if hits and os.path.abspath(hits[0]) != os.path.abspath(os.path.join(a.dst, name)):
        shutil.copy(hits[0], os.path.join(a.dst, name))
print(f"{len(files)} cases unpacked to {a.dst}/npy")

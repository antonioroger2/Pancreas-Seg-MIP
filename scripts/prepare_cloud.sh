#!/usr/bin/env bash
# Any Linux GPU runner (RTX 6000 Ada, cloud VM, Kaggle): download IDC DICOM (bit-identical to the course NIfTI, T1),
# run the T1 alignment check on all 80, preprocess (T2), unpack to npy, and enforce the same gates as the notebook.
# Usage: bash scripts/prepare_cloud.sh /scratch/work        -> /scratch/work/prep (npz) + /scratch/work/prep_npy (npy)
set -euo pipefail
W=${1:?work dir with ~25 GB free}; mkdir -p "$W"
python tools/fetch_idc.py --meta data/metadata.csv --out "$W/dicom" --workers 32
python tools/check_alignment.py --dicom "$W/dicom" --raw_labels data/raw_labels --out results/alignment_all80 --overlays 6 &
T1=$!
python -m src.data.prepare_data --dicom "$W/dicom" --labels data/raw_labels --out "$W/prep" --workers "$(nproc)"
wait $T1
python tools/unpack_npz.py --src "$W/prep" --dst "$W/prep_npy"
python - "$W" <<'PY'
import json, sys, pandas as pd
w = sys.argv[1]
q = json.load(open(f"{w}/prep/qc_summary.json")); s = json.load(open("results/alignment_all80/alignment_summary.json"))["idc"]
r = pd.read_csv("results/alignment_all80/alignment_report.csv")
assert q["n_cases"] == 80 and q["n_pass"] == 80, f"T2 QC failed: {q}"
assert json.load(open(f"{w}/prep/patient_splits.json")) == json.load(open("splits/patient_splits.json")), "splits differ"
assert s["rule"] == "T0_fx0_fy0_fz0" and not s["shape_mismatch"], f"T1 rule differs: {s}"
assert r.before_air_frac.max() <= 0.01, "label voxels in air"
print("GATES PASSED. T1 in-range<0.5 (inspect overlays):", s["failing_after"])
PY
echo "data ready: $W/prep_npy  (rm -rf $W/dicom to free ~9 GB)"

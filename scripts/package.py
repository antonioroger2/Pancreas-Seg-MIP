"""Zip (a) the code (+ TCIA labels, IDC metadata, T1 rule, splits) and optionally (b) preprocessed data, for upload as
Kaggle Datasets. (a) -> dist/pancreas-code.zip (~6 MB). (b) -> dist/pancreas-prep.zip (cases/*.npz + meta + manifest).
  python scripts/package.py            # code
  python scripts/package.py --prep prep
"""
import argparse, glob, os, zipfile

CODE = ["src/**/*.py", "tools/*.py", "scripts/*", "splits/*.json", "notebooks/kaggle_ablation.ipynb",
        "results/alignment/alignment_summary.json", "results/alignment/alignment_report.csv",
        "data/raw_labels/*.nii.gz", "data/metadata.csv", "requirements.txt", "README.md", "REPRODUCE.md", "CHANGELOG.md"]


def zip_files(path, files, root="."):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.relpath(f, root))
    print(f"{path}: {len(files)} files, {os.path.getsize(path) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prep"); a = ap.parse_args()
    files = sorted({f for pat in CODE for f in glob.glob(pat, recursive=True)
                    if "__pycache__" not in f and os.path.isfile(f)})
    assert any(f.endswith("label0001.nii.gz") for f in files), "data/raw_labels missing (tools/fetch_drive.py)"
    zip_files("dist/pancreas-code.zip", files)
    if a.prep:
        pf = sorted(glob.glob(os.path.join(a.prep, "cases", "*.npz")) + glob.glob(os.path.join(a.prep, "meta", "*.json"))
                    + [os.path.join(a.prep, f) for f in ("manifest.csv", "qc_summary.json", "patient_splits.json")
                       if os.path.exists(os.path.join(a.prep, f))])
        zip_files("dist/pancreas-prep.zip", pf, root=a.prep)


if __name__ == "__main__":
    main()

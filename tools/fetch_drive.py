"""Resumable download of the shared Drive dataset (Processed_data images/labels + raw TCIA labels).
Usage: python tools/fetch_drive.py --out data [--subset images,labels,raw_labels] [--ids 1,2,3]"""
import argparse, os, time, requests, gdown
from gdown.download_folder import _parse_embedded_folder_view

FOLDERS = {"images": "1WJT3uzXzCKIumJj5mStCoUqqq4xsNa1R",       # Processed_data/images
           "labels": "18zvJa0Hnc5E35gh60IGc_QmtbfdT_k7T",       # Processed_data/labels
           "raw_labels": "1HBeXlB-kw6lon977Dpcr5eTZullpETKB"}   # lables/ (TCIA originals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--subset", default="images,labels,raw_labels")
    ap.add_argument("--ids", default="", help="comma list of patient numbers; empty = all")
    a = ap.parse_args()
    ids = {int(i) for i in a.ids.split(",") if i}
    s = requests.Session()
    for sub in a.subset.split(","):
        _, files = _parse_embedded_folder_view(sess=s, folder_id=FOLDERS[sub], verify=True, timeout=60)
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
        for fid, name, _ in files:
            num = int("".join(c for c in name if c.isdigit()))
            dst = os.path.join(a.out, sub, name)
            if (ids and num not in ids) or os.path.exists(dst):
                continue
            for attempt in range(5):
                try:
                    gdown.download(id=fid, output=dst + ".part", quiet=True)
                    os.replace(dst + ".part", dst); print("ok", sub, name, flush=True); break
                except Exception as e:
                    print("retry", sub, name, attempt, e, flush=True); time.sleep(30 * (attempt + 1))
            else:
                print("FAILED", sub, name, flush=True)

if __name__ == "__main__":
    main()

"""Download NIH Pancreas-CT DICOM series from the public IDC S3 bucket (anonymous HTTPS).
Usage: python tools/fetch_idc.py --meta data/metadata.csv --out data/dicom --ids 1,2,3"""
import argparse, csv, os, re, time, requests
from concurrent.futures import ThreadPoolExecutor

BUCKET = "https://idc-open-data.s3.amazonaws.com"

def list_keys(prefix):
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, **({"continuation-token": token} if token else {})}
        xml = requests.get(BUCKET + "/", params=params, timeout=60).text
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)

def fetch(key, dst, tries=6):
    if os.path.exists(dst):
        return
    for t in range(tries):
        try:
            r = requests.get(f"{BUCKET}/{key}", timeout=120); r.raise_for_status(); break
        except requests.RequestException:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))
    with open(dst + ".part", "wb") as f:
        f.write(r.content)
    os.replace(dst + ".part", dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True); ap.add_argument("--out", default="data/dicom")
    ap.add_argument("--ids", default=""); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ids = {int(i) for i in a.ids.split(",") if i}
    rows = [r for r in csv.DictReader(open(a.meta)) if not ids or int(r["PatientID"][-4:]) in ids]
    for r in rows:
        prefix = r["DownloadURL"].replace("s3://idc-open-data/", "").rstrip("*")
        d = os.path.join(a.out, r["PatientID"]); os.makedirs(d, exist_ok=True)
        keys = list_keys(prefix)
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda k: fetch(k, os.path.join(d, os.path.basename(k))), keys))
        print(r["PatientID"], len(keys), "files", flush=True)

if __name__ == "__main__":
    main()

"""
evaluate.py - Final full-volume (1 mm) evaluation of a trained run on the held-out TEST set (or val).

Loads <run_dir>/config.json + best.pth, predicts every case with the run's own inference mode (R0: blind crop +
paste back; R1-R4: sliding window), thresholds at 0.5 and writes
  <run_dir>/<split>_per_patient.csv   Dice, Jaccard, precision, recall, HD95, ASSD, volumes, V_pred/V_GT
  <run_dir>/summary.json (test) or val_summary.json  mean / SD / median / min / max + inf counts
  <run_dir>/preds_<split>/<PID>.npz   bit-packed binary prediction (for figures / volumetry)
This is the ONLY script that reads the test patients (rule 2).
  python -m src.evaluate --run_dir results/R0_baseline --data /tmp/prep [--split test]
"""
import argparse, csv, json, os, time

import numpy as np
import torch

from src.config import EVAL_THRESHOLD, SW_OVERLAP_TEST, TARGET_SPACING
from src.data.dataset import load_case
from src.inference import predict_logits
from src.metrics import aggregate_metrics, compute_all_metrics
from src.models.model import build_model

COLS = ["patient", "dice", "jaccard", "precision", "recall", "hd95", "assd", "vol_pred_ml", "vol_gt_ml", "v_ratio",
        "empty_pred", "seconds"]


def load_run(run_dir, device):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    model = build_model(**cfg).to(device)
    ck = torch.load(os.path.join(run_dir, "best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    return model, cfg, ck


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True); p.add_argument("--data", required=True)
    p.add_argument("--splits", default="splits/patient_splits.json")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--overlap", type=float, default=SW_OVERLAP_TEST)
    p.add_argument("--ids", default="", help="optional comma list of patient IDs (debug only)")
    a = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ck = load_run(a.run_dir, device)
    sp = json.load(open(a.splits))
    ids = sp["test"] if a.split == "test" else sp["folds"][cfg.get("fold", 0)]["val"]
    if a.ids:
        ids = [i for i in a.ids.split(",")]
    rows = []
    for pid in ids:
        t = time.time()
        img, lbl = load_case(a.data, pid)
        prob = torch.sigmoid(predict_logits(model, img, cfg["mode"], cfg["roi"], a.overlap, cfg["crop"], device))
        mask = (prob > EVAL_THRESHOLD).cpu().numpy()
        m = compute_all_metrics(mask, np.asarray(lbl) > 0, TARGET_SPACING)
        os.makedirs(os.path.join(a.run_dir, f"preds_{a.split}"), exist_ok=True)
        np.savez_compressed(os.path.join(a.run_dir, f"preds_{a.split}", f"{pid}.npz"), bits=np.packbits(mask), shape=mask.shape)
        rows.append(dict(patient=pid, **m, seconds=round(time.time() - t, 2)))
        print(f"{pid} Dice {m['dice']:.4f} J {m['jaccard']:.4f} P {m['precision']:.3f} R {m['recall']:.3f} "
              f"HD95 {m['hd95']:.2f} ASSD {m['assd']:.2f} V {m['vol_pred_ml']:.1f}/{m['vol_gt_ml']:.1f} mL", flush=True)
    with open(os.path.join(a.run_dir, f"{a.split}_per_patient.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    summary = dict(run=cfg.get("run_name"), split=a.split, best_step=ck.get("step"), best_val_dice=ck.get("val_dice"),
                   overlap=a.overlap, mode=cfg["mode"], threshold=EVAL_THRESHOLD, **aggregate_metrics(rows))
    json.dump(summary, open(os.path.join(a.run_dir, "summary.json" if a.split == "test" else "val_summary.json"), "w"),
              indent=1)
    print(f"{a.split}: Dice {summary['dice']['mean']:.4f} +- {summary['dice']['sd']:.4f} (n={summary['n_cases']}, "
          f"empty preds {summary['n_empty_pred']})")


def load_pred(run_dir, pid, split="test"):
    z = np.load(os.path.join(run_dir, f"preds_{split}", f"{pid}.npz"))
    return np.unpackbits(z["bits"])[:int(np.prod(z["shape"]))].reshape(z["shape"]).astype(bool)


if __name__ == "__main__":
    main()

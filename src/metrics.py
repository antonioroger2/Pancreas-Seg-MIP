"""
metrics.py - Spacing-aware volume metrics on the FULL 1 mm volume.

Dice, Jaccard (= D/(2-D)), precision, recall, HD95 and ASSD (mm), volumes (mL) and V_pred/V_GT.
Empty prediction (or empty GT): HD95/ASSD = inf, flagged in `empty_pred`; aggregate_metrics counts these
separately and never drops a case silently.
"""
import numpy as np
from scipy import ndimage

from src.config import EVAL_THRESHOLD, TARGET_SPACING


def _bin(x, thr):
    return x > thr if x.dtype != bool else x


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = np.count_nonzero(p & g); fp = np.count_nonzero(p & ~g); fn = np.count_nonzero(~p & g)
    denom = 2 * tp + fp + fn
    return dict(dice=1.0 if denom == 0 else 2 * tp / denom,
                jaccard=1.0 if denom == 0 else tp / (tp + fp + fn),
                precision=float("nan") if tp + fp == 0 else tp / (tp + fp),
                recall=float("nan") if tp + fn == 0 else tp / (tp + fn))


def surface_distances(pred: np.ndarray, gt: np.ndarray, spacing=TARGET_SPACING):
    """Distances (mm) from each surface voxel of pred to gt's surface region and vice versa.
    Computed on the union bounding box (+2 voxels): exact, because every voxel of both masks lies inside it."""
    p, g = pred.astype(bool), gt.astype(bool)
    idx = np.argwhere(p | g)
    lo, hi = np.maximum(idx.min(0) - 2, 0), np.minimum(idx.max(0) + 3, p.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    p, g = p[sl], g[sl]
    st = ndimage.generate_binary_structure(3, 1)
    sp_ = p & ~ndimage.binary_erosion(p, st, border_value=0)
    sg = g & ~ndimage.binary_erosion(g, st, border_value=0)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp_, sampling=spacing)
    return dt_g[sp_], dt_p[sg]


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray, spacing=TARGET_SPACING,
                        threshold: float = EVAL_THRESHOLD) -> dict:
    """pred: probabilities or a binary mask; gt: binary. Volumes use the voxel volume from `spacing`."""
    p, g = _bin(pred, threshold), gt.astype(bool)
    m = overlap_metrics(p, g)
    vox_ml = float(np.prod(spacing)) / 1000.0
    m.update(vol_pred_ml=np.count_nonzero(p) * vox_ml, vol_gt_ml=np.count_nonzero(g) * vox_ml,
             empty_pred=not p.any())
    m["v_ratio"] = m["vol_pred_ml"] / m["vol_gt_ml"] if m["vol_gt_ml"] > 0 else float("nan")
    if p.any() and g.any():
        d_pg, d_gp = surface_distances(p, g, spacing)
        m["hd95"] = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
        m["assd"] = float((d_pg.mean() + d_gp.mean()) / 2.0)     # medpy / Review-1 definition
    else:
        m["hd95"] = m["assd"] = 0.0 if not (p.any() or g.any()) else float("inf")
    return m


def aggregate_metrics(rows: list) -> dict:
    """mean / SD (ddof=1) / median / min / max per metric over cases; non-finite values excluded from the
    statistics but counted in n_inf_<metric>."""
    out = dict(n_cases=len(rows), n_empty_pred=int(sum(bool(r["empty_pred"]) for r in rows)))
    for k in ("dice", "jaccard", "precision", "recall", "hd95", "assd", "vol_pred_ml", "vol_gt_ml", "v_ratio"):
        v = np.array([r[k] for r in rows], dtype=float)
        f = v[np.isfinite(v)]
        out[k] = dict(mean=float(f.mean()) if len(f) else float("nan"),
                      sd=float(f.std(ddof=1)) if len(f) > 1 else float("nan"),
                      median=float(np.median(f)) if len(f) else float("nan"),
                      min=float(f.min()) if len(f) else float("nan"), max=float(f.max()) if len(f) else float("nan"),
                      n=int(len(f)))
        out[f"n_inf_{k}"] = int(np.isinf(v).sum())
    return out


# single-metric wrappers (kept for the original test suite)
def compute_dice(pred, target, threshold=EVAL_THRESHOLD):
    return compute_all_metrics(pred, target > 0.5, threshold=threshold)["dice"]


def compute_assd(pred, target, voxel_spacing=TARGET_SPACING, threshold=EVAL_THRESHOLD):
    return compute_all_metrics(pred, target > 0.5, voxel_spacing, threshold)["assd"]


def compute_hd95(pred, target, voxel_spacing=TARGET_SPACING, threshold=EVAL_THRESHOLD):
    return compute_all_metrics(pred, target > 0.5, voxel_spacing, threshold)["hd95"]

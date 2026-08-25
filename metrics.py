"""
metrics.py - Volume-Level Evaluation Metrics for 3D Segmentation.

Implements: Volumetric DSC, ASSD (mm), HD95 (mm)

Reference:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942
"""

import numpy as np
from scipy import ndimage
from config import EVAL_THRESHOLD, TARGET_SPACING


def compute_dice(pred: np.ndarray, target: np.ndarray, threshold: float = EVAL_THRESHOLD) -> float:
    """Compute volumetric Dice Similarity Coefficient (DSC)."""
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)

    intersection = np.sum(pred_bin & target_bin)
    total = np.sum(pred_bin) + np.sum(target_bin)

    if total == 0:
        return 1.0
    return (2.0 * intersection) / total


def _get_surface_points(binary_mask: np.ndarray) -> np.ndarray:
    """Extract surface voxel coordinates from a binary mask."""
    if np.sum(binary_mask) == 0:
        return np.array([]).reshape(0, 3)

    struct = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(binary_mask, structure=struct, border_value=0)
    surface = binary_mask.astype(bool) & ~eroded
    return np.argwhere(surface)


def _surface_distances(pred_bin: np.ndarray, target_bin: np.ndarray,
                       voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> tuple:
    """Compute symmetric surface distances between two binary masks."""
    pred_surface = _get_surface_points(pred_bin)
    target_surface = _get_surface_points(target_bin)

    if len(pred_surface) == 0 or len(target_surface) == 0:
        return np.array([]), np.array([])

    target_distance_map = ndimage.distance_transform_edt(
        ~target_bin.astype(bool), sampling=voxel_spacing
    )
    pred_distance_map = ndimage.distance_transform_edt(
        ~pred_bin.astype(bool), sampling=voxel_spacing
    )

    dist_pred_to_target = target_distance_map[tuple(pred_surface.T)]
    dist_target_to_pred = pred_distance_map[tuple(target_surface.T)]

    return dist_pred_to_target, dist_target_to_pred


def compute_assd(pred: np.ndarray, target: np.ndarray,
                 voxel_spacing: tuple = TARGET_SPACING,
                 threshold: float = EVAL_THRESHOLD) -> float:
    """Compute Average Symmetric Surface Distance (ASSD) in mm."""
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)

    if np.sum(pred_bin) == 0 and np.sum(target_bin) == 0:
        return 0.0
    if np.sum(pred_bin) == 0 or np.sum(target_bin) == 0:
        return float('inf')

    d_p2t, d_t2p = _surface_distances(pred_bin, target_bin, voxel_spacing)
    if len(d_p2t) == 0 or len(d_t2p) == 0:
        return float('inf')

    return float((np.mean(d_p2t) + np.mean(d_t2p)) / 2.0)


def compute_hd95(pred: np.ndarray, target: np.ndarray,
                 voxel_spacing: tuple = TARGET_SPACING,
                 threshold: float = EVAL_THRESHOLD) -> float:
    """Compute 95th Percentile Hausdorff Distance (HD95) in mm."""
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)

    if np.sum(pred_bin) == 0 and np.sum(target_bin) == 0:
        return 0.0
    if np.sum(pred_bin) == 0 or np.sum(target_bin) == 0:
        return float('inf')

    d_p2t, d_t2p = _surface_distances(pred_bin, target_bin, voxel_spacing)
    if len(d_p2t) == 0 or len(d_t2p) == 0:
        return float('inf')

    return float(max(np.percentile(d_p2t, 95), np.percentile(d_t2p, 95)))


def compute_all_metrics(pred: np.ndarray, target: np.ndarray,
                        voxel_spacing: tuple = TARGET_SPACING,
                        threshold: float = EVAL_THRESHOLD) -> dict:
    """Compute all evaluation metrics for a single volume."""
    return {
        'dice': compute_dice(pred, target, threshold),
        'assd': compute_assd(pred, target, voxel_spacing, threshold),
        'hd95': compute_hd95(pred, target, voxel_spacing, threshold),
    }


def aggregate_metrics(metrics_list: list) -> dict:
    """Aggregate per-patient metrics into mean +/- std."""
    dices = [m['dice'] for m in metrics_list]
    assds = [m['assd'] for m in metrics_list if m['assd'] != float('inf')]
    hd95s = [m['hd95'] for m in metrics_list if m['hd95'] != float('inf')]

    return {
        'dice_mean': np.mean(dices) if dices else 0.0,
        'dice_std': np.std(dices) if dices else 0.0,
        'dice_min': np.min(dices) if dices else 0.0,
        'dice_max': np.max(dices) if dices else 0.0,
        'assd_mean': np.mean(assds) if assds else float('inf'),
        'assd_std': np.std(assds) if assds else 0.0,
        'hd95_mean': np.mean(hd95s) if hd95s else float('inf'),
        'hd95_std': np.std(hd95s) if hd95s else 0.0,
        'n_valid': len(dices),
        'n_inf_assd': sum(1 for m in metrics_list if m['assd'] == float('inf')),
        'n_inf_hd95': sum(1 for m in metrics_list if m['hd95'] == float('inf')),
    }

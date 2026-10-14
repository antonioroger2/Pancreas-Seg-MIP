"""
preprocessing.py - Paper preprocessing (Section 2.2) on FULL volumes + blind crop / paste-back helpers.

  1. Apply the T1 orientation rule to the label array (results/alignment/alignment_summary.json)
  2. Resample to 1x1x1 mm (linear CT, nearest label; label resampled on the image's own grid)
  3. Clip HU to [-100, 240] and scale to [0, 1]
No cropping and no ground-truth use here. R0's 224x224x128 crop is taken later with blind_center_crop(),
which depends only on the volume shape; paste_back() maps a crop prediction back into the full volume.

Array convention: (H, W, D) = (y, x, z), i.e. SimpleITK's (z, y, x) array transposed with (1, 2, 0).
"""

import numpy as np
import SimpleITK as sitk

from src.config import HU_MIN, HU_MAX, TARGET_SPACING


def clip_hu(volume: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    return np.clip(volume, hu_min, hu_max)


def normalize_to_01(volume: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    return ((clip_hu(volume, hu_min, hu_max) - hu_min) / (hu_max - hu_min)).astype(np.float32)


def resample_volume_sitk(img, target_spacing=TARGET_SPACING, is_label=False, reference=None):
    """Resample to target spacing, preserving physical extent. If reference is given, use its grid exactly."""
    f = sitk.ResampleImageFilter()
    if reference is not None:
        f.SetReferenceImage(reference)
    else:
        sp = np.array(img.GetSpacing()); sz = np.array(img.GetSize()); tsp = np.array(target_spacing, float)
        f.SetOutputSpacing(tsp.tolist()); f.SetSize(np.round(sz * sp / tsp).astype(int).tolist())
        f.SetOutputDirection(img.GetDirection()); f.SetOutputOrigin(img.GetOrigin())
    f.SetTransform(sitk.Transform())
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    f.SetDefaultPixelValue(0 if is_label else -1024.0)
    return f.Execute(img)


def apply_label_rule(lbl_zyx: np.ndarray, rule: str) -> np.ndarray:
    """Apply a T1 variant name 'T{t}_fx{a}_fy{b}_fz{c}' to a (z, y, x) label array."""
    t, fx, fy, fz = (int(c) for c in rule.replace("T", "").replace("_fx", " ").replace("_fy", " ")
                     .replace("_fz", " ").split())
    out = np.swapaxes(lbl_zyx, 1, 2) if t else lbl_zyx
    axes = [a for a, f in ((2, fx), (1, fy), (0, fz)) if f]
    return np.ascontiguousarray(np.flip(out, axes) if axes else out)


def preprocess_case(img: sitk.Image, lbl: sitk.Image = None, rule: str = "T0_fx0_fy0_fz0"):
    """Full-volume preprocessing. Returns image (H,W,D) float32 in [0,1], label (H,W,D) uint8 or None, sitk grid."""
    img = sitk.Cast(img, sitk.sitkFloat32)
    res_img = resample_volume_sitk(img)
    image = normalize_to_01(np.transpose(sitk.GetArrayFromImage(res_img), (1, 2, 0)))
    label = None
    if lbl is not None:
        arr = apply_label_rule(sitk.GetArrayFromImage(lbl) > 0, rule).astype(np.uint8)
        fixed = sitk.GetImageFromArray(arr)
        fixed.CopyInformation(img)          # label voxels are defined on the image's voxel grid (T1)
        res_lbl = resample_volume_sitk(fixed, is_label=True, reference=res_img)
        label = (np.transpose(sitk.GetArrayFromImage(res_lbl), (1, 2, 0)) > 0).astype(np.uint8)
    return image, label, res_img


def blind_center_crop(shape, size):
    """Start offsets of a geometric-centre crop of `size` from a volume of `shape` (negative = zero padding).
    Depends ONLY on the shape: no ground truth, no image content."""
    return tuple((s - t) // 2 if s >= t else -((t - s) // 2) for s, t in zip(shape, size))


def crop_at(vol: np.ndarray, offsets, size, fill=0):
    """Crop vol[o:o+size] per axis, zero-filling anything outside the volume."""
    out = np.full(size, fill, dtype=vol.dtype)
    src = tuple(slice(max(o, 0), min(o + t, s)) for o, t, s in zip(offsets, size, vol.shape))
    dst = tuple(slice(sl.start - o, sl.stop - o) for sl, o in zip(src, offsets))
    out[dst] = vol[src]
    return out


def paste_back(crop: np.ndarray, offsets, full_shape, fill=0):
    """Inverse of crop_at: place a crop prediction into a full-size array; everything outside = fill."""
    out = np.full(full_shape, fill, dtype=crop.dtype)
    dst = tuple(slice(max(o, 0), min(o + t, s)) for o, t, s in zip(offsets, crop.shape, full_shape))
    src = tuple(slice(sl.start - o, sl.stop - o) for sl, o in zip(dst, offsets))
    out[dst] = crop[src]
    return out


def center_crop_or_pad(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Blind geometric-centre crop / zero-pad (shape-only)."""
    return crop_at(volume, blind_center_crop(volume.shape, target_shape), target_shape)

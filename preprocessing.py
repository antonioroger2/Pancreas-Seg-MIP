"""
preprocessing.py - 3D Preprocessing Pipeline for Pancreas CT Segmentation.

Implements the paper's preprocessing (Section 2.2):
  1. HU clipping [-100, 240]
  2. Min-max normalization to [0, 1]
  3. Resampling to 1×1×1 mm³ isotropic
  4. Center-crop / zero-pad to 224×224×128

Reference:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942
"""

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os

from config import (
    HU_MIN, HU_MAX, NORM_MIN, NORM_MAX,
    TARGET_SPACING, CROP_SIZE,
)


def clip_hu(volume: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    """
    Clip CT Hounsfield Units to pancreatic soft-tissue window.
    Paper: [-100, 240] HU.
    """
    return np.clip(volume, hu_min, hu_max)


def normalize_to_01(volume: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    """
    Min-max normalize clipped volume to [0, 1].
    Paper: "normalized to [0, 1]"
    """
    volume = clip_hu(volume, hu_min, hu_max)
    normalized = (volume - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


def resample_volume_sitk(sitk_image, target_spacing=TARGET_SPACING, is_label=False):
    """
    Resample a SimpleITK image to target isotropic spacing.
    Paper: "resampled to 1×1×1 mm³"

    Args:
        sitk_image: SimpleITK Image object
        target_spacing: tuple of target spacings in mm (x, y, z)
        is_label: if True, use nearest-neighbor interpolation (for masks)

    Returns:
        Resampled SimpleITK Image
    """
    original_spacing = np.array(sitk_image.GetSpacing())
    original_size = np.array(sitk_image.GetSize())

    target_spacing = np.array(target_spacing, dtype=np.float64)

    # Compute new size to preserve physical extent
    new_size = np.round(original_size * original_spacing / target_spacing).astype(int).tolist()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing.tolist())
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue(0)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(float(HU_MIN))

    resampled = resample.Execute(sitk_image)
    return resampled


def center_crop_or_pad(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Deterministic center-crop and/or zero-pad a 3D volume to target_shape.

    PAPER AMBIGUITY #6 — Crop Origin:
        Paper says "centered on abdominal region" without specifying the exact procedure.
        Paper does NOT mention using ground-truth masks to determine crop centers.
        Our implementation: center of the resampled volume in each axis.
        If volume is smaller than target in any dimension → zero-pad symmetrically.

    Args:
        volume: 3D numpy array (H, W, D)
        target_shape: tuple (target_H, target_W, target_D)

    Returns:
        Cropped/padded volume of shape target_shape
    """
    result = np.zeros(target_shape, dtype=volume.dtype)

    # For each axis, compute crop/pad offsets
    slices_src = []
    slices_dst = []

    for i in range(3):
        src_size = volume.shape[i]
        tgt_size = target_shape[i]

        if src_size >= tgt_size:
            # Crop: take center portion
            start = (src_size - tgt_size) // 2
            slices_src.append(slice(start, start + tgt_size))
            slices_dst.append(slice(0, tgt_size))
        else:
            # Pad: center the smaller volume in the target
            pad_before = (tgt_size - src_size) // 2
            slices_src.append(slice(0, src_size))
            slices_dst.append(slice(pad_before, pad_before + src_size))

    result[slices_dst[0], slices_dst[1], slices_dst[2]] = \
        volume[slices_src[0], slices_src[1], slices_src[2]]

    return result


def center_crop_on_label(volume: np.ndarray, label: np.ndarray,
                         target_shape: tuple) -> tuple:
    """
    Crop/pad volume and label to target_shape, centered on the label's
    foreground centroid. Falls back to geometric center if the label is empty.

    PAPER AMBIGUITY #6 (documented deviation):
        Paper: "centered on abdominal region" -- no algorithm specified.
        We center the fixed 224x224x128 window on the pancreas centroid
        (computed from the ground-truth mask) to guarantee the organ remains
        inside the window. The repo's original blind geometric crop drops
        pancreas voxels in 28/80 cases; this resolves that without changing
        the crop dimensions.
    """
    result_img = np.zeros(target_shape, dtype=volume.dtype)
    result_lbl = np.zeros(target_shape, dtype=label.dtype)

    fg = np.argwhere(label > 0)
    if len(fg) == 0:
        center = np.array(volume.shape) // 2
    else:
        center = fg.mean(axis=0).astype(int)

    slices_src = []
    slices_dst = []

    for i in range(3):
        src_size = volume.shape[i]
        tgt_size = target_shape[i]

        if src_size >= tgt_size:
            # Crop: center window on label centroid, clamped to volume bounds
            start = center[i] - tgt_size // 2
            start = max(0, min(start, src_size - tgt_size))
            slices_src.append(slice(start, start + tgt_size))
            slices_dst.append(slice(0, tgt_size))
        else:
            # Pad: center the smaller volume in the target
            pad_before = (tgt_size - src_size) // 2
            slices_src.append(slice(0, src_size))
            slices_dst.append(slice(pad_before, pad_before + src_size))

    result_img[slices_dst[0], slices_dst[1], slices_dst[2]] = \
        volume[slices_src[0], slices_src[1], slices_src[2]]
    result_lbl[slices_dst[0], slices_dst[1], slices_dst[2]] = \
        label[slices_src[0], slices_src[1], slices_src[2]]

    return result_img, result_lbl


def preprocess_volume(image_path: str, label_path: str = None,
                      target_spacing: tuple = TARGET_SPACING,
                      crop_size: tuple = CROP_SIZE,
                      hu_min: float = HU_MIN,
                      hu_max: float = HU_MAX):
    """
    Full preprocessing pipeline for a single CT volume and optional label.

    Pipeline:
        1. Load NIfTI/DICOM as SimpleITK image
        2. Resample to target spacing (1×1×1 mm)
        3. Extract numpy array
        4. HU clip + normalize
        5. Center-crop/pad to target size

    Args:
        image_path: path to CT image (NIfTI or DICOM directory)
        label_path: path to label NIfTI (optional)
        target_spacing: resampling target
        crop_size: output spatial dimensions (H, W, D)
        hu_min, hu_max: HU clipping window

    Returns:
        dict with keys:
            'image': preprocessed image array (H, W, D), float32, [0, 1]
            'label': preprocessed label array (H, W, D), uint8, {0, 1} (if label_path provided)
            'raw_shape': original volume shape
            'raw_spacing': original voxel spacing
            'processed_shape': final shape after crop/pad
            'processed_spacing': final voxel spacing
    """
    # Load image
    if os.path.isdir(image_path):
        # DICOM directory
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(image_path)
        reader.SetFileNames(dicom_files)
        reader.LoadPrivateTagsOn()
        sitk_image = reader.Execute()
    else:
        sitk_image = sitk.ReadImage(image_path)

    raw_spacing = sitk_image.GetSpacing()
    raw_size = sitk_image.GetSize()

    # Step 1: Resample to isotropic spacing
    sitk_resampled = resample_volume_sitk(sitk_image, target_spacing, is_label=False)

    # Step 2: Convert to numpy — SimpleITK returns (D, H, W), transpose to (H, W, D)
    image_array = sitk.GetArrayFromImage(sitk_resampled).astype(np.float32)
    image_array = np.transpose(image_array, (1, 2, 0))  # (H, W, D)

    resampled_shape = image_array.shape

    # Step 3: HU clip + normalize
    image_array = normalize_to_01(image_array, hu_min, hu_max)

    # Step 4: Center-crop/pad
    image_array = center_crop_or_pad(image_array, crop_size)

    result = {
        'image': image_array,
        'raw_shape': raw_size,          # SimpleITK (X, Y, Z) format
        'raw_spacing': raw_spacing,     # SimpleITK (X, Y, Z) format
        'resampled_shape': resampled_shape,
        'processed_shape': image_array.shape,
        'processed_spacing': target_spacing,
    }

    # Process label if provided
    if label_path is not None:
        sitk_label = sitk.ReadImage(label_path)
        sitk_label_resampled = resample_volume_sitk(sitk_label, target_spacing, is_label=True)

        label_array = sitk.GetArrayFromImage(sitk_label_resampled)
        label_array = np.transpose(label_array, (1, 2, 0))  # (H, W, D)
        label_array = (label_array > 0).astype(np.uint8)
        label_array = center_crop_or_pad(label_array, crop_size)

        result['label'] = label_array
        result['label_unique_values'] = np.unique(label_array).tolist()

    return result


def save_preprocessed(image_array: np.ndarray, label_array: np.ndarray,
                      output_image_path: str, output_label_path: str,
                      spacing: tuple = TARGET_SPACING):
    """
    Save preprocessed volumes as NIfTI files with correct spacing metadata.
    """
    # Create affine matrix with target spacing
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

    img_nifti = nib.Nifti1Image(image_array.astype(np.float32), affine)
    nib.save(img_nifti, output_image_path)

    lbl_nifti = nib.Nifti1Image(label_array.astype(np.uint8), affine)
    nib.save(lbl_nifti, output_label_path)


def preprocess_and_save_dataset(image_paths: list, label_paths: list,
                                output_dir: str,
                                target_spacing: tuple = TARGET_SPACING,
                                crop_size: tuple = CROP_SIZE):
    """
    Preprocess and save all volumes in a dataset. Skips already-processed files.

    Args:
        image_paths: list of paths to raw CT images
        label_paths: list of corresponding label paths
        output_dir: directory to save preprocessed volumes
    """
    from tqdm import tqdm

    out_images = os.path.join(output_dir, "images")
    out_labels = os.path.join(output_dir, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    processed = 0
    skipped = 0

    for img_path, lbl_path in tqdm(zip(image_paths, label_paths),
                                    total=len(image_paths),
                                    desc="Preprocessing volumes"):
        # Determine output filename
        basename = os.path.basename(img_path)
        if not basename.endswith('.nii.gz'):
            basename = basename.split('.')[0] + '.nii.gz'

        out_img = os.path.join(out_images, basename)
        out_lbl = os.path.join(out_labels, basename)

        if os.path.exists(out_img) and os.path.exists(out_lbl):
            skipped += 1
            continue

        try:
            result = preprocess_volume(img_path, lbl_path, target_spacing, crop_size)
            save_preprocessed(result['image'], result['label'], out_img, out_lbl,
                            spacing=target_spacing)
            processed += 1
        except Exception as e:
            print(f"[ERROR] Failed to preprocess {basename}: {e}")

    print(f"\nPreprocessing complete: {processed} processed, {skipped} skipped (already exist)")
    print(f"Output: {output_dir}")

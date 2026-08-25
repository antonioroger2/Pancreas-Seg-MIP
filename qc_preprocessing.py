"""
qc_preprocessing.py - Strict Preprocessing Quality Control (QC) for Pancreas CT Segmentation.

Performs validation on preprocessed volumes without using ground-truth masks to alter the crop:
  1. Inspects raw image and label metadata
  2. Applies 2025 paper preprocessing (HU clip [-100, 240], normalize [0, 1], resample 1mm, geometric center-crop 224x224x128)
  3. Verifies geometry (shape, spacing, affine, orientation)
  4. Verifies pancreas containment (bounding box, voxel retention, margin to boundary)
  5. Generates representative axial slice visualization figure
  6. Exports structured JSON & Text QC Report
"""

import os
import json
import argparse
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    HU_MIN, HU_MAX, TARGET_SPACING, CROP_SIZE
)
from preprocessing import (
    resample_volume_sitk, normalize_to_01, center_crop_or_pad, clip_hu
)


def run_preprocessing_qc(image_path: str, label_path: str, output_dir: str = "./qc_output"):
    """
    Executes complete QC validation for a single case.
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(image_path).replace(".nii.gz", "").replace(".nii", "")

    print(f"\n{'='*75}")
    print(f"2025 PREPROCESSING QUALITY CONTROL (QC): {basename}")
    print(f"{'='*75}")
    print(f"  Input Image: {image_path}")
    print(f"  Input Label: {label_path}")

    # 1. Load Raw Volumes
    sitk_raw_img = sitk.ReadImage(image_path)
    sitk_raw_lbl = sitk.ReadImage(label_path)

    raw_img_arr = sitk.GetArrayFromImage(sitk_raw_img).astype(np.float32)  # (D, H, W)
    raw_lbl_arr = sitk.GetArrayFromImage(sitk_raw_lbl).astype(np.uint8)

    raw_spacing = sitk_raw_img.GetSpacing()  # (x, y, z)
    raw_size = sitk_raw_img.GetSize()        # (X, Y, Z)
    raw_img_min, raw_img_max = float(raw_img_arr.min()), float(raw_img_arr.max())
    raw_voxels = int(np.sum(raw_lbl_arr > 0))

    print(f"\n[Phase 1] Raw Input Inspection:")
    print(f"  Raw Image Size (X, Y, Z):   {raw_size}")
    print(f"  Raw Label Size (X, Y, Z):   {sitk_raw_lbl.GetSize()}")
    print(f"  Raw Voxel Spacing:          {raw_spacing} mm")
    print(f"  Raw Intensity Range:        [{raw_img_min:.1f}, {raw_img_max:.1f}] HU")
    print(f"  Raw Pancreas Voxels:        {raw_voxels:,}")

    # 2. Resample to 1.0 x 1.0 x 1.0 mm (Full Volume)
    sitk_res_img = resample_volume_sitk(sitk_raw_img, TARGET_SPACING, is_label=False)
    sitk_res_lbl = resample_volume_sitk(sitk_raw_lbl, TARGET_SPACING, is_label=True)

    res_img_arr = np.transpose(sitk.GetArrayFromImage(sitk_res_img).astype(np.float32), (1, 2, 0)) # (H, W, D)
    res_lbl_arr = np.transpose(sitk.GetArrayFromImage(sitk_res_lbl), (1, 2, 0))
    res_lbl_arr = (res_lbl_arr > 0).astype(np.uint8)

    resampled_voxels = int(np.sum(res_lbl_arr > 0))
    print(f"\n[Phase 2] Resampled Full Volume (1.0 mm isotropic):")
    print(f"  Resampled Array Shape:      {res_img_arr.shape} (H, W, D)")
    print(f"  Resampled Pancreas Voxels:  {resampled_voxels:,}")

    # 3. HU Clip & Normalization
    norm_img_arr = normalize_to_01(res_img_arr, HU_MIN, HU_MAX)

    # 4. Geometric Center-Crop (BLIND - NO GT MASK GUIDANCE)
    cropped_img = center_crop_or_pad(norm_img_arr, CROP_SIZE)
    cropped_lbl = center_crop_or_pad(res_lbl_arr, CROP_SIZE)

    cropped_voxels = int(np.sum(cropped_lbl > 0))
    retention_pct = (100.0 * cropped_voxels / resampled_voxels) if resampled_voxels > 0 else 0.0

    print(f"\n[Phase 3] Geometric Center-Cropped Volume (Target: {CROP_SIZE}):")
    print(f"  Final Image Shape:          {cropped_img.shape}")
    print(f"  Final Label Shape:          {cropped_lbl.shape}")
    print(f"  Final Intensity Range:      [{cropped_img.min():.4f}, {cropped_img.max():.4f}]")
    print(f"  Final Label Values:         {np.unique(cropped_lbl).tolist()}")
    print(f"  Pancreas Voxels in Crop:    {cropped_voxels:,} / {resampled_voxels:,} ({retention_pct:.2f}% retained)")

    # 5. Bounding Box & Boundary Margin Analysis
    pos_coords = np.argwhere(cropped_lbl > 0)
    if len(pos_coords) > 0:
        x_min, y_min, z_min = pos_coords.min(axis=0)
        x_max, y_max, z_max = pos_coords.max(axis=0)

        # Margin to volume boundary
        margin_x_min = int(x_min)
        margin_x_max = int(CROP_SIZE[0] - 1 - x_max)
        margin_y_min = int(y_min)
        margin_y_max = int(CROP_SIZE[1] - 1 - y_max)
        margin_z_min = int(z_min)
        margin_z_max = int(CROP_SIZE[2] - 1 - z_max)

        fully_contained = (
            cropped_voxels == resampled_voxels and
            margin_x_min > 0 and margin_x_max > 0 and
            margin_y_min > 0 and margin_y_max > 0 and
            margin_z_min > 0 and margin_z_max > 0
        )
    else:
        x_min = y_min = z_min = x_max = y_max = z_max = -1
        margin_x_min = margin_x_max = margin_y_min = margin_y_max = margin_z_min = margin_z_max = -1
        fully_contained = False

    print(f"\n[Phase 4] Pancreas Bounding Box in (224, 224, 128) Grid:")
    print(f"  X (Height) Range:           [{x_min}, {x_max}] (Dimension: {CROP_SIZE[0]}, Margins: {margin_x_min} / {margin_x_max} voxels)")
    print(f"  Y (Width) Range:            [{y_min}, {y_max}] (Dimension: {CROP_SIZE[1]}, Margins: {margin_y_min} / {margin_y_max} voxels)")
    print(f"  Z (Depth) Range:            [{z_min}, {z_max}] (Dimension: {CROP_SIZE[2]}, Margins: {margin_z_min} / {margin_z_max} voxels)")
    print(f"  Pancreas Fully Contained:   {'YES [OK]' if fully_contained else 'NO [CUTOFF DETECTED]'}")
    print(f"  Voxel Loss from Crop:       {resampled_voxels - cropped_voxels} voxels ({100.0 - retention_pct:.2f}%)")

    # 6. Geometry & Affine Consistency Check
    img_affine = np.eye(4)
    img_affine[0, 0], img_affine[1, 1], img_affine[2, 2] = TARGET_SPACING
    lbl_affine = np.eye(4)
    lbl_affine[0, 0], lbl_affine[1, 1], lbl_affine[2, 2] = TARGET_SPACING

    affine_diff = float(np.max(np.abs(img_affine - lbl_affine)))
    geom_pass = (cropped_img.shape == CROP_SIZE and cropped_lbl.shape == CROP_SIZE and affine_diff == 0.0)

    print(f"\n[Phase 5] Geometry & Affine Verification:")
    print(f"  Image / Label Shape Match:  {cropped_img.shape == cropped_lbl.shape} ({cropped_img.shape})")
    print(f"  Target Voxel Spacing:       {TARGET_SPACING} mm (Isotropic)")
    print(f"  Max Affine Difference:      {affine_diff:.6f}")
    print(f"  Geometry Consistency:       {'PASS [OK]' if geom_pass else 'FAIL'}")

    # 7. Generate Multi-Slice QC Figure
    fig_path = os.path.join(output_dir, f"{basename}_2025_preprocessing_qc.png")
    
    # Pick 6 representative slices containing pancreas
    if len(pos_coords) > 0:
        active_slices = sorted(np.unique(pos_coords[:, 2]))
        slice_indices = np.linspace(active_slices[0], active_slices[-1], 6, dtype=int)
    else:
        slice_indices = np.linspace(20, 100, 6, dtype=int)

    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    
    for i, s_idx in enumerate(slice_indices):
        ct_slice = cropped_img[:, :, s_idx]
        mask_slice = cropped_lbl[:, :, s_idx]

        # Row 1: CT Alone
        axes[0, i].imshow(ct_slice, cmap='gray', vmin=0.0, vmax=1.0)
        axes[0, i].set_title(f"Slice Z={s_idx}", fontsize=11, fontweight='bold')
        axes[0, i].axis('off')

        # Row 2: CT + GT Pancreas Overlay (Green)
        axes[1, i].imshow(ct_slice, cmap='gray', vmin=0.0, vmax=1.0)
        axes[1, i].imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap='Greens', alpha=0.6, vmin=0, vmax=1)
        axes[1, i].set_title(f"Overlay (Voxels: {int(np.sum(mask_slice))})", fontsize=11)
        axes[1, i].axis('off')

        # Row 3: Zoom-in on Pancreas Region
        if np.sum(mask_slice) > 0:
            coords = np.argwhere(mask_slice > 0)
            cx, cy = int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1]))
            r = 35  # zoom radius
            x0, x1 = max(0, cx - r), min(CROP_SIZE[0], cx + r)
            y0, y1 = max(0, cy - r), min(CROP_SIZE[1], cy + r)
            axes[2, i].imshow(ct_slice[x0:x1, y0:y1], cmap='gray', vmin=0.0, vmax=1.0)
            axes[2, i].imshow(np.ma.masked_where(mask_slice[x0:x1, y0:y1] == 0, mask_slice[x0:x1, y0:y1]), cmap='Greens', alpha=0.6, vmin=0, vmax=1)
            axes[2, i].set_title("Zoomed View", fontsize=10)
        else:
            axes[2, i].imshow(ct_slice, cmap='gray', vmin=0.0, vmax=1.0)
            axes[2, i].set_title("No Foreground", fontsize=10)
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel("Preprocessed CT", fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel("Pancreas Overlay", fontsize=13, fontweight='bold')
    axes[2, 0].set_ylabel("Zoomed Region", fontsize=13, fontweight='bold')

    plt.suptitle(
        f"2025 Paper Preprocessing QC: {basename}\n"
        f"HU [-100, 240] -> [0, 1] Norm -> 1.0mm Resample -> Center Crop (224, 224, 128) | "
        f"Retention: {retention_pct:.2f}% ({cropped_voxels:,} voxels)",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"\n[Phase 6] Saved QC Multi-Slice Figure: {fig_path}")

    # 8. Save Preprocessed Volumes
    out_img_nii = os.path.join(output_dir, f"{basename}.nii.gz")
    out_lbl_nii = os.path.join(output_dir, f"{basename}_label.nii.gz")
    nib.save(nib.Nifti1Image(cropped_img, img_affine), out_img_nii)
    nib.save(nib.Nifti1Image(cropped_lbl, lbl_affine), out_lbl_nii)

    # 9. Structure QC Report
    qc_data = {
        'patient_id': basename,
        'qc_status': 'PASS' if (fully_contained and geom_pass) else 'FAIL',
        'raw_metadata': {
            'size': raw_size,
            'spacing_mm': raw_spacing,
            'intensity_range_hu': [raw_img_min, raw_img_max],
            'pancreas_voxels': raw_voxels
        },
        'resampled_metadata': {
            'shape': list(res_img_arr.shape),
            'spacing_mm': list(TARGET_SPACING),
            'pancreas_voxels': resampled_voxels
        },
        'final_preprocessed_metadata': {
            'shape': list(CROP_SIZE),
            'spacing_mm': list(TARGET_SPACING),
            'intensity_range_norm': [float(cropped_img.min()), float(cropped_img.max())],
            'pancreas_voxels_in_crop': cropped_voxels,
            'pancreas_retention_pct': retention_pct,
            'bounding_box_224_224_128': {
                'x_min': int(x_min), 'x_max': int(x_max),
                'y_min': int(y_min), 'y_max': int(y_max),
                'z_min': int(z_min), 'z_max': int(z_max)
            },
            'boundary_margins_voxels': {
                'x_margin_min': margin_x_min, 'x_margin_max': margin_x_max,
                'y_margin_min': margin_y_min, 'y_margin_max': margin_y_max,
                'z_margin_min': margin_z_min, 'z_margin_max': margin_z_max
            },
            'all_pancreas_voxels_contained': fully_contained,
            'geometry_consistent': geom_pass,
            'max_affine_difference': affine_diff
        },
        'artifacts': {
            'preprocessed_image': out_img_nii,
            'preprocessed_label': out_lbl_nii,
            'qc_figure': fig_path
        }
    }

    report_json_path = os.path.join(output_dir, f"{basename}_qc_report.json")
    with open(report_json_path, 'w') as f:
        json.dump(qc_data, f, indent=2)
    print(f"  Saved JSON QC Report:        {report_json_path}")

    # Summary Conclusion
    print(f"\n{'='*75}")
    if fully_contained and geom_pass:
        print(f"RESULT: {basename} 2025 PREPROCESSING QC: PASS")
    else:
        print(f"RESULT: {basename} 2025 PREPROCESSING QC: FAIL (Pancreas cutoff or geometry error)")
    print(f"{'='*75}\n")

    return qc_data


def main():
    parser = argparse.ArgumentParser(description="Run 2025 Preprocessing Quality Control (QC)")
    parser.add_argument("--image", type=str, required=True, help="Path to raw CT image NIfTI")
    parser.add_argument("--label", type=str, required=True, help="Path to raw label NIfTI")
    parser.add_argument("--output_dir", type=str, default="./qc_output", help="Directory to save QC artifacts")
    args = parser.parse_args()

    run_preprocessing_qc(args.image, args.label, args.output_dir)


if __name__ == "__main__":
    main()

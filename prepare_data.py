"""
prepare_data.py - DICOM to NIfTI Converter for TCIA Pancreas-CT Dataset

Converts the nested DICOM folder layout downloaded from TCIA into the flat
NIfTI directory structure expected by train.py.

TCIA Download Layout (nested):
    raw_dicom_dir/
        PANCREAS_0001/
            <StudyInstanceUID>/
                <SeriesInstanceUID>/
                    000001.dcm
                    000002.dcm
                    ...
        PANCREAS_0002/
            ...

Annotations Layout (from TCIA "Image Analyses" download):
    raw_labels_dir/
        label0001.nii.gz   (or PANCREAS_0001.nii.gz)
        label0002.nii.gz
        ...

Output (what train.py expects):
    output_dir/
        images/
            PANCREAS_0001.nii.gz
            PANCREAS_0002.nii.gz
            ...
        labels/
            PANCREAS_0001.nii.gz
            PANCREAS_0002.nii.gz
            ...

Usage (Google Colab):
    !python prepare_data.py \
        --dicom_dir ./raw_dicom \
        --labels_dir ./raw_labels \
        --output_dir ./data

Usage (only DICOM conversion, masks already in NIfTI):
    !python prepare_data.py \
        --dicom_dir ./raw_dicom \
        --labels_dir ./annotations \
        --output_dir ./data
"""

import os
import re
import glob
import argparse
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def find_dicom_series(patient_dir):
    """
    Walk a patient's nested directory tree and locate the DICOM series folder.
    TCIA nests as: PatientID / StudyInstanceUID / SeriesInstanceUID / *.dcm
    Returns the path to the deepest directory containing .dcm files.
    """
    for root, dirs, files in os.walk(patient_dir):
        dcm_files = [f for f in files if f.endswith(".dcm") or f.endswith(".DCM")]
        if len(dcm_files) > 1:  # A valid CT series has many slices
            return root
    return None


def convert_dicom_to_nifti(dicom_series_path, output_nifti_path):
    """
    Reads a DICOM series from a directory and writes a single 3D NIfTI volume.
    Uses SimpleITK which correctly handles slice ordering, spacing, and orientation.
    """
    reader = sitk.ImageSeriesReader()
    dicom_filenames = reader.GetGDCMSeriesFileNames(dicom_series_path)

    if len(dicom_filenames) == 0:
        print(f"  [WARNING] No DICOM files found in: {dicom_series_path}")
        return False

    reader.SetFileNames(dicom_filenames)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    try:
        image = reader.Execute()
    except RuntimeError as e:
        print(f"  [ERROR] Failed to read DICOM series: {e}")
        return False

    sitk.WriteImage(image, output_nifti_path, useCompression=True)
    return True


def extract_patient_id(folder_name):
    """
    Extracts the numeric patient ID from folder names like 'PANCREAS_0001'.
    Returns the zero-padded string (e.g. '0001').
    """
    match = re.search(r"(\d+)", folder_name)
    if match:
        return match.group(1).zfill(4)
    return None


def find_matching_label(patient_id, labels_dir):
    """
    Finds the matching ground-truth label file for a given patient ID.
    Handles common TCIA naming conventions:
        - label0001.nii.gz
        - PANCREAS_0001.nii.gz
        - label_0001.nii.gz
    """
    patterns = [
        f"label{patient_id}.nii*",
        f"label_{patient_id}.nii*",
        f"PANCREAS_{patient_id}.nii*",
        f"pancreas_{patient_id}.nii*",
        f"*{patient_id}*.nii*",
    ]

    for pattern in patterns:
        matches = glob.glob(os.path.join(labels_dir, pattern))
        if matches:
            return matches[0]
    return None


def prepare_dataset(dicom_dir, labels_dir, output_dir):
    """
    Main pipeline: discovers all patient DICOM folders, converts each to NIfTI,
    pairs with the matching annotation mask, and writes both into the flat
    output_dir/images/ and output_dir/labels/ structure.
    """
    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Discover patient folders (top-level directories in dicom_dir)
    patient_folders = sorted([
        d for d in os.listdir(dicom_dir)
        if os.path.isdir(os.path.join(dicom_dir, d)) and not d.startswith(".")
    ])

    if len(patient_folders) == 0:
        print(f"[ERROR] No patient folders found in: {dicom_dir}")
        return

    print(f"Found {len(patient_folders)} patient folders in: {dicom_dir}")
    print(f"Looking for annotations in: {labels_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    converted = 0
    skipped = 0
    missing_labels = []

    for folder_name in tqdm(patient_folders, desc="Converting DICOM to NIfTI"):
        patient_path = os.path.join(dicom_dir, folder_name)
        patient_id = extract_patient_id(folder_name)

        if patient_id is None:
            print(f"  [SKIP] Cannot parse patient ID from: {folder_name}")
            skipped += 1
            continue

        output_name = f"PANCREAS_{patient_id}.nii.gz"
        output_image_path = os.path.join(images_out, output_name)

        # Skip if already converted
        if os.path.exists(output_image_path):
            print(f"  [EXISTS] {output_name} already converted, skipping.")
            converted += 1
            continue

        # Find DICOM series inside nested structure
        series_path = find_dicom_series(patient_path)
        if series_path is None:
            print(f"  [SKIP] No DICOM series found for: {folder_name}")
            skipped += 1
            continue

        # Convert DICOM -> NIfTI
        success = convert_dicom_to_nifti(series_path, output_image_path)
        if not success:
            skipped += 1
            continue

        # Find and copy matching label
        if labels_dir and os.path.isdir(labels_dir):
            label_path = find_matching_label(patient_id, labels_dir)
            if label_path:
                output_label_path = os.path.join(labels_out, output_name)
                if not os.path.exists(output_label_path):
                    # Read and re-save to ensure consistent format
                    label_img = sitk.ReadImage(label_path)
                    sitk.WriteImage(label_img, output_label_path, useCompression=True)
            else:
                missing_labels.append(patient_id)

        converted += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"CONVERSION COMPLETE")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Total:     {len(patient_folders)}")
    print(f"\n  Images saved to: {images_out}")
    print(f"  Labels saved to: {labels_out}")

    num_images = len(glob.glob(os.path.join(images_out, "*.nii*")))
    num_labels = len(glob.glob(os.path.join(labels_out, "*.nii*")))
    print(f"\n  NIfTI images found: {num_images}")
    print(f"  NIfTI labels found: {num_labels}")

    if missing_labels:
        print(f"\n  [WARNING] Missing labels for {len(missing_labels)} patients: {missing_labels[:10]}...")
        print("  Make sure you downloaded the annotations from TCIA 'Image Analyses'.")

    if num_images != num_labels:
        print(f"\n  [WARNING] Image/Label count mismatch ({num_images} vs {num_labels}).")
        print("  The training script requires exactly 1 label per image.")

    if num_images > 0 and num_images == num_labels:
        print(f"\n  [OK] Dataset ready for training! Run:")
        print(f"    python train.py --data_dir {output_dir}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert TCIA Pancreas-CT DICOM downloads to NIfTI for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert DICOM scans and pair with downloaded annotation masks
  python prepare_data.py --dicom_dir ./raw_dicom --labels_dir ./raw_labels --output_dir ./data

  # Convert DICOM scans only (if you already have NIfTI labels elsewhere)
  python prepare_data.py --dicom_dir ./raw_dicom --labels_dir ./existing_labels --output_dir ./data

  # Google Colab example
  !python prepare_data.py \\
      --dicom_dir /content/drive/MyDrive/TCIA_Download \\
      --labels_dir /content/drive/MyDrive/TCIA_Labels \\
      --output_dir ./data
        """
    )
    parser.add_argument("--dicom_dir", type=str, required=True,
                        help="Root directory containing patient DICOM folders (e.g. PANCREAS_0001/)")
    parser.add_argument("--labels_dir", type=str, required=True,
                        help="Directory containing ground-truth NIfTI label files (e.g. label0001.nii.gz)")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory (will create images/ and labels/ subdirectories)")

    args = parser.parse_args()

    if not os.path.isdir(args.dicom_dir):
        print(f"[ERROR] DICOM directory does not exist: {args.dicom_dir}")
        return
    if not os.path.isdir(args.labels_dir):
        print(f"[ERROR] Labels directory does not exist: {args.labels_dir}")
        return

    prepare_dataset(args.dicom_dir, args.labels_dir, args.output_dir)


if __name__ == "__main__":
    main()

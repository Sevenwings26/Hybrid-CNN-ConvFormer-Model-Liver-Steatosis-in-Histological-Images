# stain_normalization.py
import os
import cv2
import numpy as np
# import torch
import torchstain
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import csv
import shutil


# ============================
# CONFIGURATION
# ==============================
REFERENCE_IMAGE_PATH = r"C:\Users\wings\sevenwingsInc\01_clients\26-03_convFormer_ML+CNN\convFormer-model\steatosis_extracted\steatosis\training\1\208_0_9_56.png"

INPUT_CSV             = "dataset_validation_details.csv"
OUTPUT_DIR            = r"C:\Users\wings\sevenwingsInc\01_clients\26-03_convFormer_ML+CNN\convFormer-model\steatosis_normalized\training"  # update this

NORMALIZED_CSV        = "buildFiles/normalized_manifest.csv"
FAILED_CSV            = "buildFiles/normalization_failed.csv"

TARGET_SIZE           = (512, 512)   # resize after normalization


# ============================================================
# SETUP
# ============================================================
def setup_output_dirs(base_dir, grades=["0", "1", "2", "3"]):
    """
    Creates grade-separated output subfolders.
    Preserves label structure lost during consolidation.
    
    Output structure:
        steatosis_normalized/training/
            0/
            1/
            2/
            3/
    """
    for grade in grades:
        os.makedirs(os.path.join(base_dir, grade), exist_ok=True)
    print(f"Output directories ready under: {base_dir}")


def load_normalizer(reference_path):
    """
    Initializes the Macenko stain normalizer and fits
    it to the reference image.

    Args:
        reference_path (str): Path to reference H&E tile.

    Returns:
        normalizer: Fitted MacenkoNormalizer instance.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)  # torchstain expects [0,255]
    ])

    ref_img   = Image.open(reference_path).convert("RGB")
    ref_tensor = transform(ref_img)

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    normalizer.fit(ref_tensor)

    print(f"Normalizer fitted to reference: {reference_path}")
    return normalizer, transform


# ============================================================
# NORMALIZATION PIPELINE
# ============================================================
def normalize_dataset(input_csv, output_dir, reference_path):
    """
    Reads validated image manifest, applies Macenko stain
    normalization to each tile, resizes to TARGET_SIZE,
    and saves to grade-separated output folders.

    Failures are logged separately for review.
    """
    setup_output_dirs(output_dir)
    normalizer, transform = load_normalizer(reference_path)

    normalized_rows = []
    failed_rows     = []


    # Load valid tiles from manifest
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            row for row in reader
            if row["status"] == "valid"
            and row["label"] in ["0", "1", "2", "3"]
        ]

    print(f"\nNormalizing {len(rows)} valid tiles...")

    for row in tqdm(rows, desc="Stain Normalizing"):
        src_path = row["file"]
        label    = row["label"]
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, label, filename)

        try:
            # --- Load & convert ---
            img    = Image.open(src_path).convert("RGB")
            tensor = transform(img)

            # --- Normalize ---
            norm_tensor, _, _ = normalizer.normalize(tensor, stains=False)

            # --- Safety: ensure correct shape [3, H, W] ---
            # torchstain can return unexpected shapes; 
            # squeeze away any extra dimensions
            if norm_tensor.dim() == 2:
                # Grayscale edge case — skip
                raise ValueError(
                    f"Unexpected 2D tensor shape: {norm_tensor.shape}"
                )

            # If shape is [H, W, 3] transpose to [3, H, W]
            if norm_tensor.shape[-1] == 3 and norm_tensor.dim() == 3:
                norm_tensor = norm_tensor.permute(2, 0, 1)

            # Clamp values to valid uint8 range
            norm_tensor = norm_tensor.clamp(0, 255)

            # Verify final shape before conversion
            if norm_tensor.shape[0] != 3:
                raise ValueError(
                    f"Unexpected channel count: {norm_tensor.shape}"
                )

            # --- Convert to numpy HWC uint8 ---
            norm_array = norm_tensor.numpy().astype(np.uint8)
            norm_array = np.transpose(norm_array, (1, 2, 0))  # CHW → HWC

            # --- RGB → BGR for OpenCV ---
            norm_bgr = cv2.cvtColor(norm_array, cv2.COLOR_RGB2BGR)

            # --- Resize to target ---
            resized = cv2.resize(
                norm_bgr, TARGET_SIZE,
                interpolation=cv2.INTER_LINEAR
            )

            # --- Save ---
            cv2.imwrite(dst_path, resized)
            normalized_rows.append({
                "file": src_path,
                "label": label,
                "source_file": filename,
                "status": "normalized"
            })

        except Exception as e:
            failed_rows.append({
                "file":   src_path,
                "label":  label,
                "reason": str(e)
            })

    # --- Save manifests ---
    _save_normalized_csv(normalized_rows, NORMALIZED_CSV)
    _save_failed_csv(failed_rows, FAILED_CSV)

    # --- Final report ---
    print(f"\n=== NORMALIZATION COMPLETE ===")
    print(f"Successfully normalized : {len(normalized_rows)}")
    print(f"Failed                  : {len(failed_rows)}")
    print(f"Output saved to         : {output_dir}")

    return normalized_rows, failed_rows


# ============================================================
# SAVE OUTPUTS
# ============================================================
def _save_normalized_csv(rows, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "label", "source_file", "status"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Normalized manifest saved: {filename}")


def _save_failed_csv(rows, filename):
    if not rows:
        print("No failures recorded.")
        return
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "label", "reason"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Failed tiles logged to  : {filename}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    normalized, failed = normalize_dataset(
        input_csv      = INPUT_CSV,
        output_dir     = OUTPUT_DIR,
        reference_path = REFERENCE_IMAGE_PATH
    )

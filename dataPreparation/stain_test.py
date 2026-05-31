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
OUTPUT_DIR            = r"C:\Users\wings\sevenwingsInc\01_clients\26-03_convFormer_ML+CNN\convFormer-model\steatosis_normalized\test"  # update this

NORMALIZED_CSV        = "buildFiles/normalized_manifest.csv"
FAILED_CSV            = "buildFiles/normalization_failed.csv"

TARGET_SIZE           = (512, 512)   # resize after normalization




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


def test_normalization(input_csv, output_dir, reference_path, n=5):
    """
    Test normalization on first N tiles before full run.
    """
    setup_output_dirs(output_dir)
    normalizer, transform = load_normalizer(reference_path)

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            row for row in reader
            if row["status"] == "valid"
            and row["label"] in ["0", "1", "2", "3"]
        ][:n]

    print(f"\nTesting on {len(rows)} tiles...")
    passed = 0
    for row in rows:
        src_path = row["file"]
        label    = row["label"]
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, label, f"TEST_{filename}")

        try:
            img    = Image.open(src_path).convert("RGB")
            tensor = transform(img)
            norm_tensor, _, _ = normalizer.normalize(tensor, stains=False)

            print(f"  Raw tensor shape : {norm_tensor.shape}")
            print(f"  Raw tensor dtype : {norm_tensor.dtype}")

            if norm_tensor.shape[-1] == 3 and norm_tensor.dim() == 3:
                norm_tensor = norm_tensor.permute(2, 0, 1)

            norm_tensor = norm_tensor.clamp(0, 255)
            norm_array  = norm_tensor.numpy().astype(np.uint8)
            norm_array  = np.transpose(norm_array, (1, 2, 0))
            norm_bgr    = cv2.cvtColor(norm_array, cv2.COLOR_RGB2BGR)
            resized     = cv2.resize(norm_bgr, TARGET_SIZE)

            cv2.imwrite(dst_path, resized)
            print(f"  ✅ PASSED — saved to {dst_path}")
            passed += 1

        except Exception as e:
            print(f"  ❌ FAILED — {os.path.basename(src_path)}: {e}")

    print(f"\nTest result: {passed}/{n} passed")


# Then in __main__, comment out the full run and test first:
if __name__ == "__main__":
    test_normalization(
        input_csv      = INPUT_CSV,
        output_dir     = OUTPUT_DIR,
        reference_path = REFERENCE_IMAGE_PATH,
        n=5
    )

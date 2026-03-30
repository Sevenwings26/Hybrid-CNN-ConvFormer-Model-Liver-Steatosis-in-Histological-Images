import os
import cv2
import numpy as np
from tqdm import tqdm # to show progress bar during validation
import hashlib
from save_process import save_summary_to_csv

DATASET_PATH = r"HEPASS_algorithm_dataset\Original Images\TRAIN_extracted\TRAIN"  # path to dataset folder
VALID_EXT = [".png", ".jpg", ".jpeg", ".tif"]

MIN_SIZE = 224
TISSUE_THRESHOLD = 0.5   # % of image that must contain tissue
FAT_MIN_AREA = 50        # minimum pixel area for fat vacuole

def is_valid_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return False
        h, w = img.shape[:2]
        return h >= MIN_SIZE and w >= MIN_SIZE
    except:
        return False

def compute_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

def tissue_coverage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    tissue_ratio = np.sum(thresh > 0) / thresh.size
    return tissue_ratio

def detect_fat_vacuoles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect bright (white) circular-ish regions
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fat_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > FAT_MIN_AREA:
            fat_regions.append(area)

    return len(fat_regions), fat_regions

# VALIDATE DATASET
def validate_dataset(dataset_path):
    seen_hashes = set()

    report = {
        "valid": [],
        "invalid": [],
        "low_tissue": [],
        "no_fat_detected": [],
        "duplicates": []
    }

    for file in tqdm(os.listdir(dataset_path)):
        if not any(file.lower().endswith(ext) for ext in VALID_EXT):
            continue

        path = os.path.join(dataset_path, file)

        # 1. Basic validation
        if not is_valid_image(path):
            report["invalid"].append(file)
            continue

        img = cv2.imread(path)

        # 2. Duplicate check
        img_hash = compute_hash(img)
        if img_hash in seen_hashes:
            report["duplicates"].append(file)
            continue
        seen_hashes.add(img_hash)

        # 3. Tissue coverage
        tissue_ratio = tissue_coverage(img)
        if tissue_ratio < TISSUE_THRESHOLD:
            report["low_tissue"].append(file)
            continue

        # 4. Fat detection
        fat_count, fat_areas = detect_fat_vacuoles(img)
        if fat_count == 0:
            report["no_fat_detected"].append(file)
        else:
            report["valid"].append({
                "file": file,
                "fat_count": fat_count,
                "avg_fat_area": np.mean(fat_areas)
            })

    return report

# SAVE REPORT TO JSON ===
import json
def save_report(report, filename):
    with open(filename, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Report saved to: {filename}")


if __name__ == "__main__":
    report = validate_dataset(DATASET_PATH)

    print("\n=== DATASET REPORT ===")
    for k, v in report.items():
        print(f"{k}: {len(v)}")
        
    # save_report(report, r"convFormer-model\dataset_validation_report.json")
    save_report(report, "dataset_validation_report.json")
    save_summary_to_csv(report, "dataset_summary.csv")

    
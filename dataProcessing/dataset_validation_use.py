import os
import hashlib
import cv2
import numpy as np
from tqdm import tqdm # to show progress bar during validation
import csv
import numpy as np
import cv2
from tqdm import tqdm
from save_process import save_summary_to_csv, save_validation_csv

DATASET_PATH = r"C:\Users\wings\sevenwingsInc\01_clients\26-03_convFormer_ML+CNN\convFormer-model\steatosis_extracted\steatosis\training"   # Updated path

VALID_EXT = [".png", ".jpg", ".jpeg", ".tif"]

MIN_SIZE = 224
TISSUE_THRESHOLD = 0.5   # % of image that must contain tissue
FAT_MIN_AREA = 50        # minimum pixel area for fat vacuole
TISSUE_THRESHOLD_DEFAULT = 0.30   # relaxed from 0.50
TISSUE_THRESHOLD_BY_GRADE = {
    "0": 0.50,   # normal — expect dense tissue
    "1": 0.40,   # mild steatosis
    "2": 0.30,   # moderate steatosis
    "3": 0.15,   # severe — vacuoles dominate, threshold must be low
}


def get_all_image_paths(dataset_path):
    """
    Recursively scans all folders under dataset_path
    and returns a list of valid image file paths.
    """
    image_paths = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in VALID_EXT):
                image_paths.append(os.path.join(root, file))

    return image_paths


def is_valid_image(path):
    """
    Validates that an image can be successfully loaded and
    meets the minimum size requirements.

    Validation checks:
        1. Image file is readable by OpenCV.
        2. Image dimensions are greater than or equal to
           MIN_SIZE.

    Args:
        path (str):
            Full path to the image file.

    Returns:
        bool:
            True if the image is valid.
            False if the image is corrupt, unreadable,
            or below the minimum size threshold.
    """
    try:
        img = cv2.imread(path)

        if img is None:
            return False

        h, w = img.shape[:2]

        return h >= MIN_SIZE and w >= MIN_SIZE

    except Exception:
        return False


def compute_hash(image):
    """
    Generates a unique MD5 hash for an image.

    This hash is used to identify duplicate images within the dataset. Images with identical pixel content will produce the same hash value.

    Args:
        image (numpy.ndarray):
            Image loaded using OpenCV.

    Returns:
        str:
            MD5 hash string representing the image.
    """
    return hashlib.md5(image.tobytes()).hexdigest()


def tissue_coverage_steatosis(image, grade=None):
    """
    HSV-based tissue coverage estimation.
    Uses grade-aware thresholds because severe steatosis
    (Grade 3) tiles are dominated by fat vacuoles which
    appear as low-saturation white regions — structurally
    similar to background in colorimetric terms.
    
    Returns:
        tuple: (tissue_ratio, threshold_used)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    background_mask = saturation < 10
    background_ratio = np.sum(background_mask) / background_mask.size
    tissue_ratio = 1.0 - background_ratio
    
    threshold = TISSUE_THRESHOLD_BY_GRADE.get(
        str(grade), 
        TISSUE_THRESHOLD_DEFAULT
    )
    return tissue_ratio, threshold

# def tissue_coverage(image):
#     """
#     Estimates the proportion of tissue present in a histopathology image.

#     The image is converted to grayscale and thresholded to separate tissue regions from bright background regions. The resulting ratio provides a simple measure of tissue coverage.

#     Args:
#         image (numpy.ndarray):
#             Input histopathology image.

#     Returns:
#         float:
#             Tissue coverage ratio between 0 and 1.

#             Example:
#                 0.85 = 85% tissue coverage
#                 0.15 = Mostly background
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
#     tissue_ratio = np.sum(thresh > 0) / thresh.size
#     return tissue_ratio


# def tissue_coverage_steatosis(image):
#     """
#     Revised tissue coverage for steatosis slides.
#     Fat vacuoles (bright white) are valid tissue content,
#     not background. We exclude only pure white background
#     at the image border level, not vacuole white.
    
#     Strategy: use saturation channel from HSV.
#     Real background has near-zero saturation.
#     Tissue AND fat vacuoles both have structure.
#     """
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     saturation = hsv[:, :, 1]
    
#     # Background is truly colorless (saturation < 10)
#     # Tissue and vacuoles both have some saturation structure
#     background_mask = saturation < 10
#     background_ratio = np.sum(background_mask) / background_mask.size
#     tissue_ratio = 1.0 - background_ratio
    
#     return tissue_ratio


def detect_fat_vacuoles(image):
    """
    Detects potential fat vacuoles in a histopathology
    image using a simple image-processing heuristic.

    Fat vacuoles often appear as bright circular or
    near-circular empty regions within tissue sections.
    This function identifies bright regions above a
    predefined threshold and filters them by area.

    NOTE:
        This method provides an approximate estimate
        and should not be considered a clinical-grade
        steatosis detection algorithm.

    Processing Steps:
        1. Convert image to grayscale.
        2. Threshold bright regions.
        3. Extract contours.
        4. Filter contours by minimum area.
        5. Count remaining candidate vacuoles.

    Args:
        image (numpy.ndarray):
            Input histopathology image.

    Returns:
        tuple:
            (
                fat_count,
                fat_regions
            )

            fat_count (int):
                Number of detected candidate vacuoles.

            fat_regions (list):
                Area of each detected vacuole region.
    """
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


def validate_dataset(dataset_path):
    """
    Validate all images in the dataset and generate
    a detailed validation report.

    Returns:
        dict: Validation report
    """

    seen_hashes = set()

    report = {
        "valid": [],
        "invalid": [],
        "low_tissue": [],
        "no_fat_detected": [],
        "duplicates": [],
        "excluded": []
    }

    csv_rows = []
    image_paths = get_all_image_paths(dataset_path)

    for path in tqdm(image_paths, desc="Validating Images"):

        IGNORE_GRADES = {"ignore"}  # expand if needed

        # In validate_dataset(), after extracting label:
        label = os.path.basename(os.path.dirname(path))
        if label.lower() in IGNORE_GRADES:
            report["excluded"].append(path)  # add "excluded" key to report
            csv_rows.append({
                "file": path,
                "label": label,
                "status": "excluded",
                "tissue_ratio": None,
                "fat_count": None,
                "avg_fat_area": None
            })
            continue
        img = cv2.imread(path)

        # -----------------------------
        # 2. Duplicate Detection
        # -----------------------------
        img_hash = compute_hash(img)
        if img_hash in seen_hashes:
            report["duplicates"].append(path)
            csv_rows.append({
                "file": path,
                "label": label,
                "status": "duplicate",
                "tissue_ratio": None,
                "fat_count": None,
                "avg_fat_area": None
            })

            continue

        seen_hashes.add(img_hash)

        # -----------------------------
        # 3. Tissue Coverage (grade-aware)
        # -----------------------------
        tissue_ratio, tissue_threshold = tissue_coverage_steatosis(img, grade=label)
        
        if tissue_ratio < tissue_threshold:
            report["low_tissue"].append(path)
            csv_rows.append({
                "file": path,
                "label": label,
                "status": "low_tissue",
                "tissue_ratio": round(tissue_ratio, 4),
                "fat_count": None,
                "avg_fat_area": None,
                "threshold_used": tissue_threshold   # add this field
            })
            continue

        # -----------------------------
        # 4. Fat Vacuole Detection
        # -----------------------------
        fat_count, fat_areas = detect_fat_vacuoles(img)
        avg_fat_area = (
            round(float(np.mean(fat_areas)), 2)
            if fat_areas else 0
        )
        if fat_count == 0:
            # Grade_0 (normal tissue) legitimately has no fat.
            # Reclassify as valid instead of no_fat_detected.
            if label == "0":
                report["valid"].append({
                    "file": path,
                    "label": label,
                    "tissue_ratio": tissue_ratio,
                    "fat_count": 0,
                    "avg_fat_area": 0
                })
                csv_rows.append({
                    "file": path,
                    "label": label,
                    "status": "valid",
                    "tissue_ratio": round(tissue_ratio, 4),
                    "fat_count": 0,
                    "avg_fat_area": 0,
                    "threshold_used": tissue_threshold
                })
            else:
                # Non-zero grades with no fat detected are 
                # genuinely suspicious — keep excluded
                report["no_fat_detected"].append(path)
                csv_rows.append({
                    "file": path,
                    "label": label,
                    "status": "no_fat_detected",
                    "tissue_ratio": round(tissue_ratio, 4),
                    "fat_count": 0,
                    "avg_fat_area": 0,
                    "threshold_used": tissue_threshold
                })
        else:
            report["valid"].append({
                "file": path,
                "label": label,
                "tissue_ratio": tissue_ratio,
                "fat_count": fat_count,
                "avg_fat_area": avg_fat_area
            })
            csv_rows.append({
                "file": path,
                "label": label,
                "status": "valid",
                "tissue_ratio": round(tissue_ratio, 4),
                "fat_count": fat_count,
                "avg_fat_area": avg_fat_area,
                "threshold_used": tissue_threshold
            })

    # Save detailed CSV report
    save_validation_csv(csv_rows)

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
        
    save_report(report, "dataset_validation_report1.json")
    save_summary_to_csv(report)

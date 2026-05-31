# augmentation.py

import os
import cv2
import numpy as np
import csv
import random
from tqdm import tqdm
from PIL import Image, ImageEnhance
import json
from scipy.ndimage import gaussian_filter

# ============================================================
# CONFIGURATION
# ============================================================
SPLIT_MANIFEST      = "buildFiles/split_manifest.csv"
AUGMENTED_OUTPUT    = r"C:\Users\wings\sevenwingsInc\01_clients\26-03_convFormer_ML+CNN\convFormer-model\steatosis_augmented"
AUGMENTED_MANIFEST  = "buildFiles/augmented_manifest.csv"
AUGMENTED_SUMMARY   = "buildFiles/augmented_summary.csv"

AUGMENTATION_FACTOR = 4        # generate N augmented copies per tile
TARGET_SIZE         = (256, 256)  # final spatial resolution per §3.5
RANDOM_SEED         = 42
GRADES              = ["0", "1", "2", "3"]

# Color jitter ranges (aligned with §3.5)
BRIGHTNESS_RANGE    = (0.8, 1.2)   # ±20%
CONTRAST_RANGE      = (0.8, 1.2)   # ±20%
SATURATION_RANGE    = (0.9, 1.1)   # ±10%


# ============================================================
# SETUP
# ============================================================
def setup_augmented_dirs(base_dir):
    """
    Creates output structure:
        steatosis_augmented/
            train/  0/ 1/ 2/ 3/   ← augmented + originals
            val/    0/ 1/ 2/ 3/   ← originals only (copied)
            test/   0/ 1/ 2/ 3/   ← originals only (copied)
    """
    for split in ["train", "val", "test"]:
        for grade in GRADES:
            os.makedirs(
                os.path.join(base_dir, split, grade),
                exist_ok=True
            )
    print(f"Augmented directories ready: {base_dir}")


# ============================================================
# AUGMENTATION TRANSFORMS
# ============================================================
def random_rotation(img):
    """
    Rotates image by a random multiple of 90 degrees.
    Preserves full image content — no cropping artifacts.
    """
    angle = random.choice([0, 90, 180, 270])
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    else:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def random_flip(img):
    """
    Randomly applies horizontal flip, vertical flip,
    both, or neither.
    """
    choice = random.randint(0, 3)
    if choice == 0:
        return img
    elif choice == 1:
        return cv2.flip(img, 1)   # horizontal
    elif choice == 2:
        return cv2.flip(img, 0)   # vertical
    else:
        return cv2.flip(img, -1)  # both


def color_jitter(img):
    """
    Applies random brightness, contrast, and saturation
    adjustments to simulate residual stain variability
    that persists after Macenko normalization.
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Brightness
    bf = random.uniform(*BRIGHTNESS_RANGE)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(bf)

    # Contrast
    cf = random.uniform(*CONTRAST_RANGE)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(cf)

    # Saturation
    sf = random.uniform(*SATURATION_RANGE)
    pil_img = ImageEnhance.Color(pil_img).enhance(sf)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def elastic_deformation(img, alpha=34, sigma=4, seed=None):
    """
    Applies elastic deformation to simulate tissue
    distortion artifacts introduced during biopsy
    sectioning and slide preparation.

    Args:
        alpha (float): Deformation magnitude. Higher =
                       more distortion. Default 34.
        sigma (float): Smoothness of deformation field.
                       Higher = smoother warp. Default 4.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = img.shape[:2]

    # Random displacement fields
    dx = (np.random.rand(h, w) * 2 - 1)
    dy = (np.random.rand(h, w) * 2 - 1)

    # Smooth with Gaussian
    from scipy.ndimage import gaussian_filter
    dx = gaussian_filter(dx, sigma) * alpha
    dy = gaussian_filter(dy, sigma) * alpha

    # Build remap grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    # Apply remap
    deformed = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return deformed


def random_resized_crop(img, target_size=TARGET_SIZE):
    """
    Crops a random region (80-100% of image area) then
    resizes to target_size. Adds spatial diversity while
    preserving tissue content per §3.5.
    """
    h, w   = img.shape[:2]
    scale  = random.uniform(0.80, 1.0)
    new_h  = int(h * scale)
    new_w  = int(w * scale)
    top    = random.randint(0, h - new_h)
    left   = random.randint(0, w - new_w)
    cropped = img[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, target_size,
                      interpolation=cv2.INTER_LINEAR)


def augment_image(img):
    """
    Applies full augmentation pipeline to a single image.
    Order follows §3.5: rotation → flip → color → elastic → crop.
    """
    img = random_rotation(img)
    img = random_flip(img)
    img = color_jitter(img)
    img = elastic_deformation(img)
    img = random_resized_crop(img)
    return img


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_augmentation(split_manifest, output_dir, aug_factor):
    """
    Reads split manifest. For training tiles: saves original
    + N augmented copies. For val/test: copies original only.
    """
    setup_augmented_dirs(output_dir)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load manifest
    with open(split_manifest, "r", encoding="utf-8") as f:
        reader  = csv.DictReader(f)
        all_rows = list(reader)

    train_rows = [r for r in all_rows if r["split"] == "train"]
    other_rows = [r for r in all_rows if r["split"] != "train"]

    manifest_out = []
    grade_aug_counts = {g: 0 for g in GRADES}

    # --- TRAINING: original + augmented copies ---
    print(f"\nAugmenting {len(train_rows)} training tiles "
          f"(factor={aug_factor})...")

    for row in tqdm(train_rows, desc="Augmenting train"):
        src   = row["file"]
        grade = row["label"]
        fname = os.path.splitext(os.path.basename(src))[0]
        img   = cv2.imread(src)

        if img is None:
            continue

        # Save original (resized to TARGET_SIZE)
        orig_resized = cv2.resize(img, TARGET_SIZE)
        orig_dst = os.path.join(
            output_dir, "train", grade, f"{fname}_orig.png"
        )
        cv2.imwrite(orig_dst, orig_resized)
        manifest_out.append({
            "file": orig_dst, "label": grade,
            "split": "train", "aug_type": "original"
        })

        # Save N augmented copies
        for i in range(aug_factor):
            aug_img = augment_image(img.copy())
            aug_dst = os.path.join(
                output_dir, "train", grade, f"{fname}_aug{i+1}.png"
            )
            cv2.imwrite(aug_dst, aug_img)
            manifest_out.append({
                "file": aug_dst, "label": grade,
                "split": "train", "aug_type": f"aug_{i+1}"
            })
            grade_aug_counts[grade] += 1

    # --- VAL + TEST: copy originals only ---
    print(f"\nCopying {len(other_rows)} val/test tiles (no augmentation)...")

    for row in tqdm(other_rows, desc="Copying val/test"):
        src   = row["file"]
        grade = row["label"]
        split = row["split"]
        fname = os.path.basename(src)
        dst   = os.path.join(output_dir, split, grade, fname)

        img = cv2.imread(src)
        if img is None:
            continue
        resized = cv2.resize(img, TARGET_SIZE)
        cv2.imwrite(dst, resized)
        manifest_out.append({
            "file": dst, "label": grade,
            "split": split, "aug_type": "original"
        })

    # --- Save outputs ---
    _save_augmented_manifest(manifest_out)
    _save_augmented_summary(manifest_out, grade_aug_counts)

    return manifest_out


# ============================================================
# SAVE OUTPUTS
# ============================================================
def _save_augmented_manifest(rows, filename=AUGMENTED_MANIFEST):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "label", "split", "aug_type"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAugmented manifest saved: {filename}")


def _save_augmented_summary(rows, grade_aug_counts,
                            filename=AUGMENTED_SUMMARY):
    """
    Produces per-grade, per-split tile counts.
    Becomes Table 3 in your preliminary data.
    """
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["split"]][row["label"]] += 1

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Grade", "Clinical Meaning",
            "Train (orig+aug)", "Val", "Test", "Total"
        ])

        clinical = {
            "0": "Normal (<5% fat)",
            "1": "Mild (5-33% fat)",
            "2": "Moderate (34-66% fat)",
            "3": "Severe (>66% fat)"
        }

        totals = {"train": 0, "val": 0, "test": 0}
        for grade in GRADES:
            tr = counts["train"][grade]
            vl = counts["val"][grade]
            te = counts["test"][grade]
            writer.writerow([
                f"Grade_{grade}", clinical[grade],
                tr, vl, te, tr + vl + te
            ])
            totals["train"] += tr
            totals["val"]   += vl
            totals["test"]  += te

        grand = sum(totals.values())
        writer.writerow([
            "TOTAL", "",
            totals["train"], totals["val"],
            totals["test"], grand
        ])

    print(f"Augmented summary saved: {filename}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("=== AUGMENTATION PIPELINE ===\n")

    # Install scipy if needed: pip install scipy
    run_augmentation(
        split_manifest = SPLIT_MANIFEST,
        output_dir     = AUGMENTED_OUTPUT,
        aug_factor     = AUGMENTATION_FACTOR
    )

    print("\n=== AUGMENTATION COMPLETE ===")

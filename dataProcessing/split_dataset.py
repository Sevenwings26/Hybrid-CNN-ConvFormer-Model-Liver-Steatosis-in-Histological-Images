# split_dataset.py

import os
import shutil
import csv
import random
from collections import defaultdict
from tqdm import tqdm

# ======================
# CONFIGURATION
# ======================

NORMALIZED_MANIFEST  = "buildFiles/normalized_manifest.csv"  # output from stain_normalization.py
SPLIT_OUTPUT_DIR     = r"C:\Users\wings\sevenwingsInc\01_clients\26-03_convFormer_ML+CNN\convFormer-model\steatosis_split"

# 75% train, 10% val, 15% test — document this in §3.7.4
TRAIN_RATIO = 0.75   
VAL_RATIO   = 0.10
TEST_RATIO  = 0.15

RANDOM_SEED = 42  # reproducibility — document this in §3.7.4
GRADES      = ["0", "1", "2", "3"]


# ============================================================
# SETUP
# ============================================================
def setup_split_dirs(base_dir, splits=["train", "val", "test"]):
    """
    Creates output directory structure:
        steatosis_split/
            train/  0/ 1/ 2/ 3/
            val/    0/ 1/ 2/ 3/
            test/   0/ 1/ 2/ 3/
    """
    for split in splits:
        for grade in GRADES:
            os.makedirs(os.path.join(base_dir, split, grade), exist_ok=True)
    print(f"Split directories created under: {base_dir}")


# ============================================================
# LOAD MANIFEST
# ============================================================
def load_manifest_by_grade(manifest_csv):
    """
    Reads normalized_manifest.csv and groups
    file paths by grade label.

    Returns:
        dict: {grade: [file_path, ...]}
    """
    grade_files = defaultdict(list)

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if label in GRADES:
                grade_files[label].append(row["file"])

    for grade in GRADES:
        print(f"  Grade {grade}: {len(grade_files[grade])} tiles loaded")

    return grade_files


# ============================================================
# STRATIFIED SPLIT
# ============================================================
def stratified_split(grade_files, seed=RANDOM_SEED):
    """
    Performs stratified train/val/test split per grade.
    Shuffles within each grade before splitting to prevent
    ordering bias.

    Returns:
        dict: {
            "train": {grade: [paths]},
            "val":   {grade: [paths]},
            "test":  {grade: [paths]}
        }
    """
    random.seed(seed)
    splits = {"train": {}, "val": {}, "test": {}}

    for grade, files in grade_files.items():
        random.shuffle(files)
        n       = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        # test gets the remainder to ensure no tile is lost
        # n_test  = n - n_train - n_val

        splits["train"][grade] = files[:n_train]
        splits["val"][grade]   = files[n_train:n_train + n_val]
        splits["test"][grade]  = files[n_train + n_val:]

        print(f"  Grade {grade} → "
              f"train={len(splits['train'][grade])}  "
              f"val={len(splits['val'][grade])}  "
              f"test={len(splits['test'][grade])}")

    return splits


# ============================================================
# COPY FILES
# ============================================================
def copy_splits(splits, output_dir):
    """
    Copies tiles into their respective split/grade folders.
    Uses copy2 to preserve file metadata.
    """
    manifest_rows = []

    for split_name, grade_dict in splits.items():
        print(f"\nCopying {split_name} set...")
        for grade, files in grade_dict.items():
            dest_dir = os.path.join(output_dir, split_name, grade)
            for src_path in tqdm(files, desc=f"  {split_name}/grade_{grade}"):
                filename = os.path.basename(src_path)
                dst_path = os.path.join(dest_dir, filename)
                shutil.copy2(src_path, dst_path)
                manifest_rows.append({
                    "file":       dst_path,
                    "label":      grade,
                    "split":      split_name,
                    "source":     src_path
                })

    return manifest_rows


# ===================================
# SAVE SPLIT MANIFEST + SUMMARY
# ===================================
def save_split_manifest(rows, filename="buildFiles/split_manifest.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "label", "split", "source"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSplit manifest saved: {filename}")


def save_split_summary(splits, filename="buildFiles/split_summary.csv"):
    """
    Saves a human-readable split summary.
    This becomes Table 2 in your preliminary data.
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Grade", "Clinical Meaning",
            "Train", "Val", "Test", "Total"
        ])

        clinical = {
            "0": "Normal (<5% fat)",
            "1": "Mild (5-33% fat)",
            "2": "Moderate (34-66% fat)",
            "3": "Severe (>66% fat)"
        }

        totals = {"train": 0, "val": 0, "test": 0}

        for grade in GRADES:
            tr = len(splits["train"][grade])
            vl = len(splits["val"][grade])
            te = len(splits["test"][grade])
            writer.writerow([
                f"Grade_{grade}",
                clinical[grade],
                tr, vl, te, tr + vl + te
            ])
            totals["train"] += tr
            totals["val"]   += vl
            totals["test"]  += te

        # Totals row
        writer.writerow([
            "TOTAL", "",
            totals["train"],
            totals["val"],
            totals["test"],
            sum(totals.values())
        ])

        # Percentage row
        grand = sum(totals.values())
        writer.writerow([
            "PERCENTAGE", "",
            f"{round(totals['train']/grand*100, 1)}%",
            f"{round(totals['val']/grand*100, 1)}%",
            f"{round(totals['test']/grand*100, 1)}%",
            "100%"
        ])

    print(f"Split summary saved: {filename}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("=== DATASET SPLITTING ===\n")

    setup_split_dirs(SPLIT_OUTPUT_DIR)

    print("\nLoading manifest by grade...")
    grade_files = load_manifest_by_grade(NORMALIZED_MANIFEST)

    print("\nPerforming stratified split...")
    splits = stratified_split(grade_files)

    print("\nCopying files to split directories...")
    manifest_rows = copy_splits(splits, SPLIT_OUTPUT_DIR)

    save_split_manifest(manifest_rows)
    save_split_summary(splits)

    print("\n=== SPLIT COMPLETE ===")
    print(f"Total tiles distributed: {len(manifest_rows)}")


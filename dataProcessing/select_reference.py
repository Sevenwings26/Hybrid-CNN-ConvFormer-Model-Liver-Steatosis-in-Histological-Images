# select_reference.py
import cv2
import os
import csv
import numpy as np


def stain_variance(image):
    """
    Measures color diversity.
    Higher values generally indicate
    stronger and more representative staining.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.std(hsv[:, :, 1]) + np.std(hsv[:, :, 2])


def rank_reference_candidates(valid_csv, grade_filter="1", top_k=10):
    candidates = []
    with open(valid_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "valid":
                continue
            if row["label"] != grade_filter:
                continue

            path = row["file"]
            img = cv2.imread(path)

            if img is None:
                continue

            tissue_ratio = float(row["tissue_ratio"])
            fat_count = float(row["fat_count"])
            stain_score = stain_variance(img)

            score = (tissue_ratio * 0.6 + stain_score * 0.3 - fat_count * 0.1)

            candidates.append({
                "path": path,
                "score": score,
                "tissue": tissue_ratio,
                "fat_count": fat_count,
                "stain_score": stain_score
            })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    return candidates[:top_k]


def preview_ranked_candidates(valid_csv, grade_filter="1", top_k=10, output_txt="reference_candidates.txt"):

    candidates = rank_reference_candidates(valid_csv, grade_filter, top_k)

    # Save ranking to text file
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Macenko Reference Candidate Ranking\n")
        f.write("=" * 80 + "\n\n")
        for idx, c in enumerate(candidates, 1):
            f.write(f"Rank       : {idx}\n")
            f.write(f"Score      : {c['score']:.2f}\n")
            f.write(f"Tissue     : {c['tissue']:.3f}\n")
            f.write(f"Fat Count  : {c['fat_count']}\n")
            f.write(f"Path       : {c['path']}\n")
            f.write("-" * 80 + "\n")

    print(f"\nRanking saved to: {output_txt}")

    # Visual inspection
    for idx, c in enumerate(candidates, 1):

        print("\n---------------------")
        print(f"Rank: {idx}")
        print(f"Score: {c['score']:.2f}")
        print(f"Tissue: {c['tissue']:.3f}")
        print(f"Fat Count: {c['fat_count']}")
        print(f"Path: {c['path']}")

        img = cv2.imread(c["path"])

        cv2.imshow(
            f"Rank {idx}",
            img
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    preview_ranked_candidates("dataset_validation_details.csv")

    candidates = rank_reference_candidates("dataset_validation_details.csv")
    best_reference = candidates[0]["path"]

    with open("selected_reference.txt", "w", encoding="utf-8") as f:
        f.write(best_reference)

    print(f"\nSuggested reference image:\n{best_reference}")

# import cv2
# import os

# def preview_candidates(valid_csv, grade_filter="1", n=5):
#     """
#     Opens the first N valid tiles from a given grade
#     for visual inspection. Pick the clearest, most
#     representative one as your Macenko reference.
#     """
#     import csv
#     candidates = []

#     with open(valid_csv, "r") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             if row["label"] == grade_filter and row["status"] == "valid":
#                 candidates.append(row["file"])
#             if len(candidates) >= n:
#                 break

#     for path in candidates:
#         img = cv2.imread(path)
#         cv2.imshow(f"Candidate: {os.path.basename(path)}", img)
#         print(f"Path: {path}")
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     preview_candidates("dataset_validation_details.csv", grade_filter="1", n=5)
    

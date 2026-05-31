import csv
from collections import defaultdict


def save_summary_to_csv(report, filename="dataset_summary.csv"):
    """
    Saves a summary of the validation report to CSV.
    Includes:
        - Per-status counts (valid, invalid, etc.)
        - Total image count
        - Per-grade distribution across statuses
    """

    # --- Status counts ---
    summary = {
        "valid": len(report["valid"]),
        "invalid": len(report["invalid"]),
        "low_tissue": len(report["low_tissue"]),
        "no_fat_detected": len(report["no_fat_detected"]),
        "duplicates": len(report["duplicates"])
    }
    total = sum(summary.values())

    # --- Grade distribution from valid entries ---
    # valid entries are dicts; others are path strings
    grade_distribution = defaultdict(lambda: defaultdict(int))

    for status, entries in report.items():
        for entry in entries:
            if isinstance(entry, dict):
                label = entry.get("label", "unknown")
            else:
                # For path strings (invalid, low_tissue, duplicates)
                # Extract label from parent folder name
                import os
                label = os.path.basename(os.path.dirname(entry))
            grade_distribution[label][status] += 1

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # --- Section 1: Overall Summary ---
        writer.writerow(["=== OVERALL SUMMARY ==="])
        writer.writerow(["Category", "Count", "Percentage"])
        for category, count in summary.items():
            pct = round((count / total) * 100, 2) if total > 0 else 0
            writer.writerow([category, count, f"{pct}%"])
        writer.writerow(["TOTAL", total, "100%"])
        writer.writerow([])  # blank line

        # --- Section 2: Grade Distribution ---
        writer.writerow(["=== GRADE DISTRIBUTION ==="])
        all_statuses = [
            "valid", "low_tissue", "no_fat_detected",
            "invalid", "duplicates"
        ]
        writer.writerow(["grade"] + all_statuses + ["grade_total"])

        grand_total_check = 0
        for grade in sorted(grade_distribution.keys()):
            row = [f"Grade_{grade}"]
            grade_total = 0
            for status in all_statuses:
                count = grade_distribution[grade][status]
                row.append(count)
                grade_total += count
            row.append(grade_total)
            grand_total_check += grade_total
            writer.writerow(row)

        # Grade totals footer
        writer.writerow([])
        writer.writerow(["Grand Total (cross-check)", grand_total_check])

    print(f"Summary saved to: {filename}")
    print(f"Total images processed: {total}")


def save_validation_csv(rows, filename="dataset_validation_details.csv"):
    """
    Saves per-image validation results to CSV.
    """
    total = len(rows)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f,
            fieldnames=[
                "file",
                "label",
                "status",
                "tissue_ratio",
                "fat_count",
                "avg_fat_area",
                "threshold_used"  # <--- ADD THIS LINE
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

        # Update the total row to include the new field as well
        writer.writerow({
            "file": "--- TOTAL IMAGES ---",
            "label": total,
            "status": "",
            "tissue_ratio": "",
            "fat_count": "",
            "avg_fat_area": "",
            "threshold_used": "" # <--- ADD THIS LINE
        })

    print(f"\nDetailed report saved to: {filename}")
    print(f"Total rows written: {total}")
    
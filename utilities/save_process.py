import csv

def save_summary_to_csv(report, filename="dataset_summary.csv"):
    # Prepare summary counts
    summary = {
        "valid": len(report["valid"]),
        "invalid": len(report["invalid"]),
        "low_tissue": len(report["low_tissue"]),
        "no_fat_detected": len(report["no_fat_detected"]),
        "duplicates": len(report["duplicates"])
    }

    # Write to CSV
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Count"])  # header
        for k, v in summary.items():
            writer.writerow([k, v])

    print(f"Summary saved to: {filename}")

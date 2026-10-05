"""
dataset_distribution.py

Audit the chest X-ray dataset before preprocessing or training.

Expected folders:
    CPSC597Proj/
        dataset_distribution.py
        chest_xray_dataset/
            train/
            val/
            test/

Each dataset split should contain NORMAL and PNEUMONIA folders.

Run from the CPSC597Proj terminal:
    python .\\dataset_distribution.py

Outputs are saved in results/audit/.
"""

import hashlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# Locate the project folder containing this Python file.
ROOT = Path(__file__).resolve().parent

# Set the input dataset folder and output folder.
DATA = ROOT / "chest_xray_dataset"
OUT = ROOT / "results" / "audit"

# Define the expected dataset splits, classes, and image extensions.
SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
EXTS = {".jpeg", ".jpg", ".png"}


def patient_id(name: str):
    """
    Extract a candidate patient identifier from an image filename.

    Examples:
        person123_bacteria_456.jpeg -> person123
        IM-0115-0001.jpeg -> IM-0115
        NORMAL2-IM-0373-0001.jpeg -> NORMAL2-IM-0373

    Preserve NORMAL2 so different filename groups are not merged.

    These identifiers are based on filename patterns. They are not
    verified patient identities.
    """
    match = re.match(r"(person\d+)_", name, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    match = re.match(
        r"((?:NORMAL2-)?IM-\d+)-\d+",
        name,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    return None


def pneumonia_type(name: str):
    """Identify the pneumonia subtype indicated by the filename."""
    name = name.lower()

    if "bacteria" in name:
        return "bacterial"

    if "virus" in name:
        return "viral"

    return None


def md5(path: Path) -> str:
    """
    Calculate a hash of the file's contents.

    Matching hashes indicate identical file contents. This does not
    detect all visually identical images saved with different encoding.
    """
    file_hash = hashlib.md5()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            file_hash.update(chunk)

    return file_hash.hexdigest()


def main():
    # Check every required folder before starting the audit.
    # Stop instead of producing an incomplete audit if a folder is missing.
    for split in SPLITS:
        for label in CLASSES:
            folder = DATA / split / label

            if not folder.is_dir():
                raise SystemExit(
                    f"Required dataset folder not found: {folder}"
                )

    # Create the output folder if it does not already exist.
    OUT.mkdir(parents=True, exist_ok=True)

    # Store readable image information and unreadable file details.
    rows = []
    corrupt = []

    # Inspect images in each split and class.
    for split in SPLITS:
        for label in CLASSES:
            folder = DATA / split / label

            for path in sorted(folder.iterdir()):
                # Ignore directories and unsupported file extensions.
                if not path.is_file():
                    continue

                if path.suffix.lower() not in EXTS:
                    continue

                try:
                    # Verify the image's file structure.
                    with Image.open(path) as image:
                        image.verify()

                    # Reopen after verify() and fully decode the pixels.
                    # Loading can reveal errors not caught by verify().
                    with Image.open(path) as image:
                        image.load()
                        width, height = image.size
                        mode = image.mode

                    # Calculate the hash inside the error-handling block.
                    file_hash = md5(path)

                except Exception as error:
                    corrupt.append({
                        "path": str(path.relative_to(ROOT)),
                        "error": repr(error),
                    })
                    continue

                # Extract a subtype only for pneumonia images.
                subtype = None
                if label == "PNEUMONIA":
                    subtype = pneumonia_type(path.name)

                # Record one inventory entry per readable image.
                rows.append({
                    "split": split,
                    "label": label,
                    "file": path.name,
                    "path": str(path.relative_to(ROOT)),
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "md5": file_hash,
                    "patient": patient_id(path.name),
                    "ptype": subtype,
                })

    # Save unreadable file details, even if none were found.
    corrupt_df = pd.DataFrame(corrupt, columns=["path", "error"])
    corrupt_df.to_csv(OUT / "unreadable_images.csv", index=False)

    # Stop clearly if no readable images were found.
    df = pd.DataFrame(rows)

    if df.empty:
        raise SystemExit(
            "No readable images found. Check your dataset files. "
            "Any unreadable file details were saved in results/audit/."
        )

    # Save the full inventory for later cleaning and splitting.
    df.to_csv(OUT / "image_inventory.csv", index=False)

    # Count readable images in each split and class.
    # Reindex both axes so missing classes receive a count of zero.
    dist = (
        df.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=SPLITS, columns=CLASSES, fill_value=0)
        .fillna(0)
        .astype(int)
    )

    dist["total"] = dist[CLASSES].sum(axis=1)

    # Avoid division by zero when a split has no readable images.
    totals = dist["total"].replace(0, float("nan"))
    dist["pneumonia_pct"] = (
        100 * dist["PNEUMONIA"] / totals
    ).round(1)

    dist.to_csv(OUT / "class_distribution.csv")

    # Generate a class distribution chart for the project report.
    ax = dist[CLASSES].plot(
        kind="bar",
        figsize=(7, 4),
        rot=0,
    )
    ax.set_title("Readable chest X-ray images per split and class")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of images")

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(OUT / "class_distribution.png", dpi=200)
    plt.close()

    # Find all images belonging to identical-file duplicate groups.
    duplicates = df[df.duplicated("md5", keep=False)]
    duplicate_groups = duplicates.groupby("md5")

    # Identify duplicate groups appearing in multiple dataset splits.
    cross_split_hashes = []

    for file_hash, group in duplicate_groups:
        if group["split"].nunique() > 1:
            cross_split_hashes.append(file_hash)

    cross_split_duplicates = duplicates[
        duplicates["md5"].isin(cross_split_hashes)
    ]

    # Save duplicate details so individual files can be reviewed.
    duplicates.to_csv(OUT / "duplicate_images.csv", index=False)
    cross_split_duplicates.to_csv(
        OUT / "cross_split_duplicates.csv",
        index=False,
    )

    # Check candidate filename IDs for possible overlap between splits.
    # Matching IDs are a review flag, not proof of patient leakage.
    has_pid = df.dropna(subset=["patient"])
    pid_splits = has_pid.groupby("patient")["split"].nunique()
    overlapping_ids = pid_splits[pid_splits > 1]

    patient_overlap = has_pid[
        has_pid["patient"].isin(overlapping_ids.index)
    ]
    patient_overlap.to_csv(
        OUT / "candidate_patient_overlap.csv",
        index=False,
    )

    # Build a readable summary of the audit findings.
    lines = [
        f"Total readable images: {len(df)}",
        f"Corrupt/unreadable files: {len(corrupt)}",
    ]

    for item in corrupt:
        lines.append(f"  {item['path']}: {item['error']}")

    lines.append(
        "\nReadable images per split/class:\n" + dist.to_string()
    )

    # Summarize bacterial and viral filename labels when available.
    pneumonia = df[df["label"] == "PNEUMONIA"]

    if not pneumonia.empty:
        subtype_counts = (
            pneumonia.groupby(["split", "ptype"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
        lines.append(
            "\nPneumonia subtype counts (from filenames):\n"
            + subtype_counts.to_string()
        )

    lines.append(
        "\nImage width and height summary:\n"
        + df[["width", "height"]].describe().round(0).to_string()
    )
    lines.append(
        "\nColor modes:\n" + df["mode"].value_counts().to_string()
    )
    lines.append(
        f"\nIdentical-file duplicates (same MD5): "
        f"{len(duplicates)} images in "
        f"{duplicate_groups.ngroups} groups"
    )
    lines.append(
        f"Duplicate groups spanning multiple splits: "
        f"{len(cross_split_hashes)}"
    )
    lines.append(
        f"Candidate patient IDs parsed from filenames: "
        f"{has_pid['patient'].nunique()} unique IDs "
        f"({len(has_pid)}/{len(df)} images matched)"
    )
    lines.append(
        f"Candidate IDs appearing in multiple splits: "
        f"{len(overlapping_ids)}"
    )

    if not overlapping_ids.empty:
        examples = ", ".join(overlapping_ids.index[:10])
        lines.append(f"  Example overlapping IDs: {examples}")

    lines.append(
        "\nLimitations: Filename-derived IDs are not verified patient "
        "identities. MD5 detects identical file contents, not all "
        "visually identical or near-duplicate images."
    )

    # Save and print the final summary.
    summary = "\n".join(lines)
    (OUT / "audit_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)
    print(f"\nSaved results to {OUT}")


# Run the audit when this file is executed directly.
if __name__ == "__main__":
    main()
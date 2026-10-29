#!/usr/bin/env python3
"""Validate a YOLO segmentation dataset before training.

Checks:
- train/val/test image-label parity;
- allowed class IDs (0, 1);
- each non-empty label line has enough polygon coordinates;
- normalized polygon coordinates stay in [0, 1];
- reports class-instance counts and obstacle-positive image counts.

Usage:
    python scripts/validate_yolo_dataset.py data/fire_accessibility_source
    python scripts/validate_yolo_dataset.py data/fire_accessibility_augmented
"""

from collections import Counter
from pathlib import Path
import argparse
import sys

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VALID_CLASSES = {0, 1}
SPLITS = ("train", "val", "test")


def image_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def validate_label(path: Path):
    counts = Counter()
    obstacle = False
    errors = []

    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()

        if len(parts) < 7:
            errors.append(
                f"{path}:{line_no}: segmentation line has fewer than 3 points"
            )
            continue

        try:
            cls = int(float(parts[0]))
        except ValueError:
            errors.append(f"{path}:{line_no}: invalid class id '{parts[0]}'")
            continue

        if cls not in VALID_CLASSES:
            errors.append(f"{path}:{line_no}: unexpected class id {cls}")
        counts[cls] += 1
        obstacle = obstacle or cls == 1

        coords = parts[1:]
        if len(coords) % 2 != 0:
            errors.append(f"{path}:{line_no}: odd number of polygon coordinates")
            continue

        try:
            values = [float(x) for x in coords]
        except ValueError:
            errors.append(f"{path}:{line_no}: non-numeric polygon coordinate")
            continue

        bad = [v for v in values if not 0.0 <= v <= 1.0]
        if bad:
            errors.append(
                f"{path}:{line_no}: coordinate outside [0,1], e.g. {bad[0]}"
            )

    return counts, obstacle, errors


def validate_split(root: Path, split: str):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    images = image_files(img_dir)
    labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}
    errors = []

    for stem in sorted(img_stems - lbl_stems):
        errors.append(f"{split}: image has no label: {stem}")
    for stem in sorted(lbl_stems - img_stems):
        errors.append(f"{split}: label has no image: {stem}")

    class_counts = Counter()
    obstacle_images = 0
    empty_labels = 0

    for label in labels:
        if label.stat().st_size == 0:
            empty_labels += 1
        counts, obstacle, label_errors = validate_label(label)
        class_counts.update(counts)
        obstacle_images += int(obstacle)
        errors.extend(label_errors)

    return {
        "images": len(images),
        "labels": len(labels),
        "empty_labels": empty_labels,
        "instances": class_counts,
        "obstacle_images": obstacle_images,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="YOLO dataset root")
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(args.root)

    all_errors = []
    print(f"Dataset: {args.root}\n")

    for split in SPLITS:
        result = validate_split(args.root, split)
        all_errors.extend(result["errors"])
        print(
            f"{split:5s}: images={result['images']}, labels={result['labels']}, "
            f"fire_extinguisher instances={result['instances'][0]}, "
            f"obstacle instances={result['instances'][1]}, "
            f"obstacle-positive images={result['obstacle_images']}, "
            f"empty labels={result['empty_labels']}"
        )

    if all_errors:
        print("\nVALIDATION FAILED")
        for err in all_errors[:50]:
            print(f"[ERROR] {err}")
        if len(all_errors) > 50:
            print(f"... and {len(all_errors) - 50} more errors")
        sys.exit(1)

    print("\nVALIDATION OK")


if __name__ == "__main__":
    main()

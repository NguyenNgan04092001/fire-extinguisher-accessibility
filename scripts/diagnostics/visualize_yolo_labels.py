#!/usr/bin/env python3
"""Render a small random sample of YOLO segmentation labels.

Examples:
    python scripts/visualize_yolo_labels.py data/base_yolo --split train
    python scripts/visualize_yolo_labels.py data/fire_accessibility_source --split val
"""

from pathlib import Path
import argparse
import random

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = {0: "fire_extinguisher", 1: "obstacle"}


def collect_images(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/annotation_check"))
    args = parser.parse_args()

    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Missing dataset split directories: {image_dir} / {label_dir}"
        )

    images = collect_images(image_dir)
    random.seed(args.seed)
    selected = random.sample(images, min(args.n, len(images)))

    output_dir = args.output / f"{args.dataset_root.name}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in selected:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[SKIP] Cannot read {image_path}")
            continue

        h, w = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"[SKIP] Missing label {label_path}")
            continue

        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) < 7:
                continue

            class_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
            if len(coords) % 2 != 0:
                continue

            points = np.array(
                [[int(coords[i] * w), int(coords[i + 1] * h)]
                 for i in range(0, len(coords), 2)],
                dtype=np.int32,
            )

            # Use different automatically selected OpenCV colors per class.
            color = (0, 255, 0) if class_id == 0 else (0, 165, 255)
            cv2.polylines(image, [points], True, color, 2)

            overlay = image.copy()
            cv2.fillPoly(overlay, [points], color)
            image = cv2.addWeighted(overlay, 0.20, image, 0.80, 0)

            x0, y0 = points[0]
            name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            cv2.putText(
                image,
                name,
                (int(x0), max(20, int(y0) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        output_path = output_dir / image_path.name
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Cannot write {output_path}")
        print(f"Saved: {output_path}")

    print(f"\nCheck rendered samples in: {output_dir}")


if __name__ == "__main__":
    main()

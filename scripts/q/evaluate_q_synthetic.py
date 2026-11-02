#!/usr/bin/env python3

from pathlib import Path
import csv
import re

import numpy as np

from scripts.q.compute_occlusion_q_instances import (
    read_image,
    parse_yolo_seg,
    polygon_mask,
    infer_base_stem,
)


IMAGE_DIR = Path("data/cvat_synthetic/images")
LABEL_DIR = Path("annotations/cvat_synthetic/labels/train")

OUT_INSTANCE = Path("outputs/q_evaluation_instances.csv")
OUT_IMAGE = Path("outputs/q_evaluation_images.csv")


def get_scenario(stem):
    m = re.search(r"_(S[123])$", stem)

    if not m:
        raise ValueError(
            f"Cannot infer scenario: {stem}"
        )

    return m.group(1)


def find_image(stem):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = IMAGE_DIR / f"{stem}{ext}"

        if path.exists():
            return path

    return None


def main():
    OUT_INSTANCE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = sorted(
        LABEL_DIR.glob("*.txt")
    )

    instance_rows = []
    image_rows = []

    print(
        f"Synthetic labels found: {len(labels)}"
    )
    print()

    for idx, label_path in enumerate(
        labels,
        start=1,
    ):
        stem = label_path.stem

        image_path = find_image(stem)

        if image_path is None:
            print(
                f"[SKIP] image not found: {stem}"
            )
            continue

        scenario = get_scenario(stem)

        base_stem = infer_base_stem(
            image_path.name
        )

        base_label = Path(
            "annotations/cvat_base/labels/train"
        ) / f"{base_stem}.txt"

        if not base_label.exists():
            print(
                f"[SKIP] base label missing: "
                f"{base_label}"
            )
            continue

        image = read_image(image_path)

        h, w = image.shape[:2]

        base_objects = parse_yolo_seg(
            base_label
        )

        synth_objects = parse_yolo_seg(
            label_path
        )

        # -------------------------
        # OBSTACLE MASK
        # -------------------------

        obstacle_mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        obstacle_instances = 0

        for cls_id, polygon in synth_objects:
            if cls_id != 1:
                continue

            obstacle_mask |= polygon_mask(
                polygon,
                h,
                w,
            )

            obstacle_instances += 1

        # -------------------------
        # FIRE INSTANCES
        # -------------------------

        q_values = []
        fire_id = 0

        for cls_id, polygon in base_objects:
            if cls_id != 0:
                continue

            fire_id += 1

            fire_mask = polygon_mask(
                polygon,
                h,
                w,
            )

            overlap = (
                fire_mask
                & obstacle_mask
            )

            fire_area = int(
                fire_mask.sum()
            )

            occluded_area = int(
                overlap.sum()
            )

            q = (
                occluded_area / fire_area
                if fire_area > 0
                else 0.0
            )

            q_values.append(q)

            instance_rows.append({
                "image": image_path.name,
                "scenario": scenario,
                "base": base_stem,
                "instance_id": fire_id,
                "fire_area": fire_area,
                "occluded_area": occluded_area,
                "q": q,
            })

        if not q_values:
            print(
                f"[SKIP] no fire instances: "
                f"{stem}"
            )
            continue

        q_max = max(q_values)
        q_mean = float(np.mean(q_values))

        image_rows.append({
            "image": image_path.name,
            "scenario": scenario,
            "base": base_stem,
            "fire_instances": len(q_values),
            "obstacle_instances": obstacle_instances,
            "q_max": q_max,
            "q_mean": q_mean,
        })

        print(
            f"[{idx:02d}/{len(labels)}] "
            f"{image_path.name:22s} "
            f"{scenario} "
            f"fire={len(q_values)} "
            f"q_max={q_max:.3f} "
            f"q_mean={q_mean:.3f}"
        )

    # -------------------------
    # SAVE INSTANCE CSV
    # -------------------------

    with OUT_INSTANCE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "scenario",
                "base",
                "instance_id",
                "fire_area",
                "occluded_area",
                "q",
            ],
        )

        writer.writeheader()
        writer.writerows(instance_rows)

    # -------------------------
    # SAVE IMAGE CSV
    # -------------------------

    with OUT_IMAGE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "scenario",
                "base",
                "fire_instances",
                "obstacle_instances",
                "q_max",
                "q_mean",
            ],
        )

        writer.writeheader()
        writer.writerows(image_rows)

    # -------------------------
    # SUMMARY
    # -------------------------

    print()
    print("=" * 70)
    print("Q SUMMARY — IMAGE LEVEL (q_max)")
    print("=" * 70)

    for scenario in ["S1", "S2", "S3"]:

        values = [
            r["q_max"]
            for r in image_rows
            if r["scenario"] == scenario
        ]

        if not values:
            continue

        arr = np.asarray(
            values,
            dtype=float,
        )

        print(
            f"{scenario}: "
            f"n={len(arr):2d} "
            f"mean={arr.mean():.3f} "
            f"median={np.median(arr):.3f} "
            f"min={arr.min():.3f} "
            f"max={arr.max():.3f}"
        )

    print()
    print(
        f"Image rows    : {len(image_rows)}"
    )

    print(
        f"Instance rows : {len(instance_rows)}"
    )

    print()
    print(f"Saved: {OUT_IMAGE}")
    print(f"Saved: {OUT_INSTANCE}")


if __name__ == "__main__":
    main()
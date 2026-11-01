#!/usr/bin/env python3

from pathlib import Path
import csv

import cv2
import numpy as np

from compute_occlusion_q_instances import (
    read_image,
    parse_yolo_seg,
    polygon_mask,
    infer_base_stem,
)


IMAGE_DIR = Path("data/cvat_synthetic/images")
BASE_LABEL_DIR = Path("annotations/cvat_base/labels/train")

OUT_DIR = Path("outputs/accessibility_gt")
VIS_DIR = OUT_DIR / "images"
CSV_OUT = OUT_DIR / "accessibility_gt_template.csv"


def find_image(name):
    path = IMAGE_DIR / name

    if path.exists():
        return path

    return None


def main():
    source_csv = Path(
        "outputs/qr_synthetic_instances.csv"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with source_csv.open(
        encoding="utf-8"
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    output_rows = []

    grouped = {}

    for row in rows:
        grouped.setdefault(
            row["image"],
            []
        ).append(row)

    for image_name, image_rows in grouped.items():

        image_path = find_image(
            image_name
        )

        if image_path is None:
            print(
                "[SKIP]",
                image_name,
            )
            continue

        image = read_image(
            image_path
        )

        h, w = image.shape[:2]

        base_stem = infer_base_stem(
            image_name
        )

        base_label = (
            BASE_LABEL_DIR
            / f"{base_stem}.txt"
        )

        objects = parse_yolo_seg(
            base_label
        )

        fire_instances = [
            poly
            for cls_id, poly in objects
            if cls_id == 0
        ]

        for row in image_rows:

            instance_id = int(
                row["instance_id"]
            )

            if instance_id > len(
                fire_instances
            ):
                continue

            polygon = fire_instances[
                instance_id - 1
            ]

            mask = polygon_mask(
                polygon,
                h,
                w,
            )

            ys, xs = np.where(
                mask > 0
            )

            if len(xs) == 0:
                continue

            x1 = int(xs.min())
            x2 = int(xs.max())

            y1 = int(ys.min())
            y2 = int(ys.max())

            vis = image.copy()

            # Highlight target extinguisher
            contours, _ = cv2.findContours(
                (mask * 255),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            cv2.drawContours(
                vis,
                contours,
                -1,
                (0, 255, 0),
                3,
            )

            text = (
                f"INSTANCE {instance_id}"
            )

            cv2.putText(
                vis,
                text,
                (
                    max(x1, 5),
                    max(y1 - 10, 25),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            output_name = (
                f"{Path(image_name).stem}"
                f"_fire{instance_id}.jpg"
            )

            cv2.imwrite(
                str(
                    VIS_DIR
                    / output_name
                ),
                vis,
            )

            output_rows.append({
                "image": image_name,
                "instance_id": instance_id,
                "visualization": output_name,
                "accessibility_gt": "",
                "notes": "",
            })

    with CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "instance_id",
                "visualization",
                "accessibility_gt",
                "notes",
            ],
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    print()
    print("=" * 60)
    print("ACCESSIBILITY GT PREPARATION")
    print("=" * 60)

    print(
        f"Instances: {len(output_rows)}"
    )

    print(
        f"Images saved: {VIS_DIR}"
    )

    print(
        f"CSV saved: {CSV_OUT}"
    )

    print()
    print(
        "Labels:"
    )

    print(
        "0 = accessible"
    )

    print(
        "1 = partially_blocked"
    )

    print(
        "2 = severely_blocked"
    )


if __name__ == "__main__":
    main()
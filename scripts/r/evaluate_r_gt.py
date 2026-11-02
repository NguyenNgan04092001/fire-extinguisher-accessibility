#!/usr/bin/env python3

from pathlib import Path
import csv

import cv2
import numpy as np

from scripts.r.compute_access_obstruction import create_access_zones


META = Path("metadata/real_extension_v3.csv")
IMAGE_ROOT = Path("data/real_extension_v3/selected")
ANN_ROOT = Path("annotations/cvat_real_extension_v3")


def load_gt_mask(label_path, image_shape, class_id):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        p = line.split()

        if not p:
            continue

        if int(float(p[0])) != class_id:
            continue

        coords = list(map(float, p[1:]))

        pts = np.array(
            [
                [
                    coords[i] * w,
                    coords[i + 1] * h,
                ]
                for i in range(0, len(coords), 2)
            ],
            dtype=np.int32,
        )

        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)

    return mask


def main():

    with META.open(
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    results = []

    for i, row in enumerate(rows, 1):

        name = row["file_name"]
        scenario = row["scenario"]
        split = row["split"]

        image_path = IMAGE_ROOT / scenario / name

        label_path = (
            ANN_ROOT
            / split
            / "labels"
            / f"{Path(name).stem}.txt"
        )

        image = cv2.imread(str(image_path))

        fire = load_gt_mask(
            label_path,
            image.shape,
            class_id=0,
        )

        obstacle = load_gt_mask(
            label_path,
            image.shape,
            class_id=1,
        )

        z1, z2, info = create_access_zones(fire)

        z1_overlap = int(
            (obstacle & z1).sum()
        )

        z2_overlap = int(
            (obstacle & z2).sum()
        )

        r1 = (
            z1_overlap
            / max(int(z1.sum()), 1)
        )

        if info["z2_valid"]:

            r2 = (
                z2_overlap
                / max(int(z2.sum()), 1)
            )

            r = 0.7 * r1 + 0.3 * r2

        else:
            r2 = None
            r = r1

        results.append(
            (scenario, r)
        )

        print(
            f"[{i:02d}/28] "
            f"{name} {scenario} "
            f"r={r:.3f} "
            f"r1={r1:.3f} "
            f"r2="
            f"{'NA' if r2 is None else f'{r2:.3f}'}"
        )

    print()
    print("=" * 60)
    print("GT OBSTACLE SUMMARY")
    print("=" * 60)

    for scenario in ["S0", "S1", "S2", "S3"]:

        values = [
            r
            for s, r in results
            if s == scenario
        ]

        print(
            f"{scenario}: "
            f"n={len(values)}, "
            f"mean={np.mean(values):.3f}, "
            f"median={np.median(values):.3f}, "
            f"min={np.min(values):.3f}, "
            f"max={np.max(values):.3f}"
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from pathlib import Path
import csv
import re

import numpy as np

from compute_occlusion_q_instances import (
    read_image,
    parse_yolo_seg,
    polygon_mask,
    infer_base_stem,
)

from compute_access_obstruction import (
    create_access_zones,
)


IMAGE_DIR = Path("data/cvat_synthetic/images")
LABEL_DIR = Path("annotations/cvat_synthetic/labels/train")

OUT = Path("outputs/qr_synthetic_instances.csv")


def find_image(stem):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = IMAGE_DIR / f"{stem}{ext}"

        if path.exists():
            return path

    return None


def scenario_from_stem(stem):
    m = re.search(r"_(S[123])$", stem)

    if not m:
        return "?"

    return m.group(1)


def build_obstacle_mask(
    objects,
    h,
    w,
):
    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    for cls_id, polygon in objects:
        if cls_id != 1:
            continue

        mask |= polygon_mask(
            polygon,
            h,
            w,
        )

    return mask


def compute_r(
    fire_mask,
    obstacle_mask,
):
    z1, z2, info = create_access_zones(
        fire_mask
    )

    z1_area = int(
        (z1 > 0).sum()
    )

    z2_area = int(
        (z2 > 0).sum()
    )

    r1 = (
        int(
            (
                (obstacle_mask > 0)
                & (z1 > 0)
            ).sum()
        )
        / max(z1_area, 1)
    )

    r2 = (
        int(
            (
                (obstacle_mask > 0)
                & (z2 > 0)
            ).sum()
        )
        / max(z2_area, 1)
    )

    if info["z2_valid"]:
        r = (
            0.7 * r1
            + 0.3 * r2
        )
    else:
        r = r1

    return (
        float(r1),
        float(r2),
        float(r),
        bool(info["z2_valid"]),
    )


def main():
    rows = []

    labels = sorted(
        LABEL_DIR.glob("*.txt")
    )

    print(
        f"Synthetic images: {len(labels)}"
    )
    print()

    for idx, synth_label in enumerate(
        labels,
        1,
    ):
        stem = synth_label.stem

        image_path = find_image(stem)

        if image_path is None:
            print(
                f"[SKIP] missing image: {stem}"
            )
            continue

        base_stem = infer_base_stem(
            image_path.name
        )

        base_label = (
            Path(
                "annotations/"
                "cvat_base/"
                "labels/"
                "train"
            )
            / f"{base_stem}.txt"
        )

        image = read_image(
            image_path
        )

        h, w = image.shape[:2]

        base_objects = parse_yolo_seg(
            base_label
        )

        synth_objects = parse_yolo_seg(
            synth_label
        )

        obstacle_mask = build_obstacle_mask(
            synth_objects,
            h,
            w,
        )

        scenario = scenario_from_stem(
            stem
        )

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

            # q
            fire_area = int(
                fire_mask.sum()
            )

            overlap_area = int(
                (
                    fire_mask
                    & obstacle_mask
                ).sum()
            )

            q = (
                overlap_area / fire_area
                if fire_area > 0
                else 0.0
            )

            # r
            r1, r2, r, z2_valid = (
                compute_r(
                    fire_mask,
                    obstacle_mask,
                )
            )

            rows.append({
                "image": image_path.name,
                "scenario_old": scenario,
                "base": base_stem,
                "instance_id": fire_id,
                "q": q,
                "r1": r1,
                "r2": r2,
                "r": r,
                "z2_valid": int(
                    z2_valid
                ),
            })

            print(
                f"[{idx:02d}/{len(labels)}] "
                f"{image_path.name:22s} "
                f"fire#{fire_id} "
                f"q={q:.3f} "
                f"r1={r1:.3f} "
                f"r2={r2:.3f} "
                f"r={r:.3f}"
            )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "scenario_old",
                "base",
                "instance_id",
                "q",
                "r1",
                "r2",
                "r",
                "z2_valid",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    q = np.asarray(
        [x["q"] for x in rows],
        dtype=float,
    )

    r = np.asarray(
        [x["r"] for x in rows],
        dtype=float,
    )

    print()
    print("=" * 70)
    print("SYNTHETIC Q + R DATASET")
    print("=" * 70)

    print(
        f"Images    : "
        f"{len(set(x['image'] for x in rows))}"
    )

    print(
        f"Instances : {len(rows)}"
    )

    print()
    print(
        f"q mean    : {q.mean():.4f}"
    )

    print(
        f"q median  : {np.median(q):.4f}"
    )

    print(
        f"q max     : {q.max():.4f}"
    )

    print()
    print(
        f"r mean    : {r.mean():.4f}"
    )

    print(
        f"r median  : {np.median(r):.4f}"
    )

    print(
        f"r max     : {r.max():.4f}"
    )

    print()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
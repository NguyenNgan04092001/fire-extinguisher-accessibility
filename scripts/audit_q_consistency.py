#!/usr/bin/env python3

from pathlib import Path
import csv

import numpy as np

from compute_occlusion_q_instances import (
    read_image,
    parse_yolo_seg,
    polygon_mask,
    infer_base_stem,
)


IMAGE_DIR = Path("data/cvat_synthetic/images")
LABEL_DIR = Path("annotations/cvat_synthetic/labels/train")

OUT = Path("outputs/q_annotation_audit.csv")


def build_union_mask(objects, class_id, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)

    for cls, polygon in objects:
        if cls != class_id:
            continue

        mask |= polygon_mask(
            polygon,
            h,
            w,
        )

    return mask


def find_image(stem):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = IMAGE_DIR / f"{stem}{ext}"

        if path.exists():
            return path

    return None


def main():
    rows = []

    labels = sorted(
        LABEL_DIR.glob("*.txt")
    )

    for idx, synth_label in enumerate(labels, 1):
        stem = synth_label.stem

        synth_image = find_image(stem)

        if synth_image is None:
            print("[SKIP]", stem)
            continue

        base_stem = infer_base_stem(
            synth_image.name
        )

        base_label = (
            Path("annotations/cvat_base/labels/train")
            / f"{base_stem}.txt"
        )

        image = read_image(
            synth_image
        )

        h, w = image.shape[:2]

        base_objects = parse_yolo_seg(
            base_label
        )

        synth_objects = parse_yolo_seg(
            synth_label
        )

        # Full extinguisher before synthetic obstruction
        full_fire = build_union_mask(
            base_objects,
            class_id=0,
            h=h,
            w=w,
        )

        # Visible extinguisher annotated in synthetic image
        visible_fire = build_union_mask(
            synth_objects,
            class_id=0,
            h=h,
            w=w,
        )

        # Synthetic obstacle
        obstacle = build_union_mask(
            synth_objects,
            class_id=1,
            h=h,
            w=w,
        )

        full_area = int(
            full_fire.sum()
        )

        visible_inside_full = int(
            (full_fire & visible_fire).sum()
        )

        obstacle_inside_full = int(
            (full_fire & obstacle).sum()
        )

        if full_area == 0:
            continue

        q_obstacle = (
            obstacle_inside_full
            / full_area
        )

        q_visible = (
            1.0
            - visible_inside_full / full_area
        )

        diff = abs(
            q_obstacle - q_visible
        )

        rows.append({
            "image": synth_image.name,
            "full_area": full_area,
            "q_obstacle": q_obstacle,
            "q_visible": q_visible,
            "abs_diff": diff,
        })

        print(
            f"[{idx:02d}/{len(labels)}] "
            f"{synth_image.name:22s} "
            f"q_obs={q_obstacle:.3f} "
            f"q_vis={q_visible:.3f} "
            f"diff={diff:.3f}"
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
                "full_area",
                "q_obstacle",
                "q_visible",
                "abs_diff",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    diffs = np.array(
        [r["abs_diff"] for r in rows],
        dtype=float,
    )

    print()
    print("=" * 70)
    print("Q ANNOTATION CONSISTENCY")
    print("=" * 70)

    print(
        f"Images       : {len(rows)}"
    )

    print(
        f"Mean diff    : {diffs.mean():.4f}"
    )

    print(
        f"Median diff  : {np.median(diffs):.4f}"
    )

    print(
        f"Max diff     : {diffs.max():.4f}"
    )

    print(
        f"diff <= 0.05 : {(diffs <= 0.05).sum()}"
    )

    print(
        f"diff <= 0.10 : {(diffs <= 0.10).sum()}"
    )

    print(
        f"diff > 0.10  : {(diffs > 0.10).sum()}"
    )

    print()
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
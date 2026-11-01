#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import re

import cv2
import numpy as np


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def parse_yolo_seg(path):
    objects = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            cls_id = int(float(parts[0]))

            coords = np.asarray(
                list(map(float, parts[1:])),
                dtype=np.float32,
            ).reshape(-1, 2)

            objects.append((cls_id, coords))

    return objects


def polygon_mask(poly, h, w):
    pts = np.zeros_like(poly, dtype=np.int32)

    pts[:, 0] = np.clip(
        np.round(poly[:, 0] * w),
        0,
        w - 1,
    )

    pts[:, 1] = np.clip(
        np.round(poly[:, 1] * h),
        0,
        h - 1,
    )

    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(
        mask,
        [pts],
        1,
    )

    return mask


def infer_base_stem(filename):
    match = re.match(
        r"^(base_\d{4})_S\d+",
        Path(filename).stem,
    )

    if not match:
        raise ValueError(filename)

    return match.group(1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--synthetic-image",
        required=True,
    )

    parser.add_argument(
        "--synthetic-label",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        default="outputs/occlusion_q_instances",
    )

    args = parser.parse_args()

    synth_image_path = Path(args.synthetic_image)
    synth_label_path = Path(args.synthetic_label)

    base_stem = infer_base_stem(
        synth_image_path.name
    )

    base_image_path = Path(
        f"data/base_diverse/{base_stem}.jpg"
    )

    base_label_path = Path(
        f"annotations/cvat_base/labels/train/"
        f"{base_stem}.txt"
    )

    image = read_image(synth_image_path)

    h, w = image.shape[:2]

    base_objects = parse_yolo_seg(
        base_label_path
    )

    synth_objects = parse_yolo_seg(
        synth_label_path
    )

    # --------------------------------
    # OBSTACLE MASK
    # --------------------------------

    obstacle_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    obstacle_count = 0

    for cls_id, polygon in synth_objects:
        if cls_id != 1:
            continue

        obstacle_mask |= polygon_mask(
            polygon,
            h,
            w,
        )

        obstacle_count += 1

    # --------------------------------
    # EACH FIRE INSTANCE SEPARATELY
    # --------------------------------

    results = []

    fire_index = 0

    visualization = image.astype(
        np.float32
    ).copy()

    for cls_id, polygon in base_objects:
        if cls_id != 0:
            continue

        fire_index += 1

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

        overlap_area = int(
            overlap.sum()
        )

        q = (
            overlap_area / fire_area
            if fire_area > 0
            else 0.0
        )

        ys, xs = np.where(
            fire_mask > 0
        )

        if len(xs) > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())
        else:
            cx = cy = 0

        results.append({
            "instance_id": fire_index,
            "fire_area": fire_area,
            "occluded_area": overlap_area,
            "q": round(q, 6),
            "center_x": cx,
            "center_y": cy,
        })

        # red = full fire
        m = fire_mask > 0
        visualization[m] = (
            0.6 * visualization[m]
            + 0.4 * np.array(
                [0, 0, 255],
                dtype=np.float32,
            )
        )

        # green = occluded part
        m = overlap > 0
        visualization[m] = (
            0.25 * visualization[m]
            + 0.75 * np.array(
                [0, 255, 0],
                dtype=np.float32,
            )
        )

        cv2.putText(
            visualization,
            f"#{fire_index} q={q:.3f}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # blue = obstacles
    m = obstacle_mask > 0

    visualization[m] = (
        0.7 * visualization[m]
        + 0.3 * np.array(
            [255, 0, 0],
            dtype=np.float32,
        )
    )

    # redraw overlap on top
    for cls_id, polygon in base_objects:
        if cls_id != 0:
            continue

        fire_mask = polygon_mask(
            polygon,
            h,
            w,
        )

        overlap = (
            fire_mask
            & obstacle_mask
        )

        m = overlap > 0

        visualization[m] = (
            0.2 * visualization[m]
            + 0.8 * np.array(
                [0, 255, 0],
                dtype=np.float32,
            )
        )

    visualization = np.clip(
        visualization,
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------
    # SAVE
    # --------------------------------

    outdir = (
        Path(args.outdir)
        / synth_image_path.stem
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(outdir / "obstacle_mask.png"),
        obstacle_mask * 255,
    )

    cv2.imwrite(
        str(outdir / "q_instances.jpg"),
        visualization,
    )

    metadata = {
        "synthetic_image": str(
            synth_image_path
        ),
        "base_image": str(
            base_image_path
        ),
        "base_label": str(
            base_label_path
        ),
        "obstacle_instances": obstacle_count,
        "fire_instances": len(results),
        "instances": results,
    }

    with open(
        outdir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Image: {synth_image_path.name}"
    )

    print(
        f"Fire instances: {len(results)}"
    )

    for r in results:
        print(
            f"  fire #{r['instance_id']}: "
            f"area={r['fire_area']} "
            f"occluded={r['occluded_area']} "
            f"q={r['q']:.4f}"
        )

    print()
    print(f"Saved: {outdir}")


if __name__ == "__main__":
    main()
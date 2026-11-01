#!/usr/bin/env python3

from pathlib import Path
import csv
import re

import cv2
import numpy as np
from ultralytics import YOLO

from compute_occlusion_q_instances import (
    read_image,
    parse_yolo_seg,
    polygon_mask,
    infer_base_stem,
)


IMAGE_DIR = Path("data/cvat_synthetic/images")
LABEL_DIR = Path("annotations/cvat_synthetic/labels/train")

MODEL_PATH = "models/best.pt"
OUT = Path("outputs/q_evaluation_yolo.csv")

CONF = 0.25


def find_image(stem):
    for ext in [".png", ".jpg", ".jpeg"]:
        p = IMAGE_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_full_fire_mask(base_objects, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)

    for cls_id, poly in base_objects:
        if cls_id != 0:
            continue

        mask |= polygon_mask(
            poly,
            h,
            w,
        )

    return mask


def build_obstacle_mask(synth_objects, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)

    for cls_id, poly in synth_objects:
        if cls_id != 1:
            continue

        mask |= polygon_mask(
            poly,
            h,
            w,
        )

    return mask


def predict_fire_mask(model, image_path, h, w):
    result = model.predict(
        source=str(image_path),
        conf=CONF,
        imgsz=640,
        device="cpu",
        retina_masks=True,
        verbose=False,
    )[0]

    mask_union = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    if (
        result.boxes is None
        or result.masks is None
    ):
        return mask_union, 0

    masks = result.masks.data.cpu().numpy()

    count = 0

    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i])

        if cls_id != 0:
            continue

        mask = (
            masks[i] > 0.5
        ).astype(np.uint8)

        if mask.shape != (h, w):
            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_union |= mask
        count += 1

    return mask_union, count


def main():
    model = YOLO(MODEL_PATH)

    labels = sorted(
        LABEL_DIR.glob("*.txt")
    )

    rows = []

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
            continue

        base_stem = infer_base_stem(
            image_path.name
        )

        base_label = (
            Path("annotations/cvat_base/labels/train")
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

        full_fire = build_full_fire_mask(
            base_objects,
            h,
            w,
        )

        obstacle = build_obstacle_mask(
            synth_objects,
            h,
            w,
        )

        pred_fire, n_pred = predict_fire_mask(
            model,
            image_path,
            h,
            w,
        )

        full_area = int(
            full_fire.sum()
        )

        if full_area == 0:
            continue

        # -------------------------
        # Ground-truth q
        # -------------------------

        gt_occluded = int(
            (full_fire & obstacle).sum()
        )

        q_gt = (
            gt_occluded / full_area
        )

        # -------------------------
        # Predicted visible fraction
        # -------------------------

        visible_pred = int(
            (full_fire & pred_fire).sum()
        )

        visible_fraction = (
            visible_pred / full_area
        )

        q_pred = 1.0 - visible_fraction

        q_pred = float(
            np.clip(
                q_pred,
                0.0,
                1.0,
            )
        )

        error = abs(
            q_pred - q_gt
        )

        scenario_match = re.search(
            r"_(S[123])$",
            stem,
        )

        scenario = (
            scenario_match.group(1)
            if scenario_match
            else "?"
        )

        rows.append({
            "image": image_path.name,
            "scenario": scenario,
            "q_gt": q_gt,
            "q_pred": q_pred,
            "error": error,
            "pred_fire_instances": n_pred,
        })

        print(
            f"[{idx:02d}/{len(labels)}] "
            f"{image_path.name:22s} "
            f"{scenario} "
            f"GT={q_gt:.3f} "
            f"PRED={q_pred:.3f} "
            f"err={error:.3f} "
            f"n={n_pred}"
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
                "scenario",
                "q_gt",
                "q_pred",
                "error",
                "pred_fire_instances",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    gt = np.asarray(
        [r["q_gt"] for r in rows],
        dtype=float,
    )

    pred = np.asarray(
        [r["q_pred"] for r in rows],
        dtype=float,
    )

    mae = np.mean(
        np.abs(pred - gt)
    )

    rmse = np.sqrt(
        np.mean(
            (pred - gt) ** 2
        )
    )

    if (
        len(gt) > 1
        and np.std(gt) > 0
        and np.std(pred) > 0
    ):
        corr = np.corrcoef(
            gt,
            pred,
        )[0, 1]
    else:
        corr = float("nan")

    print()
    print("=" * 70)
    print("YOLO Q EVALUATION")
    print("=" * 70)

    print(
        f"Images      : {len(rows)}"
    )

    print(
        f"MAE         : {mae:.4f}"
    )

    print(
        f"RMSE        : {rmse:.4f}"
    )

    print(
        f"Correlation : {corr:.4f}"
    )

    print()
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
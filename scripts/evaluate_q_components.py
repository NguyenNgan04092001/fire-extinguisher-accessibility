#!/usr/bin/env python3

from pathlib import Path
import csv

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
OUT = Path("outputs/q_components.csv")

CONF = 0.25


def find_image(stem):
    for ext in [".png", ".jpg", ".jpeg"]:
        p = IMAGE_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def gt_mask(objects, class_id, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)

    for cls, poly in objects:
        if cls != class_id:
            continue

        mask |= polygon_mask(
            poly,
            h,
            w,
        )

    return mask


def predict_masks(model, image_path, h, w):
    result = model.predict(
        source=str(image_path),
        conf=CONF,
        imgsz=640,
        device="cpu",
        retina_masks=True,
        verbose=False,
    )[0]

    fire = np.zeros((h, w), dtype=np.uint8)
    obstacle = np.zeros((h, w), dtype=np.uint8)

    if (
        result.boxes is None
        or result.masks is None
    ):
        return fire, obstacle

    masks = result.masks.data.cpu().numpy()

    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i])

        mask = (
            masks[i] > 0.5
        ).astype(np.uint8)

        if mask.shape != (h, w):
            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        if cls_id == 0:
            fire |= mask

        elif cls_id == 1:
            obstacle |= mask

    return fire, obstacle


def compute_q(fire, obstacle):
    fire_area = int(fire.sum())

    if fire_area == 0:
        return 0.0

    overlap = int(
        (fire & obstacle).sum()
    )

    return overlap / fire_area


def metrics(gt, pred):
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mae = np.mean(
        np.abs(gt - pred)
    )

    rmse = np.sqrt(
        np.mean(
            (gt - pred) ** 2
        )
    )

    if (
        np.std(gt) > 0
        and np.std(pred) > 0
    ):
        corr = np.corrcoef(
            gt,
            pred,
        )[0, 1]
    else:
        corr = float("nan")

    return mae, rmse, corr


def main():
    model = YOLO(MODEL_PATH)

    labels = sorted(
        LABEL_DIR.glob("*.txt")
    )

    rows = []

    for idx, synth_label in enumerate(labels, 1):
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

        image = read_image(image_path)
        h, w = image.shape[:2]

        base_objects = parse_yolo_seg(
            base_label
        )

        synth_objects = parse_yolo_seg(
            synth_label
        )

        gt_fire = gt_mask(
            base_objects,
            0,
            h,
            w,
        )

        gt_obstacle = gt_mask(
            synth_objects,
            1,
            h,
            w,
        )

        pred_fire, pred_obstacle = predict_masks(
            model,
            image_path,
            h,
            w,
        )

        # Ground truth q
        q_gt = compute_q(
            gt_fire,
            gt_obstacle,
        )

        # A: GT fire + predicted obstacle
        q_A = compute_q(
            gt_fire,
            pred_obstacle,
        )

        # B: predicted fire + GT obstacle
        q_B = compute_q(
            pred_fire,
            gt_obstacle,
        )

        # C: fully predicted
        q_C = compute_q(
            pred_fire,
            pred_obstacle,
        )

        rows.append({
            "image": image_path.name,
            "q_gt": q_gt,
            "q_A_gtfire_predobs": q_A,
            "q_B_predfire_gtobs": q_B,
            "q_C_predfire_predobs": q_C,
        })

        print(
            f"[{idx:02d}/{len(labels)}] "
            f"{image_path.name:22s} "
            f"GT={q_gt:.3f} "
            f"A={q_A:.3f} "
            f"B={q_B:.3f} "
            f"C={q_C:.3f}"
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
                "q_gt",
                "q_A_gtfire_predobs",
                "q_B_predfire_gtobs",
                "q_C_predfire_predobs",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    gt = [
        r["q_gt"]
        for r in rows
    ]

    variants = {
        "A: GT fire + Pred obstacle":
            [r["q_A_gtfire_predobs"] for r in rows],

        "B: Pred fire + GT obstacle":
            [r["q_B_predfire_gtobs"] for r in rows],

        "C: Pred fire + Pred obstacle":
            [r["q_C_predfire_predobs"] for r in rows],
    }

    print()
    print("=" * 76)
    print("Q COMPONENT ANALYSIS")
    print("=" * 76)

    for name, pred in variants.items():
        mae, rmse, corr = metrics(
            gt,
            pred,
        )

        print()
        print(name)
        print(f"  MAE         = {mae:.4f}")
        print(f"  RMSE        = {rmse:.4f}")
        print(f"  Correlation = {corr:.4f}")

    print()
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
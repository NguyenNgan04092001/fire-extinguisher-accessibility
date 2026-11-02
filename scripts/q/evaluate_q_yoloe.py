#!/usr/bin/env python3

from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE

from scripts.q.compute_occlusion_q_instances import (
    read_image,
    parse_yolo_seg,
    polygon_mask,
    infer_base_stem,
)

from scripts.r.evaluate_r_yoloe import (
    MODEL_NAME,
    PROMPTS,
)


IMAGE_DIR = Path("data/cvat_synthetic/images")
LABEL_DIR = Path("annotations/cvat_synthetic/labels/train")

FIRE_MODEL_PATH = "models/best.pt"

OUT = Path("outputs/q_evaluation_yoloe.csv")

CONF = 0.20
IMGSZ = 640


def find_image(stem):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = IMAGE_DIR / f"{stem}{ext}"

        if path.exists():
            return path

    return None


def build_gt_mask(objects, class_id, h, w):
    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    for cls, poly in objects:
        if cls != class_id:
            continue

        mask |= polygon_mask(
            poly,
            h,
            w,
        )

    return mask


def predict_fire_mask(
    model,
    image_path,
    h,
    w,
):
    result = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=IMGSZ,
        device="cpu",
        retina_masks=True,
        verbose=False,
    )[0]

    fire_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    count = 0

    if (
        result.boxes is None
        or result.masks is None
    ):
        return fire_mask, count

    masks = result.masks.data.cpu().numpy()

    for i in range(len(result.boxes)):
        cls_id = int(
            result.boxes.cls[i]
        )

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

        fire_mask |= mask

        count += 1

    return fire_mask, count


def predict_yoloe_obstacle_mask(
    model,
    image_path,
    h,
    w,
):
    result = model.predict(
        source=str(image_path),
        conf=CONF,
        imgsz=IMGSZ,
        device="cpu",
        retina_masks=True,
        verbose=False,
    )[0]

    obstacle_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    detected_names = []

    if (
        result.boxes is None
        or result.masks is None
    ):
        return obstacle_mask, detected_names

    masks = result.masks.data.cpu().numpy()

    for i in range(len(result.boxes)):
        cls_id = int(
            result.boxes.cls[i]
        )

        conf = float(
            result.boxes.conf[i]
        )

        class_name = result.names[
            cls_id
        ]

        mask = (
            masks[i] > 0.5
        ).astype(np.uint8)

        if mask.shape != (h, w):
            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        area = int(
            mask.sum()
        )

        if area < 100:
            continue

        # Avoid enormous masks such as
        # walls/floor/background-like regions.
        image_area = h * w

        if area / image_area > 0.40:
            continue

        obstacle_mask |= mask

        detected_names.append(
            f"{class_name}:{conf:.2f}"
        )

    return (
        obstacle_mask,
        detected_names,
    )


def compute_q(
    fire_mask,
    obstacle_mask,
):
    fire_area = int(
        fire_mask.sum()
    )

    if fire_area == 0:
        return 0.0

    overlap = int(
        (
            fire_mask
            & obstacle_mask
        ).sum()
    )

    return overlap / fire_area


def calculate_metrics(gt, pred):
    gt = np.asarray(
        gt,
        dtype=float,
    )

    pred = np.asarray(
        pred,
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

    return mae, rmse, corr


def main():
    print("Loading fire model...")

    fire_model = YOLO(
        FIRE_MODEL_PATH
    )

    print("Loading YOLOE...")

    yoloe_model = YOLOE(
        MODEL_NAME
    )

    print("Setting prompts...")

    yoloe_model.set_classes(
        PROMPTS
    )

    print(
        f"Prompts: {len(PROMPTS)}"
    )

    print(
        f"YOLOE confidence: {CONF}"
    )

    print()

    labels = sorted(
        LABEL_DIR.glob("*.txt")
    )

    rows = []

    for idx, synth_label in enumerate(
        labels,
        start=1,
    ):
        stem = synth_label.stem

        image_path = find_image(
            stem
        )

        if image_path is None:
            print(
                f"[SKIP] image missing: "
                f"{stem}"
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

        # -------------------------
        # GROUND TRUTH
        # -------------------------

        gt_fire = build_gt_mask(
            base_objects,
            class_id=0,
            h=h,
            w=w,
        )

        gt_obstacle = build_gt_mask(
            synth_objects,
            class_id=1,
            h=h,
            w=w,
        )

        q_gt = compute_q(
            gt_fire,
            gt_obstacle,
        )

        # -------------------------
        # PREDICTED FIRE
        # -------------------------

        pred_fire, n_fire = (
            predict_fire_mask(
                fire_model,
                image_path,
                h,
                w,
            )
        )

        # -------------------------
        # YOLOE OBSTACLE
        # -------------------------

        pred_obstacle, names = (
            predict_yoloe_obstacle_mask(
                yoloe_model,
                image_path,
                h,
                w,
            )
        )

        # -------------------------
        # FULL AUTOMATIC Q
        # -------------------------

        q_pred = compute_q(
            pred_fire,
            pred_obstacle,
        )

        error = abs(
            q_pred - q_gt
        )

        rows.append({
            "image": image_path.name,
            "q_gt": q_gt,
            "q_pred": q_pred,
            "error": error,
            "fire_predictions": n_fire,
            "yoloe_objects":
                ";".join(names),
        })

        print(
            f"[{idx:02d}/{len(labels)}] "
            f"{image_path.name:22s} "
            f"GT={q_gt:.3f} "
            f"YOLOE={q_pred:.3f} "
            f"err={error:.3f} "
            f"fire={n_fire} "
            f"objects=[{'; '.join(names)}]"
        )

    # -------------------------
    # SAVE CSV
    # -------------------------

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
                "q_pred",
                "error",
                "fire_predictions",
                "yoloe_objects",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # -------------------------
    # METRICS
    # -------------------------

    gt = [
        r["q_gt"]
        for r in rows
    ]

    pred = [
        r["q_pred"]
        for r in rows
    ]

    mae, rmse, corr = (
        calculate_metrics(
            gt,
            pred,
        )
    )

    print()
    print("=" * 72)
    print("YOLOE Q EVALUATION")
    print("=" * 72)

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

    print(
        "Reference oracle:"
    )

    print(
        "Pred fire + GT obstacle:"
    )

    print(
        "MAE=0.0124, "
        "RMSE=0.0341, "
        "corr=0.9749"
    )

    print()

    print(
        f"Saved: {OUT}"
    )


if __name__ == "__main__":
    main()
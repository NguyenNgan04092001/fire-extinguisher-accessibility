#!/usr/bin/env python3

from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLOE

from compute_access_obstruction import create_access_zones
from evaluate_r_gt import load_gt_mask
from evaluate_r_yoloe import MODEL_NAME, PROMPTS


META = Path("metadata/real_extension_v3.csv")
IMAGE_ROOT = Path("data/real_extension_v3/selected")
ANN_ROOT = Path("annotations/cvat_real_extension_v3")

# YOLOE chạy một lần ở threshold thấp,
# sau đó ta sweep confidence bằng Python.
INFERENCE_CONF = 0.05

THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]


def compute_r(mask, z1, z2, z2_valid):
    r1 = int((mask & z1).sum()) / max(int(z1.sum()), 1)

    if z2_valid:
        r2 = int((mask & z2).sum()) / max(int(z2.sum()), 1)
        r = 0.7 * r1 + 0.3 * r2
    else:
        r2 = None
        r = r1

    return r


def main():
    model = YOLOE(MODEL_NAME)
    model.set_classes(PROMPTS)

    with META.open(encoding="utf-8", newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["split"] in {"train", "val"}
        ]

    print(f"Tuning images: {len(rows)}")
    print("Splits: train + val only")
    print()

    samples = []

    for idx, row in enumerate(rows, 1):
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
        h, w = image.shape[:2]

        fire = load_gt_mask(
            label_path,
            image.shape,
            class_id=0,
        )

        gt_obstacle = load_gt_mask(
            label_path,
            image.shape,
            class_id=1,
        )

        z1, z2, info = create_access_zones(fire)

        r_gt = compute_r(
            gt_obstacle,
            z1,
            z2,
            info["z2_valid"],
        )

        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=INFERENCE_CONF,
            iou=0.7,
            retina_masks=True,
            device="cpu",
            verbose=False,
        )[0]

        detections = []

        if (
            result.boxes is not None
            and result.masks is not None
        ):
            masks = result.masks.data.cpu().numpy()

            fire_area = int(fire.sum())

            for i in range(len(result.boxes)):
                confidence = float(result.boxes.conf[i])
                cls_id = int(result.boxes.cls[i])
                class_name = result.names[cls_id]

                mask = (masks[i] > 0.5).astype(np.uint8)

                if mask.shape != (h, w):
                    mask = cv2.resize(
                        mask,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                area = int(mask.sum())

                if area == 0:
                    continue

                fire_inter = int((mask & fire).sum())

                if (
                    fire_inter / max(area, 1) > 0.50
                    or fire_inter / max(fire_area, 1) > 0.30
                ):
                    continue

                z1_inter = int((mask & z1).sum())
                z2_inter = int((mask & z2).sum())

                if z1_inter + z2_inter < 25:
                    continue

                detections.append({
                    "confidence": confidence,
                    "class_name": class_name,
                    "mask": mask,
                })

        samples.append({
            "name": name,
            "scenario": scenario,
            "r_gt": r_gt,
            "z1": z1,
            "z2": z2,
            "z2_valid": info["z2_valid"],
            "fire": fire,
            "detections": detections,
        })

        print(
            f"[{idx:02d}/{len(rows)}] "
            f"{name} {scenario} "
            f"detections={len(detections)}"
        )

    print()
    print("=" * 76)
    print("CONFIDENCE SWEEP — TRAIN + VAL ONLY")
    print("=" * 76)

    best = None

    for threshold in THRESHOLDS:
        preds = []
        gts = []

        for sample in samples:
            obstacle = np.zeros_like(
                sample["fire"],
                dtype=np.uint8,
            )

            for det in sample["detections"]:
                if det["confidence"] >= threshold:
                    obstacle = np.maximum(
                        obstacle,
                        det["mask"],
                    )

            obstacle[sample["fire"] > 0] = 0

            r_pred = compute_r(
                obstacle,
                sample["z1"],
                sample["z2"],
                sample["z2_valid"],
            )

            preds.append(r_pred)
            gts.append(sample["r_gt"])

        preds = np.asarray(preds, dtype=float)
        gts = np.asarray(gts, dtype=float)

        mae = np.mean(np.abs(preds - gts))
        rmse = np.sqrt(np.mean((preds - gts) ** 2))

        if np.std(preds) > 0 and np.std(gts) > 0:
            corr = np.corrcoef(preds, gts)[0, 1]
        else:
            corr = float("nan")

        print(
            f"conf={threshold:.2f} "
            f"MAE={mae:.4f} "
            f"RMSE={rmse:.4f} "
            f"corr={corr:.4f}"
        )

        # Primary selection criterion = MAE
        if best is None or mae < best["mae"]:
            best = {
                "threshold": threshold,
                "mae": mae,
                "rmse": rmse,
                "corr": corr,
            }

    print()
    print("=" * 76)
    print("BEST VALIDATION SETTING")
    print("=" * 76)
    print(
        f"conf = {best['threshold']:.2f}\n"
        f"MAE  = {best['mae']:.4f}\n"
        f"RMSE = {best['rmse']:.4f}\n"
        f"corr = {best['corr']:.4f}"
    )


if __name__ == "__main__":
    main()
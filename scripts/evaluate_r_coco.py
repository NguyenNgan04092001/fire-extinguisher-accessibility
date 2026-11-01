#!/usr/bin/env python3

from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO

from compute_access_obstruction import create_access_zones
from evaluate_r_gt import load_gt_mask


META = Path("metadata/real_extension_v3.csv")
IMAGE_ROOT = Path("data/real_extension_v3/selected")
ANN_ROOT = Path("annotations/cvat_real_extension_v3")

OUTPUT_CSV = Path("outputs/r_evaluation_coco.csv")

# COCO-pretrained instance segmentation model
COCO_MODEL = "yolo11n-seg.pt"

CONF = 0.25
IMGSZ = 640

# Ignore tiny accidental intersections with access zones
MIN_ZONE_PIXELS = 25


def compute_r(
    obstacle_mask,
    z1,
    z2,
    z2_valid,
):
    z1_overlap = int(
        (obstacle_mask & z1).sum()
    )

    z2_overlap = int(
        (obstacle_mask & z2).sum()
    )

    r1 = (
        z1_overlap
        / max(int(z1.sum()), 1)
    )

    if z2_valid:
        r2 = (
            z2_overlap
            / max(int(z2.sum()), 1)
        )

        r = 0.7 * r1 + 0.3 * r2

    else:
        r2 = None
        r = r1

    return r1, r2, r


def get_coco_obstacle_mask(
    model,
    image_path,
    fire_mask,
    z1,
    z2,
):
    """
    Run COCO-pretrained YOLO11 segmentation.

    A detected COCO object becomes a potential obstacle only
    when its segmentation mask intersects Z1 or Z2.

    Object identity itself does NOT determine obstruction.
    """

    h, w = fire_mask.shape

    result = model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        conf=CONF,
        iou=0.7,
        retina_masks=True,
        device="cpu",
        verbose=False,
    )[0]

    obstacle_union = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    candidates = []

    if (
        result.boxes is None
        or result.masks is None
        or len(result.boxes) == 0
    ):
        return obstacle_union, candidates

    masks = result.masks.data.cpu().numpy()

    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i])
        conf = float(result.boxes.conf[i])

        class_name = result.names[cls_id]

        mask = (
            masks[i] > 0.5
        ).astype(np.uint8)

        # Safety in case mask resolution differs
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        area = int(mask.sum())

        if area == 0:
            continue

        fire_inter = int(
            (mask & fire_mask).sum()
        )

        fire_area = int(
            fire_mask.sum()
        )

        # Reject a COCO object if it is probably the
        # extinguisher itself misclassified as e.g. bottle.
        overlap_object = (
            fire_inter / max(area, 1)
        )

        overlap_fire = (
            fire_inter / max(fire_area, 1)
        )

        if (
            overlap_object > 0.50
            or overlap_fire > 0.30
        ):
            continue

        inter_z1 = int(
            (mask & z1).sum()
        )

        inter_z2 = int(
            (mask & z2).sum()
        )

        zone_intersection = (
            inter_z1 + inter_z2
        )

        if zone_intersection < MIN_ZONE_PIXELS:
            continue

        # Spatial relation determines whether this
        # COCO object is considered a potential obstacle.
        obstacle_union = np.maximum(
            obstacle_union,
            mask,
        )

        candidates.append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": conf,
            "area": area,
            "z1_overlap": inter_z1,
            "z2_overlap": inter_z2,
            "zone_overlap": zone_intersection,
        })

    # Never count the extinguisher itself
    obstacle_union[
        fire_mask > 0
    ] = 0

    return obstacle_union, candidates


def main():
    print("Loading COCO segmentation model...")
    coco_model = YOLO(COCO_MODEL)

    with META.open(
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    output_rows = []

    coco_values = []
    gt_values = []

    print()

    for idx, row in enumerate(
        rows,
        start=1,
    ):
        name = row["file_name"]
        scenario = row["scenario"]
        split = row["split"]

        image_path = (
            IMAGE_ROOT
            / scenario
            / name
        )

        label_path = (
            ANN_ROOT
            / split
            / "labels"
            / f"{Path(name).stem}.txt"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        # ------------------------------------------
        # GT fire mask
        # ------------------------------------------

        fire_mask = load_gt_mask(
            label_path,
            image.shape,
            class_id=0,
        )

        if fire_mask.sum() == 0:
            raise RuntimeError(
                f"No GT fire mask: {name}"
            )

        # ------------------------------------------
        # Access zones
        # ------------------------------------------

        z1, z2, info = (
            create_access_zones(
                fire_mask
            )
        )

        # ------------------------------------------
        # COCO object masks
        # ------------------------------------------

        coco_mask, candidates = (
            get_coco_obstacle_mask(
                model=coco_model,
                image_path=image_path,
                fire_mask=fire_mask,
                z1=z1,
                z2=z2,
            )
        )

        r1_coco, r2_coco, r_coco = (
            compute_r(
                coco_mask,
                z1,
                z2,
                info["z2_valid"],
            )
        )

        # ------------------------------------------
        # GT obstacle reference
        # ------------------------------------------

        gt_obstacle = load_gt_mask(
            label_path,
            image.shape,
            class_id=1,
        )

        r1_gt, r2_gt, r_gt = (
            compute_r(
                gt_obstacle,
                z1,
                z2,
                info["z2_valid"],
            )
        )

        error = abs(
            r_coco - r_gt
        )

        coco_values.append(r_coco)
        gt_values.append(r_gt)

        classes = sorted(
            set(
                c["class_name"]
                for c in candidates
            )
        )

        class_text = ";".join(classes)

        print(
            f"[{idx:02d}/28] "
            f"{name} {scenario} "
            f"COCO={r_coco:.3f} "
            f"GT={r_gt:.3f} "
            f"err={error:.3f} "
            f"n_obj={len(candidates):2d} "
            f"[{class_text}]"
        )

        output_rows.append({
            "file_name": name,
            "scenario": scenario,
            "split": split,

            "z2_valid": (
                info["z2_valid"]
            ),

            "candidate_count": (
                len(candidates)
            ),

            "candidate_classes": (
                class_text
            ),

            "r1_coco": round(
                r1_coco,
                6,
            ),

            "r2_coco": (
                ""
                if r2_coco is None
                else round(
                    r2_coco,
                    6,
                )
            ),

            "r_coco": round(
                r_coco,
                6,
            ),

            "r1_gt": round(
                r1_gt,
                6,
            ),

            "r2_gt": (
                ""
                if r2_gt is None
                else round(
                    r2_gt,
                    6,
                )
            ),

            "r_gt": round(
                r_gt,
                6,
            ),

            "abs_error": round(
                error,
                6,
            ),
        })

    # ----------------------------------------------
    # Save CSV
    # ----------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "file_name",
        "scenario",
        "split",
        "z2_valid",
        "candidate_count",
        "candidate_classes",
        "r1_coco",
        "r2_coco",
        "r_coco",
        "r1_gt",
        "r2_gt",
        "r_gt",
        "abs_error",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    # ----------------------------------------------
    # Global comparison
    # ----------------------------------------------

    coco_array = np.array(
        coco_values,
        dtype=float,
    )

    gt_array = np.array(
        gt_values,
        dtype=float,
    )

    mae = np.mean(
        np.abs(
            coco_array
            - gt_array
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                coco_array
                - gt_array
            ) ** 2
        )
    )

    if (
        np.std(coco_array) > 0
        and np.std(gt_array) > 0
    ):
        correlation = np.corrcoef(
            coco_array,
            gt_array,
        )[0, 1]
    else:
        correlation = float("nan")

    print()
    print("=" * 70)
    print("COCO R EVALUATION SUMMARY")
    print("=" * 70)

    for scenario in [
        "S0",
        "S1",
        "S2",
        "S3",
    ]:

        coco_s = [
            float(r["r_coco"])
            for r in output_rows
            if r["scenario"]
            == scenario
        ]

        gt_s = [
            float(r["r_gt"])
            for r in output_rows
            if r["scenario"]
            == scenario
        ]

        print(
            f"{scenario}: "
            f"COCO mean={np.mean(coco_s):.3f}, "
            f"median={np.median(coco_s):.3f} | "
            f"GT mean={np.mean(gt_s):.3f}, "
            f"median={np.median(gt_s):.3f}"
        )

    print()
    print(
        f"MAE         = {mae:.4f}"
    )

    print(
        f"RMSE        = {rmse:.4f}"
    )

    print(
        f"Correlation = {correlation:.4f}"
    )

    print()
    print(
        "Saved:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLOE

from compute_access_obstruction import create_access_zones
from evaluate_r_gt import load_gt_mask


META = Path("metadata/real_extension_v3.csv")
IMAGE_ROOT = Path("data/real_extension_v3/selected")
ANN_ROOT = Path("annotations/cvat_real_extension_v3")

OUTPUT_CSV = Path("outputs/r_evaluation_yoloe.csv")

MODEL_NAME = "yoloe-11s-seg.pt"

CONF = 0.20
IMGSZ = 640
MIN_ZONE_PIXELS = 25


# Chỉ prompt các vật thể có khả năng là vật cản di động.
# Cố ý KHÔNG đưa cabinet, wall, door, shelf...
# vì chúng gây false positive mạnh ở S1.
PROMPTS = [
    "chair",
    "armchair",
    "sofa",
    "couch",
    "table",
    "desk",
    "bench",
    "stool",
    "suitcase",
    "luggage",
    "bag",
    "backpack",
    "box",
    "cart",
    "trolley",
    "vacuum cleaner",
    "cleaning equipment",
    "tool",
    "machine",
    "equipment",
    "musical instrument",
    "plant pot",
    "trash bin",
]


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

        r = (
            0.7 * r1
            + 0.3 * r2
        )

    else:
        r2 = None
        r = r1

    return r1, r2, r


def get_yoloe_obstacle_mask(
    model,
    image_path,
    fire_mask,
    z1,
    z2,
):
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

    masks = (
        result.masks.data
        .cpu()
        .numpy()
    )

    fire_area = int(
        fire_mask.sum()
    )

    for i in range(
        len(result.boxes)
    ):
        cls_id = int(
            result.boxes.cls[i]
        )

        confidence = float(
            result.boxes.conf[i]
        )

        class_name = (
            result.names[cls_id]
        )

        mask = (
            masks[i] > 0.5
        ).astype(np.uint8)

        if mask.shape != (h, w):
            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        area = int(mask.sum())

        if area == 0:
            continue

        # ----------------------------------
        # Reject masks that are probably
        # the fire extinguisher itself
        # ----------------------------------

        fire_inter = int(
            (mask & fire_mask).sum()
        )

        object_fire_overlap = (
            fire_inter
            / max(area, 1)
        )

        fire_coverage = (
            fire_inter
            / max(fire_area, 1)
        )

        if (
            object_fire_overlap > 0.50
            or fire_coverage > 0.30
        ):
            continue

        # ----------------------------------
        # Spatial relation with Z1/Z2
        # ----------------------------------

        inter_z1 = int(
            (mask & z1).sum()
        )

        inter_z2 = int(
            (mask & z2).sum()
        )

        zone_overlap = (
            inter_z1
            + inter_z2
        )

        if (
            zone_overlap
            < MIN_ZONE_PIXELS
        ):
            continue

        # Object identity alone does not
        # determine obstruction.
        # It becomes a candidate because
        # it intersects the access zone.
        obstacle_union = np.maximum(
            obstacle_union,
            mask,
        )

        candidates.append({
            "class_name": class_name,
            "confidence": confidence,
            "area": area,
            "z1_overlap": inter_z1,
            "z2_overlap": inter_z2,
            "zone_overlap": zone_overlap,
        })

    obstacle_union[
        fire_mask > 0
    ] = 0

    return obstacle_union, candidates


def main():

    print(
        "Loading YOLOE model..."
    )

    model = YOLOE(
        MODEL_NAME
    )

    print(
        "Setting text prompts..."
    )

    model.set_classes(
        PROMPTS
    )

    print(
        f"Prompts: {len(PROMPTS)}"
    )

    with META.open(
        encoding="utf-8",
        newline="",
    ) as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row["split"] == "test"
        ]

    output_rows = []

    yoloe_values = []
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
                f"Cannot read: {image_path}"
            )

        # ----------------------------------
        # GT fire
        # ----------------------------------

        fire_mask = load_gt_mask(
            label_path,
            image.shape,
            class_id=0,
        )

        if fire_mask.sum() == 0:
            raise RuntimeError(
                f"No GT fire: {name}"
            )

        # ----------------------------------
        # Access zones
        # ----------------------------------

        z1, z2, info = (
            create_access_zones(
                fire_mask
            )
        )

        # ----------------------------------
        # YOLOE candidate masks
        # ----------------------------------

        yoloe_mask, candidates = (
            get_yoloe_obstacle_mask(
                model=model,
                image_path=image_path,
                fire_mask=fire_mask,
                z1=z1,
                z2=z2,
            )
        )

        (
            r1_yoloe,
            r2_yoloe,
            r_yoloe,
        ) = compute_r(
            yoloe_mask,
            z1,
            z2,
            info["z2_valid"],
        )

        # ----------------------------------
        # GT reference
        # ----------------------------------

        gt_obstacle = load_gt_mask(
            label_path,
            image.shape,
            class_id=1,
        )

        (
            r1_gt,
            r2_gt,
            r_gt,
        ) = compute_r(
            gt_obstacle,
            z1,
            z2,
            info["z2_valid"],
        )

        error = abs(
            r_yoloe - r_gt
        )

        yoloe_values.append(
            r_yoloe
        )

        gt_values.append(
            r_gt
        )

        classes = sorted(
            set(
                c["class_name"]
                for c in candidates
            )
        )

        class_text = ";".join(
            classes
        )

        print(
            f"[{idx:02d}/{len(rows)}] "
            f"{name} {scenario} "
            f"YOLOE={r_yoloe:.3f} "
            f"GT={r_gt:.3f} "
            f"err={error:.3f} "
            f"n={len(candidates):2d} "
            f"[{class_text}]"
        )

        output_rows.append({
            "file_name": name,
            "scenario": scenario,
            "split": split,

            "candidate_count":
                len(candidates),

            "candidate_classes":
                class_text,

            "r1_yoloe":
                round(r1_yoloe, 6),

            "r2_yoloe":
                ""
                if r2_yoloe is None
                else round(
                    r2_yoloe,
                    6,
                ),

            "r_yoloe":
                round(r_yoloe, 6),

            "r1_gt":
                round(r1_gt, 6),

            "r2_gt":
                ""
                if r2_gt is None
                else round(
                    r2_gt,
                    6,
                ),

            "r_gt":
                round(r_gt, 6),

            "abs_error":
                round(error, 6),
        })

    # ==================================
    # SAVE CSV
    # ==================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "file_name",
        "scenario",
        "split",
        "candidate_count",
        "candidate_classes",
        "r1_yoloe",
        "r2_yoloe",
        "r_yoloe",
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

    # ==================================
    # METRICS
    # ==================================

    y = np.array(
        yoloe_values,
        dtype=float,
    )

    gt = np.array(
        gt_values,
        dtype=float,
    )

    mae = np.mean(
        np.abs(y - gt)
    )

    rmse = np.sqrt(
        np.mean(
            (y - gt) ** 2
        )
    )

    if (
        np.std(y) > 0
        and np.std(gt) > 0
    ):
        correlation = (
            np.corrcoef(
                y,
                gt,
            )[0, 1]
        )
    else:
        correlation = (
            float("nan")
        )

    print()
    print("=" * 70)
    print(
        "YOLOE R EVALUATION SUMMARY"
    )
    print("=" * 70)

    for scenario in [
        "S0",
        "S1",
        "S2",
        "S3",
    ]:

        pred_s = [
            float(r["r_yoloe"])
            for r in output_rows
            if (
                r["scenario"]
                == scenario
            )
        ]

        gt_s = [
            float(r["r_gt"])
            for r in output_rows
            if (
                r["scenario"]
                == scenario
            )
        ]

        print(
            f"{scenario}: "
            f"YOLOE mean="
            f"{np.mean(pred_s):.3f}, "
            f"median="
            f"{np.median(pred_s):.3f} | "
            f"GT mean="
            f"{np.mean(gt_s):.3f}, "
            f"median="
            f"{np.median(gt_s):.3f}"
        )

    print()

    print(
        f"MAE         = "
        f"{mae:.4f}"
    )

    print(
        f"RMSE        = "
        f"{rmse:.4f}"
    )

    print(
        f"Correlation = "
        f"{correlation:.4f}"
    )

    print()

    print(
        "Saved:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
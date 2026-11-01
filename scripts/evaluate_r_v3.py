#!/usr/bin/env python3

from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO, FastSAM

from compute_access_obstruction import (
    predict_fire_mask,
    create_access_zones,
    compute_obstruction_mask,
)

def load_gt_fire_mask(label_path, image_shape):
    h, w = image_shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.split()

        if not parts:
            continue

        cls_id = int(float(parts[0]))

        if cls_id != 0:
            continue

        coords = list(
            map(float, parts[1:])
        )

        pts = np.array(
            [
                [
                    coords[i] * w,
                    coords[i + 1] * h,
                ]
                for i in range(
                    0,
                    len(coords),
                    2,
                )
            ],
            dtype=np.int32,
        )

        if len(pts) >= 3:
            cv2.fillPoly(
                mask,
                [pts],
                1,
            )

    if mask.sum() == 0:
        raise RuntimeError(
            f"No GT fire mask: {label_path}"
        )

    return mask


MODEL_PATH = Path("models/best.pt")
METADATA_PATH = Path("metadata/real_extension_v3.csv")
IMAGE_ROOT = Path("data/real_extension_v3/selected")
OUTPUT_PATH = Path("outputs/r_evaluation_v3.csv")



def main():
    fire_model = YOLO(str(MODEL_PATH))
    fastsam_model = FastSAM("FastSAM-s.pt")

    with METADATA_PATH.open(
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    results = []

    for n, row in enumerate(rows, start=1):
        name = row["file_name"]
        scenario = row["scenario"]
        split = row["split"]

        image_path = (
            IMAGE_ROOT
            / scenario
            / name
        )

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[{n:02d}/28] {name}: READ ERROR")
            continue

        try:
            label_path = (
                Path("annotations/cvat_real_extension_v3")
                / split
                / "labels"
                / f"{Path(name).stem}.txt"
            )

            fire_mask = load_gt_fire_mask(
                label_path,
                image.shape,
            )

            fire_conf = 1.0

        except Exception as exc:
            print(
                f"[{n:02d}/28] "
                f"{name} {scenario}: "
                f"FIRE NOT DETECTED"
            )

            results.append({
                "file_name": name,
                "scenario": scenario,
                "split": split,
                "fire_confidence": "",
                "z2_visibility": "",
                "z2_valid": "",
                "fastsam_mask_id": "",
                "fastsam_score": "",
                "r1": "",
                "r2": "",
                "r": "",
                "status": "fire_not_detected",
            })

            continue

        z1, z2, info = create_access_zones(
            fire_mask
        )

        obstacle_mask, obstacle_info = (
            compute_obstruction_mask(
                image_path=image_path,
                fire_mask=fire_mask,
                z1=z1,
                z2=z2,
                fastsam_model=fastsam_model,
            )
        )

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

        if info["z2_valid"]:
            r2 = (
                z2_overlap
                / max(int(z2.sum()), 1)
            )

            r = 0.7 * r1 + 0.3 * r2

        else:
            r2 = None
            r = r1

        mask_id = (
            obstacle_info["id"]
            if obstacle_info
            else None
        )

        score = (
            obstacle_info["score"]
            if obstacle_info
            else 0.0
        )

        print(
            f"[{n:02d}/28] "
            f"{name} {scenario} "
            f"r={r:.3f} "
            f"r1={r1:.3f} "
            f"r2="
            f"{'NA' if r2 is None else f'{r2:.3f}'} "
            f"score={score:.4f}"
        )

        results.append({
            "file_name": name,
            "scenario": scenario,
            "split": split,
            "fire_confidence": round(
                fire_conf, 6
            ),
            "z2_visibility": round(
                info["z2_visibility"], 6
            ),
            "z2_valid": info["z2_valid"],
            "fastsam_mask_id": (
                "" if mask_id is None else mask_id
            ),
            "fastsam_score": round(
                score, 6
            ),
            "r1": round(r1, 6),
            "r2": (
                ""
                if r2 is None
                else round(r2, 6)
            ),
            "r": round(r, 6),
            "status": "ok",
        })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "file_name",
        "scenario",
        "split",
        "fire_confidence",
        "z2_visibility",
        "z2_valid",
        "fastsam_mask_id",
        "fastsam_score",
        "r1",
        "r2",
        "r",
        "status",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 60)
    print("R EVALUATION SUMMARY")
    print("=" * 60)

    for scenario in ["S0", "S1", "S2", "S3"]:
        values = [
            float(x["r"])
            for x in results
            if x["scenario"] == scenario
            and x["status"] == "ok"
        ]

        failures = sum(
            1
            for x in results
            if x["scenario"] == scenario
            and x["status"] != "ok"
        )

        if values:
            print(
                f"{scenario}: "
                f"n={len(values)}, "
                f"mean={np.mean(values):.3f}, "
                f"median={np.median(values):.3f}, "
                f"min={np.min(values):.3f}, "
                f"max={np.max(values):.3f}, "
                f"fail={failures}"
            )
        else:
            print(
                f"{scenario}: "
                f"no valid results, "
                f"fail={failures}"
            )

    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
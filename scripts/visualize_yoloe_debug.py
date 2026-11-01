from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

from compute_access_obstruction import create_access_zones
from evaluate_r_gt import load_gt_mask
from evaluate_r_yoloe import (
    MODEL_NAME,
    PROMPTS,
    get_yoloe_obstacle_mask,
)


IMAGE_ROOT = Path("data/real_extension_v3/selected")
ANN_ROOT = Path("annotations/cvat_real_extension_v3")
OUT = Path("outputs/yoloe_debug")

CASES = [
    ("S1", "test", "v3_0017.png"),
    ("S1", "val",  "v3_0018.png"),
    ("S1", "test", "v3_0020.png"),
    ("S2", "test", "v3_0023.png"),
    ("S3", "train", "v3_0025.png"),
    ("S3", "test", "v3_0026.png"),
]


def overlay(image, mask, color, alpha):
    m = mask.astype(bool)

    image[m] = (
        (1 - alpha) * image[m]
        + alpha * np.array(color)
    ).astype(np.uint8)


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLOE(MODEL_NAME)
    model.set_classes(PROMPTS)

    for scenario, split, name in CASES:
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

        z1, z2, info = create_access_zones(
            fire
        )

        pred_obstacle, candidates = (
            get_yoloe_obstacle_mask(
                model=model,
                image_path=image_path,
                fire_mask=fire,
                z1=z1,
                z2=z2,
            )
        )

        # -------------------------
        # LEFT = GT
        # -------------------------

        gt_vis = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        overlay(
            gt_vis,
            z2,
            [255, 255, 0],
            0.25,
        )

        overlay(
            gt_vis,
            z1,
            [255, 0, 0],
            0.30,
        )

        overlay(
            gt_vis,
            gt_obstacle,
            [0, 0, 255],
            0.65,
        )

        overlay(
            gt_vis,
            fire,
            [0, 255, 0],
            0.70,
        )

        # -------------------------
        # RIGHT = YOLOE
        # -------------------------

        pred_vis = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        overlay(
            pred_vis,
            z2,
            [255, 255, 0],
            0.25,
        )

        overlay(
            pred_vis,
            z1,
            [255, 0, 0],
            0.30,
        )

        overlay(
            pred_vis,
            pred_obstacle,
            [255, 0, 255],
            0.65,
        )

        overlay(
            pred_vis,
            fire,
            [0, 255, 0],
            0.70,
        )

        names = sorted(
            set(
                c["class_name"]
                for c in candidates
            )
        )

        print(
            name,
            "->",
            names
        )

        combined = np.hstack([
            gt_vis,
            pred_vis,
        ])

        combined = cv2.cvtColor(
            combined,
            cv2.COLOR_RGB2BGR,
        )

        cv2.imwrite(
            str(
                OUT
                / f"{Path(name).stem}_debug.jpg"
            ),
            combined,
        )

    print("Saved:", OUT)


if __name__ == "__main__":
    main()
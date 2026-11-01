#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
from ultralytics import YOLO, FastSAM
import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL = Path("models/best.pt")
DEFAULT_OUTPUT = Path("outputs/access_obstruction")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create directional access zones Z1/Z2 around a fire extinguisher."
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image path.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="YOLO segmentation model.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Fire extinguisher confidence threshold.",
    )

    return parser.parse_args()


def predict_fire_mask(
    model: YOLO,
    image_path: Path,
    image_shape,
    conf: float,
):
    """
    Detect fire extinguishers and return the highest-confidence
    class-0 segmentation mask.
    """

    h, w = image_shape[:2]

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=conf,
        device="cpu",
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError(
            f"No detections found in {image_path.name}"
        )

    fire_ids = []

    for i, cls_id in enumerate(result.boxes.cls):
        if int(cls_id) == 0:
            fire_ids.append(i)

    if not fire_ids:
        raise RuntimeError(
            f"No fire_extinguisher detected in {image_path.name} "
            f"at conf={conf}"
        )

    if result.masks is None:
        raise RuntimeError(
            "YOLO detected fire_extinguisher but returned no segmentation mask."
        )

    # Highest-confidence extinguisher
    best_id = max(
        fire_ids,
        key=lambda i: float(result.boxes.conf[i]),
    )

    confidence = float(result.boxes.conf[best_id])

    polygon = result.masks.xy[best_id]

    if polygon is None or len(polygon) < 3:
        raise RuntimeError("Invalid extinguisher segmentation polygon.")

    polygon = polygon.astype(np.int32)

    fire_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(
        fire_mask,
        [polygon],
        1,
    )

    return fire_mask, confidence


def create_access_zones(fire_mask):
    """
    Create directional access zones.

    Z1:
        manipulation zone around the extinguisher,
        representing space required to reach/grip/extract it.

    Z2:
        approach corridor extending downward in image space,
        representing the assumed direction from which a person approaches.

    Current assumption:
        approach direction = toward the bottom of the image.
    """

    h, w = fire_mask.shape

    ys, xs = np.where(fire_mask > 0)

    if len(xs) == 0:
        raise RuntimeError("Empty fire extinguisher mask.")

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    fire_width = x2 - x1 + 1
    fire_height = y2 - y1 + 1

    cx = (x1 + x2) // 2

    # Characteristic extinguisher scale
    s = max(fire_width, fire_height)

    # --------------------------------------------------
    # Z1: manipulation zone
    # use width/height separately
    # --------------------------------------------------

    z1 = np.zeros((h, w), dtype=np.uint8)

    margin_x = round(0.50 * fire_width)
    margin_top = round(0.15 * fire_height)
    margin_bottom = round(0.25 * fire_height)

    z1_left = max(0, x1 - margin_x)
    z1_right = min(w, x2 + margin_x)

    z1_top = max(0, y1 - margin_top)
    z1_bottom = min(h, y2 + margin_bottom)

    z1[
        z1_top:z1_bottom,
        z1_left:z1_right,
    ] = 1

    z1[fire_mask > 0] = 0

    # --------------------------------------------------
    # Z2: directional approach corridor
    #
    # Approximation:
    # a person/camera approaches from the lower-central
    # part of the image toward the extinguisher.
    # --------------------------------------------------

    z2 = np.zeros((h, w), dtype=np.uint8)

    # Bottom-center of image = approximate approach origin
    approach_x = w // 2
    approach_y = h - 1

    # Start from extinguisher centre
    fire_cy = (y1 + y2) // 2

    dx = approach_x - cx
    dy = approach_y - fire_cy

    distance = float(
        np.hypot(dx, dy)
    )

    # Maximum corridor length relative to extinguisher height
    max_length = 1.50 * fire_height

    corridor_length = min(
        distance,
        max_length,
    )

    # Corridor width
    corridor_width = max(
        1.70 * fire_width,
        0.45 * fire_height,
    )

    # Corridor is unreliable if essentially no approach
    # space is visible in front of the extinguisher
    minimum_length = 0.30 * fire_height

    if distance < 1.0 or corridor_length < minimum_length:

        z2_visibility = 0.0
        z2_valid = False

    else:

        ux = dx / distance
        uy = dy / distance

        end_x = cx + ux * corridor_length
        end_y = fire_cy + uy * corridor_length

        middle_x = (cx + end_x) / 2
        middle_y = (fire_cy + end_y) / 2

        angle = np.degrees(
            np.arctan2(uy, ux)
        )

        rect = (
            (middle_x, middle_y),
            (corridor_length, corridor_width),
            angle,
        )

        points = cv2.boxPoints(rect)
        points = np.int32(points)

        expected_z2_area = max(
            1,
            int(cv2.contourArea(points)),
        )

        raw_z2 = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        cv2.fillPoly(
            raw_z2,
            [points],
            1,
        )

        visible_before_exclusion = int(
            raw_z2.sum()
        )

        z2_visibility = (
            visible_before_exclusion
            / expected_z2_area
        )

        z2 = raw_z2

        # Z1 has priority
        z2[z1 > 0] = 0

        # Never count extinguisher itself
        z2[fire_mask > 0] = 0

        z2_valid = (
            z2_visibility >= 0.50
        )

    info = {
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
        "fire_width": fire_width,
        "fire_height": fire_height,
        "z1_area": int(z1.sum()),
        "z2_area": int(z2.sum()),
        "z2_visibility": float(z2_visibility),
        "z2_valid": bool(z2_valid),
        "approach_anchor": [
            int(approach_x),
            int(approach_y),
        ],
        "approach_length": float(
            corridor_length
        ),
    }

    return z1, z2, info


def create_visualization(
    image_bgr,
    fire_mask,
    z1,
    z2,
):
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    vis = image_rgb.copy()

    # Z2 = yellow
    mask = z2.astype(bool)

    vis[mask] = (
        0.65 * vis[mask]
        + 0.35 * np.array([255, 255, 0])
    ).astype(np.uint8)

    # Z1 = red
    mask = z1.astype(bool)

    vis[mask] = (
        0.55 * vis[mask]
        + 0.45 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # Fire = green
    mask = fire_mask.astype(bool)

    vis[mask] = (
        0.30 * vis[mask]
        + 0.70 * np.array([0, 255, 0])
    ).astype(np.uint8)

    return cv2.cvtColor(
        vis,
        cv2.COLOR_RGB2BGR,
    )


def save_binary_mask(path, mask):
    cv2.imwrite(
        str(path),
        mask.astype(np.uint8) * 255,
    )

def compute_obstruction_mask(
    image_path,
    fire_mask,
    z1,
    z2,
    fastsam_model=None,
):
    """
    Use FastSAM to find surrounding object masks.
    Rank candidate masks by how specifically they occupy
    the access zones Z1/Z2.
    """

    h, w = fire_mask.shape
    image_area = h * w

    if fastsam_model is None:
        fastsam_model = FastSAM("FastSAM-s.pt")

    result = fastsam_model(
        str(image_path),
        device="cpu",
        retina_masks=True,
        imgsz=1024,
        conf=0.25,
        iou=0.9,
        verbose=False,
    )[0]

    if result.masks is None:
        return np.zeros_like(fire_mask), None

    fire_area = int(fire_mask.sum())
    zone_area = int(z1.sum() + z2.sum())

    candidates = []

    for idx, polygon in enumerate(result.masks.xy):
        pts = np.asarray(polygon, dtype=np.int32)

        if len(pts) < 3:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        area = int(mask.sum())

        if area < 250:
            continue

        # reject huge scene/background masks
        if area / image_area > 0.35:
            continue

        inter_fire = int((mask & fire_mask).sum())
        inter_z1 = int((mask & z1).sum())
        inter_z2 = int((mask & z2).sum())

        if inter_z1 == 0 and inter_z2 == 0:
            continue

        fire_overlap = (
            inter_fire / max(fire_area, 1)
        )

        # reject extinguisher-related masks
        if fire_overlap > 0.40:
            continue

        zone_intersection = inter_z1 + inter_z2

        specificity = (
            zone_intersection / max(area, 1)
        )

        coverage = (
            zone_intersection / max(zone_area, 1)
        )

        score = specificity * coverage

        candidates.append({
            "id": idx,
            "mask": mask,
            "area": area,
            "z1": inter_z1,
            "z2": inter_z2,
            "specificity": specificity,
            "coverage": coverage,
            "score": score,
        })

    if not candidates:
        return np.zeros_like(fire_mask), None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    best = candidates[0]

    return best["mask"], best


def main():
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}"
        )

    image = cv2.imread(str(args.image))

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {args.image}"
        )

    print(f"Image : {args.image}")
    print(f"Model : {args.model}")

    model = YOLO(str(args.model))

    fire_mask, confidence = predict_fire_mask(
        model=model,
        image_path=args.image,
        image_shape=image.shape,
        conf=args.conf,
    )

    z1, z2, info = create_access_zones(
        fire_mask
    )

    obstacle_mask, obstacle_info = compute_obstruction_mask(
        args.image,
        fire_mask,
        z1,
        z2,
    )

    z1_overlap = int(
        (obstacle_mask & z1).sum()
    )

    z2_overlap = int(
        (obstacle_mask & z2).sum()
    )

    r1 = (
        z1_overlap / max(int(z1.sum()), 1)
    )

    if info["z2_valid"]:
        r2 = (
            z2_overlap / max(int(z2.sum()), 1)
        )

        r = 0.7 * r1 + 0.3 * r2
    else:
        r2 = None
        r = r1

    info["r1"] = float(r1)
    info["r2"] = None if r2 is None else float(r2)
    info["r"] = float(r)

    if obstacle_info is not None:
        info["fastsam_mask_id"] = obstacle_info["id"]
        info["fastsam_score"] = float(
            obstacle_info["score"]
        )

    stem = args.image.stem

    output_dir = args.output / stem
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_binary_mask(
        output_dir / "fire_mask.png",
        fire_mask,
    )

    save_binary_mask(
        output_dir / "z1_mask.png",
        z1,
    )

    save_binary_mask(
        output_dir / "z2_mask.png",
        z2,
    )

    save_binary_mask(
        output_dir / "obstacle_mask.png",
        obstacle_mask,
    )

    visualization = create_visualization(
        image,
        fire_mask,
        z1,
        z2,
    )

    cv2.imwrite(
        str(output_dir / "access_zones.jpg"),
        visualization,
    )

    info["image"] = str(args.image)
    info["fire_confidence"] = confidence

    with (
        output_dir / "metadata.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            info,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Fire extinguisher detected")
    print(f"confidence = {confidence:.4f}")
    print(
        f"bbox = "
        f"{info['fire_width']} x "
        f"{info['fire_height']}"
    )
    print(f"Z2 visibility = {info['z2_visibility']:.3f}")
    print(f"Z2 valid = {info['z2_valid']}")
    print(f"Z1 area = {info['z1_area']}")
    print(f"Z2 area = {info['z2_area']}")

    print()
    print(f"Saved to: {output_dir}")
    print("  fire_mask.png")
    print("  z1_mask.png")
    print("  z2_mask.png")
    print("  access_zones.jpg")
    print("  metadata.json")

    print()
    print("Access obstruction")
    print(f"r1 = {r1:.4f}")

    if r2 is not None:
        print(f"r2 = {r2:.4f}")
    else:
        print("r2 = N/A")

    print(f"r  = {r:.4f}")

    if obstacle_info is not None:
        print(
            f"FastSAM mask = {obstacle_info['id']}, "
            f"score = {obstacle_info['score']:.4f}"
        )


if __name__ == "__main__":
    main()
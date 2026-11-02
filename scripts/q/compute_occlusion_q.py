# scripts/compute_occlusion_q.py
#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import re

import cv2
import numpy as np


def load_image_shape(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    return img, h, w


def parse_yolo_seg_file(label_path: Path):
    rows = []
    with label_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            if len(coords) < 6 or len(coords) % 2 != 0:
                raise ValueError(
                    f"Invalid polygon in {label_path} line {line_no}"
                )
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            rows.append((cls, pts))
    return rows


def polygon_to_pixels(poly_norm, w, h):
    pts = np.empty_like(poly_norm, dtype=np.int32)
    pts[:, 0] = np.clip(np.round(poly_norm[:, 0] * w), 0, w - 1).astype(np.int32)
    pts[:, 1] = np.clip(np.round(poly_norm[:, 1] * h), 0, h - 1).astype(np.int32)
    return pts


def build_mask(objects, target_class, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    count = 0
    for cls, poly in objects:
        if cls != target_class:
            continue
        pts = polygon_to_pixels(poly, w, h)
        cv2.fillPoly(mask, [pts], 255)
        count += 1
    return mask, count


def overlay_visual(image, base_fire_mask, obstacle_mask, overlap_mask):
    vis = image.astype(np.float32).copy()

    alpha = 0.40

    # BGR colors
    fire_color = np.array([0, 0, 255], dtype=np.float32)       # red
    obstacle_color = np.array([255, 0, 0], dtype=np.float32)   # blue
    overlap_color = np.array([0, 255, 0], dtype=np.float32)    # green

    # Full original extinguisher mask
    m = base_fire_mask > 0
    vis[m] = (
        (1.0 - alpha) * vis[m]
        + alpha * fire_color
    )

    # Synthetic obstacle mask
    m = obstacle_mask > 0
    vis[m] = (
        (1.0 - alpha) * vis[m]
        + alpha * obstacle_color
    )

    # Occluded extinguisher region
    # Draw last so overlap is clearly visible
    m = overlap_mask > 0
    vis[m] = (
        (1.0 - 0.70) * vis[m]
        + 0.70 * overlap_color
    )

    return np.clip(vis, 0, 255).astype(np.uint8)

def infer_base_name_from_synthetic(synth_name: str):
    m = re.match(r"^(base_\d{4})_S\d+\.(jpg|jpeg|png)$", synth_name, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Cannot infer base name from synthetic image name: {synth_name}"
        )
    return m.group(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compute q = occluded fraction of extinguisher using base full fire mask and synthetic obstacle mask."
    )
    parser.add_argument(
        "--synthetic-image",
        required=True,
        help="Path to synthetic image, e.g. data/cvat_synthetic/images/base_0015_S2.png",
    )
    parser.add_argument(
        "--synthetic-label",
        required=True,
        help="Path to synthetic label, e.g. annotations/cvat_synthetic/labels/train/base_0015_S2.txt",
    )
    parser.add_argument(
        "--base-image",
        default=None,
        help="Optional explicit base image path. If omitted, inferred from synthetic name.",
    )
    parser.add_argument(
        "--base-label",
        default=None,
        help="Optional explicit base label path. If omitted, inferred from synthetic name.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/occlusion_q",
        help="Output directory root",
    )
    args = parser.parse_args()

    synthetic_image = Path(args.synthetic_image)
    synthetic_label = Path(args.synthetic_label)

    if not synthetic_image.exists():
        raise FileNotFoundError(f"Missing synthetic image: {synthetic_image}")
    if not synthetic_label.exists():
        raise FileNotFoundError(f"Missing synthetic label: {synthetic_label}")

    base_stem = infer_base_name_from_synthetic(synthetic_image.name)

    base_image = Path(args.base_image) if args.base_image else Path(f"data/base_diverse/{base_stem}.jpg")
    base_label = Path(args.base_label) if args.base_label else Path(f"annotations/cvat_base/labels/train/{base_stem}.txt")

    if not base_image.exists():
        raise FileNotFoundError(f"Missing base image: {base_image}")
    if not base_label.exists():
        raise FileNotFoundError(f"Missing base label: {base_label}")

    synth_img, h_s, w_s = load_image_shape(synthetic_image)
    _, h_b, w_b = load_image_shape(base_image)

    if (h_s, w_s) != (h_b, w_b):
        raise ValueError(
            f"Image size mismatch: synthetic={w_s}x{h_s}, base={w_b}x{h_b}"
        )

    base_objs = parse_yolo_seg_file(base_label)
    synth_objs = parse_yolo_seg_file(synthetic_label)

    base_fire_mask, n_fire = build_mask(base_objs, target_class=0, w=w_s, h=h_s)
    obstacle_mask, n_obs = build_mask(synth_objs, target_class=1, w=w_s, h=h_s)

    if n_fire == 0:
        raise RuntimeError("Base label has no fire_extinguisher polygons (class 0)")
    if n_obs == 0:
        raise RuntimeError("Synthetic label has no obstacle polygons (class 1)")

    overlap_mask = cv2.bitwise_and(base_fire_mask, obstacle_mask)

    fire_area = int(np.count_nonzero(base_fire_mask))
    obstacle_area = int(np.count_nonzero(obstacle_mask))
    overlap_area = int(np.count_nonzero(overlap_mask))

    q = overlap_area / fire_area if fire_area > 0 else 0.0

    outdir = Path(args.outdir) / synthetic_image.stem
    outdir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(outdir / "base_fire_mask.png"), base_fire_mask)
    cv2.imwrite(str(outdir / "synthetic_obstacle_mask.png"), obstacle_mask)
    cv2.imwrite(str(outdir / "occluded_fire_mask.png"), overlap_mask)

    vis = overlay_visual(synth_img, base_fire_mask, obstacle_mask, overlap_mask)
    cv2.imwrite(str(outdir / "q_visualization.jpg"), vis)

    meta = {
        "synthetic_image": str(synthetic_image),
        "synthetic_label": str(synthetic_label),
        "base_image": str(base_image),
        "base_label": str(base_label),
        "image_width": w_s,
        "image_height": h_s,
        "num_base_fire_instances": n_fire,
        "num_synthetic_obstacle_instances": n_obs,
        "fire_area": fire_area,
        "obstacle_area": obstacle_area,
        "occluded_fire_area": overlap_area,
        "q": round(q, 6),
    }

    with (outdir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Base image   : {base_image}")
    print(f"Base label   : {base_label}")
    print(f"Synth image  : {synthetic_image}")
    print(f"Synth label  : {synthetic_label}")
    print()
    print(f"Fire area            = {fire_area}")
    print(f"Obstacle area        = {obstacle_area}")
    print(f"Occluded fire area   = {overlap_area}")
    print(f"q                    = {q:.4f}")
    print()
    print(f"Saved to: {outdir}")
    print("  base_fire_mask.png")
    print("  synthetic_obstacle_mask.png")
    print("  occluded_fire_mask.png")
    print("  q_visualization.jpg")
    print("  metadata.json")


if __name__ == "__main__":
    main()
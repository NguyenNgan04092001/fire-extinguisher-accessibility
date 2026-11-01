#!/usr/bin/env python3
from pathlib import Path
import csv
import random

import cv2
import numpy as np


# =========================
# CẤU HÌNH
# =========================

BASE_DATASET_DIR = Path("data/base_yolo")
OBSTACLE_ROOT = Path("assets/obstacles")
OUTPUT_DIR = Path("data/synthetic_s1")

SPLITS = ["train", "val", "test"]

# Mỗi ảnh base sẽ sinh 1 ảnh S1
# Nếu muốn nhiều hơn, sau này có thể đổi thành >1.
VARIANTS_PER_IMAGE = 1

# Mức che khuất mong muốn cho S1
MIN_Q = 0.15
MAX_Q = 0.45

# Thử tối đa bao nhiêu lần để đặt obstacle sao cho q nằm trong khoảng trên
MAX_TRIES = 60

# Tỷ lệ chiều cao obstacle so với chiều cao bbox của bình
# obstacle_height = scale * extinguisher_bbox_height
OBSTACLE_SCALE_MIN = 0.45
OBSTACLE_SCALE_MAX = 1.10

# Dịch chuyển ngẫu nhiên quanh tâm bình (theo tỉ lệ bbox)
SHIFT_X_RATIO = 0.35
SHIFT_Y_RATIO = 0.35

RANDOM_SEED = 42

# Loại bỏ polygon quá nhỏ sau khi che
MIN_VISIBLE_MASK_AREA = 25


# =========================
# HÀM HỖ TRỢ
# =========================

def collect_obstacles():
    obstacles = []

    for category_dir in sorted(OBSTACLE_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        for png_path in sorted(category_dir.glob("*.png")):
            obstacles.append({
                "category": category,
                "path": png_path,
            })

    return obstacles


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return img


def load_rgba(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Không đọc được obstacle: {path}")

    # Nếu PNG không có alpha thì tự thêm alpha full
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        b, g, r = cv2.split(img)
        alpha = np.full_like(b, 255)
        img = cv2.merge([b, g, r, alpha])

    return img


def parse_yolo_polygon_file(label_path: Path, img_w: int, img_h: int):
    """
    Đọc file YOLO segmentation.
    Trả về list dict:
    {
        "class_id": int,
        "points_px": np.ndarray shape [N,2],
        "mask": np.ndarray bool / uint8
    }
    """
    instances = []

    with label_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()

        if len(parts) < 7:
            continue

        class_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        pts = []
        for i in range(0, len(coords), 2):
            x = int(round(coords[i] * img_w))
            y = int(round(coords[i + 1] * img_h))
            x = max(0, min(img_w - 1, x))
            y = max(0, min(img_h - 1, y))
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        instances.append({
            "class_id": class_id,
            "points_px": pts,
            "mask": mask,
        })

    return instances


def mask_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def choose_target_instance(instances):
    """
    Chọn bình lớn nhất trong ảnh làm target cho S1.
    """
    if not instances:
        return None

    areas = [mask_area(x["mask"]) for x in instances]
    idx = int(np.argmax(areas))
    return idx


def resize_obstacle_to_target(obstacle_rgba, target_bbox_h):
    h, w = obstacle_rgba.shape[:2]

    if h <= 0 or w <= 0:
        return obstacle_rgba

    scale = random.uniform(OBSTACLE_SCALE_MIN, OBSTACLE_SCALE_MAX)
    new_h = max(12, int(round(scale * target_bbox_h)))
    new_w = max(12, int(round(w * new_h / h)))

    resized = cv2.resize(
        obstacle_rgba,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if new_h < h else cv2.INTER_LINEAR,
    )
    return resized


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    return x1, y1, x2, y2


def place_obstacle_mask(obstacle_rgba, canvas_h, canvas_w, x_left, y_top):
    """
    Trả về:
    placed_rgb   : np.ndarray HxWx3 (chỉ vùng obstacle, còn lại 0)
    placed_alpha : np.ndarray HxW float32 [0,1]
    placed_mask  : np.ndarray HxW uint8 {0,1}
    """
    placed_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    placed_alpha = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    oh, ow = obstacle_rgba.shape[:2]

    x1 = max(0, x_left)
    y1 = max(0, y_top)
    x2 = min(canvas_w, x_left + ow)
    y2 = min(canvas_h, y_top + oh)

    if x1 >= x2 or y1 >= y2:
        return placed_rgb, placed_alpha, np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    src_x1 = x1 - x_left
    src_y1 = y1 - y_top
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)

    crop = obstacle_rgba[src_y1:src_y2, src_x1:src_x2]

    rgb = crop[:, :, :3]
    alpha = crop[:, :, 3].astype(np.float32) / 255.0

    placed_rgb[y1:y2, x1:x2] = rgb
    placed_alpha[y1:y2, x1:x2] = alpha

    placed_mask = (placed_alpha > 0.5).astype(np.uint8)

    return placed_rgb, placed_alpha, placed_mask


def alpha_blend(base_bgr, obs_rgb, obs_alpha):
    """
    base_bgr: HxWx3
    obs_rgb:  HxWx3
    obs_alpha: HxW float32 [0,1]
    """
    alpha = obs_alpha[:, :, None]
    comp = obs_rgb.astype(np.float32) * alpha + base_bgr.astype(np.float32) * (1.0 - alpha)
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    return comp


def contour_mask_to_yolo_line(mask: np.ndarray, class_id: int):
    """
    Chuyển mask -> 1 dòng YOLO segmentation.
    Nếu mask bị vỡ thành nhiều mảnh thì lấy contour lớn nhất.
    """
    if mask_area(mask) < MIN_VISIBLE_MASK_AREA:
        return None

    h, w = mask.shape

    mask_u8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < MIN_VISIBLE_MASK_AREA:
        return None

    # Xấp xỉ nhẹ để giảm số điểm
    peri = cv2.arcLength(contour, True)
    epsilon = 0.003 * peri
    contour = cv2.approxPolyDP(contour, epsilon, True)

    contour = contour.reshape(-1, 2)

    if len(contour) < 3:
        return None

    values = [str(class_id)]

    for x, y in contour:
        xn = float(x) / w
        yn = float(y) / h
        xn = min(max(xn, 0.0), 1.0)
        yn = min(max(yn, 0.0), 1.0)
        values.append(f"{xn:.6f}")
        values.append(f"{yn:.6f}")

    return " ".join(values)


def generate_one_variant(image_path: Path, label_path: Path, split: str, variant_idx: int, obstacles):
    img = load_image(image_path)
    h, w = img.shape[:2]

    instances = parse_yolo_polygon_file(label_path, w, h)

    if not instances:
        return None

    target_idx = choose_target_instance(instances)
    if target_idx is None:
        return None

    target_instance = instances[target_idx]
    target_mask = target_instance["mask"].copy()

    bbox = bbox_from_mask(target_mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1 + 1
    bbox_h = y2 - y1 + 1
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    chosen_result = None

    for _ in range(MAX_TRIES):
        obstacle_info = random.choice(obstacles)
        obstacle_rgba = load_rgba(obstacle_info["path"])
        obstacle_rgba = resize_obstacle_to_target(obstacle_rgba, bbox_h)

        oh, ow = obstacle_rgba.shape[:2]

        shift_x = int(random.uniform(-SHIFT_X_RATIO, SHIFT_X_RATIO) * bbox_w)
        shift_y = int(random.uniform(-SHIFT_Y_RATIO, SHIFT_Y_RATIO) * bbox_h)

        # đặt obstacle quanh tâm bình
        x_left = int(cx - ow / 2 + shift_x)
        y_top = int(cy - oh / 2 + shift_y)

        obs_rgb, obs_alpha, obs_mask = place_obstacle_mask(
            obstacle_rgba,
            h,
            w,
            x_left,
            y_top,
        )

        overlap = ((target_mask > 0) & (obs_mask > 0)).astype(np.uint8)

        full_area = mask_area(target_mask)
        occ_area = mask_area(overlap)

        if full_area == 0:
            continue

        q = occ_area / full_area

        if MIN_Q <= q <= MAX_Q:
            synthetic = alpha_blend(img, obs_rgb, obs_alpha)

            # Update label: phần bình còn nhìn thấy sau khi bị obstacle che
            output_lines = []
            for inst in instances:
                visible_mask = ((inst["mask"] > 0) & (obs_mask == 0)).astype(np.uint8)
                line = contour_mask_to_yolo_line(visible_mask, inst["class_id"])
                if line is not None:
                    output_lines.append(line)

            if not output_lines:
                continue

            chosen_result = {
                "synthetic_image": synthetic,
                "output_lines": output_lines,
                "q": q,
                "obstacle_type": obstacle_info["category"],
                "obstacle_file": obstacle_info["path"].name,
                "target_index": target_idx,
            }
            break

    if chosen_result is None:
        return None

    out_img_name = f"{image_path.stem}_S1_{variant_idx:02d}.jpg"
    out_lbl_name = f"{image_path.stem}_S1_{variant_idx:02d}.txt"

    out_img_path = OUTPUT_DIR / "images" / split / out_img_name
    out_lbl_path = OUTPUT_DIR / "labels" / split / out_lbl_name

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_img_path), chosen_result["synthetic_image"])

    with out_lbl_path.open("w", encoding="utf-8") as f:
        for line in chosen_result["output_lines"]:
            f.write(line + "\n")

    row = {
        "image_id": out_img_path.stem,
        "base_id": image_path.stem,
        "split": split,
        "scenario": "S1",
        "variant_idx": variant_idx,
        "target_instance_index": chosen_result["target_index"],
        "obstacle_type": chosen_result["obstacle_type"],
        "obstacle_file": chosen_result["obstacle_file"],
        "q": round(chosen_result["q"], 6),
        "output_image": str(out_img_path),
        "output_label": str(out_lbl_path),
    }

    return row


def main():
    random.seed(RANDOM_SEED)

    obstacles = collect_obstacles()

    if not obstacles:
        raise ValueError("Không tìm thấy obstacle PNG trong assets/obstacles/")

    print(f"Tìm thấy {len(obstacles)} obstacle")

    # Tạo thư mục output
    for split in SPLITS:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    total_ok = 0
    total_fail = 0

    for split in SPLITS:
        image_dir = BASE_DATASET_DIR / "images" / split
        label_dir = BASE_DATASET_DIR / "labels" / split

        image_paths = sorted(image_dir.glob("*.jpg"))

        print(f"\n===== {split.upper()} =====")
        print(f"Số ảnh base: {len(image_paths)}")

        for idx, image_path in enumerate(image_paths, start=1):
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                print(f"[BỎ QUA - THIẾU LABEL] {label_path}")
                total_fail += 1
                continue

            for variant_idx in range(1, VARIANTS_PER_IMAGE + 1):
                row = generate_one_variant(
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
                    variant_idx=variant_idx,
                    obstacles=obstacles,
                )

                if row is None:
                    print(f"[FAIL] {image_path.name}")
                    total_fail += 1
                else:
                    metadata_rows.append(row)
                    total_ok += 1
                    print(
                        f"[OK] {image_path.name} -> {Path(row['output_image']).name} "
                        f"(q={row['q']:.3f}, obstacle={row['obstacle_type']})"
                    )

    metadata_path = OUTPUT_DIR / "metadata_s1.csv"

    with metadata_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image_id",
            "base_id",
            "split",
            "scenario",
            "variant_idx",
            "target_instance_index",
            "obstacle_type",
            "obstacle_file",
            "q",
            "output_image",
            "output_label",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n========== HOÀN THÀNH ==========")
    print(f"Sinh thành công : {total_ok}")
    print(f"Thất bại        : {total_fail}")
    print(f"Metadata        : {metadata_path}")
    print(f"Output folder   : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from pathlib import Path
import shutil
import random
import cv2
import numpy as np
import sys

# ============================================================
# CONFIG
# ============================================================

SRC_ROOT = Path("data/fire_accessibility_source")
OUT_ROOT = Path("data/fire_accessibility_augmented")
YAML_PATH = Path("configs/fire_accessibility_augmented.yaml")

SEED = 42

# Tổng số phiên bản muốn có cho mỗi ảnh, TÍNH CẢ ảnh gốc.
BASE_TOTAL_VERSIONS = 4       # ảnh gốc: 1 original + 3 aug
SYN_TOTAL_VERSIONS = 10       # synthetic: 1 original + 9 aug

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def is_synthetic(stem: str) -> bool:
    return any(tag in stem for tag in ["_S1", "_S2", "_S3"])


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return img


def adjust_brightness_contrast(img):
    alpha = random.uniform(0.85, 1.15)  # contrast
    beta = random.randint(-20, 20)      # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def adjust_gamma(img):
    gamma = random.uniform(0.80, 1.20)

    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8
    )

    return cv2.LUT(img, table)


def adjust_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Hue thay đổi rất nhẹ
    hue_shift = random.uniform(-4.0, 4.0)

    # Saturation / Value thay đổi nhẹ
    sat_scale = random.uniform(0.88, 1.12)
    val_scale = random.uniform(0.90, 1.10)

    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)

    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gaussian_blur(img):
    # Blur nhẹ thôi
    k = random.choice([3, 3, 3, 5])
    sigma = random.uniform(0.2, 1.0)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)


def add_gaussian_noise(img):
    sigma = random.uniform(2.0, 8.0)

    noise = np.random.normal(
        0,
        sigma,
        img.shape
    ).astype(np.float32)

    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def mild_sharpen(img):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    amount = random.uniform(0.10, 0.30)

    out = cv2.addWeighted(
        img,
        1.0 + amount,
        blurred,
        -amount,
        0
    )

    return np.clip(out, 0, 255).astype(np.uint8)


def augment_image(img):
    """
    Chỉ photometric augmentation.
    KHÔNG flip / crop / rotate / resize.
    Vì vậy YOLO segmentation label được copy nguyên.
    """
    out = img.copy()

    # Luôn áp dụng 1 biến đổi ánh sáng/màu
    primary = random.choice([
        "brightness",
        "gamma",
        "hsv",
    ])

    if primary == "brightness":
        out = adjust_brightness_contrast(out)
    elif primary == "gamma":
        out = adjust_gamma(out)
    elif primary == "hsv":
        out = adjust_hsv(out)

    # Có xác suất thêm một biến đổi phụ
    if random.random() < 0.35:
        out = adjust_brightness_contrast(out)

    if random.random() < 0.20:
        out = gaussian_blur(out)

    if random.random() < 0.20:
        out = add_gaussian_noise(out)

    if random.random() < 0.10:
        out = mild_sharpen(out)

    return out


def save_image(path: Path, img):
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(
            str(path),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
    else:
        ok = cv2.imwrite(str(path), img)

    if not ok:
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def copy_split_unchanged(split):
    src_img_dir = SRC_ROOT / "images" / split
    src_lbl_dir = SRC_ROOT / "labels" / split

    dst_img_dir = OUT_ROOT / "images" / split
    dst_lbl_dir = OUT_ROOT / "labels" / split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for p in src_img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(p, dst_img_dir / p.name)

    for p in src_lbl_dir.glob("*.txt"):
        shutil.copy2(p, dst_lbl_dir / p.name)


# ============================================================
# MAIN
# ============================================================

def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Không thấy dataset nguồn: {SRC_ROOT}")

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    # --------------------------------------------------------
    # 1. Copy val/test unchanged
    # --------------------------------------------------------
    print("Copy val/test giữ nguyên...")

    copy_split_unchanged("val")
    copy_split_unchanged("test")

    # --------------------------------------------------------
    # 2. Build augmented train
    # --------------------------------------------------------
    src_img_dir = SRC_ROOT / "images" / "train"
    src_lbl_dir = SRC_ROOT / "labels" / "train"

    dst_img_dir = OUT_ROOT / "images" / "train"
    dst_lbl_dir = OUT_ROOT / "labels" / "train"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in src_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    original_count = 0
    aug_count = 0
    base_source_count = 0
    syn_source_count = 0
    fail_count = 0

    print(f"\nNguồn train: {len(image_paths)} ảnh")

    for idx, img_path in enumerate(image_paths, start=1):
        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"[FAIL] Thiếu label: {label_path}")
            fail_count += 1
            continue

        synthetic = is_synthetic(img_path.stem)

        if synthetic:
            total_versions = SYN_TOTAL_VERSIONS
            syn_source_count += 1
        else:
            total_versions = BASE_TOTAL_VERSIONS
            base_source_count += 1

        # ----------------------------------------------------
        # Copy original
        # ----------------------------------------------------
        shutil.copy2(
            img_path,
            dst_img_dir / img_path.name
        )

        shutil.copy2(
            label_path,
            dst_lbl_dir / label_path.name
        )

        original_count += 1

        # ----------------------------------------------------
        # Generate augmented copies
        # ----------------------------------------------------
        image = read_image(img_path)

        num_aug = total_versions - 1

        for aug_idx in range(1, num_aug + 1):
            aug_img = augment_image(image)

            aug_stem = f"{img_path.stem}_aug{aug_idx:02d}"
            aug_img_path = dst_img_dir / f"{aug_stem}{img_path.suffix.lower()}"
            aug_lbl_path = dst_lbl_dir / f"{aug_stem}.txt"

            save_image(
                aug_img_path,
                aug_img
            )

            # Photometric only -> polygon giữ nguyên
            shutil.copy2(
                label_path,
                aug_lbl_path
            )

            aug_count += 1

        print(
            f"[{idx}/{len(image_paths)}] "
            f"{img_path.name} -> {total_versions} phiên bản "
            f"({'synthetic' if synthetic else 'base'})"
        )

    # --------------------------------------------------------
    # 3. YAML
    # --------------------------------------------------------
    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)

    yaml_text = f"""path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: fire_extinguisher
  1: obstacle
"""

    YAML_PATH.write_text(
        yaml_text,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 4. Count
    # --------------------------------------------------------
    def count_images(split):
        folder = OUT_ROOT / "images" / split
        return sum(
            1 for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    def count_labels(split):
        return len(
            list(
                (OUT_ROOT / "labels" / split).glob("*.txt")
            )
        )

    print("\n" + "=" * 60)
    print("HOÀN THÀNH")
    print("=" * 60)

    print(f"Base source train         : {base_source_count}")
    print(f"Synthetic source train    : {syn_source_count}")
    print(f"Ảnh original đã giữ       : {original_count}")
    print(f"Ảnh augmentation tạo thêm : {aug_count}")
    print(f"Lỗi                       : {fail_count}")
    print()

    for split in ["train", "val", "test"]:
        print(
            f"{split:5s}: "
            f"{count_images(split)} ảnh, "
            f"{count_labels(split)} labels"
        )

    print()
    print(f"Dataset: {OUT_ROOT}")
    print(f"YAML   : {YAML_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)

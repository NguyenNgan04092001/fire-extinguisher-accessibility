#!/usr/bin/env python3
"""Create the training-only photometric augmentation dataset.

Validation and test are copied unchanged. Geometry is never changed, so YOLO
segmentation polygons can be copied without modification.
"""

from pathlib import Path
import random
import re
import shutil
import sys

import cv2
import numpy as np

SRC_ROOT = Path("data/fire_accessibility_source")
OUT_ROOT = Path("data/fire_accessibility_augmented")
YAML_PATH = Path("configs/fire_accessibility_augmented.yaml")

SEED = 42
REGULAR_TOTAL_VERSIONS = 4   # original + 3 photometric variants
SYN_TOTAL_VERSIONS = 10      # original + 9 photometric variants
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SYNTHETIC_RE = re.compile(r"_S\d+$", re.IGNORECASE)

random.seed(SEED)
np.random.seed(SEED)


def is_synthetic(stem: str) -> bool:
    """Recognize S1, S2, S3 and future S4/S5/... source images."""
    return SYNTHETIC_RE.search(stem) is not None


def image_files(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def adjust_brightness_contrast(img):
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-20, 20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def adjust_gamma(img):
    gamma = random.uniform(0.80, 1.20)
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, table)


def adjust_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-4.0, 4.0)) % 180
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * random.uniform(0.88, 1.12), 0, 255
    )
    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2] * random.uniform(0.90, 1.10), 0, 255
    )
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def gaussian_blur(img):
    k = random.choice([3, 3, 3, 5])
    return cv2.GaussianBlur(img, (k, k), sigmaX=random.uniform(0.2, 1.0))


def add_gaussian_noise(img):
    noise = np.random.normal(0, random.uniform(2.0, 8.0), img.shape)
    out = img.astype(np.float32) + noise.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def mild_sharpen(img):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    amount = random.uniform(0.10, 0.30)
    out = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_image(img):
    out = img.copy()
    primary = random.choice(("brightness", "gamma", "hsv"))
    if primary == "brightness":
        out = adjust_brightness_contrast(out)
    elif primary == "gamma":
        out = adjust_gamma(out)
    else:
        out = adjust_hsv(out)

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
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Cannot write image: {path}")


def validate_source_split(split: str):
    img_dir = SRC_ROOT / "images" / split
    lbl_dir = SRC_ROOT / "labels" / split
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing source split: {split}")

    image_stems = {p.stem for p in image_files(img_dir)}
    label_stems = {p.stem for p in lbl_dir.glob("*.txt")}
    if image_stems != label_stems:
        raise RuntimeError(
            f"Image/label mismatch in {split}: "
            f"missing labels={sorted(image_stems-label_stems)[:5]}, "
            f"missing images={sorted(label_stems-image_stems)[:5]}"
        )


def copy_split_unchanged(split: str):
    src_img = SRC_ROOT / "images" / split
    src_lbl = SRC_ROOT / "labels" / split
    dst_img = OUT_ROOT / "images" / split
    dst_lbl = OUT_ROOT / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for p in image_files(src_img):
        shutil.copy2(p, dst_img / p.name)
    for p in sorted(src_lbl.glob("*.txt")):
        shutil.copy2(p, dst_lbl / p.name)


def write_yaml():
    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    YAML_PATH.write_text(
        f"""path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: fire_extinguisher
  1: obstacle
""",
        encoding="utf-8",
    )


def count_split(split: str):
    return (
        len(image_files(OUT_ROOT / "images" / split)),
        len(list((OUT_ROOT / "labels" / split).glob("*.txt"))),
    )


def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Missing source dataset: {SRC_ROOT}")

    for split in ("train", "val", "test"):
        validate_source_split(split)

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    copy_split_unchanged("val")
    copy_split_unchanged("test")

    src_img_dir = SRC_ROOT / "images" / "train"
    src_lbl_dir = SRC_ROOT / "labels" / "train"
    dst_img_dir = OUT_ROOT / "images" / "train"
    dst_lbl_dir = OUT_ROOT / "labels" / "train"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    source_images = image_files(src_img_dir)
    regular_count = 0
    synthetic_count = 0
    aug_count = 0

    for idx, img_path in enumerate(source_images, start=1):
        label_path = src_lbl_dir / f"{img_path.stem}.txt"
        synthetic = is_synthetic(img_path.stem)
        total_versions = SYN_TOTAL_VERSIONS if synthetic else REGULAR_TOTAL_VERSIONS

        if synthetic:
            synthetic_count += 1
        else:
            regular_count += 1

        shutil.copy2(img_path, dst_img_dir / img_path.name)
        shutil.copy2(label_path, dst_lbl_dir / label_path.name)

        image = read_image(img_path)
        for aug_idx in range(1, total_versions):
            aug_stem = f"{img_path.stem}_aug{aug_idx:02d}"
            aug_img_path = dst_img_dir / f"{aug_stem}{img_path.suffix.lower()}"
            aug_lbl_path = dst_lbl_dir / f"{aug_stem}.txt"
            save_image(aug_img_path, augment_image(image))
            shutil.copy2(label_path, aug_lbl_path)
            aug_count += 1

        print(
            f"[{idx}/{len(source_images)}] {img_path.name} -> "
            f"{total_versions} versions ({'synthetic' if synthetic else 'regular'})"
        )

    write_yaml()

    print("\n" + "=" * 60)
    print("AUGMENTED DATASET READY")
    print("=" * 60)
    print(f"Regular source train      : {regular_count}")
    print(f"Synthetic source train    : {synthetic_count}")
    print(f"Original source images    : {len(source_images)}")
    print(f"Augmentations created     : {aug_count}")
    for split in ("train", "val", "test"):
        imgs, labels = count_split(split)
        print(f"{split:5s}: {imgs} images, {labels} labels")
    print(f"Dataset: {OUT_ROOT}")
    print(f"YAML   : {YAML_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        sys.exit(1)

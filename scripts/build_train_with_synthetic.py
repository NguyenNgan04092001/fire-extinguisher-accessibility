#!/usr/bin/env python3
from pathlib import Path
import shutil
import sys

BASE_ROOT = Path("data/base_yolo")
SYN_IMAGES = Path("data/cvat_synthetic/images")
SYN_LABELS = Path("annotations/cvat_synthetic/labels/train")

OUT_ROOT = Path("data/fire_accessibility_source")
YAML_PATH = Path("configs/fire_accessibility_source.yaml")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def copy_split(split: str):
    src_img = BASE_ROOT / "images" / split
    src_lbl = BASE_ROOT / "labels" / split

    dst_img = OUT_ROOT / "images" / split
    dst_lbl = OUT_ROOT / "labels" / split

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    count_img = 0
    count_lbl = 0

    for p in sorted(src_img.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(p, dst_img / p.name)
            count_img += 1

    for p in sorted(src_lbl.glob("*.txt")):
        shutil.copy2(p, dst_lbl / p.name)
        count_lbl += 1

    return count_img, count_lbl


def validate_label_classes(label_path: Path):
    classes = set()
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(int(float(parts[0])))
    return classes


def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    print("Copy bộ base...")
    for split in ["train", "val", "test"]:
        imgs, lbls = copy_split(split)
        print(f"  {split}: {imgs} ảnh, {lbls} labels")

    train_img_dir = OUT_ROOT / "images" / "train"
    train_lbl_dir = OUT_ROOT / "labels" / "train"

    syn_images = sorted(
        p for p in SYN_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    syn_ok = 0
    syn_fail = 0
    obstacle_labels = 0

    print("\nThêm synthetic vào train...")

    for img_path in syn_images:
        label_path = SYN_LABELS / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"[FAIL] Thiếu label synthetic: {label_path.name}")
            syn_fail += 1
            continue

        classes = validate_label_classes(label_path)

        if 0 not in classes:
            print(f"[WARN] {label_path.name} không có class 0 fire_extinguisher")

        if 1 in classes:
            obstacle_labels += 1
        else:
            print(f"[WARN] {label_path.name} không có class 1 obstacle")

        dst_img = train_img_dir / img_path.name
        dst_lbl = train_lbl_dir / label_path.name

        if dst_img.exists() or dst_lbl.exists():
            print(f"[FAIL] Trùng tên: {img_path.stem}")
            syn_fail += 1
            continue

        shutil.copy2(img_path, dst_img)
        shutil.copy2(label_path, dst_lbl)
        syn_ok += 1

    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)

    yaml_text = f"""path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: fire_extinguisher
  1: obstacle
"""
    YAML_PATH.write_text(yaml_text, encoding="utf-8")

    def count_images(split):
        folder = OUT_ROOT / "images" / split
        return sum(
            1 for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    def count_labels(split):
        return len(list((OUT_ROOT / "labels" / split).glob("*.txt")))

    print("\n" + "=" * 60)
    print("HOÀN THÀNH")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        print(
            f"{split:5s}: {count_images(split)} ảnh, "
            f"{count_labels(split)} labels"
        )

    print()
    print(f"Synthetic thêm thành công : {syn_ok}")
    print(f"Synthetic lỗi             : {syn_fail}")
    print(f"Label có obstacle class 1 : {obstacle_labels}")
    print(f"Dataset                   : {OUT_ROOT}")
    print(f"YAML                      : {YAML_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)

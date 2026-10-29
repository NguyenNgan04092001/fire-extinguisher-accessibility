#!/usr/bin/env python3
from pathlib import Path
import shutil
import zipfile

SRC_IMAGES = Path("data/cvat_synthetic/images")
SRC_LABELS = Path("data/cvat_synthetic/labels")

BUILD_DIR = Path("data/cvat_synthetic_yolo")
ZIP_PATH = Path("data/cvat_synthetic_yolo.zip")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def main():
    if not SRC_IMAGES.exists():
        raise FileNotFoundError(f"Không thấy: {SRC_IMAGES}")
    if not SRC_LABELS.exists():
        raise FileNotFoundError(f"Không thấy: {SRC_LABELS}")

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    images_train = BUILD_DIR / "images" / "train"
    labels_train = BUILD_DIR / "labels" / "train"

    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in SRC_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        raise RuntimeError("Không có ảnh synthetic hợp lệ.")

    train_lines = []
    copied = 0

    for img_path in image_files:
        label_path = SRC_LABELS / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"[BỎ QUA] Thiếu label: {label_path.name}")
            continue

        dst_img = images_train / img_path.name
        dst_lbl = labels_train / label_path.name

        shutil.copy2(img_path, dst_img)
        shutil.copy2(label_path, dst_lbl)

        train_lines.append(f"images/train/{img_path.name}")
        copied += 1

    data_yaml = """path: .
train: train.txt

names:
  0: fire_extinguisher
  1: obstacle
"""

    (BUILD_DIR / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (BUILD_DIR / "train.txt").write_text(
        "\n".join(train_lines) + "\n",
        encoding="utf-8"
    )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in BUILD_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(BUILD_DIR))

    print("\n========== HOÀN THÀNH ==========")
    print(f"Ảnh + label đã đóng gói: {copied}")
    print(f"Folder: {BUILD_DIR}")
    print(f"ZIP   : {ZIP_PATH}")


if __name__ == "__main__":
    main()




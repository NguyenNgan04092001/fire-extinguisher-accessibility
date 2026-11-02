from pathlib import Path
import csv
import shutil


# =========================
# ĐƯỜNG DẪN
# =========================

IMAGE_DIR = Path("data/base_diverse")
LABEL_DIR = Path("annotations/cvat_base/labels/train")
METADATA_CSV = Path("data/metadata_base.csv")

OUTPUT_DIR = Path("data/base_yolo")


def main():
    # Tạo thư mục output
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    with METADATA_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            base_id = row["base_id"]
            file_name = row["file_name"]
            split = row["split"]

            image_src = IMAGE_DIR / file_name
            label_src = LABEL_DIR / f"{base_id}.txt"

            image_dst = OUTPUT_DIR / "images" / split / file_name
            label_dst = OUTPUT_DIR / "labels" / split / f"{base_id}.txt"

            # Kiểm tra ảnh
            if not image_src.exists():
                print(f"[THIẾU ẢNH] {image_src}")
                continue

            # Kiểm tra label
            if not label_src.exists():
                print(f"[THIẾU LABEL] {label_src}")
                continue

            # Copy ảnh
            shutil.copy2(image_src, image_dst)

            # Copy label
            shutil.copy2(label_src, label_dst)

            counts[split] += 1

    print("\n========== KẾT QUẢ ==========")
    print(f"Train : {counts['train']}")
    print(f"Val   : {counts['val']}")
    print(f"Test  : {counts['test']}")
    print(f"Tổng  : {sum(counts.values())}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
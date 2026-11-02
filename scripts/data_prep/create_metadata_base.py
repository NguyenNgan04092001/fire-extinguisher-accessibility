from pathlib import Path
import csv
import random


# =========================
# CẤU HÌNH
# =========================

IMAGE_DIR = Path("data/base_diverse")   # sửa thành thư mục ảnh của bạn
OUTPUT_CSV = Path("data/metadata_base.csv")

SOURCE = "mendeley"

TRAIN_COUNT = 123
VAL_COUNT = 26
TEST_COUNT = 27

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # Tìm toàn bộ ảnh
    image_files = sorted(
        [
            p for p in IMAGE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    total = len(image_files)

    print(f"Tìm thấy {total} ảnh trong: {IMAGE_DIR}")

    expected_total = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

    if total != expected_total:
        raise ValueError(
            f"Số ảnh hiện tại là {total}, "
            f"nhưng cấu hình split yêu cầu {expected_total} ảnh."
        )

    # Shuffle cố định
    random.seed(RANDOM_SEED)
    shuffled_files = image_files.copy()
    random.shuffle(shuffled_files)

    # Chia split
    train_files = shuffled_files[:TRAIN_COUNT]

    val_start = TRAIN_COUNT
    val_end = TRAIN_COUNT + VAL_COUNT
    val_files = shuffled_files[val_start:val_end]

    test_files = shuffled_files[val_end:]

    # Mapping filename -> split
    split_map = {}

    for p in train_files:
        split_map[p.name] = "train"

    for p in val_files:
        split_map[p.name] = "val"

    for p in test_files:
        split_map[p.name] = "test"

    # Sắp xếp lại theo tên file để CSV dễ đọc
    rows = []

    for image_path in image_files:
        base_id = image_path.stem

        rows.append({
            "base_id": base_id,
            "file_name": image_path.name,
            "source": SOURCE,
            "split": split_map[image_path.name],
            "selected": "yes",
            "notes": ""
        })

    # Tạo thư mục nếu chưa có
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Ghi CSV
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fieldnames = [
            "base_id",
            "file_name",
            "source",
            "split",
            "selected",
            "notes"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n========== KẾT QUẢ ==========")
    print(f"Tổng ảnh : {total}")
    print(f"Train    : {len(train_files)}")
    print(f"Val      : {len(val_files)}")
    print(f"Test     : {len(test_files)}")
    print(f"CSV      : {OUTPUT_CSV}")

    print("\nVí dụ 5 dòng đầu:")

    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
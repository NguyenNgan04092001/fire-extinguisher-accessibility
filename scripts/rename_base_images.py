from pathlib import Path
import csv

# ====== CẤU HÌNH ======
INPUT_DIR = Path("data/base_diverse") 
PREFIX = "base"
START_INDEX = 1
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_OUTPUT = Path("results/rename_mapping.csv")
# ======================

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    )

    if not files:
        print(f"Không tìm thấy ảnh trong {INPUT_DIR}")
        return

    print(f"Tìm thấy {len(files)} ảnh.")

    # Bước 1: đổi sang tên tạm để tránh trùng tên
    temp_files = []
    for i, old_path in enumerate(files, start=1):
        temp_path = old_path.with_name(f"__temp__{i:04d}{old_path.suffix.lower()}")
        old_path.rename(temp_path)
        temp_files.append((old_path.name, temp_path))

    # Bước 2: đổi sang tên cuối cùng
    mapping = []
    for idx, (old_name, temp_path) in enumerate(temp_files, start=START_INDEX):
        new_name = f"{PREFIX}_{idx:04d}{temp_path.suffix.lower()}"
        new_path = temp_path.with_name(new_name)
        temp_path.rename(new_path)

        mapping.append({
            "old_name": old_name,
            "new_name": new_name
        })

    # Lưu mapping ra CSV
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["old_name", "new_name"])
        writer.writeheader()
        writer.writerows(mapping)

    print("Đổi tên xong.")
    print(f"File mapping: {CSV_OUTPUT}")

if __name__ == "__main__":
    main()
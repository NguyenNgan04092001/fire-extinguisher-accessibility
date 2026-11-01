#!/usr/bin/env python3
from pathlib import Path

ROOT = Path("assets/obstacles")
CATEGORIES = ["chair", "backpack", "suitcase"]

def rename_category(category: str):
    folder = ROOT / category

    if not folder.exists():
        print(f"[BỎ QUA] Không tìm thấy: {folder}")
        return

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )

    if not files:
        print(f"[BỎ QUA] Không có PNG trong: {folder}")
        return

    print(f"\n===== {category.upper()} =====")
    print(f"Tìm thấy {len(files)} ảnh")

    temp_files = []

    for i, old_path in enumerate(files, start=1):
        temp_path = folder / f"__tmp__{category}_{i:04d}.png"
        old_path.rename(temp_path)
        temp_files.append(temp_path)

    for i, temp_path in enumerate(temp_files, start=1):
        new_name = f"{category}_{i:03d}.png"
        new_path = folder / new_name
        temp_path.rename(new_path)
        print(new_name)

    print(f"Đã đổi tên {len(temp_files)} ảnh {category}")

def main():
    for category in CATEGORIES:
        rename_category(category)

    print("\n========== HOÀN THÀNH ==========")
    for category in CATEGORIES:
        folder = ROOT / category
        if folder.exists():
            count = len(list(folder.glob("*.png")))
            print(f"{category}: {count}")

if __name__ == "__main__":
    main()

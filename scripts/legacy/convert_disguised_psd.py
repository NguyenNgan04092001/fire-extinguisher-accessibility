#!/usr/bin/env python3
"""
Chuyển các file PSD bị đặt nhầm đuôi .png/.jpg thành PNG thật.

Ví dụ:
    base_0001_S1.png  (nội dung thực tế là PSD)
sẽ được:
    - backup PSD gốc vào data/manual_composite_psd_backup/...
    - ghi lại base_0001_S1.png dưới dạng PNG thật

Chỉ đụng tới file có magic header PSD = b"8BPS".
Các PNG/JPEG thật được giữ nguyên.
"""

from pathlib import Path
import shutil
import sys

from psd_tools import PSDImage


ROOT = Path("data/manual_composite")
BACKUP_ROOT = Path("data/manual_composite_psd_backup")

SPLITS = ["train", "val", "test"]
SCENARIOS = ["S1", "S2", "S3"]

EXTS = {".png", ".jpg", ".jpeg"}


def is_psd_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"8BPS"
    except OSError:
        return False


def main():
    converted = 0
    skipped = 0
    failed = 0

    for split in SPLITS:
        for scenario in SCENARIOS:
            folder = ROOT / split / scenario

            if not folder.exists():
                continue

            print(f"\n===== {split.upper()} / {scenario} =====")

            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in EXTS:
                    continue

                if not is_psd_file(path):
                    skipped += 1
                    continue

                try:
                    # Backup file PSD gốc trước
                    backup_dir = BACKUP_ROOT / split / scenario
                    backup_dir.mkdir(parents=True, exist_ok=True)

                    backup_path = backup_dir / f"{path.stem}.psd"

                    if not backup_path.exists():
                        shutil.copy2(path, backup_path)

                    # Đọc PSD dựa trên header, không quan tâm extension
                    psd = PSDImage.open(path)

                    # Flatten/composite toàn bộ layer
                    image = psd.composite()

                    if image is None:
                        raise RuntimeError("Không composite được PSD")

                    # Luôn xuất PNG thật.
                    # Nếu file hiện tại là .jpg thì tạo tên .png.
                    output_path = path.with_suffix(".png")

                    # Tránh mode không tương thích khi save
                    if image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGBA")

                    image.save(output_path, format="PNG")

                    # Nếu file PSD giả mang đuôi .jpg/.jpeg,
                    # xóa file giả sau khi đã tạo PNG thành công.
                    if path != output_path and path.exists():
                        path.unlink()

                    converted += 1

                    print(
                        f"[OK] {path.name} -> {output_path.name} "
                        f"({image.width}x{image.height})"
                    )

                except Exception as e:
                    failed += 1
                    print(f"[FAIL] {path}: {type(e).__name__}: {e}")

    print("\n" + "=" * 55)
    print("HOÀN THÀNH")
    print("=" * 55)
    print(f"PSD đã chuyển thành PNG thật : {converted}")
    print(f"Ảnh thật bỏ qua             : {skipped}")
    print(f"Chuyển thất bại             : {failed}")
    print(f"Backup PSD                  : {BACKUP_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)

from pathlib import Path
import csv
import shutil
import cv2
import numpy as np

MANUAL_ROOT = Path("data/manual_composite")
BASE_ROOT = Path("data/base_yolo")
MISMATCH_CSV = Path("data/cvat_synthetic/size_mismatches.csv")
BACKUP_ROOT = Path("data/manual_composite_orientation_backup")


def mean_abs_diff(img1, img2):
    if img1.shape != img2.shape:
        return 1e18

    diff = cv2.absdiff(img1, img2)
    return float(np.mean(diff))


def choose_best_rotation(syn_img, base_img):
    cand_cw = cv2.rotate(syn_img, cv2.ROTATE_90_CLOCKWISE)
    cand_ccw = cv2.rotate(syn_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    score_cw = mean_abs_diff(cand_cw, base_img)
    score_ccw = mean_abs_diff(cand_ccw, base_img)

    if score_cw <= score_ccw:
        return cand_cw, "cw", score_cw, score_ccw
    else:
        return cand_ccw, "ccw", score_ccw, score_cw


def main():
    if not MISMATCH_CSV.exists():
        raise FileNotFoundError(f"Không thấy file: {MISMATCH_CSV}")

    fixed = 0
    failed = 0

    with MISMATCH_CSV.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            split = row["split"]
            scenario = row["scenario"]
            synthetic_file = row["synthetic_file"]
            base_image_name = row["base_image"]

            syn_path = MANUAL_ROOT / split / scenario / synthetic_file
            base_path = BASE_ROOT / "images" / split / base_image_name

            if not syn_path.exists():
                print(f"[FAIL] Không thấy synthetic: {syn_path}")
                failed += 1
                continue

            if not base_path.exists():
                print(f"[FAIL] Không thấy base: {base_path}")
                failed += 1
                continue

            syn_img = cv2.imread(str(syn_path), cv2.IMREAD_COLOR)
            base_img = cv2.imread(str(base_path), cv2.IMREAD_COLOR)

            if syn_img is None:
                print(f"[FAIL] Không đọc được synthetic: {syn_path}")
                failed += 1
                continue

            if base_img is None:
                print(f"[FAIL] Không đọc được base: {base_path}")
                failed += 1
                continue

            # chỉ xử lý trường hợp width-height bị đảo
            sh, sw = syn_img.shape[:2]
            bh, bw = base_img.shape[:2]

            if not (sw == bh and sh == bw):
                print(f"[SKIP] Không phải case xoay 90 độ: {syn_path.name}")
                continue

            fixed_img, direction, best_score, other_score = choose_best_rotation(
                syn_img, base_img
            )

            # backup file cũ
            backup_dir = BACKUP_ROOT / split / scenario
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / syn_path.name

            if not backup_path.exists():
                shutil.copy2(syn_path, backup_path)

            # ghi đè file đã xoay
            ok = cv2.imwrite(str(syn_path), fixed_img)

            if not ok:
                print(f"[FAIL] Không ghi được file: {syn_path}")
                failed += 1
                continue

            fixed += 1
            print(
                f"[OK] {syn_path.name} | rotate={direction} | "
                f"score_best={best_score:.2f} | score_other={other_score:.2f}"
            )

    print("\n========== HOÀN THÀNH ==========")
    print(f"Đã sửa      : {fixed}")
    print(f"Thất bại    : {failed}")
    print(f"Backup ở    : {BACKUP_ROOT}")


if __name__ == "__main__":
    main()
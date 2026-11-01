#!/usr/bin/env python3
"""
Chọn N ảnh base đa dạng nhất bằng embedding ResNet18 + Farthest Point Sampling.

Mục tiêu:
- Input: tập ảnh đã loại near-duplicate bằng pHash.
- Trích xuất embedding thị giác bằng ResNet18 pretrained.
- Lọc nhẹ ảnh quá mờ (tùy chọn).
- Chọn N ảnh trải rộng nhất trong không gian embedding.
- KHÔNG xóa ảnh gốc.
- Copy ảnh được chọn sang output.
- Xuất CSV ghi lại ảnh được chọn.

Ví dụ:
python scripts/legacy/select_diverse_base.py \
    --input data/base_selected \
    --output data/base_diverse_200 \
    --n 200 \
    --device mps
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def sharpness_score(path: Path) -> float:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_feature_extractor(device: torch.device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Bỏ fully-connected layer; output còn vector 512 chiều.
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    preprocess = weights.transforms()
    return model, preprocess


@torch.inference_mode()
def extract_embeddings(
    image_paths: list[Path],
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 32,
):
    embeddings = []
    valid_paths = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]

        tensors = []
        accepted_paths = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(preprocess(img))
                accepted_paths.append(path)
            except Exception as exc:
                print(f"[BỎ QUA] {path}: {exc}")

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        feats = model(batch)

        # L2 normalization để cosine similarity tương ứng dot product.
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)

        embeddings.append(feats.cpu().numpy())
        valid_paths.extend(accepted_paths)

        done = min(start + batch_size, len(image_paths))
        print(f"Embedding: {done}/{len(image_paths)}")

    if not embeddings:
        return [], np.empty((0, 512), dtype=np.float32)

    return valid_paths, np.concatenate(embeddings, axis=0)


def farthest_point_sampling(features: np.ndarray, n_select: int) -> list[int]:
    """
    Greedy farthest-point sampling dùng cosine distance.

    Vì feature đã L2-normalized:
        cosine_similarity = x dot y
        cosine_distance   = 1 - similarity
    """
    n = len(features)
    if n_select >= n:
        return list(range(n))

    # Bắt đầu từ ảnh gần "trung tâm" embedding nhất.
    centroid = features.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    similarities_to_center = features @ centroid
    first = int(np.argmax(similarities_to_center))

    selected = [first]

    # Khoảng cách nhỏ nhất của mỗi ảnh tới bất kỳ ảnh đã chọn nào.
    min_dist = 1.0 - (features @ features[first])

    for _ in range(1, n_select):
        idx = int(np.argmax(min_dist))
        selected.append(idx)

        dist_to_new = 1.0 - (features @ features[idx])
        min_dist = np.minimum(min_dist, dist_to_new)

        # Không chọn lại index cũ.
        min_dist[selected] = -1.0

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, mps, cpu hoặc cuda",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=20.0,
        help="Bỏ ảnh quá mờ. Đặt 0 để tắt lọc blur.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/base_selection.csv"),
    )
    args = parser.parse_args()

    image_paths = collect_images(args.input)
    print(f"Tìm thấy {len(image_paths)} ảnh.")

    # Lọc ảnh cực mờ trước khi chọn diversity.
    candidates = []
    quality_rows = {}

    for i, path in enumerate(image_paths, 1):
        sharpness = sharpness_score(path)
        quality_rows[path] = sharpness

        if sharpness >= args.min_sharpness:
            candidates.append(path)

        if i % 100 == 0:
            print(f"Kiểm tra độ nét: {i}/{len(image_paths)}")

    print(f"Ảnh đạt ngưỡng độ nét: {len(candidates)}/{len(image_paths)}")

    if len(candidates) < args.n:
        print(
            f"Cảnh báo: chỉ còn {len(candidates)} ảnh, "
            f"nhỏ hơn n={args.n}. Sẽ chọn toàn bộ ảnh còn lại."
        )

    device = choose_device(args.device)
    print(f"Device: {device}")

    model, preprocess = build_feature_extractor(device)

    paths, features = extract_embeddings(
        candidates,
        model,
        preprocess,
        device,
        batch_size=args.batch_size,
    )

    n_select = min(args.n, len(paths))
    selected_indices = farthest_point_sampling(features, n_select)

    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for rank, idx in enumerate(selected_indices, start=1):
        src = paths[idx]
        dst_name = f"{rank:03d}_{src.name}"
        dst = args.output / dst_name
        shutil.copy2(src, dst)

        rows.append({
            "rank": rank,
            "source": str(src),
            "selected_file": str(dst),
            "sharpness": round(quality_rows[src], 3),
        })

    with args.report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "source", "selected_file", "sharpness"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== KẾT QUẢ ==========")
    print(f"Ảnh đầu vào       : {len(image_paths)}")
    print(f"Ảnh đủ độ nét     : {len(candidates)}")
    print(f"Ảnh base đã chọn  : {n_select}")
    print(f"Output             : {args.output}")
    print(f"CSV                : {args.report}")
    print("\nBước cuối: xem thủ công 200 ảnh và loại các ảnh không phù hợp với đề tài.")


if __name__ == "__main__":
    main()

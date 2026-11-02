#!/usr/bin/env python3

from pathlib import Path
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Path to YOLO segmentation model",
    )

    parser.add_argument(
        "--outdir",
        default="outputs/inference_test",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLO(args.model)

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    print(f"Image: {image_path}")

    if result.boxes is None:
        print("Predictions: 0")
    else:
        print(
            f"Predictions: {len(result.boxes)}"
        )

        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            print(
                f"{i}: "
                f"class={cls_id} "
                f"name={result.names[cls_id]} "
                f"conf={conf:.4f}"
            )

    vis = result.plot()

    output_path = (
        outdir
        / f"{image_path.stem}_prediction.jpg"
    )

    import cv2

    cv2.imwrite(
        str(output_path),
        vis,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
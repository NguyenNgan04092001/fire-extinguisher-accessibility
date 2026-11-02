#!/usr/bin/env python3

from pathlib import Path
import csv

import cv2


CSV_PATH = Path(
    "outputs/accessibility_gt/accessibility_gt_template.csv"
)

IMAGE_DIR = Path(
    "outputs/accessibility_gt/images"
)


def save_rows(rows):
    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "instance_id",
                "visualization",
                "accessibility_gt",
                "notes",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    with CSV_PATH.open(
        encoding="utf-8"
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    total = len(rows)

    for i, row in enumerate(rows):

        # Skip already labeled rows
        if row["accessibility_gt"] in {
            "0", "1", "2"
        }:
            continue

        image_path = (
            IMAGE_DIR
            / row["visualization"]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                "Cannot read:",
                image_path,
            )
            continue

        display = image.copy()

        # Instructions
        text1 = (
            f"{i + 1}/{total} | "
            f"{row['image']} | "
            f"fire #{row['instance_id']}"
        )

        text2 = (
            "0=accessible  "
            "1=partial  "
            "2=severe  "
            "S=skip  Q=quit"
        )

        cv2.putText(
            display,
            text1,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            text2,
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Accessibility GT",
            display,
        )

        while True:
            key = (
                cv2.waitKey(0)
                & 0xFF
            )

            if key in [
                ord("0"),
                ord("1"),
                ord("2"),
            ]:
                label = chr(key)

                row[
                    "accessibility_gt"
                ] = label

                save_rows(rows)

                print(
                    f"[{i + 1:02d}/{total}] "
                    f"{row['image']} "
                    f"fire#{row['instance_id']} "
                    f"-> {label}"
                )

                break

            elif key in [
                ord("s"),
                ord("S"),
            ]:
                print(
                    f"[SKIP] "
                    f"{row['image']} "
                    f"fire#{row['instance_id']}"
                )
                break

            elif key in [
                ord("q"),
                ord("Q"),
                27,
            ]:
                save_rows(rows)
                cv2.destroyAllWindows()

                print()
                print(
                    "Progress saved."
                )
                return

    save_rows(rows)
    cv2.destroyAllWindows()

    labeled = sum(
        r["accessibility_gt"]
        in {"0", "1", "2"}
        for r in rows
    )

    print()
    print("=" * 60)
    print("ACCESSIBILITY GT LABELING COMPLETE")
    print("=" * 60)

    print(
        f"Labeled: {labeled}/{total}"
    )

    print(
        f"Saved: {CSV_PATH}"
    )


if __name__ == "__main__":
    main()
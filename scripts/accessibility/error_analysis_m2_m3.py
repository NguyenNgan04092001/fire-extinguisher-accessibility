#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


CSV = Path(
    "outputs/m1_m2_m3_predictions.csv"
)

OUT = Path(
    "outputs/m2_m3_error_analysis.csv"
)


def main():
    df = pd.read_csv(CSV)

    df["M2_correct"] = (
        df["pred_M2_r"]
        == df["accessibility_gt"]
    )

    df["M3_correct"] = (
        df["pred_M3_qr"]
        == df["accessibility_gt"]
    )

    def category(row):
        if (
            not row["M2_correct"]
            and row["M3_correct"]
        ):
            return "M3_fixed_M2"

        if (
            row["M2_correct"]
            and not row["M3_correct"]
        ):
            return "M3_broke_M2"

        if (
            row["M2_correct"]
            and row["M3_correct"]
        ):
            return "both_correct"

        return "both_wrong"

    df["comparison"] = df.apply(
        category,
        axis=1,
    )

    print("=" * 72)
    print("M2 vs M3 ERROR ANALYSIS")
    print("=" * 72)

    print()
    print(
        df["comparison"]
        .value_counts()
    )

    print()
    print("=" * 72)
    print("CASES WHERE M3 FIXED M2")
    print("=" * 72)

    fixed = df[
        df["comparison"]
        == "M3_fixed_M2"
    ].copy()

    for _, row in fixed.iterrows():
        print(
            f"{row['image']} "
            f"fire#{row['instance_id']} | "
            f"GT={row['accessibility_gt']} "
            f"M2={row['pred_M2_r']} "
            f"M3={row['pred_M3_qr']} | "
            f"q={row['q']:.3f} "
            f"r={row['r']:.3f}"
        )

    print()
    print("=" * 72)
    print("CASES WHERE M3 BROKE M2")
    print("=" * 72)

    broke = df[
        df["comparison"]
        == "M3_broke_M2"
    ].copy()

    for _, row in broke.iterrows():
        print(
            f"{row['image']} "
            f"fire#{row['instance_id']} | "
            f"GT={row['accessibility_gt']} "
            f"M2={row['pred_M2_r']} "
            f"M3={row['pred_M3_qr']} | "
            f"q={row['q']:.3f} "
            f"r={row['r']:.3f}"
        )

    df.to_csv(
        OUT,
        index=False,
    )

    print()
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
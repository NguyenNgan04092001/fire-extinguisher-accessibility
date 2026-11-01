#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


CSV = Path(
    "outputs/m1_m2_m3_predictions.csv"
)

N_BOOT = 10000
SEED = 42


def metrics(y, pred):
    return {
        "accuracy": accuracy_score(
            y,
            pred,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred,
            ),
        "macro_f1": f1_score(
            y,
            pred,
            average="macro",
        ),
    }


def main():
    df = pd.read_csv(CSV)

    y = df[
        "accessibility_gt"
    ].to_numpy()

    p2 = df[
        "pred_M2_r"
    ].to_numpy()

    p3 = df[
        "pred_M3_qr"
    ].to_numpy()

    groups = df[
        "image"
    ].to_numpy()

    unique_groups = np.unique(
        groups
    )

    m2 = metrics(y, p2)
    m3 = metrics(y, p3)

    print("=" * 70)
    print("OBSERVED PERFORMANCE")
    print("=" * 70)

    for key in m2:
        print(
            f"{key:20s} "
            f"M2={m2[key]:.4f} "
            f"M3={m3[key]:.4f} "
            f"delta={m3[key]-m2[key]:+.4f}"
        )

    rng = np.random.default_rng(
        SEED
    )

    deltas = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }

    # Cluster bootstrap by image,
    # not by individual extinguisher.
    for _ in range(N_BOOT):

        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )

        indices = []

        for g in sampled_groups:
            idx = np.where(
                groups == g
            )[0]

            indices.extend(
                idx.tolist()
            )

        indices = np.asarray(
            indices,
            dtype=int,
        )

        y_b = y[indices]
        p2_b = p2[indices]
        p3_b = p3[indices]

        m2_b = metrics(
            y_b,
            p2_b,
        )

        m3_b = metrics(
            y_b,
            p3_b,
        )

        for key in deltas:
            deltas[key].append(
                m3_b[key]
                - m2_b[key]
            )

    print()
    print("=" * 70)
    print("CLUSTER BOOTSTRAP: M3 - M2")
    print("=" * 70)

    for key, values in deltas.items():

        arr = np.asarray(
            values,
            dtype=float,
        )

        low, high = np.percentile(
            arr,
            [2.5, 97.5],
        )

        probability_positive = (
            arr > 0
        ).mean()

        print()
        print(key)

        print(
            f"  mean delta = "
            f"{arr.mean():+.4f}"
        )

        print(
            f"  95% CI     = "
            f"[{low:+.4f}, "
            f"{high:+.4f}]"
        )

        print(
            f"  P(delta>0) = "
            f"{probability_positive:.4f}"
        )


if __name__ == "__main__":
    main()
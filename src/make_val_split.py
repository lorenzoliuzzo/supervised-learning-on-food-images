# Run: python src/make_val_split.py [food251] [--seed 251]
#
# Writes splits/val_split.csv once. Ablations and checkpoint selection read
# val-dev only; val-test is touched once, for the report's headline number.
# The split must be a committed file, not recomputed per run -- a split that
# drifts between runs is worse than no split.
from __future__ import annotations

import argparse
import pathlib
import random

import pandas as pd


def stratified_split(labels: dict[str, int], seed: int) -> dict[str, str]:
    # Per-class shuffle-and-halve, not a single global shuffle: a global split
    # would let small classes land entirely on one side by chance. The class
    # with 2 images still gets exactly one of each; odd counts give dev the
    # extra image.
    by_class: dict[int, list[str]] = {}
    for image, label in labels.items():
        by_class.setdefault(label, []).append(image)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for label in sorted(by_class):
        images = sorted(by_class[label])
        rng.shuffle(images)
        cut = (len(images) + 1) // 2
        for image in images[:cut]:
            assignment[image] = "dev"
        for image in images[cut:]:
            assignment[image] = "test"
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default="food251", type=pathlib.Path)
    parser.add_argument("--seed", default=251, type=int)
    parser.add_argument("--out", default="splits/val_split.csv", type=pathlib.Path)
    args = parser.parse_args()

    val_labels = pd.read_csv(args.data / "meta" / "val_labels.csv")
    labels = dict(zip(val_labels["img_name"], val_labels["label"], strict=True))

    assignment = stratified_split(labels, args.seed)
    dev = sum(1 for s in assignment.values() if s == "dev")
    print(f"{len(assignment):,} val images -> {dev:,} dev / {len(assignment) - dev:,} test")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = val_labels.copy()
    out["split"] = out["img_name"].map(assignment)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

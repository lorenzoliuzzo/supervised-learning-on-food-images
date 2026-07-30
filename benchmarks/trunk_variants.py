# Run: python benchmarks/trunk_variants.py
#
# Produces the architecture table in plans/2026-07-30-training-roadmap.md. The
# question it answers is where to spend the parameter headroom the GAP head freed
# -- and the answer is not the one parameter counting suggests, so the numbers
# need to be re-derivable rather than quoted.
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench import BUDGET, measure, report  # noqa: E402

from model import FoodCNN, ResidualBlock  # noqa: E402

SIZE, BATCH = 176, 160


def trunk(
    widths: tuple[int, ...],
    blocks: tuple[int, ...],
    *,
    pool: bool = True,
    num_classes: int = 251,
) -> nn.Module:
    # Reuses the repo's own ResidualBlock so the comparison is against FoodCNN's
    # actual building block, not a lookalike.
    layers: list[nn.Module] = [
        nn.Conv2d(3, widths[0], kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(widths[0]),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    in_channels = widths[0]
    for i, (width, count) in enumerate(zip(widths, blocks, strict=True)):
        stage = [ResidualBlock(in_channels, width, stride=1 if i == 0 else 2)]
        stage += [ResidualBlock(width, width) for _ in range(count - 1)]
        layers.append(nn.Sequential(*stage))
        in_channels = width

    return nn.Sequential(
        nn.Sequential(*layers),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(in_channels, num_classes),
    )


def final_map(stages: int, pool: bool) -> int:
    resolution = SIZE // 2
    if pool:
        resolution //= 2
    for _ in range(stages - 1):
        resolution = (resolution + 1) // 2
    return resolution


def census() -> None:
    model = FoodCNN()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    features = sum(p.numel() for p in model.features.parameters())
    classifier = sum(p.numel() for p in model.classifier.parameters())

    print(f"current FoodCNN: {total:,} params, {100 * total / BUDGET:.1f}% of budget, "
          f"headroom {BUDGET - total:,}")
    print(f"  features {features:,} ({100 * features / total:.1f}%)  "
          f"classifier {classifier:,} ({100 * classifier / total:.1f}%)")

    biases = sum(m.bias.numel() for m in model.modules()
                 if isinstance(m, nn.Conv2d) and m.bias is not None)
    print(f"  conv biases made redundant by a following BatchNorm: {biases:,}\n")


VARIANTS: list[tuple[str, tuple[int, ...], tuple[int, ...], bool]] = [
    ("baseline  [2,2,2,1] 64-512",      (64, 128, 256, 512), (2, 2, 2, 1), True),
    ("deeper@11 [2,2,3,1] 64-512",      (64, 128, 256, 512), (2, 2, 3, 1), True),
    ("deeper@11 [2,2,4,1] 64-512",      (64, 128, 256, 512), (2, 2, 4, 1), True),
    ("deeper@22 [2,3,3,1] 64-512",      (64, 128, 256, 512), (2, 3, 3, 1), True),
    ("deeper@6  [2,2,2,2] 64-448",      (64, 128, 256, 448), (2, 2, 2, 2), True),
    ("wider@11  [2,2,2,1] 64-320-512",  (64, 128, 320, 512), (2, 2, 2, 1), True),
    ("wider all [2,2,2,1] 72-576",      (72, 144, 288, 576), (2, 2, 2, 1), True),
    ("5 stages  [2,2,2,2,1] 48-512",    (48, 96, 192, 320, 512), (2, 2, 2, 2, 1), True),
    # The 5-stage winner ends at a 3x3 map. Dropping the stem's MaxPool is the
    # obvious way to keep resolution at the same parameter count -- and costs 3x.
    ("5 stages  [2,2,2,2,1] no MaxPool", (48, 96, 192, 320, 512), (2, 2, 2, 2, 1), False),
    ("4 stages  [2,2,3,1] no MaxPool",  (64, 128, 256, 512), (2, 2, 3, 1), False),
]


def main() -> None:
    census()
    print(f"{SIZE} px, batch {BATCH}, bf16 + channels_last, budget {BUDGET:,}\n")
    for label, widths, blocks, pool in VARIANTS:
        model = trunk(widths, blocks, pool=pool)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if params >= BUDGET:
            print(f"  {label:36s} {params / 1e6:5.2f}M  OVER BUDGET, skipped")
            continue
        size = final_map(len(widths), pool)
        report(f"{label} -> {size}x{size}", measure(model, SIZE, BATCH))


if __name__ == "__main__":
    main()

# Run: python benchmarks/trunk_variants.py
#
# Produces the architecture table in plans/2026-07-30-training-roadmap.md. The
# question it answers is where to spend the parameter headroom the GAP head freed
# -- and the answer is not the one parameter counting suggests, so the numbers
# need to be re-derivable rather than quoted.
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench import BUDGET, amp_dtype, measure, report  # noqa: E402

from model import FoodCNN, ResidualBlock  # noqa: E402

SIZE, BATCH = 176, 160


class ConcatPool(nn.Module):
    # Average pooling reports how much of a feature is present across the whole
    # map; max pooling reports the strongest single response. For fine-grained
    # classes separated by one localised cue -- a garnish, a crust -- averaging
    # over a 6x6 map dilutes exactly the evidence that separates them.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [
            nn.functional.adaptive_avg_pool2d(x, 1),
            nn.functional.adaptive_max_pool2d(x, 1),
        ]
        return torch.cat(pooled, dim=1)


class SpatialAttentionPool(nn.Module):
    # A weighted average over the final map, with the weights predicted from the
    # map itself, so pooling can concentrate on the discriminative region instead
    # of spending equal weight on background. Max pooling is the hard version of
    # this (all weight on one cell); this is the soft, learned version.
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c = x.shape[0], x.shape[1]
        # Softmax over the flattened map, not a sigmoid gate: the weights have to
        # sum to one for this to stay a pooling operation on the same scale as
        # GAP, which is what lets the rest of the head stay unchanged.
        weights = self.score(x).flatten(2).softmax(dim=-1)
        return (x.flatten(2) * weights).sum(dim=-1).reshape(n, c, 1, 1)


def build_head(kind: str, channels: int, num_classes: int) -> nn.Module:
    if kind == "gap":
        pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
        features = channels
    elif kind == "gap+gmp":
        pool = ConcatPool()
        features = channels * 2
    elif kind == "attention":
        pool = SpatialAttentionPool(channels)
        features = channels
    else:
        raise ValueError(f"unknown head: {kind}")

    return nn.Sequential(
        pool,
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(features, num_classes),
    )


def trunk(
    widths: tuple[int, ...],
    blocks: tuple[int, ...],
    *,
    pool: bool = True,
    head: str = "gap",
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
        build_head(head, in_channels, num_classes),
    )


@dataclass(frozen=True)
class Variant:
    # `key` is what the command line selects on, `label` is what the roadmap
    # tables print. They are separate because the labels contain spaces and the
    # block-count notation, which make for miserable CLI arguments.
    key: str
    label: str
    widths: tuple[int, ...]
    blocks: tuple[int, ...]
    pool: bool = True
    head: str = "gap"

    def build(self, num_classes: int = 251) -> nn.Module:
        return trunk(self.widths, self.blocks, pool=self.pool, head=self.head,
                     num_classes=num_classes)


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


BASELINE_WIDTHS = (64, 128, 256, 512)
BASELINE_BLOCKS = (2, 2, 2, 1)

VARIANTS: list[Variant] = [
    Variant("baseline", "baseline  [2,2,2,1] 64-512", BASELINE_WIDTHS, BASELINE_BLOCKS),
    Variant("deep-3", "deeper@11 [2,2,3,1] 64-512", (64, 128, 256, 512), (2, 2, 3, 1)),
    Variant("deep-4", "deeper@11 [2,2,4,1] 64-512", (64, 128, 256, 512), (2, 2, 4, 1)),
    Variant("deep-22", "deeper@22 [2,3,3,1] 64-512", (64, 128, 256, 512), (2, 3, 3, 1)),
    Variant("deep-6", "deeper@6  [2,2,2,2] 64-448", (64, 128, 256, 448), (2, 2, 2, 2)),
    Variant("wide-11", "wider@11  [2,2,2,1] 64-320-512", (64, 128, 320, 512), (2, 2, 2, 1)),
    Variant("wide-all", "wider all [2,2,2,1] 72-576", (72, 144, 288, 576), (2, 2, 2, 1)),
    Variant("5stage", "5 stages  [2,2,2,2,1] 48-512", (48, 96, 192, 320, 512), (2, 2, 2, 2, 1)),
    # The 5-stage winner ends at a 3x3 map. Dropping the stem's MaxPool is the
    # obvious way to keep resolution at the same parameter count -- and costs 3x.
    Variant("5stage-nopool", "5 stages  [2,2,2,2,1] no MaxPool",
            (48, 96, 192, 320, 512), (2, 2, 2, 2, 1), pool=False),
    Variant("deep-3-nopool", "4 stages  [2,2,3,1] no MaxPool",
            (64, 128, 256, 512), (2, 2, 3, 1), pool=False),

    # Below baseline (#28). Phase C only ever swept baseline and up, so the flat
    # region it found says nothing about where capacity starts binding. Depth,
    # stage count and downsample structure are held identical to baseline so
    # width is the only variable -- the 5-stage row above already showed that
    # changing the downsample structure swamps any width effect. Widths stay
    # multiples of 32: the 72-576 row is 46% slower than the 5-stage variant at
    # fewer parameters, purely for falling off the tensor-core fast path.
    Variant("narrow-384", "narrow    [2,2,2,1] 64-384", (64, 128, 256, 384), BASELINE_BLOCKS),
    Variant("narrow-256", "narrow    [2,2,2,1] 32-256", (32, 64, 128, 256), BASELINE_BLOCKS),

    # Heads (#29), on the baseline trunk so the head is the only variable.
    Variant("head-gapgmp", "head      GAP+GMP concat", BASELINE_WIDTHS, BASELINE_BLOCKS,
            head="gap+gmp"),
    Variant("head-attn", "head      spatial attention", BASELINE_WIDTHS, BASELINE_BLOCKS,
            head="attention"),
]

BY_KEY: dict[str, Variant] = {v.key: v for v in VARIANTS}


def main() -> None:
    census()
    dtype_name = str(amp_dtype()).removeprefix("torch.")
    print(f"{SIZE} px, batch {BATCH}, {dtype_name} + channels_last, budget {BUDGET:,}\n")
    for variant in VARIANTS:
        model = variant.build()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if params >= BUDGET:
            print(f"  {variant.label:36s} {params / 1e6:5.2f}M  OVER BUDGET, skipped")
            continue
        size = final_map(len(variant.widths), variant.pool)
        report(f"{variant.label} -> {size}x{size}", measure(model, SIZE, BATCH))


if __name__ == "__main__":
    main()

# Run: python benchmarks/loader_throughput.py [food251]
#
# Answers whether augmentation is free. It is only free while the loader stays
# ahead of the GPU, and the trunk got fast enough that this stopped being
# obvious -- so the comparison is measured, not assumed.
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench import measure  # noqa: E402

from model import FoodCNN  # noqa: E402

SIZE, BATCH = 176, 160


class DecodeOnly(Dataset):
    def __init__(self, files: list[Path], size: int, *, augment: bool) -> None:
        self.files = files
        pipeline: list[object] = [
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
        ]
        if augment:
            pipeline.append(transforms.TrivialAugmentWide())
        pipeline.append(transforms.ToTensor())
        self.transform = transforms.Compose(pipeline)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


def image_stats(files: list[Path], sample: int = 300) -> None:
    chosen = random.sample(files, min(sample, len(files)))
    sizes = [f.stat().st_size for f in chosen]
    dims = []
    for f in chosen:
        with Image.open(f) as image:
            dims.append(image.size)

    print(f"  median file {statistics.median(sizes) / 1024:.0f} KiB, "
          f"median {statistics.median(w for w, _ in dims):.0f}x"
          f"{statistics.median(h for _, h in dims):.0f}, "
          f"median {statistics.median(w * h for w, h in dims) / 1e6:.2f} MP, "
          f"max dim {max(max(d) for d in dims)}")


def decode_rate(files: list[Path], workers: int, *, augment: bool, batches: int = 30) -> float:
    loader = DataLoader(
        DecodeOnly(files, SIZE, augment=augment),
        batch_size=128,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=6,
    )
    iterator = iter(loader)
    next(iterator)  # warm the worker pool before timing

    started = time.perf_counter()
    seen = 0
    for _ in range(batches):
        seen += next(iterator).size(0)
    return seen / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default="food251", type=Path)
    parser.add_argument("--files", default=20_000, type=int)
    args = parser.parse_args()

    files = sorted((args.data / "train_set").glob("*.jpg"))[: args.files]
    if not files:
        raise SystemExit(f"no images under {args.data / 'train_set'}")

    print(f"{len(files):,} images sampled from {args.data / 'train_set'}")
    image_stats(files)

    demand = measure(FoodCNN(), SIZE, BATCH).images_per_second
    print(f"\nGPU demand at {SIZE} px: {demand:.0f} img/s. Loader must beat this.\n")

    # Page cache makes these noisy run to run; the margin is what matters, not
    # the ordering between worker counts.
    for augment in (False, True):
        for workers in (8, 12, 16):
            rate = decode_rate(files, workers, augment=augment)
            label = "TrivialAugment" if augment else "crop+flip"
            verdict = "BELOW GPU DEMAND" if rate < demand else f"{rate / demand:.1f}x headroom"
            print(f"  {label:16s} workers {workers:2d}: {rate:7.0f} img/s   {verdict}")


if __name__ == "__main__":
    main()

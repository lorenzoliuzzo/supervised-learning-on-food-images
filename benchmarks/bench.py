from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

TRAIN_IMAGES = 118_475
BUDGET = 10_000_000


@dataclass(frozen=True)
class Throughput:
    params: int
    images_per_second: float
    peak_gib: float

    @property
    def minutes_per_epoch(self) -> float:
        return TRAIN_IMAGES / self.images_per_second / 60

    def hours(self, epochs: int) -> float:
        return epochs * TRAIN_IMAGES / self.images_per_second / 3600


def measure(
    model: nn.Module,
    size: int,
    batch: int,
    *,
    warmup: int = 8,
    steps: int = 40,
    rounds: int = 3,
) -> Throughput:
    # A full fwd+bwd+step, not just a forward pass: the backward is where the
    # activation memory that decides these rankings actually shows up.
    #
    # This GPU runs against a ~50 W SW_POWER_CAP with its clock oscillating
    # 2175-2340 MHz, so a single short timing window measures whichever power
    # state it happened to catch -- early versions of this script swung 45% on
    # one variant between sweeps. Time several longer rounds and keep the best:
    # under a power cap the fastest round is the one least interrupted, and it
    # is the figure that reproduces.
    if not torch.cuda.is_available():
        raise SystemExit("these benchmarks measure this box's GPU; no CUDA device found")

    # Release the previous variant's blocks before measuring this one, so peak
    # figures stay comparable across a sweep.
    torch.cuda.empty_cache()

    model = model.cuda().to(memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    images = torch.randn(batch, 3, size, size, device="cuda").to(memory_format=torch.channels_last)
    target = torch.randint(0, 251, (batch,), device="cuda")

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        # bf16 needs no GradScaler; on this GPU fp16's scaler is pure overhead.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            criterion(model(images), target).backward()
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    best = 0.0
    for _ in range(rounds):
        started = time.perf_counter()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        best = max(best, steps * batch / (time.perf_counter() - started))

    return Throughput(params, best, torch.cuda.max_memory_allocated() / 2**30)


def report(label: str, result: Throughput, epochs: int = 90) -> None:
    print(
        f"  {label:36s} {result.params / 1e6:5.2f}M  {result.images_per_second:6.0f} img/s  "
        f"{result.minutes_per_epoch:4.1f} min/ep  {result.hours(epochs):4.1f} h/{epochs}ep  "
        f"peak {result.peak_gib:.2f} GiB"
    )

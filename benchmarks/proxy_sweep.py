# Run: python benchmarks/proxy_sweep.py [food251]
#
# Phase C of plans/2026-07-30-training-roadmap.md: four 15-epoch proxy runs on
# real data, real labels, val-dev top-1/top-5 -- not the synthetic-batch
# throughput sweep in trunk_variants.py. Answers whether the 5-stage variant's
# 3x3 final map costs accuracy relative to the 6x6 alternatives.
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trunk_variants import VARIANTS, trunk  # noqa: E402

from main import (  # noqa: E402
    FoodX251Dataset,
    dataset_paths,
    load_val_split,
    train,
    validate,
    warmup_cosine_lr,
)
from runlog import RunLog  # noqa: E402

# The four trunks Phase C compares, pulled from trunk_variants.VARIANTS by
# label so the widths/blocks/pool settings can't drift from the synthetic
# throughput sweep those same labels are ranked in.
CANDIDATE_LABELS = {
    "baseline  [2,2,2,1] 64-512",
    "deeper@11 [2,2,4,1] 64-512",
    "deeper@6  [2,2,2,2] 64-448",
    "5 stages  [2,2,2,2,1] 48-512",
}


def run_proxy(
    label: str,
    widths: tuple[int, ...],
    blocks: tuple[int, ...],
    pool: bool,
    *,
    data_root: Path,
    val_split: Path,
    epochs: int,
    batch_size: int,
    workers: int,
    lr: float,
    log_dir: Path,
) -> tuple[float, float]:
    device = torch.device("cuda")
    model = trunk(widths, blocks, pool=pool).to(device, memory_format=torch.channels_last)
    torch.cuda.reset_peak_memory_stats()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9,
                                 weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: warmup_cosine_lr(epoch, epochs))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    (train_dir, train_labels), (val_dir, val_labels) = dataset_paths(data_root)

    train_dataset = FoodX251Dataset(
        train_dir, train_labels,
        transforms.Compose([
            transforms.RandomResizedCrop(176),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = FoodX251Dataset(
        val_dir, val_labels,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        subset=load_val_split(val_split, "dev"))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=True)

    # train()/validate() only read these fields off `args`; a plain namespace
    # avoids dragging in main.py's full distributed/checkpoint machinery for
    # a proxy run that never checkpoints or resumes.
    args = argparse.Namespace(no_accel=False, gpu=None, distributed=False,
                               world_size=1, multiprocessing_distributed=False,
                               print_freq=50)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run = RunLog(label=label.split()[0], config={
        "label": label, "params": params, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "val_dev_size": len(val_dataset),
    })

    print(f"\n=== {label} ({params / 1e6:.2f}M params) ===")
    for epoch in range(epochs):
        started = time.perf_counter()
        lr_used = optimizer.param_groups[0]['lr']
        train_loss, train_acc1, train_acc3, train_acc5 = train(
            train_loader, model, criterion, optimizer, epoch, device, args)
        val_acc1, val_acc3, val_acc5 = validate(val_loader, model, criterion, args)
        run.record(epoch, lr_used, train_loss, train_acc1, train_acc3, train_acc5,
                   val_acc1, val_acc3, val_acc5)
        scheduler.step()
        print(f"  epoch {epoch:2d}  train_loss {train_loss:.3f}  "
              f"val_acc1 {val_acc1:5.2f}  val_acc3 {val_acc3:5.2f}  val_acc5 {val_acc5:5.2f}  "
              f"({time.perf_counter() - started:.0f}s)")

    peak_vram_gib = torch.cuda.max_memory_allocated() / 2**30
    path = run.save(log_dir, peak_vram_gib=peak_vram_gib)
    print(f"  -> {path}")

    final = run.history[-1]
    return final.val_acc1, final.val_acc5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default="food251", type=Path)
    parser.add_argument("--val-split", default="splits/val_split.csv", type=Path)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--log-dir", default="runs/phaseC", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this sweep trains on real data; no CUDA device found")

    candidates = [v for v in VARIANTS if v[0] in CANDIDATE_LABELS]
    if len(candidates) != len(CANDIDATE_LABELS):
        raise SystemExit("trunk_variants.VARIANTS no longer contains all four Phase C labels")

    results = []
    for label, widths, blocks, pool in candidates:
        acc1, acc5 = run_proxy(
            label, widths, blocks, pool,
            data_root=args.data, val_split=args.val_split, epochs=args.epochs,
            batch_size=args.batch_size, workers=args.workers, lr=args.lr,
            log_dir=args.log_dir,
        )
        results.append((label, acc1, acc5))

    print(f"\n{'variant':36s} {'val-dev top1':>12s} {'val-dev top5':>12s}")
    for label, acc1, acc5 in sorted(results, key=lambda r: -r[1]):
        print(f"  {label:36s} {acc1:10.2f}%  {acc5:10.2f}%")


if __name__ == "__main__":
    main()

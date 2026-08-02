# Run: python benchmarks/proxy_sweep.py [food251] --variants baseline narrow-384
#
# The 15-epoch proxy protocol of plans/2026-07-30-training-roadmap.md, on real
# data, real labels, val-dev top-1/top-5 -- not the synthetic-batch throughput
# sweep in trunk_variants.py. Ranks trunk and head variants against each other
# cheaply enough to be routine; the full 90-epoch run confirms once.
#
# Phase C used it to answer whether the 5-stage variant's 3x3 final map costs
# accuracy (it does). Which variants run is now a command-line choice, because
# #28 and #29 ask the same question of trunks narrower than baseline and of
# pooling heads, and hardcoding one phase's shortlist made that a code edit.
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trunk_variants import BY_KEY, Variant  # noqa: E402

from main import (  # noqa: E402
    FoodX251Dataset,
    dataset_paths,
    load_val_split,
    train,
    validate,
    warmup_cosine_lr,
)
from main import parser as main_parser  # noqa: E402
from runlog import RunLog  # noqa: E402

# The four trunks Phase C compared. Kept as the default so `proxy_sweep.py` with
# no --variants still reproduces that table; every definition still comes from
# trunk_variants so the widths/blocks/head can't drift from the synthetic
# throughput sweep the same keys are ranked in.
PHASE_C_KEYS = ["baseline", "deep-4", "deep-6", "5stage"]


def predict_correct(
    loader: torch.utils.data.DataLoader, model: nn.Module, device: torch.device
) -> np.ndarray:
    # bf16, matching training, where validate() above scores in fp32. On a
    # 6,063-image val-dev that moves a couple of borderline images, so this
    # vector's mean can sit ~0.03 points off the logged val_acc1. Every vector
    # is produced here the same way, so pairing two of them is consistent;
    # pairing one against a checkpoint re-scored in fp32 is not, which is why
    # the control leg gets re-run through this sweep rather than reused.
    model.eval()
    correct: list[np.ndarray] = []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model(images).argmax(dim=1)
            correct.append((preds.cpu() == target).numpy().astype(np.int8))
    return np.concatenate(correct)


def run_proxy(
    variant: Variant,
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
    model = variant.build().to(device, memory_format=torch.channels_last)
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

    # Every default from main.py's own parser, then the few fields a proxy run
    # differs on. Hand-listing the fields train()/validate() happen to read is
    # what broke this sweep silently once already: Phase D added --mix, train()
    # started reading args.mix, and nothing here knew until it crashed a leg in.
    args = main_parser.parse_args([])
    args.distributed = False
    args.multiprocessing_distributed = False
    args.gpu = None
    args.batch_size = batch_size
    args.workers = workers

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run = RunLog(label=variant.key, config={
        "label": variant.label, "key": variant.key, "head": variant.head,
        "widths": list(variant.widths), "blocks": list(variant.blocks),
        "params": params, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "val_dev_size": len(val_dataset),
    })

    print(f"\n=== {variant.label} ({params / 1e6:.2f}M params) ===")
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

    # A point-estimate accuracy gap can't say whether a comparison had the power
    # to detect a real difference, and these proxies routinely land within a
    # point of each other. Saving the per-image correctness vector lets
    # significance_test.py pair variants afterwards without this sweep having to
    # keep checkpoints around -- which it deliberately doesn't, and which
    # wouldn't load into FoodCNN anyway once the head or widths differ.
    predictions_path = log_dir / f"{variant.key}-correct.npy"
    np.save(predictions_path, predict_correct(val_loader, model, device))
    print(f"  -> {predictions_path}")

    final = run.history[-1]
    return final.val_acc1, final.val_acc5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default="food251", type=Path)
    parser.add_argument("--val-split", default="splits/val_split.csv", type=Path)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=8, type=int)
    # Defaults are the recipe Phase D settled on, not the 0.1 Phase C ran at.
    # Phase C's numbers predate the LR search and are not comparable to anything
    # measured since; a sweep that silently reproduced them would invite exactly
    # that mistake.
    parser.add_argument("--lr", default=0.8, type=float)
    parser.add_argument("--log-dir", default="runs/proxy", type=Path)
    parser.add_argument("--variants", nargs="+", default=PHASE_C_KEYS, metavar="KEY",
                        help=f"trunk_variants keys to run; one of {sorted(BY_KEY)}")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this sweep trains on real data; no CUDA device found")

    unknown = [key for key in args.variants if key not in BY_KEY]
    if unknown:
        raise SystemExit(f"unknown variant key(s) {unknown}; known keys: {sorted(BY_KEY)}")

    args.log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for key in args.variants:
        variant = BY_KEY[key]
        acc1, acc5 = run_proxy(
            variant,
            data_root=args.data, val_split=args.val_split, epochs=args.epochs,
            batch_size=args.batch_size, workers=args.workers, lr=args.lr,
            log_dir=args.log_dir,
        )
        results.append((variant.label, acc1, acc5))

    print(f"\n{'variant':36s} {'val-dev top1':>12s} {'val-dev top5':>12s}")
    for label, acc1, acc5 in sorted(results, key=lambda r: -r[1]):
        print(f"  {label:36s} {acc1:10.2f}%  {acc5:10.2f}%")


if __name__ == "__main__":
    main()

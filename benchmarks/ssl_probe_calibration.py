# Run: python benchmarks/ssl_probe_calibration.py [checkpoint.pth.tar ...] [--data food251]
#
# Produces the anchors src/simsiam.py prints beside feat_std and effective
# rank, and the calibration table in plans/2026-07-30-training-roadmap.md.
#
# The collapse gate originally read feat_std against 1/sqrt(512) = 0.0442.
# That reference describes SimSiam's *projector* output; collapse_metrics runs
# on the encoder's, which is non-negative because ResidualBlock applies ReLU
# after the add, so every feature vector sits in the positive orthant of the
# unit sphere where per-dimension std is structurally lower. No encoder in this
# project can reach it -- which made the gate read a working representation as
# half-collapsed.
#
# The fix is not a different constant but two empirical anchors: a randomly
# initialized encoder (the floor a pretrain must beat) and the 63.83%
# supervised checkpoint (a representation known to be useful). Both are scored
# through the identical probe, so the numbers are comparable by construction.
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import (  # noqa: E402
    FoodX251Dataset,
    dataset_paths,
    load_val_split,
    select_amp_dtype,
)
from model import FoodCNN  # noqa: E402

# build_eval_transform is imported rather than reimplemented on purpose: an
# anchor is only comparable to a probe reading if both saw the identical
# pipeline, so the two must not be able to drift apart.
from simsiam import (  # noqa: E402
    build_eval_transform,
    collapse_metrics,
    extract_features,
    knn_accuracy,
)

RANDOM_INIT = "random"
SUPERVISED = "checkpoints/full-90ep-lr0.8-best.pth.tar"
BANK_SEED = 251  # same bank every probe, matching src/simsiam.py


@dataclass
class Probe:
    label: str
    knn_acc1: float
    feat_std: float
    effective_rank: float
    mean_norm: float
    frac_nonneg: float


def load_encoder(source: str, device: torch.device, seed: int = 0) -> FoodCNN:
    # The random-init anchor is a *sample*, not a constant: unseeded, it moved
    # 2.80% -> 3.36% kNN and 4.8 -> 4.4 effective rank between two runs of this
    # script. That is the same order as a real pretrain's first probe interval,
    # so an unseeded floor can flatter or damn a run by chance. Seed it.
    torch.manual_seed(seed)
    model = FoodCNN(num_classes=251)
    if source != RANDOM_INIT:
        state = torch.load(source, map_location="cpu", weights_only=False)["state_dict"]
        # A SimSiam checkpoint stores the whole SimSiamModel; a supervised one
        # stores the bare FoodCNN. Strip the prefix so either loads here.
        if any(key.startswith("encoder.") for key in state):
            state = {
                key.removeprefix("encoder."): value
                for key, value in state.items()
                if key.startswith("encoder.")
            }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{source}: {len(missing)} missing and {len(unexpected)} unexpected keys; "
                "this is not a FoodCNN or SimSiamModel checkpoint")
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


@torch.no_grad()
def probe(label: str, source: str, bank_loader, query_loader, device, amp_dtype,
          seed: int = 0) -> Probe:
    encoder = load_encoder(source, device, seed)
    bank, bank_labels = extract_features(bank_loader, encoder, device, amp_dtype)
    query, query_labels = extract_features(query_loader, encoder, device, amp_dtype)

    acc1 = knn_accuracy(bank, bank_labels, query, query_labels, k=20, temperature=0.07)
    feat_std, effective_rank = collapse_metrics(query)
    # ||mean(f)|| is the positive-orthant effect made visible: 1.0 means every
    # normalized vector points the same way, 0.0 means they cancel isotropically.
    # It is why feat_std cannot approach 1/sqrt(d) here.
    result = Probe(
        label=label,
        knn_acc1=acc1,
        feat_std=feat_std,
        effective_rank=effective_rank,
        mean_norm=query.mean(dim=0).norm().item(),
        frac_nonneg=(query >= 0).float().mean().item(),
    )
    del bank, query, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*",
                       help="extra checkpoints to probe alongside the two anchors")
    parser.add_argument("--data", default="food251")
    parser.add_argument("--bank-size", default=25000, type=int,
                       help="train images in the kNN bank (0 = all). The probe is "
                            "loader-bound, so the full 118,475 costs minutes")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--val-split", default="splits/val_split.csv")
    parser.add_argument("--val-subset", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--seed", default=0, type=int,
                       help="seeds the random-init anchor, which is otherwise a "
                            "different encoder on every run (default: 0)")
    args = parser.parse_args()

    device = torch.accelerator.current_accelerator()
    amp_dtype = select_amp_dtype(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_transform = build_eval_transform(normalize)
    (train_dir, train_labels), (val_dir, val_labels) = dataset_paths(args.data)

    bank_dataset = FoodX251Dataset(train_dir, train_labels, eval_transform)
    if args.bank_size and args.bank_size < len(bank_dataset):
        indices = torch.randperm(
            len(bank_dataset), generator=torch.Generator().manual_seed(BANK_SEED)
        )[:args.bank_size]
        bank_dataset = torch.utils.data.Subset(bank_dataset, indices.tolist())

    bank_loader = torch.utils.data.DataLoader(
        bank_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    query_loader = torch.utils.data.DataLoader(
        FoodX251Dataset(val_dir, val_labels, eval_transform,
                        subset=load_val_split(args.val_split, args.val_subset)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    sources = [("random init", RANDOM_INIT), ("supervised 90ep", SUPERVISED)]
    sources += [(Path(path).stem, path) for path in args.checkpoints]

    results = [
        probe(label, source, bank_loader, query_loader, device, amp_dtype, args.seed)
        for label, source in sources
    ]

    print(f"\nbank {len(bank_dataset)} train images, queries val-{args.val_subset}, "
          f"k=20, t=0.07, {amp_dtype}")
    print(f"{'encoder':<24} {'kNN top-1':>10} {'feat_std':>9} "
          f"{'eff. rank':>10} {'||mean||':>9} {'frac>=0':>8}")
    for r in results:
        print(f"{r.label:<24} {r.knn_acc1:>9.2f}% {r.feat_std:>9.4f} "
              f"{r.effective_rank:>7.1f}/512 {r.mean_norm:>9.4f} {r.frac_nonneg:>8.3f}")
    print(f"\n(1/sqrt(512) = {1 / 512 ** 0.5:.4f} -- unreachable here; see the header)")


if __name__ == "__main__":
    main()

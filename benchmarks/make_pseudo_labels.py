# Run: python benchmarks/make_pseudo_labels.py checkpoints/full-90ep-lr0.8-best.pth.tar
#
# Pseudo-labels food251/test_set with a trained checkpoint and writes
# image_name,label,confidence to splits/pseudo_labels.csv, for main.py
# --pseudo-labels to train on alongside train_set.
#
# test_set is the only unlabeled pool this project has -- iFood 2019 never
# released its labels, so food251/meta/ holds train_labels.csv and
# val_labels.csv only. It is 28,377 images against the 118,475 in train_set,
# so **+24%**, which is what bounds the whole idea: self-training cannot
# invent information the teacher lacks, it can only propagate what the teacher
# already knows onto 24% more pixels.
#
# So the number that decides whether training on this is worth any GPU time is
# not the mean confidence, it is the survivor count per threshold, and how
# accurate those survivors are. This script prints both. The accuracy column
# is measured on val-dev -- images with real labels that the teacher was
# selected on -- so read it as an optimistic bound on pseudo-label precision,
# not an estimate of it.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import (  # noqa: E402
    FoodX251Dataset,
    dataset_paths,
    load_val_split,
    select_amp_dtype,
)
from model import FoodCNN  # noqa: E402

# Shared with the SSL probe so a pseudo-label and a probe reading cannot drift
# apart on preprocessing.
from simsiam import UnlabeledImageDataset, build_eval_transform  # noqa: E402

THRESHOLDS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def load_teacher(checkpoint_path: str, device: torch.device) -> FoodCNN:
    model = FoodCNN(num_classes=251)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    # A simsiam.py checkpoint wraps the FoodCNN; a main.py one is bare.
    if any(key.startswith("encoder.") for key in state):
        state = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict(
    model: FoodCNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    confidences: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(images)
        # softmax in fp32: under autocast the logits come back in a reduced
        # dtype, and a 0.95 confidence threshold is well inside fp16's spacing
        # near 1.0, so thresholding the reduced-precision softmax would move
        # survivors in and out of the set for numerical reasons alone.
        probabilities = F.softmax(logits.float(), dim=1)
        batch_conf, batch_pred = probabilities.max(dim=1)
        confidences.append(batch_conf.cpu())
        predictions.append(batch_pred.cpu())
    return torch.cat(confidences), torch.cat(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="food251")
    parser.add_argument("--out", default="splits/pseudo_labels.csv")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("-j", "--workers", default=8, type=int)
    parser.add_argument("--val-split", default="splits/val_split.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = select_amp_dtype(device)
    model = load_teacher(args.checkpoint, device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # build_eval_transform fixes main.py's Resize(256)/CenterCrop(224) -- the
    # FixRes pair the teacher's 63.83% was measured at. A pseudo-label taken at
    # any other resolution would not be the label that checkpoint earns.
    transform = build_eval_transform(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    # Calibration first: val-dev has real labels, so it says how often this
    # teacher is right *at each threshold*. Without it the survivor count
    # below is just a count of things the model feels strongly about.
    (_, _), (val_dir, val_labels) = dataset_paths(args.data)
    val_dataset = FoodX251Dataset(
        val_dir, val_labels, transform, subset=load_val_split(args.val_split, "dev")
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    val_conf, val_pred = predict(model, val_loader, device, amp_dtype)
    val_truth = torch.tensor(val_dataset.labels, dtype=val_pred.dtype)
    print(f"=> teacher on val-dev ({len(val_dataset)} images): "
          f"top-1 {100.0 * (val_pred == val_truth).float().mean():.2f}%")

    test_dir = Path(args.data) / "test_set"
    test_dataset = UnlabeledImageDataset([test_dir], transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    print(f"=> pseudo-labeling {len(test_dataset)} unlabeled images from {test_dir}")
    conf, pred = predict(model, test_loader, device, amp_dtype)

    print(f"\n{'threshold':>10} {'kept':>8} {'% of pool':>10} "
          f"{'val-dev precision':>18} {'classes':>8}")
    for threshold in THRESHOLDS:
        keep = conf >= threshold
        val_keep = val_conf >= threshold
        precision = (
            100.0 * (val_pred[val_keep] == val_truth[val_keep]).float().mean()
            if val_keep.any() else float("nan")
        )
        print(f"{threshold:>10.2f} {int(keep.sum()):>8} "
              f"{100.0 * keep.float().mean():>9.1f}% {precision:>17.2f}% "
              f"{len(pred[keep].unique()):>8}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Every image is written with its confidence; main.py thresholds at train
    # time. Regenerating this file costs a GPU pass, so the choice of
    # threshold should not be baked into it.
    pd.DataFrame({
        "image_name": [Path(p).name for p in test_dataset.paths],
        "label": pred.tolist(),
        "confidence": conf.tolist(),
    }).to_csv(out_path, index=False)
    print(f"\n=> wrote {len(pred)} rows to {out_path}")


if __name__ == "__main__":
    main()

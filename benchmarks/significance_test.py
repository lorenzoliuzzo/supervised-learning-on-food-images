# Run: python benchmarks/significance_test.py <control.pth.tar> <other.pth.tar> [food251] --split dev
#
# Paired McNemar's test between two checkpoints on the same val split. Two
# checkpoints are scored on identical images, so their predictions are
# paired, not independent samples -- a two-proportion z-test would ignore
# that pairing and overstate significance. McNemar's only uses the
# discordant pairs (control right/other wrong vs control wrong/other right),
# which is exactly the information a point-estimate accuracy gap discards.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_errors import run_inference  # noqa: E402

from main import FoodX251Dataset, dataset_paths, load_val_split  # noqa: E402
from model import FoodCNN  # noqa: E402

NUM_CLASSES = 251


def correctness(source: Path, loader: torch.utils.data.DataLoader, device: torch.device) -> np.ndarray:
    # `.npy` is the per-image correctness vector proxy_sweep.py saves at the end
    # of a run. Proxy variants change the head or the stage widths, so they don't
    # load into FoodCNN and can't be re-scored from a checkpoint here -- the
    # sweep has to hand over its own predictions. Same val-dev images in the same
    # order either way, which is what makes the pairing valid.
    if source.suffix == ".npy":
        return np.load(source).astype(int)

    model = FoodCNN(num_classes=NUM_CLASSES)
    ck = torch.load(source, map_location=device)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    result = run_inference(model, loader, device)
    return (result["targets"] == result["preds"]).astype(int)


def mcnemar(control_correct: np.ndarray, other_correct: np.ndarray) -> dict[str, float | int]:
    b = int(((control_correct == 1) & (other_correct == 0)).sum())
    c = int(((control_correct == 0) & (other_correct == 1)).sum())
    n = b + c
    p_value = 1.0 if n == 0 else binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return {"b_control_only": b, "c_other_only": c, "n_discordant": n, "p_value": p_value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("other", type=Path)
    parser.add_argument("data", nargs="?", default="food251", type=Path)
    parser.add_argument("--val-split", default="splits/val_split.csv", type=Path)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    _, (val_dir, val_labels) = dataset_paths(args.data)
    dataset = FoodX251Dataset(
        val_dir, val_labels,
        transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), normalize,
        ]),
        subset=load_val_split(args.val_split, args.split),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"val-{args.split} n={len(dataset)}")

    control_correct = correctness(args.control, loader, device)
    other_correct = correctness(args.other, loader, device)
    # McNemar's is only valid on paired observations. A saved correctness vector
    # carries no record of which split produced it, so a length mismatch is the
    # one chance to catch a val-dev vector being paired against a val-test one.
    if len(control_correct) != len(other_correct) or len(control_correct) != len(dataset):
        raise SystemExit(
            f"not paired: {len(control_correct)} and {len(other_correct)} predictions "
            f"against {len(dataset)} val-{args.split} images -- these were scored on "
            "different splits")

    result = mcnemar(control_correct, other_correct)

    control_acc, other_acc = control_correct.mean() * 100, other_correct.mean() * 100
    print(f"{args.control.name}: {control_acc:.2f}%")
    print(f"{args.other.name}: {other_acc:.2f}%  (delta {other_acc - control_acc:+.2f})")
    print(f"discordant pairs: control-only={result['b_control_only']} "
          f"other-only={result['c_other_only']} (n={result['n_discordant']})")
    print(f"McNemar exact two-sided p-value: {result['p_value']:.4f}")


if __name__ == "__main__":
    main()

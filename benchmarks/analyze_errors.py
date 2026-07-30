# Run: python benchmarks/analyze_errors.py <checkpoint.pth.tar> [food251] --split dev
#
# Per-class accuracy, a classification report, the most-confused class pairs,
# a confidence histogram, and a t-SNE of the penultimate features -- the
# standard post-hoc diagnostics for an image classifier, run against a real
# trained checkpoint instead of guessed at from aggregate top-1/top-5.
#
# Defaults to CPU and val-dev: this is meant to run alongside a concurrent
# GPU training job without contending for it, and val-test is only ever
# touched once, for the report's headline number.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as tf  # noqa: E402
import torch.utils.data  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import FoodX251Dataset, dataset_paths, load_val_split  # noqa: E402
from model import FoodCNN  # noqa: E402

NUM_CLASSES = 251


def load_class_names(path: Path, num_classes: int = NUM_CLASSES) -> list[str]:
    names = [""] * num_classes
    for line in path.read_text().splitlines():
        index, name = line.split(maxsplit=1)
        names[int(index)] = name.replace("_", " ")
    return names


def top_confused_pairs(
    confusion: np.ndarray, class_names: list[str], n: int = 20
) -> list[tuple[str, str, int]]:
    # Off-diagonal entries only -- the diagonal is correct predictions, not
    # confusion. Sorted descending so the worst mix-ups come first.
    off_diagonal = confusion.copy()
    np.fill_diagonal(off_diagonal, 0)
    flat_indices = np.argsort(off_diagonal, axis=None)[::-1][:n]

    pairs = []
    for flat_index in flat_indices:
        true_index, pred_index = np.unravel_index(flat_index, confusion.shape)
        # Read the count from off_diagonal, not confusion: a diagonal cell
        # sorts to the same rank as a genuinely-zero off-diagonal cell (both
        # are 0 here), and reading the original confusion matrix at that
        # position would silently resurrect it with its real (large) count.
        count = int(off_diagonal[true_index, pred_index])
        if count == 0:
            break
        pairs.append((class_names[true_index], class_names[pred_index], count))
    return pairs


def per_class_accuracy(confusion: np.ndarray) -> np.ndarray:
    row_sums = confusion.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = np.diag(confusion) / row_sums
    return np.nan_to_num(accuracy)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    penultimate: list[torch.Tensor] = []

    # A forward hook keeps this decoupled from FoodCNN's internals -- no need
    # to reimplement `features -> avgpool` here, so it can't drift from the
    # real forward pass.
    def capture(_module: torch.nn.Module, _input: object, output: torch.Tensor) -> None:
        penultimate.append(output.detach().flatten(1).cpu())

    handle = model.avgpool.register_forward_hook(capture)

    all_targets, all_preds, all_confidences = [], [], []
    for images, target in loader:
        images = images.to(device)
        output = model(images)
        probs = tf.softmax(output, dim=1)
        confidence, pred = probs.max(dim=1)

        all_targets.append(target.numpy())
        all_preds.append(pred.cpu().numpy())
        all_confidences.append(confidence.cpu().numpy())

    handle.remove()
    return {
        "targets": np.concatenate(all_targets),
        "preds": np.concatenate(all_preds),
        "confidences": np.concatenate(all_confidences),
        "features": torch.cat(penultimate).numpy(),
    }


def plot_worst_classes(
    accuracy: np.ndarray, class_names: list[str], out_path: Path, n: int = 30
) -> None:
    order = np.argsort(accuracy)[:n]
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.25)))
    ax.barh([class_names[i] for i in order], accuracy[order] * 100)
    ax.set_xlabel("accuracy (%)")
    ax.set_title(f"{n} worst classes by accuracy")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confidence_histogram(confidences: np.ndarray, correct: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(confidences[correct], bins=30, alpha=0.6, label="correct", density=True)
    ax.hist(confidences[~correct], bins=30, alpha=0.6, label="incorrect", density=True)
    ax.set_xlabel("model confidence (max softmax probability)")
    ax.set_ylabel("density")
    ax.set_title("prediction confidence: correct vs incorrect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tsne(
    features: np.ndarray, correct: np.ndarray, out_path: Path, *, samples: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    index = rng.choice(len(features), size=min(samples, len(features)), replace=False)

    embedded = TSNE(n_components=2, init="pca", random_state=seed).fit_transform(features[index])

    fig, ax = plt.subplots(figsize=(7, 6))
    subset_correct = correct[index]
    ax.scatter(embedded[subset_correct, 0], embedded[subset_correct, 1],
               s=6, alpha=0.5, label="correct")
    ax.scatter(embedded[~subset_correct, 0], embedded[~subset_correct, 1],
               s=6, alpha=0.5, label="incorrect")
    ax.set_title(f"t-SNE of penultimate features (n={len(index):,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", nargs="?", default="food251", type=Path)
    parser.add_argument("--val-split", default="splits/val_split.csv", type=Path)
    parser.add_argument("--split", default="dev", choices=["dev", "test"],
                         help="val-test should only be touched once, for the report's "
                              "headline number -- default is val-dev")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                         help="default cpu so this can run alongside a concurrent GPU "
                              "training job without contending for it")
    parser.add_argument("--out", default="runs/analysis", type=Path)
    parser.add_argument("--top-n", default=20, type=int)
    parser.add_argument("--tsne-samples", default=2000, type=int)
    parser.add_argument("--seed", default=251, type=int)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.data / "meta" / "class_list.txt")

    model = FoodCNN(num_classes=NUM_CLASSES)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    _, (val_dir, val_labels) = dataset_paths(args.data)
    dataset = FoodX251Dataset(
        val_dir, val_labels,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        subset=load_val_split(args.val_split, args.split),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print(f"running inference on {len(dataset):,} images ({args.split}) from checkpoint "
          f"epoch {checkpoint.get('epoch', '?')}...")
    result = run_inference(model, loader, device)
    targets, preds = result["targets"], result["preds"]
    correct = targets == preds
    print(f"overall accuracy: {correct.mean() * 100:.2f}%\n")

    confusion = confusion_matrix(targets, preds, labels=range(NUM_CLASSES))
    accuracy = per_class_accuracy(confusion)

    report = classification_report(targets, preds, labels=range(NUM_CLASSES),
                                    target_names=class_names, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(args.out / "classification_report.csv")
    macro, weighted = report["macro avg"], report["weighted avg"]
    print(f"macro avg    precision {macro['precision']:.3f}  recall {macro['recall']:.3f}  "
          f"f1 {macro['f1-score']:.3f}")
    print(f"weighted avg precision {weighted['precision']:.3f}  recall {weighted['recall']:.3f}  "
          f"f1 {weighted['f1-score']:.3f}")

    pairs = top_confused_pairs(confusion, class_names, n=args.top_n)
    pd.DataFrame(pairs, columns=["true", "predicted", "count"]).to_csv(
        args.out / "top_confused_pairs.csv", index=False)
    print(f"\ntop {len(pairs)} most confused pairs (true -> predicted, count):")
    for true_name, pred_name, count in pairs:
        print(f"  {true_name:30s} -> {pred_name:30s}  {count}")

    plot_worst_classes(accuracy, class_names, args.out / "worst_classes.png")
    plot_confidence_histogram(result["confidences"], correct, args.out / "confidence_histogram.png")
    plot_tsne(result["features"], correct, args.out / "tsne.png",
              samples=args.tsne_samples, seed=args.seed)
    print(f"\nwrote classification_report.csv, top_confused_pairs.csv and 3 plots to {args.out}")


if __name__ == "__main__":
    main()

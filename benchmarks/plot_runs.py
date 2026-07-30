# Run: python benchmarks/plot_runs.py runs/phaseD/*.json --out runs/plots
#
# Training history sits in runs/*.json and until now was only ever read via
# one-off scripts printing a final number. A final-epoch comparison hides how
# runs got there -- this turns the full history into per-run learning curves
# plus a cross-run overlay for whichever metric is being compared.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRIC_CHOICES = [
    "lr", "train_loss", "train_acc1", "train_acc3", "train_acc5",
    "val_acc1", "val_acc3", "val_acc5",
]


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _series(history: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    # Older run logs predate top-3/lr tracking (deliberately not backfilled --
    # see plans/2026-07-30-training-roadmap.md) and simply lack those keys.
    # Skip missing epochs instead of crashing so old and new logs both plot.
    points = [(r["epoch"], r[key]) for r in history if key in r]
    return [e for e, _ in points], [v for _, v in points]


def plot_learning_curves(run: dict[str, Any], out_path: Path) -> None:
    history = run["history"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(*_series(history, "train_loss"), label="train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("loss")
    axes[0].legend()

    for key, label in (
        ("train_acc1", "train top-1"),
        ("val_acc1", "val top-1"),
        ("val_acc3", "val top-3"),
        ("val_acc5", "val top-5"),
    ):
        epochs, values = _series(history, key)
        if values:
            axes[1].plot(epochs, values, label=label)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title("accuracy")
    axes[1].legend()

    fig.suptitle(run["label"])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_comparison(runs: list[dict[str, Any]], metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in runs:
        epochs, values = _series(run["history"], metric)
        if not values:
            print(f"  skipping '{run['label']}': no '{metric}' in this run's log")
            continue
        ax.plot(epochs, values, label=run["label"])
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"comparison: {metric}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path, help="run JSON files, e.g. runs/phaseD/*.json")
    parser.add_argument("--out", default="runs/plots", type=Path)
    parser.add_argument("--metric", default="val_acc1", choices=METRIC_CHOICES,
                         help="metric to overlay in the multi-run comparison plot (default: val_acc1)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(p) for p in args.runs]

    for run, path in zip(runs, args.runs, strict=True):
        out_path = args.out / f"{path.stem}-curves.png"
        plot_learning_curves(run, out_path)
        print(f"  wrote {out_path}")

    if len(runs) > 1:
        out_path = args.out / f"comparison-{args.metric}.png"
        plot_comparison(runs, args.metric, out_path)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()

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
    "knn_acc1", "feat_std", "effective_rank",
]

# A self-supervised run has no accuracy to plot, so its probe metrics are only
# readable against the two encoders benchmarks/ssl_probe_calibration.py scores:
# a random-init floor and the 63.83% supervised checkpoint. Drawn as reference
# lines, because an absolute effective rank means nothing without them.
SSL_ANCHORS = {
    "knn_acc1": (2.80, 61.16),
    "feat_std": (0.0074, 0.0341),
    "effective_rank": (4.8, 184.7),
}


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _series(history: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    # Older run logs predate top-3/lr tracking (deliberately not backfilled --
    # see plans/2026-07-30-training-roadmap.md) and simply lack those keys.
    # Skip missing epochs instead of crashing so old and new logs both plot.
    #
    # None is skipped as well as absent: an SSL run records every key every
    # epoch but only fills the probe metrics on probe epochs, so knn_acc1 is
    # present-but-None on the epochs in between.
    points = [(r["epoch"], r[key]) for r in history if r.get(key) is not None]
    return [e for e, _ in points], [v for _, v in points]


def _draw_anchors(ax, metric: str) -> None:
    anchors = SSL_ANCHORS.get(metric)
    if anchors is None:
        return
    random_init, supervised = anchors
    ax.axhline(random_init, color="grey", linestyle=":", linewidth=1)
    ax.axhline(supervised, color="green", linestyle=":", linewidth=1)
    ax.annotate("random init", (0.01, random_init), xycoords=("axes fraction", "data"),
                fontsize=7, color="grey", va="bottom")
    ax.annotate("supervised 63.83%", (0.01, supervised), xycoords=("axes fraction", "data"),
                fontsize=7, color="green", va="top")


def plot_learning_curves(run: dict[str, Any], out_path: Path) -> None:
    history = run["history"]
    is_ssl = bool(_series(history, "knn_acc1")[1] or _series(history, "effective_rank")[1])

    fig, axes = plt.subplots(1, 3 if is_ssl else 2, figsize=(15 if is_ssl else 11, 4))

    axes[0].plot(*_series(history, "train_loss"), label="train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("loss")
    if is_ssl:
        # -1.0 is the exact loss of the collapsed solution the stop-gradient
        # exists to prevent, so proximity to it is the thing to look at.
        axes[0].axhline(-1.0, color="red", linestyle=":", linewidth=1)
        axes[0].annotate("collapse", (0.01, -1.0), xycoords=("axes fraction", "data"),
                         fontsize=7, color="red", va="bottom")
    axes[0].legend()

    if is_ssl:
        panels = [("knn_acc1", "kNN top-1 (%)", "kNN probe"),
                  ("effective_rank", "effective rank / 512", "dimensions used")]
    else:
        panels = [(None, "accuracy (%)", "accuracy")]

    for ax, (key, ylabel, title) in zip(axes[1:], panels, strict=True):
        if key is None:
            for series_key, label in (
                ("train_acc1", "train top-1"),
                ("val_acc1", "val top-1"),
                ("val_acc3", "val top-3"),
                ("val_acc5", "val top-5"),
            ):
                epochs, values = _series(history, series_key)
                if values:
                    ax.plot(epochs, values, label=label)
        else:
            ax.plot(*_series(history, key), marker="o", label=key)
            _draw_anchors(ax, key)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

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
        ax.plot(epochs, values, marker="o" if metric in SSL_ANCHORS else None,
                label=run["label"])
    _draw_anchors(ax, metric)
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

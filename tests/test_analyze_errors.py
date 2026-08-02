import pathlib

import numpy as np
import pytest
import torch
from analyze_errors import (
    default_out_dir,
    load_class_names,
    per_class_accuracy,
    plot_confidence_histogram,
    plot_tsne,
    plot_worst_classes,
    run_inference,
    symmetric_confusion_candidates,
    top_confused_pairs,
)
from torch.utils.data import DataLoader, TensorDataset

from model import FoodCNN


def test_default_out_dir_strips_best_and_both_extensions() -> None:
    assert default_out_dir(pathlib.Path("checkpoints/phaseD-gce-best.pth.tar")) == pathlib.Path(
        "runs/analysis/phaseD-gce")
    assert default_out_dir(pathlib.Path("checkpoints/phaseD-gce.pth.tar")) == pathlib.Path(
        "runs/analysis/phaseD-gce")


def test_default_out_dir_keeps_dots_inside_the_run_label() -> None:
    # Path.stem would turn "phaseD-lr0.8" into "phaseD-lr0" and collapse the
    # whole learning-rate sweep into one directory.
    assert default_out_dir(pathlib.Path("checkpoints/phaseD-lr0.8-best.pth.tar")) == pathlib.Path(
        "runs/analysis/phaseD-lr0.8")


def test_default_out_dir_differs_per_checkpoint() -> None:
    # The point of the default: two checkpoints analyzed back to back must not
    # write over each other.
    assert default_out_dir(pathlib.Path("checkpoints/phaseD-gce-best.pth.tar")) != default_out_dir(
        pathlib.Path("checkpoints/phaseD-mixup-best.pth.tar"))


def test_load_class_names_parses_index_and_replaces_underscores(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "class_list.txt"
    path.write_text("0 macaron\n1 beef_bourguignonne\n2 club_sandwich\n")

    names = load_class_names(path, num_classes=3)

    assert names == ["macaron", "beef bourguignonne", "club sandwich"]


def test_top_confused_pairs_ignores_the_diagonal() -> None:
    # class 0 <-> 1 confused 5 times, class 1 -> 2 confused 3 times; the
    # diagonal (10, 8, 6 correct predictions) must never show up as "confusion".
    confusion = np.array([
        [10, 5, 0],
        [0, 8, 3],
        [0, 0, 6],
    ])
    pairs = top_confused_pairs(confusion, ["a", "b", "c"], n=10)

    assert ("a", "b", 5) in pairs
    assert ("b", "c", 3) in pairs
    assert not any(true == pred for true, pred, _ in pairs)
    assert len(pairs) == 2  # only two nonzero off-diagonal entries exist


def test_top_confused_pairs_respects_n() -> None:
    confusion = np.array([
        [0, 5, 4],
        [3, 0, 2],
        [1, 1, 0],
    ])
    pairs = top_confused_pairs(confusion, ["a", "b", "c"], n=2)

    assert len(pairs) == 2
    assert pairs[0][2] == 5  # sorted descending by count


def test_symmetric_confusion_requires_both_directions_nonzero() -> None:
    # "a" is confused for "b" a lot, but "b" is never confused for "a" --
    # that's an ordinary one-way mix-up (e.g. a common class as a catch-all
    # wrong answer), not evidence the two labels are the same thing.
    confusion = np.array([
        [0, 20, 0],
        [0, 10, 0],
        [0, 0, 10],
    ])
    candidates = symmetric_confusion_candidates(confusion, ["a", "b", "c"], min_rate=0.01)

    assert candidates == []


def test_symmetric_confusion_flags_a_genuine_two_way_pair() -> None:
    # "a" and "b" are confused for each other at a high, roughly symmetric
    # rate -- the signature of a possible near-duplicate class, distinct
    # from top_confused_pairs' one-directional ranking.
    confusion = np.array([
        [5, 8, 0],
        [7, 5, 0],
        [0, 0, 20],
    ])
    candidates = symmetric_confusion_candidates(confusion, ["a", "b", "c"], min_rate=0.1)

    assert len(candidates) == 1
    class_a, class_b, a_to_b, b_to_a, rate = candidates[0]
    assert {class_a, class_b} == {"a", "b"}
    assert {a_to_b, b_to_a} == {8, 7}
    assert rate == pytest.approx(15 / 25)


def test_symmetric_confusion_respects_min_rate() -> None:
    confusion = np.array([
        [90, 1, 0],
        [1, 90, 0],
        [0, 0, 20],
    ])
    candidates = symmetric_confusion_candidates(confusion, ["a", "b", "c"], min_rate=0.5)

    assert candidates == []


def test_per_class_accuracy_matches_manual_computation() -> None:
    confusion = np.array([
        [8, 2],
        [1, 9],
    ])
    accuracy = per_class_accuracy(confusion)

    assert accuracy == pytest.approx([0.8, 0.9])


def test_per_class_accuracy_handles_a_class_with_zero_samples() -> None:
    # A class absent from this val split (row sum 0) must not produce NaN or
    # raise -- it should just read as 0 accuracy, not "undefined".
    confusion = np.array([
        [0, 0],
        [2, 8],
    ])
    accuracy = per_class_accuracy(confusion)

    assert accuracy[0] == 0.0
    assert not np.isnan(accuracy).any()


def test_run_inference_returns_expected_shapes_and_types() -> None:
    torch.manual_seed(0)
    model = FoodCNN(num_classes=4)
    images = torch.randn(6, 3, 176, 176)
    targets = torch.randint(0, 4, (6,))
    loader = DataLoader(TensorDataset(images, targets), batch_size=4)

    result = run_inference(model, loader, torch.device("cpu"))

    assert result["targets"].shape == (6,)
    assert result["preds"].shape == (6,)
    assert result["confidences"].shape == (6,)
    assert result["features"].shape == (6, 512)
    assert (result["confidences"] >= 0).all() and (result["confidences"] <= 1).all()


def test_plot_worst_classes_writes_a_file(tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "worst.png"
    accuracy = np.linspace(0.1, 0.9, 10)
    plot_worst_classes(accuracy, [f"class_{i}" for i in range(10)], out_path, n=5)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_confidence_histogram_writes_a_file(tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "confidence.png"
    rng = np.random.default_rng(0)
    confidences = rng.random(100)
    correct = rng.random(100) > 0.5

    plot_confidence_histogram(confidences, correct, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_tsne_writes_a_file(tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "tsne.png"
    rng = np.random.default_rng(0)
    features = rng.random((50, 16))
    correct = rng.random(50) > 0.5

    plot_tsne(features, correct, out_path, samples=50, seed=0)

    assert out_path.exists()
    assert out_path.stat().st_size > 0

import pathlib

from plot_runs import _series, plot_comparison, plot_learning_curves


def _run(label: str, *, with_top3: bool = True) -> dict:
    history = []
    for epoch in range(3):
        record = {
            "epoch": epoch,
            "train_loss": 5.0 - epoch,
            "train_acc1": epoch * 10.0,
            "val_acc1": epoch * 12.0,
            "val_acc5": epoch * 20.0,
        }
        if with_top3:
            record["train_acc3"] = epoch * 15.0
            record["val_acc3"] = epoch * 16.0
            record["lr"] = 0.1
        history.append(record)
    return {"label": label, "history": history}


def test_series_skips_epochs_missing_the_key() -> None:
    # Older run logs predate top-3/lr tracking and simply lack those keys --
    # the plotter has to degrade gracefully, not crash on a real log.
    history = [
        {"epoch": 0, "val_acc1": 10.0},
        {"epoch": 1, "val_acc1": 20.0, "val_acc3": 25.0},
    ]
    epochs, values = _series(history, "val_acc3")
    assert epochs == [1]
    assert values == [25.0]


def test_series_returns_empty_when_key_never_present() -> None:
    history = [{"epoch": 0, "val_acc1": 10.0}]
    assert _series(history, "lr") == ([], [])


def test_plot_learning_curves_writes_a_nonempty_file(tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "curves.png"
    plot_learning_curves(_run("baseline"), out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_learning_curves_handles_a_pre_top3_run(tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "curves.png"
    plot_learning_curves(_run("old-run", with_top3=False), out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_comparison_skips_runs_missing_the_metric(tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "comparison.png"
    runs = [_run("new-run", with_top3=True), _run("old-run", with_top3=False)]

    # Must not raise even though "old-run" has no val_acc3 -- it should be
    # silently skipped, not crash the whole comparison.
    plot_comparison(runs, "val_acc3", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0

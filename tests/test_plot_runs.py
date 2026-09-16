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


def _ssl_run(label: str = "gate", probe_freq: int = 2) -> dict:
    # An SSL log writes every key every epoch but only fills the probe metrics
    # on probe epochs, so the gaps are None rather than absent keys.
    history = []
    for epoch in range(4):
        probed = (epoch + 1) % probe_freq == 0
        history.append({
            "epoch": epoch,
            "lr": 0.05,
            "train_loss": -0.4 - epoch * 0.05,
            "train_acc1": None,
            "val_acc1": None,
            "knn_acc1": 3.0 + epoch if probed else None,
            "feat_std": 0.01 + epoch * 0.001 if probed else None,
            "effective_rank": 5.0 + epoch * 2 if probed else None,
        })
    return {"label": label, "history": history}


def test_series_skips_present_but_none_values() -> None:
    # An SSL run records knn_acc1 on every epoch and fills it only on probe
    # epochs; `key in record` is True for the None ones, so filtering on
    # presence alone would hand matplotlib a list full of None.
    epochs, values = _series(_ssl_run()["history"], "knn_acc1")

    assert epochs == [1, 3]
    assert values == [4.0, 6.0]


def test_ssl_run_plots_probe_panels(tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "ssl-curves.png"
    plot_learning_curves(_ssl_run(), out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_ssl_comparison_plots_effective_rank(tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "cmp.png"
    plot_comparison([_ssl_run("a"), _ssl_run("b")], "effective_rank", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_supervised_run_still_plots_two_panels(tmp_path: pathlib.Path) -> None:
    # The SSL layout must not leak into a supervised run, which has no probe
    # metrics at all and would otherwise get an empty third panel.
    out_path = tmp_path / "sup.png"
    plot_learning_curves(_run("baseline"), out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


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

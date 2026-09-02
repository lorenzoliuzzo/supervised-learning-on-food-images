import json
import pathlib

from runlog import RunLog


def test_config_hash_is_deterministic_and_order_independent() -> None:
    a = RunLog(label="x", config={"lr": 0.1, "batch": 256})
    b = RunLog(label="x", config={"batch": 256, "lr": 0.1})

    assert a.config_hash() == b.config_hash()


def test_config_hash_differs_for_different_configs() -> None:
    a = RunLog(label="x", config={"lr": 0.1})
    b = RunLog(label="x", config={"lr": 0.2})

    assert a.config_hash() != b.config_hash()


def test_save_writes_history_and_metadata(tmp_path: pathlib.Path) -> None:
    run = RunLog(label="baseline", config={"lr": 0.1, "epochs": 15})
    run.record(epoch=0, lr=0.02, train_loss=5.0, train_acc1=1.0, train_acc3=3.0, train_acc5=5.0,
               val_acc1=0.5, val_acc3=2.0, val_acc5=3.0)
    run.record(epoch=1, lr=0.04, train_loss=4.5, train_acc1=2.0, train_acc3=4.0, train_acc5=7.0,
               val_acc1=1.0, val_acc3=2.5, val_acc5=4.0)

    path = run.save(tmp_path, peak_vram_gib=1.23)
    payload = json.loads(path.read_text())

    assert payload["label"] == "baseline"
    assert payload["config"] == {"lr": 0.1, "epochs": 15}
    assert payload["config_hash"] == run.config_hash()
    assert payload["peak_vram_gib"] == 1.23
    assert len(payload["history"]) == 2
    assert payload["history"][1] == {
        "epoch": 1, "lr": 0.04, "train_loss": 4.5, "train_acc1": 2.0, "train_acc3": 4.0,
        "train_acc5": 7.0, "val_acc1": 1.0, "val_acc3": 2.5, "val_acc5": 4.0,
    }


def test_restore_carries_the_pre_restart_history(tmp_path: pathlib.Path) -> None:
    # A resumed run used to log only the epochs after the restart, so
    # plot_runs.py drew a curve starting mid-training with no sign of it.
    original = RunLog(label="phaseD", config={"lr": 0.8})
    original.record(epoch=0, lr=0.16, train_loss=5.0, train_acc1=1.0, train_acc3=3.0,
                    train_acc5=5.0, val_acc1=0.5, val_acc3=2.0, val_acc5=3.0)
    original.record(epoch=1, lr=0.32, train_loss=4.5, train_acc1=2.0, train_acc3=4.0,
                    train_acc5=7.0, val_acc1=1.0, val_acc3=2.5, val_acc5=4.0)

    resumed = RunLog.restore("phaseD", {"lr": 0.8}, original.to_checkpoint())
    resumed.record(epoch=2, lr=0.48, train_loss=4.0, train_acc1=3.0, train_acc3=5.0,
                   train_acc5=8.0, val_acc1=1.5, val_acc3=3.0, val_acc5=5.0)

    payload = json.loads(resumed.save(tmp_path).read_text())

    assert [r["epoch"] for r in payload["history"]] == [0, 1, 2]
    assert payload["history"][0]["train_loss"] == 5.0


def test_restore_keeps_ssl_records_with_their_none_accuracies() -> None:
    # record_ssl leaves the six accuracy fields at None; EpochRecord(**record)
    # has to round-trip that rather than choke on the missing numbers.
    original = RunLog(label="simsiam", config={})
    original.record_ssl(epoch=0, lr=0.05, loss=-0.42)

    resumed = RunLog.restore("simsiam", {}, original.to_checkpoint())

    assert len(resumed.history) == 1
    assert resumed.history[0].train_loss == -0.42
    assert resumed.history[0].val_acc1 is None


def test_restore_accumulates_wall_clock_across_restarts(tmp_path: pathlib.Path) -> None:
    original = RunLog(label="phaseD", config={})
    checkpoint = original.to_checkpoint()
    checkpoint["run_wall_clock_s"] = 3600.0  # an hour already spent

    resumed = RunLog.restore("phaseD", {}, checkpoint)
    payload = json.loads(resumed.save(tmp_path).read_text())

    assert payload["wall_clock_s"] >= 3600.0


def test_restore_from_a_checkpoint_predating_run_history(tmp_path: pathlib.Path) -> None:
    # Checkpoints written before this existed carry neither key; restoring
    # from one has to behave like a fresh log, not raise.
    resumed = RunLog.restore("phaseD", {"lr": 0.8}, {"epoch": 30, "best_acc1": 40.0})

    assert resumed.history == []
    assert resumed.resumed_wall_clock_s == 0.0
    assert json.loads(resumed.save(tmp_path).read_text())["history"] == []


def test_save_path_is_named_from_the_config_hash(tmp_path: pathlib.Path) -> None:
    run = RunLog(label="baseline", config={"lr": 0.1})
    path = run.save(tmp_path)

    assert path.name == f"{run.config_hash()}-baseline.json"

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


def test_save_path_is_named_from_the_config_hash(tmp_path: pathlib.Path) -> None:
    run = RunLog(label="baseline", config={"lr": 0.1})
    path = run.save(tmp_path)

    assert path.name == f"{run.config_hash()}-baseline.json"

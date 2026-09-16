from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpochRecord:
    epoch: int
    lr: float
    train_loss: float
    # None for SSL runs (record_ssl), which have no classification accuracy.
    train_acc1: float | None = None
    train_acc3: float | None = None
    train_acc5: float | None = None
    val_acc1: float | None = None
    val_acc3: float | None = None
    val_acc5: float | None = None
    # SSL-only (record_ssl). knn_acc1 is the frozen-feature probe; feat_std and
    # effective_rank are the collapse diagnostics -- SimSiam's loss descends
    # even when the representation has collapsed, so the loss alone can't tell
    # a working pretrain from a dead one. None on epochs where no probe ran.
    knn_acc1: float | None = None
    feat_std: float | None = None
    effective_rank: float | None = None


@dataclass
class RunLog:
    label: str
    config: dict[str, Any]
    history: list[EpochRecord] = field(default_factory=list)
    # Seconds already spent by the run this one continues, so wall_clock_s
    # covers the whole run rather than just the tail since the last restart.
    resumed_wall_clock_s: float = 0.0
    _started: float = field(default_factory=time.perf_counter, repr=False)

    @classmethod
    def restore(cls, label: str, config: dict[str, Any], checkpoint: dict[str, Any]) -> RunLog:
        # A resumed run used to start its log from empty, so runs/*.json held
        # only the epochs after the restart and plot_runs.py drew a curve
        # beginning mid-training with nothing to say it was truncated.
        # Checkpoints written before this carry neither key and restore as a
        # fresh log, which is the old behaviour.
        return cls(
            label=label,
            config=config,
            history=[EpochRecord(**record) for record in checkpoint.get('run_history', [])],
            resumed_wall_clock_s=checkpoint.get('run_wall_clock_s', 0.0),
        )

    def to_checkpoint(self) -> dict[str, Any]:
        # Carried inside the checkpoint rather than re-read from runs/*.json:
        # the log is only written at the end of a run, so a run that died
        # mid-training -- the reason to resume at all -- never wrote one.
        return {
            'run_history': [vars(r) for r in self.history],
            'run_wall_clock_s': self.elapsed(),
        }

    def elapsed(self) -> float:
        return self.resumed_wall_clock_s + (time.perf_counter() - self._started)

    def config_hash(self) -> str:
        # Config, not label, is what should collide when two runs are the
        # same experiment -- the label is just a human-readable tag.
        blob = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def record(
        self,
        epoch: int,
        lr: float,
        train_loss: float,
        train_acc1: float,
        train_acc3: float,
        train_acc5: float,
        val_acc1: float,
        val_acc3: float,
        val_acc5: float,
    ) -> None:
        self.history.append(
            EpochRecord(epoch, lr, train_loss, train_acc1, train_acc3, train_acc5,
                        val_acc1, val_acc3, val_acc5)
        )

    def record_ssl(
        self,
        epoch: int,
        lr: float,
        loss: float,
        knn_acc1: float | None = None,
        feat_std: float | None = None,
        effective_rank: float | None = None,
    ) -> None:
        # SimSiam pretraining (src/simsiam.py) has no classification accuracy
        # to log -- the six acc fields stay at their None default.
        self.history.append(
            EpochRecord(epoch, lr, loss, knn_acc1=knn_acc1, feat_std=feat_std,
                        effective_rank=effective_rank)
        )

    def save(self, directory: Path, *, peak_vram_gib: float = 0.0) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": self.label,
            "config": self.config,
            "config_hash": self.config_hash(),
            "wall_clock_s": self.elapsed(),
            "peak_vram_gib": peak_vram_gib,
            "history": [vars(r) for r in self.history],
        }
        path = directory / f"{self.config_hash()}-{self.label}.json"
        path.write_text(json.dumps(payload, indent=2))
        return path

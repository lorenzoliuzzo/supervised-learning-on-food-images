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
    _started: float = field(default_factory=time.perf_counter, repr=False)

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
            "wall_clock_s": time.perf_counter() - self._started,
            "peak_vram_gib": peak_vram_gib,
            "history": [vars(r) for r in self.history],
        }
        path = directory / f"{self.config_hash()}-{self.label}.json"
        path.write_text(json.dumps(payload, indent=2))
        return path

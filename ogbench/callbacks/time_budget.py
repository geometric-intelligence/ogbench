"""Stop training gracefully once a wall-clock budget is spent."""

from __future__ import annotations

import logging
import time
from typing import Any

import lightning.pytorch as pl
import psutil

log = logging.getLogger(__name__)


class TimeBudget(pl.Callback):
    """End ``trainer.fit`` at a wall-clock deadline so the run still tests its best checkpoint.

    Lightning's ``max_time`` only sets ``trainer.should_stop``, which the fit loop ignores until
    ``min_epochs`` is reached. Once the budget is spent this callback also lifts ``min_epochs``
    and ``min_steps``, so training ends after the current batch and ``trainer.test`` runs on the
    best checkpoint saved so far.

    The deadline is either an absolute Unix time (``deadline``, set by the search launcher so
    data loading counts) or ``budget_seconds`` after this process started. With neither set the
    callback does nothing.
    """

    def __init__(
        self,
        budget_seconds: float | str | None = None,
        deadline: float | str | None = None,
    ) -> None:
        super().__init__()
        self.deadline = self._resolve_deadline(budget_seconds, deadline)
        self.hit = False
        self.stopped_at_epoch: int | None = None

    @staticmethod
    def _optional_float(value: float | str | None) -> float | None:
        if value is None or (isinstance(value, str) and value.strip().lower() in {'', 'null'}):
            return None
        return float(value)

    @classmethod
    def _resolve_deadline(
        cls, budget_seconds: float | str | None, deadline: float | str | None
    ) -> float | None:
        absolute = cls._optional_float(deadline)
        if absolute is not None:
            return absolute
        budget = cls._optional_float(budget_seconds)
        if budget is None:
            return None
        if budget <= 0:
            raise ValueError(f'budget_seconds must be positive, got {budget}')
        return psutil.Process().create_time() + budget

    @property
    def active(self) -> bool:
        return self.deadline is not None

    def _stop_if_spent(self, trainer: pl.Trainer) -> None:
        if self.deadline is None or self.hit or time.time() < self.deadline:
            return
        self.hit = True
        self.stopped_at_epoch = trainer.current_epoch
        trainer.fit_loop.min_epochs = 0
        trainer.fit_loop.epoch_loop.min_steps = None
        trainer.should_stop = True
        log.info(
            'Time budget spent at epoch %d; stopping training and testing the best checkpoint',
            trainer.current_epoch,
        )

    def on_train_batch_end(self, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        self._stop_if_spent(trainer)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._stop_if_spent(trainer)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.sanity_checking:
            self._stop_if_spent(trainer)

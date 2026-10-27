"""Tests for the graceful wall-clock stop callback."""

from __future__ import annotations

import time
from pathlib import Path

import lightning.pytorch as pl
import psutil
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader, TensorDataset

from ogbench.callbacks.time_budget import TimeBudget
from ogbench.utils.config_resolvers import register_all_resolvers
from ogbench.utils.hparam_search import DEADLINE_ENV_VAR

CONFIG_DIR = Path(__file__).resolve().parents[2] / 'configs'


class _TinyModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        features, target = batch
        return torch.nn.functional.mse_loss(self.layer(features), target)

    def test_step(self, batch, batch_idx):
        features, target = batch
        self.log('test/loss', torch.nn.functional.mse_loss(self.layer(features), target))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    dataset = TensorDataset(torch.randn(32, 4, generator=generator), torch.randn(32, 1))
    return DataLoader(dataset, batch_size=8)


def _trainer(callback: TimeBudget) -> pl.Trainer:
    return pl.Trainer(
        min_epochs=50,
        max_epochs=200,
        accelerator='cpu',
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def test_spent_budget_overrides_min_epochs_and_still_tests() -> None:
    callback = TimeBudget(deadline=time.time() - 1)
    trainer = _trainer(callback)
    model = _TinyModule()

    trainer.fit(model, train_dataloaders=_loader())
    results = trainer.test(model, dataloaders=_loader(), verbose=False)

    assert callback.hit is True
    assert callback.stopped_at_epoch == 0
    assert trainer.current_epoch <= 1
    assert trainer.global_step == 1
    assert 'test/loss' in results[0]


def test_unspent_budget_leaves_training_alone() -> None:
    callback = TimeBudget(deadline=time.time() + 3600)
    trainer = _trainer(callback)
    trainer.fit_loop.max_epochs = 2
    trainer.fit_loop.min_epochs = 2

    trainer.fit(_TinyModule(), train_dataloaders=_loader())

    assert callback.hit is False
    assert trainer.current_epoch == 2


@pytest.mark.parametrize('deadline', [None, '', 'null'])
def test_callback_is_inactive_without_budget_or_deadline(deadline) -> None:
    callback = TimeBudget(budget_seconds=None, deadline=deadline)

    assert callback.active is False
    assert callback.deadline is None


def test_budget_counts_from_process_start() -> None:
    callback = TimeBudget(budget_seconds=120)

    assert callback.deadline == pytest.approx(psutil.Process().create_time() + 120)


def test_absolute_deadline_takes_precedence_and_accepts_env_strings() -> None:
    callback = TimeBudget(budget_seconds=120, deadline='1700000000.5')

    assert callback.deadline == 1700000000.5


def test_non_positive_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match='must be positive'):
        TimeBudget(budget_seconds=0)


@pytest.mark.parametrize('experiment', ['omics_readout', 'no_readout'])
def test_composed_config_reads_launcher_deadline(
    experiment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_all_resolvers()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base='1.3'):
        cfg = compose('train.yaml', overrides=[f'experiment={experiment}'])

    monkeypatch.delenv(DEADLINE_ENV_VAR, raising=False)
    assert instantiate(cfg.callbacks.time_budget).active is False

    monkeypatch.setenv(DEADLINE_ENV_VAR, '1700000000.5')
    assert instantiate(cfg.callbacks.time_budget).deadline == 1700000000.5

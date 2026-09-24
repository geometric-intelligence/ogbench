"""Tests for transferring ratio-0.5 configurations to ratio-0.3 cells."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.optuna_search import OptunaSearchConfig, _stable_hash, build_outer_cells
from scripts.ratio03_prepare import (
    best_trial,
    cell_key,
    lookup,
    sampled_from_values,
    summarize_trials,
)

SOURCE_CONFIG = Path('configs/hparams_search/sep24_factorial_optuna.yaml')
TARGET_CONFIG = Path('configs/hparams_search/sep24_ratio03_transfer.yaml')


@pytest.fixture(scope='module')
def source() -> OptunaSearchConfig:
    return OptunaSearchConfig.from_yaml(SOURCE_CONFIG)


def _values(lr: float = 0.001, weight_decay: float = 0.0) -> dict:
    return {
        'optimizer.parameters.lr': lr,
        'optimizer.parameters.weight_decay': weight_decay,
        'model.backbone.dropout': 0.2,
        'model.feature_encoder.out_channels': 64,
        'model.backbone.num_layers': 3,
    }


def _run(study: str, trial: int, fold: int, objective: float, values: dict) -> dict:
    return {
        'run_id': f'{study}-{trial}-{fold}',
        'run_name': f'{study}_trial{trial:04d}_fold{fold}_attempt1',
        'study_name': study,
        'trial_number': trial,
        'fold': fold,
        'attempt': 1,
        'objective': objective,
        'runtime': 100.0,
        'created_at': f'2026-09-20T00:00:{trial:02d}',
        'config_values': json.dumps(values, sort_keys=True),
    }


def test_every_target_cell_has_a_ratio_05_source(source: OptunaSearchConfig) -> None:
    target = OptunaSearchConfig.from_yaml(TARGET_CONFIG)
    source_keys = {
        cell_key(cell)
        for cell in build_outer_cells(source)
        if cell.values['dataset.loader.parameters.node_sample_ratio'] == 0.5
    }

    assert {cell_key(cell) for cell in build_outer_cells(target)} == source_keys


def test_lookup_reads_nested_hydra_keys() -> None:
    config = {'optimizer': {'parameters': {'lr': 0.01}}}

    assert lookup(config, 'optimizer.parameters.lr') == 0.01
    with pytest.raises(KeyError):
        lookup(config, 'model.backbone.dropout')


def test_sampled_values_are_canonical_and_in_space(source: OptunaSearchConfig) -> None:
    sampled = sampled_from_values(source, 'gcn', _values(weight_decay=0))

    assert sampled is not None
    assert isinstance(sampled['optimizer.parameters.weight_decay'], float)
    assert sampled_from_values(source, 'gcn', {**_values(), 'model.backbone.num_layers': 9}) is None
    assert sampled_from_values(source, 'gcn', {'optimizer.parameters.lr': 0.001}) is None


def test_retried_folds_are_grouped_by_parameters(source: OptunaSearchConfig) -> None:
    cell = next(
        cell
        for cell in build_outer_cells(source, models=['gcn'], datasets=['parkinsons'])
        if cell.values['dataset.loader.parameters.node_sample_ratio'] == 0.5
    )
    study = cell.study_name
    best, other = _values(lr=0.001), _values(lr=0.0001)
    rows = [_run(study, 3, fold, 0.7, best) for fold in (0, 1, 2)]
    rows += [_run(study, 9, fold, 0.8, best) for fold in (3, 4)]
    rows += [_run(study, 1, fold, 0.6, other) for fold in range(5)]
    rows += [_run(study, 5, 0, 0.99, _values(lr=0.005))]

    trials = summarize_trials(source, pd.DataFrame(rows), {study: cell})[study]
    chosen = best_trial(trials, complete_only=True)

    assert chosen is not None
    assert chosen.trial_numbers == (3, 9)
    assert chosen.mean == pytest.approx(0.74)
    assert chosen.param_hash == _stable_hash(sampled_from_values(source, 'gcn', best))
    partial = best_trial(trials, complete_only=False)
    assert partial is not None and partial.fold_scores == {0: 0.99}

"""Tests for the resumable Optuna ablation launcher."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from optuna.trial import TrialState

from scripts import optuna_search
from scripts.optuna_search import (
    ADJACENCY_THRESHOLD,
    OptunaSearchConfig,
    RunLedger,
    _cache_configs,
    _enqueue_failed_trials,
    _load_or_create_study,
    _objective,
    build_outer_cells,
    run_search,
)

CONFIG_PATH = Path('configs/hparams_search/optuna_smoke_test.yaml')


@pytest.fixture
def search_config(tmp_path: Path) -> OptunaSearchConfig:
    config = OptunaSearchConfig.from_yaml(CONFIG_PATH)
    config.output_dir = tmp_path / 'output'
    config.storage = f'sqlite:///{tmp_path / "studies.db"}'
    return config


def test_config_builds_separate_outer_cell_with_resolved_threshold(
    search_config: OptunaSearchConfig,
) -> None:
    cells = build_outer_cells(search_config)

    assert len(cells) == 1
    cell = cells[0]
    assert cell.model == 'gcn'
    assert cell.dataset == 'motrpac'
    assert cell.values[ADJACENCY_THRESHOLD] == pytest.approx(0.0572)
    assert 'wgcna' in cell.study_name
    assert search_config.folds == [0, 1, 2, 3, 4]


def test_outer_cell_exclusions_are_applied(search_config: OptunaSearchConfig) -> None:
    search_config.ablations['experiment'] = ['omics_readout', 'no_readout']
    search_config.exclude_cells = [{'model': 'gcn', 'experiment': 'omics_readout'}]

    cells = build_outer_cells(search_config)

    assert len(cells) == 1
    assert cells[0].values['experiment'] == 'no_readout'


def test_cache_configs_include_every_fold_setting(
    search_config: OptunaSearchConfig,
) -> None:
    loaders = _cache_configs(search_config, build_outer_cells(search_config))

    assert len(loaders) == 5
    assert {loader.parameters.fold for loader in loaders} == {0, 1, 2, 3, 4}
    assert all(loader.parameters.split_type == 'k-fold' for loader in loaders)
    assert all(loader.parameters.k == 5 for loader in loaders)


def test_config_rejects_incomplete_fold_rotation(tmp_path: Path) -> None:
    raw = CONFIG_PATH.read_text().replace('folds: [0, 1, 2, 3, 4]', 'folds: [0, 1, 2, 3]')
    path = tmp_path / 'invalid.yaml'
    path.write_text(raw)

    with pytest.raises(ValueError, match='every fold exactly once'):
        OptunaSearchConfig.from_yaml(path)


def test_runtime_paths_are_independent_of_launch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / 'project'
    config_dir = project / 'configs' / 'hparams_search'
    config_dir.mkdir(parents=True)
    (project / '.project-root').touch()
    raw = CONFIG_PATH.read_text().replace(
        'thresholds_from: multi_dataset_grid_search.yaml',
        f'thresholds_from: {Path("configs/hparams_search/multi_dataset_grid_search.yaml").resolve()}',
    )
    raw = raw.replace(
        './search_results/optuna_smoke_test',
        './outputs',
    )
    path = config_dir / 'search.yaml'
    path.write_text(raw)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    config = OptunaSearchConfig.from_yaml(path)

    assert config.output_dir == project / 'outputs'
    assert config.storage == f'sqlite:///{project / "outputs" / "studies.db"}'


def test_cache_randomness_is_reset_deterministically() -> None:
    optuna_search._seed_cache_randomness(42)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    optuna_search._seed_cache_randomness(42)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert first == second


def test_objective_aggregates_folds_and_reuses_completed_fold_records(
    search_config: OptunaSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = build_outer_cells(search_config)[0]
    ledger_path = search_config.output_dir / 'ledger.db'
    calls: list[int] = []

    def fake_training(overrides, **kwargs):
        fold_override = next(
            value for value in overrides if value.startswith('dataset.split_params.data_seed=')
        )
        fold = int(fold_override.rsplit('=', maxsplit=1)[1])
        calls.append(fold)
        return True, None, {'objective': 0.5 + fold / 10}

    monkeypatch.setattr(optuna_search, 'run_training', fake_training)
    study = optuna.create_study(direction='maximize', study_name=cell.study_name)
    study.optimize(
        lambda trial: _objective(
            trial,
            config=search_config,
            cell=cell,
            ledger_path=ledger_path,
            gpu_queue=None,
        ),
        n_trials=1,
    )
    study.enqueue_trial(study.trials[0].params)
    study.optimize(
        lambda trial: _objective(
            trial,
            config=search_config,
            cell=cell,
            ledger_path=ledger_path,
            gpu_queue=None,
        ),
        n_trials=1,
    )

    assert calls == [0, 1, 2, 3, 4]
    assert study.trials[0].value == pytest.approx(0.7)
    assert study.trials[0].user_attrs['fold_std'] == pytest.approx(0.1414213562)
    assert study.trials[1].user_attrs['reused_folds'] == [0, 1, 2, 3, 4]


def test_failed_trial_retry_only_reruns_failed_and_missing_folds(
    search_config: OptunaSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = build_outer_cells(search_config)[0]
    ledger_path = search_config.output_dir / 'ledger.db'
    calls: list[int] = []
    failed_once = False

    def flaky_training(overrides, **kwargs):
        nonlocal failed_once
        fold_override = next(
            value for value in overrides if value.startswith('dataset.split_params.data_seed=')
        )
        fold = int(fold_override.rsplit('=', maxsplit=1)[1])
        calls.append(fold)
        if fold == 2 and not failed_once:
            failed_once = True
            return False, 'synthetic failure', None
        return True, None, {'objective': 0.5 + fold / 10}

    monkeypatch.setattr(optuna_search, 'run_training', flaky_training)
    study = optuna.create_study(direction='maximize', study_name=cell.study_name)
    study.optimize(
        lambda trial: _objective(
            trial,
            config=search_config,
            cell=cell,
            ledger_path=ledger_path,
            gpu_queue=None,
        ),
        n_trials=1,
        catch=(optuna_search.FoldExecutionError,),
    )
    assert study.trials[0].state == TrialState.FAIL

    retry_sources = _enqueue_failed_trials(
        study,
        RunLedger(ledger_path),
        max_attempts=search_config.max_retries + 1,
    )
    assert len(retry_sources) == 1
    retry_sources = _enqueue_failed_trials(
        study,
        RunLedger(ledger_path),
        max_attempts=search_config.max_retries + 1,
    )
    assert len(retry_sources) == 1
    assert sum(trial.state == TrialState.WAITING for trial in study.trials) == 1
    study.optimize(
        lambda trial: _objective(
            trial,
            config=search_config,
            cell=cell,
            ledger_path=ledger_path,
            gpu_queue=None,
            retry_sources=retry_sources,
        ),
        n_trials=1,
        catch=(optuna_search.FoldExecutionError,),
    )

    assert calls == [0, 1, 2, 2, 3, 4]
    assert study.trials[-1].state == TrialState.COMPLETE
    assert study.trials[-1].user_attrs['retry_of'] == 0
    assert study.trials[-1].user_attrs['reused_folds'] == [0, 1]


def test_study_resume_and_fingerprint_protection(
    search_config: OptunaSearchConfig,
) -> None:
    cell = build_outer_cells(search_config)[0]
    first = _load_or_create_study(search_config, cell)
    resumed = _load_or_create_study(search_config, cell)

    assert first.study_name == resumed.study_name
    changed = replace(search_config, objective_metric='best_val/accuracy')
    with pytest.raises(ValueError, match='different search configuration'):
        _load_or_create_study(changed, cell)


def test_interrupted_trial_is_recorded_only_once(
    search_config: OptunaSearchConfig,
) -> None:
    cell = build_outer_cells(search_config)[0]
    study = optuna.create_study(direction='maximize', study_name=cell.study_name)
    trial = study.ask()
    sampled = optuna_search._sample_parameters(trial, search_config, cell.model)
    trial.set_user_attr('sampled_params', sampled)
    study.tell(trial, state=TrialState.FAIL)
    ledger = RunLedger(search_config.output_dir / 'ledger.db')

    optuna_search._record_interrupted_trials(study, search_config, cell, ledger)
    optuna_search._record_interrupted_trials(study, search_config, cell, ledger)

    attempts = ledger.all_attempts()
    assert len(attempts) == 1
    assert attempts[0]['attempt'] == 1


def test_run_search_resumes_without_repeating_completed_trials(
    search_config: OptunaSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_config.n_trials = 1
    calls = 0

    def fake_training(overrides, **kwargs):
        nonlocal calls
        calls += 1
        return True, None, {'objective': 0.75}

    monkeypatch.setattr(optuna_search, 'run_training', fake_training)
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    first = run_search(search_config, skip_warmup=True)
    second = run_search(search_config, skip_warmup=True)

    assert calls == 5
    assert len(first) == 1
    assert len(second) == 1
    assert first.iloc[0]['objective'] == pytest.approx(0.75)
    assert (search_config.output_dir / 'trials.csv').is_file()
    assert (search_config.output_dir / 'best_trials.csv').is_file()
    assert (search_config.output_dir / 'fold_attempts.csv').is_file()
    assert (search_config.output_dir / 'failures.csv').is_file()


def test_empty_best_results_preserve_csv_schema() -> None:
    trials = pd.DataFrame([{'study_name': 'failed', 'state': 'FAIL', 'objective': None}])

    best = optuna_search._best_rows(trials, 'maximize')

    assert best.empty
    assert list(best.columns) == list(trials.columns)


def test_ledger_keeps_attempt_history_and_reports_only_unresolved_failures(
    tmp_path: Path,
) -> None:
    ledger = RunLedger(tmp_path / 'ledger.db')
    common = {
        'study_name': 'study',
        'param_hash': 'params',
        'params': {'lr': 0.1},
        'fold': 2,
        'training_seed': 42,
        'elapsed_time': 1.0,
        'log_path': tmp_path / 'run.log',
        'trial_number': 0,
        'gpu': None,
    }
    ledger.record(
        **common,
        attempt=1,
        status='failed',
        metric=None,
        error='failure',
    )
    assert len(ledger.unresolved_failures()) == 1
    assert ledger.retryable('study', 'params', 1, expected_folds=[2, 3, 4]) is False
    ledger.record(
        **common,
        attempt=2,
        status='success',
        metric=0.8,
        error=None,
    )

    assert ledger.next_attempt('study', 'params', 2, 42) == 3
    assert ledger.successful_metric('study', 'params', 2, 42) == pytest.approx(0.8)
    assert ledger.unresolved_failures() == []


def test_structured_categorical_is_decoded_for_hydra(
    tmp_path: Path,
) -> None:
    config_text = CONFIG_PATH.read_text().replace(
        'model.backbone.num_layers: [2]',
        (
            'model.backbone.num_layers:\n'
            '      type: categorical\n'
            '      choices:\n'
            '        - [2, 3]\n'
            '        - [3, 4]'
        ),
    )
    config_text = config_text.replace(
        'thresholds_from: multi_dataset_grid_search.yaml',
        f'thresholds_from: {Path("configs/hparams_search/multi_dataset_grid_search.yaml").resolve()}',
    )
    path = tmp_path / 'structured.yaml'
    path.write_text(config_text)
    config = OptunaSearchConfig.from_yaml(path)
    study = optuna.create_study()
    trial = study.ask()

    sampled = optuna_search._sample_parameters(trial, config, 'gcn')

    assert sampled['model.backbone.num_layers'] in ([2, 3], [3, 4])
    assert json.loads(trial.params['model.backbone.num_layers']) in ([2, 3], [3, 4])

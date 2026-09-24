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

from ogbench.utils.hparam_search import GpuDevice, acquire_gpu, is_oom_failure
from scripts import optuna_search
from scripts.optuna_search import (
    ADJACENCY_METHOD,
    ADJACENCY_TARGET_CONNECTIVITY,
    ExecutionPolicy,
    OptunaSearchConfig,
    RunLedger,
    _cache_configs,
    _enqueue_failed_trials,
    _load_or_create_study,
    _objective,
    _run_study,
    _stable_hash,
    _validate_candidates,
    build_outer_cells,
    read_study_manifest,
    run_search,
    select_study_shards,
    study_shard,
    validate_sampled_parameters,
)

CONFIG_PATH = Path('configs/hparams_search/optuna_smoke_test.yaml')
SEP24_CONFIG_PATH = Path('configs/hparams_search/sep24_ofat_optuna.yaml')
SEP24_FACTORIAL_CONFIG_PATH = Path('configs/hparams_search/sep24_factorial_optuna.yaml')
RATIO03_CONFIG_PATH = Path('configs/hparams_search/sep24_ratio03_transfer.yaml')
MULTI_DATASET_OPTUNA_CONFIG_PATH = Path('configs/hparams_search/multi_dataset_optuna_search.yaml')


@pytest.fixture
def search_config(tmp_path: Path) -> OptunaSearchConfig:
    config = OptunaSearchConfig.from_yaml(CONFIG_PATH)
    config.output_dir = tmp_path / 'output'
    config.storage = f'sqlite:///{tmp_path / "studies.db"}'
    return config


def test_config_builds_outer_cell_with_fold_local_connectivity_target(
    search_config: OptunaSearchConfig,
) -> None:
    cells = build_outer_cells(search_config)

    assert len(cells) == 1
    cell = cells[0]
    assert cell.model == 'gcn'
    assert cell.dataset == 'motrpac'
    assert cell.values[ADJACENCY_TARGET_CONNECTIVITY] == pytest.approx(0.10)
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
    cell = build_outer_cells(search_config)[0]
    same_caches_other_model = replace(cell, model='gin', study_name='other-study')
    loaders = _cache_configs(search_config, [cell, same_caches_other_model])

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
    first = (random.random(), np.random.random(), torch.rand(1).item())  # nosec B311
    optuna_search._seed_cache_randomness(42)
    second = (random.random(), np.random.random(), torch.rand(1).item())  # nosec B311

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


def test_sep24_ofat_builds_426_single_axis_cells() -> None:
    config = OptunaSearchConfig.from_yaml(SEP24_CONFIG_PATH)

    cells = build_outer_cells(config)

    assert config.wgcna_target_connectivity == pytest.approx(0.10)
    assert len(cells) == 426
    baseline = config.ablation_baseline
    for cell in cells:
        model_baseline = {**baseline, **config.per_model_ablation_baseline.get(cell.model, {})}
        changed_axes = [key for key in config.ablations if cell.values[key] != model_baseline[key]]
        assert len(changed_axes) <= 1


def test_sep24_factorial_builds_all_2448_cells_with_seven_trials() -> None:
    config = OptunaSearchConfig.from_yaml(SEP24_FACTORIAL_CONFIG_PATH)

    cells = build_outer_cells(config)

    assert config.ablation_mode == 'full_factorial'
    assert config.wgcna_target_connectivity == pytest.approx(0.10)
    assert config.n_trials == 7
    assert config.n_startup_trials == 3
    assert len(cells) == 2448
    assert len(build_outer_cells(config, models=['gcn'], datasets=['parkinsons'])) == 48
    assert len(build_outer_cells(config, models=['mlp'], datasets=['parkinsons'])) == 24
    assert all(
        cell.values[ADJACENCY_TARGET_CONNECTIVITY] == pytest.approx(0.10)
        for cell in cells
        if cell.values[ADJACENCY_METHOD] == 'wgcna'
    )


def test_multi_dataset_optuna_defaults_to_fold_local_wgcna_connectivity() -> None:
    config = OptunaSearchConfig.from_yaml(MULTI_DATASET_OPTUNA_CONFIG_PATH)

    assert config.wgcna_target_connectivity == pytest.approx(0.10)
    wgcna_cells = [
        cell
        for cell in build_outer_cells(config, models=['gcn'], datasets=['motrpac'])
        if cell.values[ADJACENCY_METHOD] == 'wgcna'
    ]
    assert wgcna_cells
    assert all(
        cell.values[ADJACENCY_TARGET_CONNECTIVITY] == pytest.approx(0.10) for cell in wgcna_cells
    )


def test_sep24_mlp_uses_valid_model_specific_baseline() -> None:
    config = OptunaSearchConfig.from_yaml(SEP24_CONFIG_PATH)

    cells = build_outer_cells(config, models=['mlp'], datasets=['parkinsons'])

    assert len(cells) == 7
    assert {cell.values['experiment'] for cell in cells} == {'no_readout'}


def test_virtual_shards_are_deterministic_disjoint_and_exhaustive() -> None:
    config = OptunaSearchConfig.from_yaml(SEP24_CONFIG_PATH)
    cells = build_outer_cells(config)

    shards = [select_study_shards(cells, 7, [index]) for index in range(7)]
    names = [{cell.study_name for cell in shard} for shard in shards]

    assert sum(map(len, names)) == 426
    assert set().union(*names) == {cell.study_name for cell in cells}
    assert all(names[left].isdisjoint(names[right]) for left in range(7) for right in range(left))
    assert all(
        study_shard(cell.study_name, 7) == index
        for index, shard in enumerate(shards)
        for cell in shard
    )


def test_virtual_shard_validation(search_config: OptunaSearchConfig) -> None:
    cells = build_outer_cells(search_config)

    with pytest.raises(ValueError, match='positive'):
        select_study_shards(cells, 0, [0])
    with pytest.raises(ValueError, match='duplicates'):
        select_study_shards(cells, 2, [0, 0])
    with pytest.raises(ValueError, match='between'):
        select_study_shards(cells, 2, [2])


def test_read_study_manifest_ignores_comments_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / 'studies.txt'
    manifest.write_text('# handoff\nstudy-a\n\n study-b \n')

    assert read_study_manifest(manifest) == ['study-a', 'study-b']

    manifest.write_text('study-a\nstudy-a\n')
    with pytest.raises(ValueError, match='duplicate'):
        read_study_manifest(manifest)


def test_filtered_run_exports_only_selected_study_attempts(
    search_config: OptunaSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_config.n_trials = 1
    search_config.ablations['dataset.loader.parameters.method'] = ['variance', 'random']
    selected, excluded = build_outer_cells(search_config)
    ledger = RunLedger(search_config.output_dir / 'run_ledger.sqlite3')
    ledger.record(
        study_name=excluded.study_name,
        param_hash='excluded',
        params={'lr': 0.1},
        fold=0,
        training_seed=search_config.training_seed,
        attempt=1,
        status='failed',
        metric=None,
        elapsed_time=1.0,
        error='excluded failure',
        log_path=search_config.output_dir / 'excluded.log',
        trial_number=0,
        gpu=None,
    )

    monkeypatch.setattr(
        optuna_search,
        'run_training',
        lambda overrides, **kwargs: (True, None, {'objective': 0.75}),
    )
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)

    run_search(search_config, studies=[selected.study_name], skip_warmup=True)

    attempts = pd.read_csv(search_config.output_dir / 'fold_attempts.csv')
    failures = pd.read_csv(search_config.output_dir / 'failures.csv')
    assert set(attempts['study_name']) == {selected.study_name}
    assert failures.empty


def test_runtime_path_overrides_are_portable(
    search_config: OptunaSearchConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    optuna_search._apply_runtime_overrides(
        search_config,
        output_dir='server-output',
        storage='sqlite:///server-output/server.db',
        root_dir='server-data',
    )

    assert search_config.output_dir == tmp_path / 'server-output'
    assert search_config.storage == f'sqlite:///{tmp_path / "server-output/server.db"}'
    assert search_config.fixed['paths.root_dir'] == str(tmp_path / 'server-data')


def test_data_root_does_not_change_study_fingerprint(
    search_config: OptunaSearchConfig,
) -> None:
    original = search_config.fingerprint

    search_config.fixed['paths.root_dir'] = '/different/local/scratch/root'

    assert search_config.fingerprint == original


def test_warmup_only_does_not_open_studies_or_train(
    search_config: OptunaSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmed: list[int] = []
    monkeypatch.setattr(
        optuna_search,
        'warmup_caches',
        lambda _config, cells, n_jobs: warmed.append(len(cells)),
    )
    monkeypatch.setattr(
        optuna_search,
        '_storage',
        lambda _config: pytest.fail('storage should not be opened'),
    )
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)

    result = run_search(search_config, warmup_only=True)

    assert result.empty
    assert warmed == [1]


def _fold_of(overrides) -> int:
    value = next(item for item in overrides if item.startswith('dataset.split_params.data_seed='))
    return int(value.rsplit('=', maxsplit=1)[1])


def _ratio03_config(tmp_path: Path) -> OptunaSearchConfig:
    config = OptunaSearchConfig.from_yaml(RATIO03_CONFIG_PATH)
    config.output_dir = tmp_path / 'ratio03'
    config.storage = f'sqlite:///{tmp_path / "ratio03.db"}'
    return config


RATIO03_CANDIDATES = {
    'gcn': {
        'model.feature_encoder.out_channels': 64,
        'model.backbone.num_layers': 3,
    },
    'gatv4': {
        'model.backbone.hidden_channels': [32, 64],
        'model.backbone.heads': [8, 8],
        'model.backbone.use_layer_norm': False,
    },
    'mlp': {
        'model.backbone.hidden_channels': [16, 32, 8],
        'model.backbone.norm': None,
    },
    'gps': {
        'model.feature_encoder.out_channels': 16,
        'model.backbone.num_layers': 4,
        'model.encodings': ['RWSE'],
    },
}


def _candidate(config: OptunaSearchConfig, model: str) -> dict:
    sampled = {
        'optimizer.parameters.lr': 0.00225757786780932,
        'optimizer.parameters.weight_decay': 0.0001,
        'model.backbone.dropout': 0.30000000000000004,
        **RATIO03_CANDIDATES[model],
    }
    canonical = validate_sampled_parameters(config, model, sampled)
    return {
        'sampled_params': canonical,
        'param_hash': _stable_hash(canonical),
        'source_study': f'source-{model}',
        'source_rule': 'best_complete_r0.5',
        'source_trial_numbers': [4],
        'source_mean': 0.61,
    }


def test_ratio03_config_builds_816_transfer_cells() -> None:
    config = OptunaSearchConfig.from_yaml(RATIO03_CONFIG_PATH)

    cells = build_outer_cells(config)

    assert len(cells) == 816
    assert {cell.values['dataset.loader.parameters.node_sample_ratio'] for cell in cells} == {0.3}
    assert config.fixed['logger.wandb.project'] == 'ogbench_sep24_ratio03_transfer'
    assert all(cell.study_name.startswith('sep24tc10ratio03transfer_') for cell in cells)


@pytest.mark.parametrize('model', sorted(RATIO03_CANDIDATES))
def test_candidate_runs_fixed_parameters_once_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    config = _ratio03_config(tmp_path)
    cell = next(cell for cell in build_outer_cells(config) if cell.model == model)
    candidate = _candidate(config, model)
    calls: list[int] = []

    def fake_training(overrides, **kwargs):
        calls.append(_fold_of(overrides))
        return True, None, {'objective': 0.5}

    monkeypatch.setattr(optuna_search, 'run_training', fake_training)
    ledger_path = config.output_dir / 'run_ledger.sqlite3'

    first = _run_study(config, cell, ledger_path, None, True, candidate)
    second = _run_study(config, cell, ledger_path, None, True, candidate)

    assert calls == [0, 1, 2, 3, 4]
    assert len(first) == len(second) == 1
    assert second[0]['state'] == 'COMPLETE'
    assert json.loads(second[0]['sampled_params']) == candidate['sampled_params']
    assert json.loads(second[0]['transfer_source'])['source_study'] == f'source-{model}'


def test_failed_candidate_is_retried_only_with_retry_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _ratio03_config(tmp_path)
    cell = next(cell for cell in build_outer_cells(config) if cell.model == 'gcn')
    candidate = _candidate(config, 'gcn')
    calls: list[int] = []

    def flaky_training(overrides, **kwargs):
        fold = _fold_of(overrides)
        calls.append(fold)
        if fold == 1 and calls.count(1) == 1:
            return False, 'Timeout after 3600s', None
        return True, None, {'objective': 0.5}

    monkeypatch.setattr(optuna_search, 'run_training', flaky_training)
    ledger_path = config.output_dir / 'run_ledger.sqlite3'

    _run_study(config, cell, ledger_path, None, False, candidate)
    _run_study(config, cell, ledger_path, None, False, candidate)
    rows = _run_study(config, cell, ledger_path, None, True, candidate)

    assert calls == [0, 1, 1, 2, 3, 4]
    assert [row['state'] for row in rows] == ['FAIL', 'COMPLETE']


def test_candidate_validation_rejects_bad_parameters(tmp_path: Path) -> None:
    config = _ratio03_config(tmp_path)
    cell = next(cell for cell in build_outer_cells(config) if cell.model == 'gcn')
    candidate = _candidate(config, 'gcn')

    with pytest.raises(ValueError, match='no candidate'):
        _validate_candidates(config, [cell], {})
    with pytest.raises(ValueError, match='hash mismatch'):
        _validate_candidates(config, [cell], {cell.study_name: {**candidate, 'param_hash': 'x'}})
    outside = {
        **candidate,
        'sampled_params': {**candidate['sampled_params'], 'model.backbone.num_layers': 5},
    }
    with pytest.raises(ValueError, match='num_layers'):
        _validate_candidates(config, [cell], {cell.study_name: outside})


def test_validate_sampled_parameters_canonicalizes_numeric_choices(tmp_path: Path) -> None:
    config = _ratio03_config(tmp_path)
    sampled = {**_candidate(config, 'gcn')['sampled_params']}
    sampled['optimizer.parameters.weight_decay'] = 0

    canonical = validate_sampled_parameters(config, 'gcn', sampled)

    assert isinstance(canonical['optimizer.parameters.weight_decay'], float)
    with pytest.raises(ValueError, match='step grid'):
        validate_sampled_parameters(config, 'gcn', {**sampled, 'model.backbone.dropout': 0.25})


def test_manifest_order_sets_execution_priority(
    search_config: OptunaSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_config.n_trials = 1
    search_config.ablations['dataset.loader.parameters.method'] = ['variance', 'random']
    first, second = build_outer_cells(search_config)
    order: list[str] = []

    def fake_training(overrides, **kwargs):
        group = next(item for item in overrides if item.startswith('logger.wandb.group='))
        order.append(group.split('=', maxsplit=1)[1])
        return True, None, {'objective': 0.75}

    monkeypatch.setattr(optuna_search, 'run_training', fake_training)
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)

    run_search(search_config, studies=[second.study_name, first.study_name], skip_warmup=True)

    assert order == [second.study_name] * 5 + [first.study_name] * 5


def test_acquire_gpu_skips_devices_without_enough_free_memory() -> None:
    import queue

    slots: queue.Queue = queue.Queue()
    busy, free = GpuDevice(0, '0'), GpuDevice(1, '1')
    slots.put(busy)
    slots.put(free)
    memory = {'0': 10_000, '1': 50_000}

    device = acquire_gpu(slots, 30_000, free_memory=memory.get, sleep=lambda _: None)

    assert device == free
    assert slots.get_nowait() == busy


def test_acquire_gpu_relaxes_threshold_after_waiting() -> None:
    import queue

    slots: queue.Queue = queue.Queue()
    device = GpuDevice(0, '0')
    slots.put(device)
    now = iter([0.0, 0.0, 10.0])

    acquired = acquire_gpu(
        slots,
        60_000,
        fallback_min_free_mib=30_000,
        fallback_after_seconds=5.0,
        free_memory=lambda _: 40_000,
        clock=lambda: next(now),
        sleep=lambda _: None,
    )

    assert acquired == device


def test_oom_classification(tmp_path: Path) -> None:
    log = tmp_path / 'fold.log'
    log.write_text('STDERR:\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n')

    assert is_oom_failure('Return code 1\nSTDERR: ...', log)
    assert is_oom_failure('Return code -9\nSTDERR: ')
    assert not is_oom_failure('Timeout after 3600s', log)
    assert not is_oom_failure('Return code 1\nSTDERR: KeyError', tmp_path / 'missing.log')


def test_oom_failures_retry_inline_without_using_retry_budget(
    search_config: OptunaSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = build_outer_cells(search_config)[0]
    ledger_path = search_config.output_dir / 'ledger.db'
    calls: list[int] = []

    def oom_training(overrides, **kwargs):
        fold = _fold_of(overrides)
        calls.append(fold)
        if fold == 1 and calls.count(1) <= 3:
            return False, 'Return code 1\nSTDERR: torch.OutOfMemoryError: CUDA out of memory', None
        return True, None, {'objective': 0.5}

    monkeypatch.setattr(optuna_search, 'run_training', oom_training)
    monkeypatch.setattr(optuna_search.time, 'sleep', lambda _: None)
    policy = ExecutionPolicy(oom_retries=3, oom_backoff_seconds=0)
    study = optuna.create_study(direction='maximize', study_name=cell.study_name)
    study.optimize(
        lambda trial: _objective(
            trial,
            config=search_config,
            cell=cell,
            ledger_path=ledger_path,
            gpu_queue=None,
            policy=policy,
        ),
        n_trials=1,
    )

    attempts = [row for row in RunLedger(ledger_path).all_attempts() if row['fold'] == 1]
    assert study.trials[0].state == TrialState.COMPLETE
    assert calls == [0, 1, 1, 1, 1, 2, 3, 4]
    assert [row['error'][:5] for row in attempts[:3]] == ['OOM: '] * 3
    assert attempts[-1]['status'] == 'success'


def test_oom_failure_without_policy_fails_the_trial(
    search_config: OptunaSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = build_outer_cells(search_config)[0]
    monkeypatch.setattr(
        optuna_search,
        'run_training',
        lambda overrides, **kwargs: (False, 'Return code 1\nSTDERR: CUDA out of memory', None),
    )
    study = optuna.create_study(direction='maximize', study_name=cell.study_name)
    study.optimize(
        lambda trial: _objective(
            trial,
            config=search_config,
            cell=cell,
            ledger_path=search_config.output_dir / 'ledger.db',
            gpu_queue=None,
        ),
        n_trials=1,
        catch=(optuna_search.FoldExecutionError,),
    )

    assert study.trials[0].state == TrialState.FAIL
    assert len(RunLedger(search_config.output_dir / 'ledger.db').all_attempts()) == 1


def test_retry_budget_excludes_free_oom_failures(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / 'ledger.db')
    for attempt in (1, 2):
        ledger.record(
            study_name='study',
            param_hash='params',
            params={'lr': 0.1},
            fold=0,
            training_seed=42,
            attempt=attempt,
            status='failed',
            metric=None,
            elapsed_time=1.0,
            error='OOM: Return code 1',
            log_path=tmp_path / 'run.log',
            trial_number=0,
            gpu=None,
        )

    assert ledger.attempt_budget_used('study', 'params', 0, 42, oom_retries=2) == 0
    assert ledger.attempt_budget_used('study', 'params', 0, 42, oom_retries=1) == 1
    assert ledger.retryable('study', 'params', 1, expected_folds=[0], oom_retries=2)
    assert not ledger.retryable('study', 'params', 1, expected_folds=[0], oom_retries=1)
    assert not ledger.retryable('study', 'params', 2, expected_folds=[0])


def test_export_only_reads_existing_studies_without_training(
    search_config: OptunaSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_config.n_trials = 1
    monkeypatch.setattr(
        optuna_search,
        'run_training',
        lambda overrides, **kwargs: (True, None, {'objective': 0.75}),
    )
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    run_search(search_config, skip_warmup=True)
    (search_config.output_dir / 'trials.csv').unlink()
    monkeypatch.setattr(
        optuna_search,
        'run_training',
        lambda overrides, **kwargs: pytest.fail('export-only must not train'),
    )

    trials = run_search(search_config, export_only=True)

    assert len(trials) == 1
    assert len(pd.read_csv(search_config.output_dir / 'trials.csv')) == 1


def test_export_only_requires_existing_storage(search_config: OptunaSearchConfig) -> None:
    with pytest.raises(FileNotFoundError):
        run_search(search_config, export_only=True)

"""Tests for live Optuna campaign status reporting."""

from __future__ import annotations

from pathlib import Path

from scripts.optuna_search import GpuDevice, OptunaSearchConfig, RunLedger, build_outer_cells
from scripts.optuna_status import collect_status


def test_collect_status_reports_durable_progress_without_study_database(
    tmp_path: Path,
) -> None:
    config = OptunaSearchConfig.from_yaml(
        'configs/hparams_search/optuna_smoke_test.yaml'
    )
    config.output_dir = tmp_path / 'output'
    config.storage = f'sqlite:///{config.output_dir / "studies.db"}'
    cell = build_outer_cells(config)[0]
    ledger = RunLedger(config.output_dir / 'run_ledger.sqlite3')
    common = {
        'study_name': cell.study_name,
        'param_hash': 'params',
        'params': {'lr': 0.1},
        'training_seed': config.training_seed,
        'elapsed_time': 10.0,
        'error': None,
        'log_path': config.output_dir / 'run.log',
        'trial_number': 0,
        'gpu': GpuDevice(logical_id=0, visibility_token='0'),
    }
    ledger.record(**common, fold=0, attempt=1, status='success', metric=0.8)
    ledger.record(**common, fold=1, attempt=1, status='success', metric=0.7)

    summary, attempts, failures, trials = collect_status(
        config,
        num_shards=1,
        shard_indices=[0],
        window_hours=6,
    )

    assert summary['expected_studies'] == 1
    assert summary['expected_trials'] == 2
    assert summary['expected_folds'] == 10
    assert summary['completed_folds'] == 2
    assert summary['progress_percent'] == 20
    assert summary['unresolved_failures'] == 0
    assert summary['folds_per_hour'] > 0
    assert len(attempts) == 2
    assert failures.empty
    assert trials.empty

#!/usr/bin/env python3
"""Report and export live progress for a sharded Optuna campaign."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from scripts.optuna_search import (
    OptunaSearchConfig,
    _apply_runtime_overrides,
    _atomic_write_csv,
    _best_rows,
    _trial_rows,
    build_outer_cells,
    read_study_manifest,
    select_study_shards,
    study_shard,
)


def _sqlite_path(storage: str) -> Path | None:
    prefix = 'sqlite:///'
    if not storage.startswith(prefix):
        return None
    return Path(storage.removeprefix(prefix))


def _read_attempts(output_dir: Path) -> pd.DataFrame:
    ledger_path = output_dir / 'run_ledger.sqlite3'
    if not ledger_path.is_file():
        return pd.DataFrame()
    uri = f'file:{ledger_path}?mode=ro'
    with sqlite3.connect(uri, uri=True, timeout=60) as connection:
        return pd.read_sql_query(
            """
            SELECT * FROM fold_attempts
            ORDER BY study_name, param_hash, fold, training_seed, attempt
            """,
            connection,
        )


def _unresolved_failures(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return attempts.copy()
    keys = ['study_name', 'param_hash', 'fold', 'training_seed']
    successful = {
        tuple(row)
        for row in attempts.loc[attempts['status'] == 'success', keys].itertuples(
            index=False, name=None
        )
    }
    latest = attempts.sort_values('attempt').groupby(keys, as_index=False).tail(1)
    mask = [
        row.status == 'failed'
        and (row.study_name, row.param_hash, row.fold, row.training_seed) not in successful
        for row in latest.itertuples()
    ]
    return latest.loc[mask].reset_index(drop=True)


def _successful_folds(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return attempts.copy()
    keys = ['study_name', 'param_hash', 'fold', 'training_seed']
    return (
        attempts.loc[attempts['status'] == 'success']
        .sort_values('attempt')
        .drop_duplicates(keys, keep='last')
    )


def _load_trials(
    storage: str,
    expected_cells: dict[str, Any],
) -> pd.DataFrame:
    path = _sqlite_path(storage)
    if path is not None and not path.is_file():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage)
    except (OSError, RuntimeError, sqlite3.Error):
        return pd.DataFrame()
    for summary in summaries:
        cell = expected_cells.get(summary.study_name)
        if cell is None:
            continue
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        rows.extend(_trial_rows(study, cell))
    return pd.DataFrame(rows)


def _throughput(
    successful: pd.DataFrame,
    attempts: pd.DataFrame,
    window_hours: float,
) -> tuple[float, float]:
    if successful.empty:
        return 0.0, window_hours
    now = datetime.now(UTC)
    timestamps = pd.to_datetime(successful['created_at'], utc=True)
    all_timestamps = pd.to_datetime(attempts['created_at'], utc=True)
    earliest_start = all_timestamps.min().to_pydatetime() - timedelta(
        seconds=float(attempts['elapsed_time'].max())
    )
    window_start = max(now - timedelta(hours=window_hours), earliest_start)
    recent = int((timestamps >= window_start).sum())
    observed_hours = max((now - window_start).total_seconds() / 3600, 1 / 60)
    return recent / observed_hours, observed_hours


def collect_status(
    config: OptunaSearchConfig,
    *,
    num_shards: int,
    shard_indices: list[int] | None,
    window_hours: float,
    study_names: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect campaign progress without changing trial or fold state."""
    cells = select_study_shards(build_outer_cells(config), num_shards, shard_indices)
    if study_names:
        requested = set(study_names)
        cells = [cell for cell in cells if cell.study_name in requested]
        missing = requested - {cell.study_name for cell in cells}
        if missing:
            raise ValueError(f'Unknown study filters: {sorted(missing)}')
    expected = {cell.study_name: cell for cell in cells}
    attempts = _read_attempts(config.output_dir)
    if not attempts.empty:
        attempts = attempts.loc[attempts['study_name'].isin(expected)].copy()
    successful = _successful_folds(attempts)
    failures = _unresolved_failures(attempts)
    trials = _load_trials(config.storage, expected)

    expected_trials = len(cells) * config.n_trials
    expected_folds = expected_trials * len(config.folds)
    studies_by_shard = {
        index: sum(study_shard(cell.study_name, num_shards) == index for cell in cells)
        for index in (shard_indices or range(num_shards))
    }
    expected_folds_by_shard = {
        index: count * config.n_trials * len(config.folds)
        for index, count in studies_by_shard.items()
    }
    completed_folds = len(successful)
    folds_per_hour, observed_hours = _throughput(successful, attempts, window_hours)
    remaining = max(0, expected_folds - completed_folds)
    eta_hours = None if folds_per_hour <= 0 else remaining / folds_per_hour
    state_counts = (
        trials['state'].value_counts().sort_index().to_dict() if not trials.empty else {}
    )
    summary = {
        'timestamp_utc': datetime.now(UTC).isoformat(timespec='seconds'),
        'output_dir': str(config.output_dir),
        'storage': config.storage,
        'num_shards': num_shards,
        'shard_indices': list(shard_indices or range(num_shards)),
        'expected_studies': len(cells),
        'expected_trials': expected_trials,
        'expected_folds': expected_folds,
        'studies_by_shard': studies_by_shard,
        'expected_folds_by_shard': expected_folds_by_shard,
        'completed_folds': completed_folds,
        'progress_percent': 100 * completed_folds / expected_folds,
        'unresolved_failures': len(failures),
        'folds_per_hour': folds_per_hour,
        'throughput_window_hours': observed_hours,
        'eta_hours': eta_hours,
        'trial_states': state_counts,
    }
    return summary, attempts, failures, trials


def _print_status(
    summary: dict[str, Any],
    attempts: pd.DataFrame,
    *,
    num_shards: int,
) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))
    if attempts.empty:
        return
    successful = _successful_folds(attempts)
    completed_per_shard: dict[int, int] = {}
    for index in summary['shard_indices']:
        completed_per_shard[index] = 0
    for study_name in attempts['study_name'].unique():
        index = study_shard(study_name, num_shards)
        if index in completed_per_shard:
            completed_per_shard[index] += int(
                (successful['study_name'] == study_name).sum()
            )
    for index in completed_per_shard:
        print(
            f'shard={index} completed_folds={completed_per_shard[index]} '
            f'expected_folds={summary["expected_folds_by_shard"][index]}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--storage')
    parser.add_argument('--root-dir')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-indices', nargs='+', type=int)
    parser.add_argument(
        '--studies-file',
        help='Only report study names in this newline-delimited manifest',
    )
    parser.add_argument('--window-hours', type=float, default=6.0)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    if args.window_hours <= 0:
        parser.error('--window-hours must be positive')

    config = OptunaSearchConfig.from_yaml(args.config)
    storage = args.storage or f'sqlite:///{Path(args.output_dir).resolve() / "studies.db"}'
    _apply_runtime_overrides(
        config,
        output_dir=args.output_dir,
        storage=storage,
        root_dir=args.root_dir,
    )
    summary, attempts, failures, trials = collect_status(
        config,
        num_shards=args.num_shards,
        shard_indices=args.shard_indices,
        window_hours=args.window_hours,
        study_names=read_study_manifest(args.studies_file) if args.studies_file else None,
    )
    _print_status(summary, attempts, num_shards=args.num_shards)

    if args.export:
        _atomic_write_csv(pd.DataFrame([summary]), config.output_dir / 'live_status.csv')
        _atomic_write_csv(attempts, config.output_dir / 'live_fold_attempts.csv')
        _atomic_write_csv(failures, config.output_dir / 'live_failures.csv')
        _atomic_write_csv(trials, config.output_dir / 'live_trials.csv')
        best = _best_rows(trials, config.direction) if not trials.empty else trials.copy()
        _atomic_write_csv(best, config.output_dir / 'live_best_trials.csv')
        print(f'Live exports written to {config.output_dir}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Snapshot and partition incomplete Optuna studies for server-local continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.trial import TrialState

from scripts.optuna_search import (
    OptunaSearchConfig,
    _apply_runtime_overrides,
    _stable_hash,
    build_outer_cells,
    select_study_shards,
)


@dataclass(frozen=True)
class RemainingStudy:
    """Estimated continuation work for one study."""

    study_name: str
    model: str
    dataset: str
    completed_parameter_sets: int
    remaining_folds: int
    estimated_seconds: float


def _sqlite_path(url: str) -> Path:
    prefix = 'sqlite:///'
    if not url.startswith(prefix):
        raise ValueError('Rebalance snapshots currently require SQLite storage')
    return Path(url.removeprefix(prefix)).resolve()


def backup_sqlite(source: Path, destination: Path) -> None:
    """Create a transactionally consistent SQLite backup."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with (
        sqlite3.connect(f'file:{source}?mode=ro', uri=True, timeout=60) as source_db,
        sqlite3.connect(destination, timeout=60) as destination_db,
    ):
        source_db.backup(destination_db)


def _read_attempts(path: Path) -> pd.DataFrame:
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=60) as connection:
        return pd.read_sql_query('SELECT * FROM fold_attempts', connection)


def _successful_attempts(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return attempts.copy()
    keys = ['study_name', 'param_hash', 'fold', 'training_seed']
    return (
        attempts.loc[attempts['status'] == 'success']
        .sort_values('attempt')
        .drop_duplicates(keys, keep='last')
    )


def _trial_param_hash(trial: optuna.trial.FrozenTrial) -> str | None:
    value = trial.user_attrs.get('param_hash')
    if isinstance(value, str):
        return value
    sampled = trial.user_attrs.get('sampled_params')
    return _stable_hash(sampled) if isinstance(sampled, dict) else None


def remaining_studies(
    config: OptunaSearchConfig,
    *,
    storage: str,
    ledger_path: Path,
    num_shards: int,
    shard_indices: list[int],
) -> tuple[list[RemainingStudy], int]:
    """Estimate unfinished fold work from a frozen Optuna and ledger snapshot."""
    cells = select_study_shards(
        build_outer_cells(config),
        num_shards,
        shard_indices,
    )
    attempts = _read_attempts(ledger_path)
    successful = _successful_attempts(attempts)
    elapsed = successful.loc[successful['elapsed_time'] > 0].copy()
    global_median = float(elapsed['elapsed_time'].median()) if not elapsed.empty else 60.0
    cell_by_name = {cell.study_name: cell for cell in cells}
    if not elapsed.empty:
        elapsed['model'] = elapsed['study_name'].map(
            lambda name: cell_by_name[name].model if name in cell_by_name else None
        )
        elapsed['dataset'] = elapsed['study_name'].map(
            lambda name: cell_by_name[name].dataset if name in cell_by_name else None
        )
    medians = (
        elapsed.dropna(subset=['model', 'dataset'])
        .groupby(['model', 'dataset'])['elapsed_time']
        .median()
        .to_dict()
        if not elapsed.empty
        else {}
    )

    summaries = {
        summary.study_name: summary
        for summary in optuna.study.get_all_study_summaries(storage=storage)
    }
    remaining: list[RemainingStudy] = []
    completed_studies = 0
    for cell in cells:
        summary = summaries.get(cell.study_name)
        trials = (
            optuna.load_study(study_name=cell.study_name, storage=storage).trials
            if summary is not None
            else []
        )
        complete_hashes = {
            param_hash
            for trial in trials
            if trial.state == TrialState.COMPLETE
            if (param_hash := _trial_param_hash(trial)) is not None
        }
        if len(complete_hashes) >= config.n_trials:
            completed_studies += 1
            continue

        candidate_hashes = {
            param_hash
            for trial in trials
            if (param_hash := _trial_param_hash(trial)) is not None
        } - complete_hashes
        missing_parameter_sets = max(
            0,
            config.n_trials - len(complete_hashes) - len(candidate_hashes),
        )
        study_successes = successful.loc[
            successful['study_name'] == cell.study_name
        ]
        successful_folds = (
            study_successes.groupby('param_hash')['fold'].nunique().to_dict()
            if not study_successes.empty
            else {}
        )
        remaining_folds = missing_parameter_sets * len(config.folds)
        remaining_folds += sum(
            max(0, len(config.folds) - int(successful_folds.get(param_hash, 0)))
            for param_hash in candidate_hashes
        )
        # A retry may only need to aggregate already durable folds. Keep a
        # small nonzero estimate so it is assigned to exactly one server.
        fold_equivalent = max(0.1, float(remaining_folds))
        median_seconds = float(medians.get((cell.model, cell.dataset), global_median))
        remaining.append(
            RemainingStudy(
                study_name=cell.study_name,
                model=cell.model,
                dataset=cell.dataset,
                completed_parameter_sets=len(complete_hashes),
                remaining_folds=remaining_folds,
                estimated_seconds=fold_equivalent * median_seconds,
            )
        )
    return remaining, completed_studies


def assign_weighted(
    studies: list[RemainingStudy],
    weights: dict[str, float],
) -> dict[str, list[RemainingStudy]]:
    """Greedily balance estimated work proportional to server capacity."""
    if not weights or any(weight <= 0 for weight in weights.values()):
        raise ValueError('Every server weight must be positive')
    assignments: dict[str, list[RemainingStudy]] = {server: [] for server in weights}
    loads = dict.fromkeys(weights, 0.0)
    for study in sorted(studies, key=lambda item: item.estimated_seconds, reverse=True):
        server = min(weights, key=lambda name: (loads[name] / weights[name], name))
        assignments[server].append(study)
        loads[server] += study.estimated_seconds
    return assignments


def _parse_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        try:
            name, raw_weight = value.split('=', maxsplit=1)
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(f'Invalid server weight {value!r}; expected NAME=WEIGHT') from error
        if not name or name in weights or weight <= 0:
            raise ValueError(f'Invalid or duplicate server weight: {value!r}')
        weights[name] = weight
    return weights


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def create_handoff(
    config: OptunaSearchConfig,
    *,
    source_storage: str,
    source_output_dir: Path,
    handoff_dir: Path,
    num_shards: int,
    shard_indices: list[int],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Back up campaign state and write disjoint continuation manifests."""
    if handoff_dir.exists():
        raise FileExistsError(handoff_dir)
    handoff_dir.mkdir(parents=True)
    studies_snapshot = handoff_dir / 'studies.db'
    ledger_snapshot = handoff_dir / 'run_ledger.sqlite3'
    backup_sqlite(_sqlite_path(source_storage), studies_snapshot)
    backup_sqlite(source_output_dir / 'run_ledger.sqlite3', ledger_snapshot)
    config_copy = handoff_dir / 'sep24_factorial_optuna.yaml'
    shutil.copy2(config.source_path, config_copy)

    snapshot_storage = f'sqlite:///{studies_snapshot}'
    studies, completed_studies = remaining_studies(
        config,
        storage=snapshot_storage,
        ledger_path=ledger_snapshot,
        num_shards=num_shards,
        shard_indices=shard_indices,
    )
    assignments = assign_weighted(studies, weights)
    summary: dict[str, Any] = {
        'created_at_utc': datetime.now(UTC).isoformat(timespec='seconds'),
        'config_fingerprint': config.fingerprint,
        'num_shards': num_shards,
        'source_shard_indices': shard_indices,
        'completed_studies_excluded': completed_studies,
        'incomplete_studies': len(studies),
        'remaining_folds': sum(study.remaining_folds for study in studies),
        'servers': {},
    }
    all_names: set[str] = set()
    for server, assigned in assignments.items():
        names = sorted(study.study_name for study in assigned)
        overlap = all_names.intersection(names)
        if overlap:
            raise AssertionError(f'Duplicate manifest ownership: {sorted(overlap)}')
        all_names.update(names)
        manifest = handoff_dir / f'{server}.txt'
        manifest.write_text(''.join(f'{name}\n' for name in names))
        summary['servers'][server] = {
            'weight': weights[server],
            'studies': len(assigned),
            'remaining_folds': sum(study.remaining_folds for study in assigned),
            'estimated_hours': sum(study.estimated_seconds for study in assigned) / 3600,
            'manifest': manifest.name,
        }
    if all_names != {study.study_name for study in studies}:
        raise AssertionError('Manifest union does not match incomplete studies')

    summary_path = handoff_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    checksummed = [
        studies_snapshot,
        ledger_snapshot,
        config_copy,
        summary_path,
        *(handoff_dir / f'{server}.txt' for server in assignments),
    ]
    (handoff_dir / 'SHA256SUMS').write_text(
        ''.join(f'{_sha256(path)}  {path.name}\n' for path in checksummed)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--storage', required=True)
    parser.add_argument('--handoff-dir', required=True)
    parser.add_argument('--root-dir')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-indices', nargs='+', type=int, required=True)
    parser.add_argument(
        '--server-weight',
        action='append',
        default=[],
        metavar='NAME=WEIGHT',
        help='Repeat for each destination (default: parka=2, frank=1, hall=1)',
    )
    args = parser.parse_args()

    config = OptunaSearchConfig.from_yaml(args.config)
    _apply_runtime_overrides(
        config,
        output_dir=args.output_dir,
        storage=args.storage,
        root_dir=args.root_dir,
    )
    weights = _parse_weights(args.server_weight or ['parka=2', 'frank=1', 'hall=1'])
    summary = create_handoff(
        config,
        source_storage=config.storage,
        source_output_dir=config.output_dir,
        handoff_dir=Path(args.handoff_dir).resolve(),
        num_shards=args.num_shards,
        shard_indices=args.shard_indices,
        weights=weights,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

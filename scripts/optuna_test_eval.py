#!/usr/bin/env python3
"""Evaluate the best Optuna trial of each study on the held-out test fold.

Studies are shard-local: each structural cell belongs to exactly one server.
Run this on the machine that produced the search CSV. Do not pick a global
model×dataset winner first — test every complete study's best trial so later
ablation plots stay comparable.

Default mode reuses the existing Lightning checkpoints (Frank has all 3,555
best-trial fold ckpts) and only runs ``train=false test=true``. Retrain if a
checkpoint is missing unless ``--require-ckpt`` is set.

    PYTHONUNBUFFERED=1 python scripts/optuna_test_eval.py \
      --config configs/hparams_search/sep24_factorial_optuna.yaml \
      --best-csv /scratch/louisvl/ogbench/search_results/sep24_factorial_optuna/live_best_trials.csv \
      --search-output-dir /scratch/louisvl/ogbench/search_results/sep24_factorial_optuna \
      --root-dir /scratch/louisvl/ogbench \
      --output-dir /scratch/louisvl/ogbench/search_results/sep24_factorial_optuna_test \
      --gpus 1 2 3 4 5 6 7 \
      --jobs-per-gpu 2
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from ogbench.utils.hparam_search import (
    enforce_single_thread_process,
    populate_gpu_queue,
    run_training,
    to_override,
    visible_gpu_devices,
)
from scripts.optuna_search import (
    ADJACENCY_METHOD,
    ADJACENCY_TARGET_CONNECTIVITY,
    ADJACENCY_THRESHOLD,
    EXPERIMENT,
    NODE_SAMPLE_RATIO,
    SELECTION_METHOD,
    OptunaSearchConfig,
    OuterCell,
    _apply_runtime_overrides,
    _atomic_write_csv,
    _trial_hyperparameters,
)

TEST_METRIC_KEYS = (
    'best_test/f1_macro',
    'test/f1_macro',
    'best_test/f1_weighted',
    'test/f1_weighted',
    'best_test/accuracy',
    'test/accuracy',
    'best_test/auroc',
    'test/auroc',
    'best_val/f1_macro',
    'val/f1_macro',
)


def _parse_json_or_literal(raw: Any) -> Any:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _first_metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in metrics and metrics[name] is not None:
            try:
                return float(metrics[name])
            except (TypeError, ValueError):
                continue
    return None


def _cell_from_row(row: pd.Series) -> OuterCell:
    values: dict[str, Any] = {
        EXPERIMENT: row['experiment'],
        ADJACENCY_METHOD: row[ADJACENCY_METHOD],
        NODE_SAMPLE_RATIO: row[NODE_SAMPLE_RATIO],
        SELECTION_METHOD: row[SELECTION_METHOD],
    }
    if ADJACENCY_THRESHOLD in row.index and pd.notna(row[ADJACENCY_THRESHOLD]):
        values[ADJACENCY_THRESHOLD] = row[ADJACENCY_THRESHOLD]
    if ADJACENCY_TARGET_CONNECTIVITY in row.index and pd.notna(row[ADJACENCY_TARGET_CONNECTIVITY]):
        values[ADJACENCY_TARGET_CONNECTIVITY] = row[ADJACENCY_TARGET_CONNECTIVITY]
    return OuterCell(
        model=str(row['model']),
        dataset=str(row['dataset']),
        values=values,
        study_name=str(row['study_name']),
    )


def load_best_studies(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame['state'].astype(str).str.upper() == 'COMPLETE'].copy()
    if frame.empty:
        raise ValueError('No COMPLETE studies in --best-csv')
    frame = (
        frame.sort_values('fold_mean', ascending=False)
        .drop_duplicates('study_name', keep='first')
        .reset_index(drop=True)
    )
    return frame


def find_checkpoint(
    search_output_dir: Path, study_name: str, trial_number: int, fold: int
) -> Path | None:
    study_dir = search_output_dir / 'runs' / study_name
    if not study_dir.is_dir():
        return None
    for attempt in range(1, 8):
        run_dir = (
            study_dir / f'{study_name}_trial{int(trial_number):04d}_fold{fold}_attempt{attempt}'
        )
        ckpt_dir = run_dir / 'checkpoints'
        if not ckpt_dir.is_dir():
            continue
        ckpts = sorted(ckpt_dir.glob('*.ckpt'))
        if ckpts:
            return ckpts[-1]
    return None


class TestLedger:
    """Append-only fold-level test evaluations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS test_folds (
                    study_name TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    fold INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    val_f1 REAL,
                    test_f1 REAL,
                    test_f1_weighted REAL,
                    test_accuracy REAL,
                    test_auroc REAL,
                    used_ckpt INTEGER NOT NULL,
                    ckpt_path TEXT,
                    elapsed_time REAL NOT NULL,
                    error TEXT,
                    log_path TEXT,
                    params_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (study_name, trial_number, fold, attempt)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        return connection

    def successful(self, study_name: str, trial_number: int, fold: int) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM test_folds
                WHERE study_name=? AND trial_number=? AND fold=? AND status='success'
                LIMIT 1
                """,
                (study_name, int(trial_number), int(fold)),
            ).fetchone()
        return row is not None

    def next_attempt(self, study_name: str, trial_number: int, fold: int) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(attempt), 0) AS attempt
                FROM test_folds
                WHERE study_name=? AND trial_number=? AND fold=?
                """,
                (study_name, int(trial_number), int(fold)),
            ).fetchone()
        return int(row['attempt']) + 1

    def record(self, **row: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO test_folds (
                    study_name, trial_number, fold, attempt, status,
                    val_f1, test_f1, test_f1_weighted, test_accuracy, test_auroc,
                    used_ckpt, ckpt_path, elapsed_time, error, log_path, params_json
                ) VALUES (
                    :study_name, :trial_number, :fold, :attempt, :status,
                    :val_f1, :test_f1, :test_f1_weighted, :test_accuracy, :test_auroc,
                    :used_ckpt, :ckpt_path, :elapsed_time, :error, :log_path, :params_json
                )
                ON CONFLICT(study_name, trial_number, fold, attempt) DO UPDATE SET
                    status=excluded.status,
                    val_f1=excluded.val_f1,
                    test_f1=excluded.test_f1,
                    test_f1_weighted=excluded.test_f1_weighted,
                    test_accuracy=excluded.test_accuracy,
                    test_auroc=excluded.test_auroc,
                    used_ckpt=excluded.used_ckpt,
                    ckpt_path=excluded.ckpt_path,
                    elapsed_time=excluded.elapsed_time,
                    error=excluded.error,
                    log_path=excluded.log_path,
                    params_json=excluded.params_json
                """,
                row,
            )

    def all_rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute('SELECT * FROM test_folds')]


def _eval_overrides(
    config: OptunaSearchConfig,
    cell: OuterCell,
    sampled: dict[str, Any],
    trial_number: int,
    fold: int,
    attempt: int,
    output_dir: Path,
    ckpt_path: Path | None,
) -> list[str]:
    hyperparameters = _trial_hyperparameters(cell, sampled)
    run_name = f'{cell.study_name}_trial{int(trial_number):04d}_fold{fold}_test{attempt}'
    tags = [cell.model, cell.dataset, 'optuna', 'test_eval', f'fold{fold}', *config.tags]
    parameters = {**config.fixed, **hyperparameters}
    parameters.update(
        {
            'seed': config.training_seed,
            'dataset.split_params.split_type': 'k-fold',
            'dataset.split_params.k': config.k,
            'dataset.split_params.data_seed': fold,
            'test': True,
            'train': ckpt_path is None,
            'logger.wandb.tags': tags,
            'logger.wandb.group': f'{cell.study_name}_test',
            'logger.wandb.name': run_name,
            'hydra.run.dir': str(output_dir / 'runs' / cell.study_name / run_name),
        }
    )
    if ckpt_path is not None:
        parameters['ckpt_path'] = str(ckpt_path)
    overrides = [
        f'model={cell.model}',
        f'dataset={cell.dataset}',
    ]
    overrides.extend(to_override(key, value) for key, value in parameters.items())
    return overrides


def _evaluate_fold(
    config_payload: dict[str, Any],
    row_payload: dict[str, Any],
    fold: int,
    search_output_dir: Path,
    output_dir: Path,
    ledger_path: Path,
    gpu_queue: Any,
    require_ckpt: bool,
    timeout: int,
) -> dict[str, Any]:
    enforce_single_thread_process()
    config = OptunaSearchConfig.from_yaml(config_payload['config'])
    _apply_runtime_overrides(
        config,
        output_dir=config_payload['output_dir'],
        root_dir=config_payload['root_dir'],
    )
    ledger = TestLedger(ledger_path)
    cell = _cell_from_row(pd.Series(row_payload))
    sampled = _parse_json_or_literal(row_payload['sampled_params'])
    trial_number = int(row_payload['trial_number'])
    if ledger.successful(cell.study_name, trial_number, fold):
        return {'study_name': cell.study_name, 'fold': fold, 'status': 'skipped'}

    ckpt = find_checkpoint(search_output_dir, cell.study_name, trial_number, fold)
    if ckpt is None and require_ckpt:
        attempt = ledger.next_attempt(cell.study_name, trial_number, fold)
        ledger.record(
            study_name=cell.study_name,
            trial_number=trial_number,
            fold=fold,
            attempt=attempt,
            status='failed',
            val_f1=None,
            test_f1=None,
            test_f1_weighted=None,
            test_accuracy=None,
            test_auroc=None,
            used_ckpt=0,
            ckpt_path=None,
            elapsed_time=0.0,
            error='Missing Lightning checkpoint for this best-trial fold',
            log_path=None,
            params_json=json.dumps(sampled, sort_keys=True),
        )
        return {'study_name': cell.study_name, 'fold': fold, 'status': 'failed'}

    gpu = gpu_queue.get() if gpu_queue is not None else None
    attempt = ledger.next_attempt(cell.study_name, trial_number, fold)
    log_path = (
        output_dir
        / 'logs'
        / cell.study_name
        / f'trial{trial_number:04d}_fold{fold}_attempt{attempt}.log'
    )
    overrides = _eval_overrides(
        config, cell, sampled, trial_number, fold, attempt, output_dir, ckpt
    )
    started = time.time()
    try:
        success, error, metrics = run_training(
            overrides,
            timeout=timeout,
            gpu_id=None if gpu is None else gpu.visibility_token,
            n_threads=1,
            log_path=log_path,
        )
    finally:
        if gpu_queue is not None:
            gpu_queue.put(gpu)
    metrics = metrics or {}
    test_f1 = _first_metric(metrics, 'best_test/f1_macro', 'test/f1_macro')
    if success and test_f1 is None:
        success = False
        error = 'Training/eval succeeded but did not emit test/f1_macro or best_test/f1_macro'
    ledger.record(
        study_name=cell.study_name,
        trial_number=trial_number,
        fold=fold,
        attempt=attempt,
        status='success' if success else 'failed',
        val_f1=_first_metric(metrics, 'best_val/f1_macro', 'val/f1_macro'),
        test_f1=test_f1,
        test_f1_weighted=_first_metric(metrics, 'best_test/f1_weighted', 'test/f1_weighted'),
        test_accuracy=_first_metric(metrics, 'best_test/accuracy', 'test/accuracy'),
        test_auroc=_first_metric(metrics, 'best_test/auroc', 'test/auroc'),
        used_ckpt=1 if ckpt is not None else 0,
        ckpt_path=None if ckpt is None else str(ckpt),
        elapsed_time=time.time() - started,
        error=error,
        log_path=str(log_path),
        params_json=json.dumps(sampled, sort_keys=True),
    )
    return {
        'study_name': cell.study_name,
        'fold': fold,
        'status': 'success' if success else 'failed',
        'test_f1': test_f1,
        'used_ckpt': ckpt is not None,
    }


def _export(ledger: TestLedger, best: pd.DataFrame, output_dir: Path) -> None:
    folds = pd.DataFrame(ledger.all_rows())
    _atomic_write_csv(folds, output_dir / 'live_test_folds.csv')
    if folds.empty:
        return
    success = folds.loc[folds['status'] == 'success']
    if success.empty:
        return
    latest = success.sort_values('attempt').drop_duplicates(
        ['study_name', 'trial_number', 'fold'], keep='last'
    )
    summary = latest.groupby(['study_name', 'trial_number'], as_index=False).agg(
        n_folds=('fold', 'nunique'),
        test_f1_mean=('test_f1', 'mean'),
        test_f1_std=('test_f1', 'std'),
        val_f1_mean=('val_f1', 'mean'),
        val_f1_std=('val_f1', 'std'),
    )
    merged = best.merge(summary, on=['study_name', 'trial_number'], how='left')
    _atomic_write_csv(merged, output_dir / 'live_test_best_trials.csv')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--best-csv', action='append', required=True, type=Path)
    parser.add_argument('--search-output-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--root-dir')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--jobs-per-gpu', type=int, default=2)
    parser.add_argument('--n-jobs', type=int)
    parser.add_argument('--require-ckpt', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Rewrite live_test_*.csv from the existing ledger and exit',
    )
    parser.add_argument('--limit-studies', type=int)
    args = parser.parse_args()

    config = OptunaSearchConfig.from_yaml(args.config)
    _apply_runtime_overrides(config, output_dir=str(args.output_dir), root_dir=args.root_dir)
    best = load_best_studies(args.best_csv)
    if args.limit_studies:
        best = best.head(args.limit_studies).copy()
    folds = list(config.folds)
    jobs = [(row.to_dict(), fold) for _, row in best.iterrows() for fold in folds]
    print(f'Studies: {len(best)} | fold jobs: {len(jobs)} | output: {args.output_dir}')
    missing = 0
    for row in best.itertuples():
        for fold in folds:
            if (
                find_checkpoint(
                    args.search_output_dir, row.study_name, int(row.trial_number), fold
                )
                is None
            ):
                missing += 1
    print(f'Checkpoints found for {len(jobs) - missing}/{len(jobs)} fold jobs')

    if args.export_only:
        ledger = TestLedger(args.output_dir / 'test_ledger.sqlite3')
        _export(ledger, best, args.output_dir)
        print(f'Exported {args.output_dir / "live_test_folds.csv"}')
        print(f'Exported {args.output_dir / "live_test_best_trials.csv"}')
        return

    if args.dry_run:
        sample_row = best.iloc[0]
        cell = _cell_from_row(sample_row)
        sampled = _parse_json_or_literal(sample_row['sampled_params'])
        ckpt = find_checkpoint(
            args.search_output_dir, cell.study_name, int(sample_row['trial_number']), folds[0]
        )
        overrides = _eval_overrides(
            config,
            cell,
            sampled,
            int(sample_row['trial_number']),
            folds[0],
            1,
            args.output_dir,
            ckpt,
        )
        print('Example command:')
        print('python -m ogbench.run ' + ' '.join(overrides))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = TestLedger(args.output_dir / 'test_ledger.sqlite3')
    devices = visible_gpu_devices(args.gpus)
    workers = args.n_jobs or max(1, len(devices) * args.jobs_per_gpu)
    print(f'GPUs: {[d.logical_id for d in devices] or ["CPU"]} | workers: {workers}')
    pending = [
        (row, fold)
        for row, fold in jobs
        if not ledger.successful(row['study_name'], int(row['trial_number']), fold)
    ]
    print(f'Pending fold jobs after resume: {len(pending)}')
    config_payload = {
        'config': str(Path(args.config).resolve()),
        'output_dir': str(args.output_dir.resolve()),
        'root_dir': args.root_dir,
    }
    if workers == 1 or not pending:
        for row, fold in pending:
            print(
                _evaluate_fold(
                    config_payload,
                    row,
                    fold,
                    args.search_output_dir,
                    args.output_dir,
                    ledger.path,
                    None,
                    args.require_ckpt,
                    config.timeout,
                )
            )
    else:
        manager = multiprocessing.Manager()
        gpu_queue = manager.Queue()
        populate_gpu_queue(gpu_queue, devices, args.jobs_per_gpu)
        with parallel_backend('loky', inner_max_num_threads=1):
            Parallel(n_jobs=workers, verbose=10)(
                delayed(_evaluate_fold)(
                    config_payload,
                    row,
                    fold,
                    args.search_output_dir,
                    args.output_dir,
                    ledger.path,
                    gpu_queue,
                    args.require_ckpt,
                    config.timeout,
                )
                for row, fold in pending
            )
    _export(ledger, best, args.output_dir)
    print(f'Wrote {args.output_dir / "live_test_folds.csv"}')


if __name__ == '__main__':
    main()

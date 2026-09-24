#!/usr/bin/env python3
"""Test-evaluate best Optuna trials using a pre-built checkpoint map.

Accepts both map formats produced by this repo:

  * **Filesystem map** (``optuna_make_ckpt_map.py``): ``ckpt_path`` column
    holds the full path to a ``.ckpt`` file.  Null rows are skipped.

  * **W&B manifest** (``optuna_wandb_ckpt_map.py``): ``ckpt_dir`` column holds
    the checkpoints/ directory; ``server_root`` identifies which machine owns
    those paths.  Pass ``--server-root /scratch/…`` to filter to local rows,
    then the script scans for ``*.ckpt`` files in each ``ckpt_dir``.

The output ``live_test_best_trials.csv`` is in the same format as the one
produced by ``optuna_test_eval.py`` and can be passed directly to
``plot_sep24_test_best.py --csv``.  An extra ``search_root_server`` column
records which search root each study came from.

Basic usage (runs only folds with locally found checkpoints):

    python scripts/optuna_ckpt_test_eval.py \\
        --config configs/hparams_search/sep24_factorial_optuna.yaml \\
        --ckpt-map /scratch/.../ckpt_map_frank.csv \\
        --output-dir /scratch/.../sep24_factorial_optuna_test_rebal \\
        --root-dir /scratch/... \\
        --gpus 1 2 3 4 5 6 7 \\
        --jobs-per-gpu 2

Dry-run to preview the first command without executing anything:

    python scripts/optuna_ckpt_test_eval.py ... --dry-run

Export existing ledger to CSV without running any new jobs:

    python scripts/optuna_ckpt_test_eval.py ... --export-only
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
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
from scripts.optuna_test_eval import (
    TestLedger,
    _first_metric,
    _parse_json_or_literal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def load_ckpt_map(paths: list[Path], server_root: str | None = None) -> pd.DataFrame:
    """Load and concatenate one or more ckpt_map CSVs.

    Handles two formats transparently:

    * **Filesystem map** (``optuna_make_ckpt_map.py``): has a ``ckpt_path``
      column.  Rows with null ``ckpt_path`` are dropped.

    * **W&B manifest** (``optuna_wandb_ckpt_map.py``): has a ``ckpt_dir``
      column and a ``server_root`` column.  If ``server_root`` is provided,
      only rows whose ``ckpt_dir`` starts with that prefix are kept (the
      ``ckpt_dir`` path is always correct; ``server_root`` / ``paths.root_dir``
      in W&B may reflect the config default rather than the actual server if
      ``--root-dir`` was not passed at launch).  Then the script scans each
      ``ckpt_dir`` locally for ``*.ckpt`` files to populate ``ckpt_path``.

    Deduplicates so that each (study_name, fold) pair appears at most once.
    """
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    # ---- W&B manifest: filter by ckpt_dir prefix and resolve → ckpt_path
    if 'ckpt_dir' in df.columns and 'ckpt_path' not in df.columns:
        if server_root:
            prefix = server_root.rstrip('/')
            df = df[df['ckpt_dir'].astype(str).str.startswith(prefix)].copy()
        # Scan locally for *.ckpt in each ckpt_dir
        def _resolve(ckpt_dir: str | float) -> str | None:
            if not isinstance(ckpt_dir, str) or not ckpt_dir:
                return None
            d = Path(ckpt_dir)
            if not d.is_dir():
                return None
            ckpts = sorted(d.glob('*.ckpt'))
            return str(ckpts[-1]) if ckpts else None
        df['ckpt_path'] = df['ckpt_dir'].apply(_resolve)

    # ---- filesystem map: optionally filter by ckpt_path prefix
    elif server_root and 'ckpt_path' in df.columns:
        prefix = server_root.rstrip('/')
        df = df[df['ckpt_path'].astype(str).str.startswith(prefix)].copy()

    df = df[df['ckpt_path'].notna()].copy()
    df = df.drop_duplicates(['study_name', 'fold'], keep='first').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-fold worker (spawned by joblib)
# ---------------------------------------------------------------------------


def _evaluate_fold(
    config_payload: dict[str, Any],
    row_payload: dict[str, Any],
    fold: int,
    output_dir: Path,
    ledger_path: Path,
    gpu_queue: Any,
    timeout: int,
) -> dict[str, Any]:
    """Evaluate one fold of one study's best trial.

    Unlike ``optuna_test_eval._evaluate_fold``, the checkpoint path is taken
    directly from ``row_payload['ckpt_path']`` — no filesystem search needed.
    """
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

    ckpt_path_str = row_payload.get('ckpt_path')
    ckpt = Path(ckpt_path_str) if ckpt_path_str and Path(ckpt_path_str).is_file() else None

    if ckpt is None:
        # Checkpoint was present when the map was built but has since disappeared.
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
            error=f'Checkpoint no longer found at {ckpt_path_str!r}',
            log_path=None,
            params_json=json.dumps(sampled, sort_keys=True),
        )
        return {'study_name': cell.study_name, 'fold': fold, 'status': 'failed',
                'error': 'checkpoint_missing'}

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
        used_ckpt=1,
        ckpt_path=str(ckpt),
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
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _export(ledger: TestLedger, best: pd.DataFrame, output_dir: Path) -> None:
    """Write live_test_folds.csv and live_test_best_trials.csv.

    ``best`` is the study-level DataFrame (one row per study) derived from the
    ckpt_map.  It retains the ``search_root_server`` column so the output CSV
    records which search root each study came from.
    """
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ckpt-map',
        action='append',
        required=True,
        type=Path,
        metavar='PATH',
        help='CSV produced by optuna_make_ckpt_map.py. Repeat to concatenate maps.',
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--root-dir')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--jobs-per-gpu', type=int, default=2)
    parser.add_argument('--n-jobs', type=int)
    parser.add_argument(
        '--server-root',
        metavar='PATH',
        help=(
            'Filter the manifest to rows whose server_root matches this path. '
            'Required when using a W&B manifest (optuna_wandb_ckpt_map.py). '
            'E.g. /scratch/louisvl/ogbench on Frank, /scratch/lcornelis/ogbench on Parka.'
        ),
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Rewrite live_test_*.csv from the existing ledger and exit.',
    )
    parser.add_argument('--limit-studies', type=int)
    args = parser.parse_args()

    config = OptunaSearchConfig.from_yaml(args.config)
    _apply_runtime_overrides(config, output_dir=str(args.output_dir), root_dir=args.root_dir)

    ckpt_map = load_ckpt_map(args.ckpt_map, server_root=args.server_root)
    if ckpt_map.empty:
        raise ValueError('No rows with non-null ckpt_path found in --ckpt-map')

    # Study-level summary (one row per study) — used for the export merge.
    # Keep search_root_server from the first fold row for each study.
    # The val-f1 sort column differs between the two input formats.
    _sort_col = next(
        (c for c in ('fold_mean', 'val_f1_mean') if c in ckpt_map.columns),
        None,
    )
    _drop_extra = [c for c in ('fold', 'ckpt_path') if c in ckpt_map.columns]
    study_level = ckpt_map.copy()
    if _sort_col:
        study_level = study_level.sort_values(_sort_col, ascending=False)
    study_level = (
        study_level
        .drop_duplicates('study_name', keep='first')
        .drop(columns=_drop_extra)  # fold-level columns not needed here
        .reset_index(drop=True)
    )
    if args.limit_studies:
        study_names = study_level['study_name'].head(args.limit_studies).tolist()
        study_level = study_level[study_level['study_name'].isin(study_names)].copy()
        ckpt_map = ckpt_map[ckpt_map['study_name'].isin(study_names)].copy()

    jobs = [(row.to_dict(), int(row['fold'])) for _, row in ckpt_map.iterrows()]

    # Count per server for information
    server_counts: dict[str, int] = {}
    if 'search_root_server' in ckpt_map.columns:
        server_counts = ckpt_map['search_root_server'].value_counts().to_dict()

    print(
        f'Studies: {len(study_level)} | fold jobs: {len(jobs)} '
        f'| output: {args.output_dir}'
    )
    if server_counts:
        print('  Fold jobs by search_root_server:')
        for srv, cnt in sorted(server_counts.items()):
            print(f'    {srv}: {cnt}')

    # ---- export-only -------------------------------------------------------
    if args.export_only:
        ledger = TestLedger(args.output_dir / 'test_ledger.sqlite3')
        _export(ledger, study_level, args.output_dir)
        print(f'Exported {args.output_dir / "live_test_folds.csv"}')
        print(f'Exported {args.output_dir / "live_test_best_trials.csv"}')
        return

    # ---- dry-run -----------------------------------------------------------
    if args.dry_run:
        sample_row_dict, sample_fold = jobs[0]
        cell = _cell_from_row(pd.Series(sample_row_dict))
        sampled = _parse_json_or_literal(sample_row_dict['sampled_params'])
        ckpt = Path(sample_row_dict['ckpt_path'])
        overrides = _eval_overrides(
            config,
            cell,
            sampled,
            int(sample_row_dict['trial_number']),
            sample_fold,
            1,
            args.output_dir,
            ckpt,
        )
        print('Example command:')
        print('python -m ogbench.run ' + ' '.join(overrides))
        return

    # ---- actual run --------------------------------------------------------
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
                    args.output_dir,
                    ledger.path,
                    None,
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
                    args.output_dir,
                    ledger.path,
                    gpu_queue,
                    config.timeout,
                )
                for row, fold in pending
            )

    _export(ledger, study_level, args.output_dir)
    print(f'Wrote {args.output_dir / "live_test_folds.csv"}')
    print(f'Wrote {args.output_dir / "live_test_best_trials.csv"}')


if __name__ == '__main__':
    main()

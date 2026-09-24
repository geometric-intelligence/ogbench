#!/usr/bin/env python3
"""Summarize the ratio-0.3 transfer campaign without touching running trials.

Writes results_ratio03.csv (one row per cell, in priority order) and
coverage.md to the campaign root, prints a short status, and with
--check-remaining exits 0 while any cell can still make progress and 3 when
every cell is complete or permanently failed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.optuna_search import (  # noqa: E402
    OOM_ERROR_PREFIX,
    OptunaSearchConfig,
    RunLedger,
    _apply_runtime_overrides,
    read_study_manifest,
    run_search,
)

NO_REMAINING_WORK = 3


def ledger_attempts(ledger_path: Path) -> pd.DataFrame:
    if not ledger_path.exists():
        return pd.DataFrame(
            columns=['study_name', 'param_hash', 'fold', 'status', 'error', 'created_at']
        )
    with sqlite3.connect(f'file:{ledger_path}?mode=ro', uri=True) as connection:
        return pd.read_sql_query(
            'SELECT study_name, param_hash, fold, attempt, status, error, elapsed_time, '
            'created_at FROM fold_attempts',
            connection,
        )


def cell_status(
    study_name: str,
    param_hash: str,
    complete: set[str],
    attempts: pd.DataFrame,
    ledger: RunLedger,
    config: OptunaSearchConfig,
    oom_retries: int,
) -> str:
    if study_name in complete:
        return 'complete'
    rows = attempts[attempts['study_name'] == study_name]
    if rows.empty:
        return 'pending'
    if not ledger.retryable(
        study_name,
        param_hash,
        config.max_retries + 1,
        expected_folds=config.folds,
        oom_retries=oom_retries,
    ):
        return 'failed'
    return 'in_progress'


def coverage_markdown(
    results: pd.DataFrame,
    summary: dict[str, Any],
    *,
    final: bool,
    notes: Sequence[str] = (),
) -> str:
    counts = results['status'].value_counts().to_dict()
    coverage = results.assign(done=results['status'] == 'complete').pivot_table(
        index='model', columns='dataset', values='done', aggfunc='sum', fill_value=0
    )
    totals = results.pivot_table(
        index='model', columns='dataset', values='study_name', aggfunc='count', fill_value=0
    )
    model_order = list(dict.fromkeys(results['model']))
    source_rules = results['source_rule'].value_counts().to_dict()
    table = (
        coverage.astype(int).astype(str) + '/' + totals.astype(str)
    ).reindex(model_order)
    unfinished = (
        f'- Incomplete (stopped before 5 successful folds): {counts.get("incomplete", 0)}'
        if final
        else f'- In progress: {counts.get("in_progress", 0)}'
    )
    lines = [
        '# Ratio 0.3 transfer coverage',
        '',
        f'{"Final snapshot, campaign stopped" if final else "Snapshot"}: {summary["time"]}',
        '',
        'Each cell evaluates one transferred configuration with 5-fold CV; other ratios used',
        'a 7-trial Optuna search. The configuration is the best trial of the same cell at',
        'ratio 0.5, else 0.8, else 1.0, else of the closest sibling cell (`source_rule` in',
        'results_ratio03.csv).',
        '',
        'Configuration sources: '
        + ', '.join(f'{rule} {count}' for rule, count in source_rules.items()),
        '',
        f'- Complete cells: {counts.get("complete", 0)} / {len(results)}',
        f'{unfinished}; pending: {counts.get("pending", 0)}; '
        f'permanently failed: {counts.get("failed", 0)}',
        f'- Successful folds: {summary["folds_succeeded"]} / {summary["folds_target"]} '
        f'({summary["folds_last_hour"]} in the last hour)',
        f'- Failed attempts: {summary["failed_attempts"]} '
        f'(out of memory: {summary["oom_attempts"]}, timeouts: {summary["timeouts"]})',
        *notes,
        '',
        'Completed cells per model (rows, cheapest first) and dataset (columns):',
        '',
        table.to_markdown(),
        '',
    ]
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', default='configs/hparams_search/sep24_ratio03_transfer.yaml'
    )
    parser.add_argument('--output-dir', help='Campaign root (default: training.output_dir)')
    parser.add_argument('--storage', help='Optuna storage (default: optuna.storage)')
    parser.add_argument('--prepare-dir', help='Directory with candidates.csv and priority.txt')
    parser.add_argument('--oom-retries', type=int, default=3)
    parser.add_argument('--check-remaining', action='store_true')
    parser.add_argument(
        '--final',
        action='store_true',
        help='The campaign has stopped: label unfinished cells incomplete, not in progress',
    )
    args = parser.parse_args()

    config = OptunaSearchConfig.from_yaml(args.config)
    _apply_runtime_overrides(config, output_dir=args.output_dir, storage=args.storage)
    root = config.output_dir
    prepare_dir = Path(args.prepare_dir) if args.prepare_dir else root / 'prepare'
    studies = read_study_manifest(prepare_dir / 'priority.txt')
    candidates = pd.read_csv(prepare_dir / 'candidates.csv').set_index('study_name')

    storage_path = Path(config.storage.removeprefix('sqlite:///'))
    trials = (
        run_search(config, studies=studies, export_only=True)
        if storage_path.exists()
        else pd.DataFrame()
    )
    complete_rows = (
        trials[trials['state'] == 'COMPLETE'].drop_duplicates('study_name').set_index('study_name')
        if not trials.empty
        else pd.DataFrame()
    )
    complete = set(complete_rows.index)

    ledger_path = root / 'run_ledger.sqlite3'
    attempts = ledger_attempts(ledger_path)
    ledger = RunLedger(ledger_path) if ledger_path.exists() else None
    successes = attempts[attempts['status'] == 'success']
    folds_done = successes.groupby('study_name')['fold'].nunique()

    rows = []
    for study_name in studies:
        candidate = candidates.loc[study_name]
        status = (
            cell_status(
                study_name,
                str(candidate['param_hash']),
                complete,
                attempts,
                ledger,
                config,
                args.oom_retries,
            )
            if ledger is not None
            else 'pending'
        )
        row = {
            'priority': int(candidate['priority']),
            'study_name': study_name,
            'model': candidate['model'],
            'dataset': candidate['dataset'],
            'experiment': candidate['experiment'],
            'adjacency_method': candidate['dataset.loader.parameters.adjacency_method'],
            'selection_method': candidate['dataset.loader.parameters.method'],
            'node_sample_ratio': candidate['dataset.loader.parameters.node_sample_ratio'],
            'status': status,
            'folds_done': int(folds_done.get(study_name, 0)),
            'objective': None,
            'fold_std': None,
            'fold_scores': None,
            'protocol': 'transferred_single_config',
            'source_study': candidate['source_study'],
            'source_rule': candidate['source_rule'],
            'source_objective': candidate['source_mean'],
            'sampled_params': candidate['sampled_params'],
        }
        if study_name in complete:
            trial = complete_rows.loc[study_name]
            row['objective'] = trial['objective']
            row['fold_std'] = trial['fold_std']
            row['fold_scores'] = trial['fold_scores']
        rows.append(row)
    results = pd.DataFrame(rows)
    if args.final:
        results['status'] = results['status'].replace('in_progress', 'incomplete')
    results.to_csv(root / 'results_ratio03.csv', index=False)

    now = datetime.now(UTC)
    created = pd.to_datetime(attempts['created_at'], utc=True, errors='coerce')
    last_hour = attempts[created >= now - timedelta(hours=1)]
    failed = attempts[attempts['status'] == 'failed']
    errors = failed['error'].fillna('')
    oom_attempts = int(errors.str.startswith(OOM_ERROR_PREFIX).sum())
    timeouts = int(errors.str.startswith('Timeout after').sum())
    recent_errors = last_hour[last_hour['status'] == 'failed']['error'].fillna('')
    counts = results['status'].value_counts().to_dict()
    summary = {
        'time': now.astimezone().strftime('%Y-%m-%d %H:%M %Z'),
        'cells': len(results),
        'status': counts,
        'folds_succeeded': int(len(successes)),
        'folds_target': len(results) * len(config.folds),
        'folds_last_hour': int((last_hour['status'] == 'success').sum()),
        'failed_attempts': int(len(failed)),
        'oom_attempts': oom_attempts,
        'oom_attempts_last_hour': int(recent_errors.str.startswith(OOM_ERROR_PREFIX).sum()),
        'attempts_last_hour': int(len(last_hour)),
        'timeouts': timeouts,
    }

    (root / 'coverage.md').write_text(coverage_markdown(results, summary, final=args.final))
    (root / 'status_latest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))

    if args.check_remaining:
        remaining = results['status'].isin(['pending', 'in_progress']).any()
        raise SystemExit(0 if remaining else NO_REMAINING_WORK)


if __name__ == '__main__':
    main()

"""Plot test F1 (fold mean ± std) for the Optuna-best trial in each cell.

Each study's winner was chosen on validation. This script plots that same
trial's held-out test F1. Concatenate one export per server once the shards
live in one place:

    python plotting/plot_sep24_test_best.py \\
        --csv /scratch/louisvl/ogbench/search_results/sep24_factorial_optuna_test/live_test_best_trials.csv \\
        --csv /path/to/hall/live_test_best_trials.csv \\
        --csv /path/to/parka/live_test_best_trials.csv

While a server is still evaluating, ``live_test_best_trials.csv`` does not exist
yet. Pass that server's ledger plus its search ``live_best_trials.csv``:

    python plotting/plot_sep24_test_best.py \\
        --best-csv /scratch/louisvl/ogbench/search_results/sep24_factorial_optuna/live_best_trials.csv \\
        --ledger /scratch/louisvl/ogbench/search_results/sep24_factorial_optuna_test/test_ledger.sqlite3 \\
        --min-folds 5
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_sep24_val_best import (  # noqa: E402
    _annotate_cells,
    _with_metric_columns,
    load_best_trials,
    render_sep24_factor_plots,
)

DEFAULT_TEST_CSV = Path(
    '/scratch/louisvl/ogbench/search_results/sep24_factorial_optuna_test/live_test_best_trials.csv'
)
DEFAULT_BEST_CSV = Path(
    '/scratch/louisvl/ogbench/search_results/sep24_factorial_optuna/live_best_trials.csv'
)
DEFAULT_LEDGER = Path(
    '/scratch/louisvl/ogbench/search_results/sep24_factorial_optuna_test/test_ledger.sqlite3'
)


def summarize_test_folds(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return folds
    success = folds.loc[folds['status'].astype(str) == 'success'].copy()
    if success.empty:
        return success
    latest = success.sort_values('attempt').drop_duplicates(
        ['study_name', 'trial_number', 'fold'], keep='last'
    )
    return latest.groupby(['study_name', 'trial_number'], as_index=False).agg(
        n_folds=('fold', 'nunique'),
        test_f1_mean=('test_f1', 'mean'),
        test_f1_std=('test_f1', 'std'),
        val_f1_from_eval=('val_f1', 'mean'),
    )


def _read_ledger(path: Path) -> pd.DataFrame:
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query('SELECT * FROM test_folds', connection)


def _load_study_level_test_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame['source_csv'] = str(path)
    if 'test_f1_mean' in frame.columns:
        return frame
    if {'test_f1', 'fold', 'status'} <= set(frame.columns):
        summary = summarize_test_folds(frame)
        summary['source_csv'] = str(path)
        return summary
    raise ValueError(
        f'{path} is not a live_test_best_trials.csv or live_test_folds.csv '
        f'(columns: {sorted(frame.columns)})'
    )


def _merge_test_onto_best(best: pd.DataFrame, test_summary: pd.DataFrame) -> pd.DataFrame:
    if test_summary.empty:
        return test_summary
    keep = [
        c
        for c in (
            'study_name',
            'trial_number',
            'n_folds',
            'test_f1_mean',
            'test_f1_std',
            'source_csv',
        )
        if c in test_summary.columns
    ]
    merged = best.merge(test_summary[keep], on=['study_name', 'trial_number'], how='inner')
    return merged


def load_test_trials(
    *,
    test_csvs: list[Path],
    best_csvs: list[Path],
    ledgers: list[Path],
    min_folds: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    best: pd.DataFrame | None = None
    if best_csvs:
        best = load_best_trials(best_csvs)

    for path in test_csvs:
        raw = _load_study_level_test_csv(path)
        if {'model', 'dataset', 'experiment'} <= set(
            raw.columns
        ) and 'test_f1_mean' in raw.columns:
            frames.append(_annotate_cells(raw))
        else:
            if best is None:
                raise ValueError(f'{path} has no study metadata; pass --best-csv as well')
            frames.append(_merge_test_onto_best(best, raw))

    for path in ledgers:
        if best is None:
            raise ValueError('--ledger requires --best-csv so study metadata can be joined')
        frames.append(_merge_test_onto_best(best, summarize_test_folds(_read_ledger(path))))

    if not frames:
        raise ValueError('No test inputs: pass --csv and/or --ledger + --best-csv')

    combined = pd.concat(frames, ignore_index=True)
    if 'n_folds' in combined.columns:
        combined['n_folds'] = pd.to_numeric(combined['n_folds'], errors='coerce')
        combined = combined.loc[combined['n_folds'] >= min_folds].copy()
    combined = _with_metric_columns(combined, 'test_f1_mean', 'test_f1_std')
    if combined.empty:
        raise ValueError(f'No studies with test F1 and at least {min_folds} successful folds')
    combined = (
        combined.sort_values(['n_folds', 'metric_mean'], ascending=[False, False])
        .drop_duplicates('study_name', keep='first')
        .reset_index(drop=True)
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--csv',
        action='append',
        type=Path,
        help='live_test_best_trials.csv or live_test_folds.csv (repeat per server)',
    )
    parser.add_argument(
        '--best-csv',
        action='append',
        type=Path,
        help='search live_best_trials.csv; required with --ledger or fold-level CSVs',
    )
    parser.add_argument(
        '--ledger',
        action='append',
        type=Path,
        help='test_ledger.sqlite3 (repeat per server); join with --best-csv',
    )
    parser.add_argument(
        '--min-folds',
        type=int,
        default=5,
        help='Keep a study only if this many folds have a successful test score (default: 5)',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('plotting/plots_sep24_test'),
    )
    args = parser.parse_args()

    test_csvs: list[Path] = []
    missing_csvs: list[Path] = []
    for path in args.csv or []:
        if path.exists():
            test_csvs.append(path)
        else:
            missing_csvs.append(path)
            print(f'Missing {path}; it is only written when test eval finishes.')
    best_csvs = list(args.best_csv or [])
    ledgers = list(args.ledger or [])
    if missing_csvs and not ledgers:
        sibling_ledgers = [
            path.parent / 'test_ledger.sqlite3'
            for path in missing_csvs
            if (path.parent / 'test_ledger.sqlite3').exists()
        ]
        if sibling_ledgers:
            ledgers.extend(sibling_ledgers)
            print('Falling back to test_ledger.sqlite3 next to the missing CSV(s).')
        elif DEFAULT_LEDGER.exists():
            ledgers = [DEFAULT_LEDGER]
            print(f'Falling back to {DEFAULT_LEDGER}')
    if not test_csvs and not ledgers:
        if DEFAULT_TEST_CSV.exists():
            test_csvs = [DEFAULT_TEST_CSV]
        elif DEFAULT_LEDGER.exists():
            ledgers = [DEFAULT_LEDGER]
            best_csvs = best_csvs or [DEFAULT_BEST_CSV]
        else:
            raise FileNotFoundError(
                'No --csv/--ledger given and no Frank test export or ledger at the default paths'
            )
    if ledgers and not best_csvs:
        best_csvs = [DEFAULT_BEST_CSV]

    best = load_test_trials(
        test_csvs=test_csvs,
        best_csvs=best_csvs,
        ledgers=ledgers,
        min_folds=args.min_folds,
    )
    print(
        f'Loaded {len(best)} studies with ≥{args.min_folds} test folds '
        f'from {len(test_csvs)} CSV(s) and {len(ledgers)} ledger(s); '
        f'datasets={sorted(best.data_name.unique())}; models={sorted(best.model_name.unique())}'
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_sep24_factor_plots(best, args.out_dir, split='test')
    print(f'Wrote figures and summary CSVs to {args.out_dir.resolve()}')


if __name__ == '__main__':
    main()

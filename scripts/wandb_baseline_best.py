#!/usr/bin/env python3
"""Fetch baseline results from W&B and export the best hyperparams per data setting.

For each (dataset × node_sample_ratio × feature_method × baseline_model_type × fold),
keeps the run with the highest val F1 (HPs are tuned per fold), then writes:

  * ``baseline_best_folds.csv``   — one row per (data_setting × model × fold)
                                    for the winning hyperparam config
  * ``baseline_best_trials.csv``  — one row per (data_setting × model)
                                    with mean ± std across folds

Selection is done on val F1 only. Test F1 is carried through but never used
to select.

Usage::

    python scripts/wandb_baseline_best.py \\
        --project bioshape-lab/ogbench_sep24_factorial_tc10 \\
        --output-dir plotting/baseline_results

The two CSVs are written to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRIC_SPLITS = ('train', 'val', 'test')
_METRIC_NAMES = (
    'f1_macro',
    'f1_weighted',
    'accuracy',
    'balanced_accuracy',
    'auroc',
    'pr_auc',
    'precision',
    'recall',
)


def _config_val(config: dict, key: str):
    """Get a top-level config value, returning None if absent."""
    return config.get(key)


def _extract_hyperparams(summary: dict) -> dict:
    """Return all hyperparams/{model}__{param} entries as a flat dict."""
    return {
        k[len('hyperparams/') :]: v for k, v in summary.items() if k.startswith('hyperparams/')
    }


def _model_type_from_hyperparams(hp: dict) -> str | None:
    """Derive model type from the first hyperparam key, e.g. 'elastic_net__C' → 'elastic_net'."""
    for k in hp:
        parts = k.split('__', 1)
        if len(parts) == 2:
            return parts[0]
    return None


def _extract_metrics(summary: dict) -> dict:
    """Pull train/val/test metrics from summary into a flat dict."""
    metrics: dict = {}
    for split in _METRIC_SPLITS:
        for metric in _METRIC_NAMES:
            key = f'{split}/{metric}'
            if key in summary:
                metrics[f'{split}_{metric}'] = summary[key]
        # train only has best_cv_score
        cv_key = f'{split}/best_cv_score'
        if cv_key in summary:
            metrics['train_cv_score'] = summary[cv_key]
    return metrics


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_baseline_runs(projects: list[str]) -> pd.DataFrame:
    """Fetch all finished baseline runs from W&B."""
    api = wandb.Api(timeout=120)
    records: list[dict] = []

    for project in projects:
        print(f'Querying {project} for baseline runs …', flush=True)
        filters = {
            '$and': [
                {'displayName': {'$regex': '^baseline_'}},
                {'state': 'finished'},
            ]
        }
        fetched = skipped = 0
        for run in api.runs(project, filters=filters, per_page=500):
            cfg = run.config
            summary = run.summary._json_dict

            dataset = _config_val(cfg, 'dataset')
            node_sample_ratio = _config_val(cfg, 'node_sample_ratio')
            method = _config_val(cfg, 'method')
            fold = _config_val(cfg, 'fold')
            baseline_name = _config_val(cfg, 'baseline_name')

            if any(v is None for v in (dataset, node_sample_ratio, method, fold)):
                skipped += 1
                continue

            hp = _extract_hyperparams(summary)
            # Prefer baseline_name from config; fall back to parsing run name
            if baseline_name is None:
                m = re.match(r'^baseline_([^_]+(?:_[^_]+)?)_', run.name)
                baseline_name = m.group(1) if m else 'unknown'
            # Override with model type derived from hyperparams (more reliable)
            model_from_hp = _model_type_from_hyperparams(hp)
            model_type = model_from_hp or baseline_name

            val_f1 = summary.get('val/f1_macro')
            if val_f1 is None:
                skipped += 1
                continue

            metrics = _extract_metrics(summary)

            records.append(
                {
                    'dataset': dataset,
                    'node_sample_ratio': node_sample_ratio,
                    'feature_method': method,
                    'model_type': model_type,
                    'fold': fold,
                    'hyperparams': json.dumps(hp, sort_keys=True),
                    'val_f1': val_f1,
                    'wandb_run_id': run.id,
                    **metrics,
                }
            )
            fetched += 1
            if fetched % 200 == 0:
                print(f'  … {fetched} runs', flush=True)

        print(f'  {project}: {fetched} baseline runs, {skipped} skipped')

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Normalise dtypes
    df['node_sample_ratio'] = pd.to_numeric(df['node_sample_ratio'], errors='coerce')
    df['fold'] = pd.to_numeric(df['fold'], errors='coerce').astype('Int64')
    return df


# ---------------------------------------------------------------------------
# Best-hyperparam selection
# ---------------------------------------------------------------------------

_SETTING_COLS = ['dataset', 'node_sample_ratio', 'feature_method', 'model_type']


def select_best_hyperparams(runs: pd.DataFrame) -> pd.DataFrame:
    """Keep one run per (data_setting × model_type × fold).

    Baseline HPs are chosen independently per fold (inner CV), so the winning
    ``C`` / ``l1_ratio`` often differs across folds. Requiring an identical
    hyperparam JSON therefore collapsed many settings to a single fold.

    For each fold we keep the run with the highest val F1 (handles retries).
    The five W&B fold runs for a setting are then the nested-CV estimate.
    """
    if runs.empty:
        return runs
    return (
        runs.sort_values('val_f1', ascending=False)
        .drop_duplicates(_SETTING_COLS + ['fold'], keep='first')
        .reset_index(drop=True)
    )


def summarise_folds(best_folds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level rows into per-(setting × model) summary."""
    if best_folds.empty:
        return best_folds

    # Drop the per-config-mean helper column; it equals val_f1_macro_mean after averaging.
    df = best_folds.drop(columns=['val_f1_mean'], errors='ignore')

    # Columns to aggregate: every metric col except fold itself
    metric_cols = [
        c for c in df.columns if any(c.startswith(f'{s}_') for s in _METRIC_SPLITS) and c != 'fold'
    ]

    agg: dict = {col: ['mean', 'std'] for col in metric_cols if col in df.columns}
    agg['fold'] = 'nunique'
    agg['hyperparams'] = 'first'

    summary = df.groupby(_SETTING_COLS, as_index=False).agg(agg)

    # Flatten multi-level columns: ('val_f1_macro', 'mean') → 'val_f1_macro_mean'
    # non-aggregated columns like groupby keys keep a single-level name
    summary.columns = [f'{a}_{b}' if b not in ('', 'first') else a for a, b in summary.columns]
    summary = summary.rename(columns={'fold_nunique': 'n_folds'})
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project',
        action='append',
        required=True,
        metavar='ENTITY/PROJECT',
        help=(
            'W&B project to query (repeat for multiple):\n'
            '  --project bioshape-lab/ogbench_sep24_factorial_tc10'
        ),
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        type=Path,
        help='Directory to write baseline_best_folds.csv and baseline_best_trials.csv',
    )
    args = parser.parse_args()

    runs_df = fetch_baseline_runs(args.project)
    if runs_df.empty:
        print('ERROR: no baseline runs fetched — check project name and W&B credentials')
        sys.exit(1)

    n_settings = runs_df.groupby(_SETTING_COLS).ngroups
    print(
        f'\nTotal baseline fold runs fetched: {len(runs_df)} '
        f'({runs_df["dataset"].nunique()} datasets, '
        f'{runs_df["model_type"].nunique()} model types, '
        f'{n_settings} data-setting × model combinations)'
    )

    best_folds = select_best_hyperparams(runs_df)
    print(
        f'Best-per-fold rows: {len(best_folds)} '
        f'(one run per data_setting × model_type × fold, best val F1 if retries)'
    )

    summary = summarise_folds(best_folds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = args.output_dir / 'baseline_best_folds.csv'
    trials_path = args.output_dir / 'baseline_best_trials.csv'

    best_folds.to_csv(folds_path, index=False)
    summary.to_csv(trials_path, index=False)

    print(f'\nWrote {len(best_folds)} rows → {folds_path}')
    print(f'Wrote {len(summary)} rows → {trials_path}')
    print('\nModel types found:', sorted(runs_df['model_type'].unique()))
    print('Datasets found:', sorted(runs_df['dataset'].unique()))
    print('\nSettings per model_type:')
    for mt, grp in summary.groupby('model_type'):
        print(f'  {mt}: {len(grp)} data settings')


if __name__ == '__main__':
    main()

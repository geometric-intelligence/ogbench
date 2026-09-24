"""Plot validation F1 (fold mean ± std) for the best Optuna trial in each cell.

Reads ``best_trials.csv`` / ``live_best_trials.csv`` from the Sep 24 campaign
(one row per study). Pass extra CSVs later to concatenate Hall (shards 3–4)
or Parka (shards 0–2):

    python plotting/plot_sep24_val_best.py \\
        --csv /scratch/louisvl/ogbench/search_results/sep24_factorial_optuna/live_best_trials.csv \\
        --csv /path/to/hall/live_best_trials.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CANONICAL_MODEL_ORDER = [
    'mlp',
    'gin',
    'gcn',
    'gatv2',
    'sage',
    'gps',
    'sagn',
    'chebnet',
    'gatv4',
]
MODEL_ALIASES = {'graph_sage': 'sage', 'graphsage': 'sage', 'mlagnn': 'gatv4'}
MODEL_DISPLAY = {
    'mlp': 'MLP',
    'gin': 'GIN',
    'gcn': 'GCN',
    'gatv2': 'GATv2',
    'sage': 'SAGE',
    'gps': 'GPS',
    'sagn': 'SAGN',
    'chebnet': 'ChebNet',
    'gatv4': 'MLA-GNN',
}
DATASET_ORDER = [
    'motrpac',
    'addneuromed',
    'parkinsons',
    'brca',
    'smoking',
    'tuberculosis',
]
DATASET_DISPLAY = {
    'motrpac': 'Heritage',
    'addneuromed': 'Addneuromed',
    'parkinsons': 'Parkinsons',
    'brca': 'BRCA',
    'smoking': 'Smoking',
    'tuberculosis': 'Tuberculosis',
}
MODEL_COLORS = {
    'mlp': '#5B9BD5',
    'gin': '#D62828',
    'gcn': '#E85D04',
    'gatv2': '#F48C06',
    'sage': '#FAA307',
    'gps': '#20BFC3',
    'sagn': '#7B2CBF',
    'chebnet': '#2D6A4F',
    'gatv4': '#52B788',
}
ADJ_DISPLAY = {'string': 'PPI', 'wgcna': 'Co-expression'}
READOUT_DISPLAY = {'omics_readout': 'Omics readout', 'no_readout': 'No readout'}
METHOD_DISPLAY = {
    'variance': 'Variance',
    'random': 'Random',
    'correlation': 'Correlation',
    'distance_correlation': 'Dist. corr.',
}


def _canonical_model(name: object) -> str:
    s = str(name).strip().lower().replace('-', '_').replace(' ', '_')
    return MODEL_ALIASES.get(s, s)


def load_best_trials(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame['source_csv'] = str(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    complete = raw.loc[raw['state'].astype(str).str.upper() == 'COMPLETE'].copy()
    if complete.empty:
        raise ValueError('No COMPLETE trials in the provided CSVs')

    complete = _annotate_cells(complete)
    complete['val_f1_mean'] = pd.to_numeric(complete['fold_mean'], errors='coerce')
    complete['val_f1_std'] = pd.to_numeric(complete['fold_std'], errors='coerce')
    complete = complete.loc[complete['val_f1_mean'].notna()].copy()

    # One row per study: if the same study appears in two CSVs, keep the higher objective.
    complete = (
        complete.sort_values('val_f1_mean', ascending=False)
        .drop_duplicates('study_name', keep='first')
        .reset_index(drop=True)
    )
    return _with_metric_columns(complete, 'val_f1_mean', 'val_f1_std')


def _annotate_cells(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out['model_name'] = out['model'].map(_canonical_model)
    out['data_name'] = out['dataset'].astype(str).str.strip().str.lower()
    out['experiment'] = out['experiment'].astype(str).str.strip()
    out['adjacency_method'] = (
        out['dataset.loader.parameters.adjacency_method'].astype(str).str.strip().str.lower()
    )
    out['node_sample_ratio'] = pd.to_numeric(
        out['dataset.loader.parameters.node_sample_ratio'], errors='coerce'
    )
    out['sampling_method'] = out['dataset.loader.parameters.method'].astype(str).str.strip()
    return out


def _with_metric_columns(frame: pd.DataFrame, mean_col: str, std_col: str) -> pd.DataFrame:
    out = frame.copy()
    out['metric_mean'] = pd.to_numeric(out[mean_col], errors='coerce')
    out['metric_std'] = pd.to_numeric(out[std_col], errors='coerce').fillna(0.0)
    return out.loc[out['metric_mean'].notna()].copy()


def pick_best(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        frame.sort_values('metric_mean', ascending=False)
        .drop_duplicates(group_cols, keep='first')
        .reset_index(drop=True)
    )


def _ordered_models(frame: pd.DataFrame) -> list[str]:
    found = set(frame['model_name'].dropna())
    models = [m for m in CANONICAL_MODEL_ORDER if m in found]
    models.extend(sorted(found - set(models)))
    return models


def _ordered_datasets(frame: pd.DataFrame) -> list[str]:
    found = set(frame['data_name'].dropna())
    datasets = [d for d in DATASET_ORDER if d in found]
    datasets.extend(sorted(found - set(datasets)))
    return datasets


def _ylim(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    lo, hi = min(finite), max(finite)
    pad = max(0.04, 0.08 * (hi - lo + 1e-6))
    return max(0.0, lo - pad), min(1.05, hi + pad)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_best_overall(
    best: pd.DataFrame,
    out_dir: Path,
    *,
    ylabel: str = 'Validation F1 macro (fold mean ± std)',
    title: str = (
        'Best available setting per model — validation F1\n'
        '(7-trial Optuna budget; error bars are 5-fold std; shards present in the input CSVs only)'
    ),
    out_name: str = 'best_overall_val_f1',
) -> pd.DataFrame:
    """Best cell per model × dataset (max metric among available studies)."""
    slice_best = pick_best(best, ['data_name', 'model_name'])
    models = _ordered_models(slice_best)
    datasets = _ordered_datasets(slice_best)
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(max(14, 3.4 * len(datasets)), 4.6), sharey=False
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(models))
    for ax, dataset in zip(axes, datasets, strict=True):
        means, stds = [], []
        for model in models:
            row = slice_best[
                (slice_best['data_name'] == dataset) & (slice_best['model_name'] == model)
            ]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(row.iloc[0]['metric_mean']))
                stds.append(float(row.iloc[0]['metric_std']))
        colors = [MODEL_COLORS.get(m, '#888888') for m in models]
        ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            edgecolor='black',
            linewidth=0.8,
            capsize=3,
            error_kw={'elinewidth': 1, 'capthick': 1},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=45, ha='right')
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)
        vals = [m - s for m, s in zip(means, stds, strict=True) if np.isfinite(m)]
        vals += [m + s for m, s in zip(means, stds, strict=True) if np.isfinite(m)]
        ax.set_ylim(_ylim(vals))
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
    fig.suptitle(title, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / out_name)
    return slice_best


def plot_grouped_effect(
    best: pd.DataFrame,
    *,
    factor: str,
    levels: list[str],
    labels: dict[str, str],
    hatches: dict[str, str],
    title: str,
    out_name: str,
    out_dir: Path,
    exclude_mlp: bool = False,
    ylabel: str = 'Validation F1 macro (fold mean ± std)',
) -> None:
    frame = best.copy()
    if exclude_mlp:
        frame = frame.loc[frame['model_name'] != 'mlp']
    slice_best = pick_best(frame, ['data_name', 'model_name', factor])
    models = _ordered_models(slice_best)
    datasets = _ordered_datasets(slice_best)
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(16, 4.2 * len(datasets)), 5.0))
    axes = np.atleast_1d(axes)
    bar_w = 0.35
    x = np.arange(len(models))
    for ax, dataset in zip(axes, datasets, strict=True):
        ylim_vals: list[float] = []
        for m_idx, model in enumerate(models):
            for r_idx, level in enumerate(levels):
                row = slice_best[
                    (slice_best['data_name'] == dataset)
                    & (slice_best['model_name'] == model)
                    & (slice_best[factor] == level)
                ]
                if row.empty:
                    continue
                mu = float(row.iloc[0]['metric_mean'])
                sig = float(row.iloc[0]['metric_std'])
                offset = (r_idx - 0.5) * bar_w
                ax.bar(
                    x[m_idx] + offset,
                    mu,
                    yerr=sig,
                    width=bar_w,
                    color=MODEL_COLORS.get(model, '#888888'),
                    edgecolor='black',
                    linewidth=0.8,
                    hatch=hatches[level],
                    alpha=0.95 if r_idx == 0 else 0.65,
                    capsize=2.5,
                    error_kw={'elinewidth': 1, 'capthick': 1},
                )
                ylim_vals.extend([mu - sig, mu + sig])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=45, ha='right')
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)
        ax.set_ylim(_ylim(ylim_vals))
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
    handles = [
        mpatches.Patch(
            facecolor='lightgray',
            edgecolor='black',
            hatch=hatches[level],
            alpha=0.95 if i == 0 else 0.65,
            label=labels[level],
        )
        for i, level in enumerate(levels)
    ]
    fig.legend(handles=handles, loc='upper center', ncol=len(levels), bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(title, fontweight='bold', y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir / out_name)


def plot_ratio_and_method(
    best: pd.DataFrame,
    out_dir: Path,
    *,
    ylabel: str = 'Validation F1 macro (fold mean ± std)',
    title: str = 'Best validation F1 by node sampling (any model / readout / adjacency)',
    out_name: str = 'best_val_f1_by_ratio_and_method',
) -> None:
    datasets = _ordered_datasets(best)
    ratio_best = pick_best(best, ['data_name', 'node_sample_ratio'])
    method_best = pick_best(best, ['data_name', 'sampling_method'])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ratios = sorted(ratio_best['node_sample_ratio'].dropna().unique())
    x = np.arange(len(datasets))
    width = 0.22
    for i, ratio in enumerate(ratios):
        means, stds = [], []
        for dataset in datasets:
            row = ratio_best[
                (ratio_best['data_name'] == dataset)
                & (np.isclose(ratio_best['node_sample_ratio'], ratio))
            ]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(row.iloc[0]['metric_mean']))
                stds.append(float(row.iloc[0]['metric_std']))
        axes[0].bar(
            x + (i - (len(ratios) - 1) / 2) * width,
            means,
            yerr=stds,
            width=width,
            label=f'ratio {ratio:g}',
            capsize=2.5,
            edgecolor='black',
            linewidth=0.6,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DATASET_DISPLAY.get(d, d) for d in datasets], rotation=25, ha='right')
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('Best cell at each node-sample ratio')
    axes[0].legend(frameon=True)
    axes[0].grid(axis='y', linestyle='--', alpha=0.35)
    axes[0].set_axisbelow(True)

    methods = [m for m in METHOD_DISPLAY if m in set(method_best['sampling_method'])]
    width = 0.18
    for i, method in enumerate(methods):
        means, stds = [], []
        for dataset in datasets:
            row = method_best[
                (method_best['data_name'] == dataset) & (method_best['sampling_method'] == method)
            ]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(row.iloc[0]['metric_mean']))
                stds.append(float(row.iloc[0]['metric_std']))
        axes[1].bar(
            x + (i - (len(methods) - 1) / 2) * width,
            means,
            yerr=stds,
            width=width,
            label=METHOD_DISPLAY[method],
            capsize=2.5,
            edgecolor='black',
            linewidth=0.6,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DATASET_DISPLAY.get(d, d) for d in datasets], rotation=25, ha='right')
    axes[1].set_title('Best cell at each node-selection method')
    axes[1].legend(frameon=True)
    axes[1].grid(axis='y', linestyle='--', alpha=0.35)
    axes[1].set_axisbelow(True)
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    _save(fig, out_dir / out_name)


def write_summary(
    slice_best: pd.DataFrame,
    best: pd.DataFrame,
    out_dir: Path,
    *,
    mean_col: str = 'val_f1_mean',
    std_col: str = 'val_f1_std',
    extra_cols: list[str] | None = None,
) -> None:
    if mean_col not in slice_best.columns:
        slice_best = slice_best.copy()
        slice_best[mean_col] = slice_best['metric_mean']
        slice_best[std_col] = slice_best['metric_std']
    if mean_col not in best.columns:
        best = best.copy()
        best[mean_col] = best['metric_mean']
        best[std_col] = best['metric_std']
    cols = [
        'data_name',
        'model_name',
        'experiment',
        'adjacency_method',
        'node_sample_ratio',
        'sampling_method',
        mean_col,
        std_col,
        'study_name',
        'trial_number',
        'n_cells_available',
    ]
    if extra_cols:
        cols.extend(c for c in extra_cols if c in slice_best.columns and c not in cols)
    counts = best.groupby(['data_name', 'model_name']).size().rename('n_cells_available')
    out = slice_best.merge(counts, on=['data_name', 'model_name'], how='left')
    out[cols].sort_values(['data_name', mean_col], ascending=[True, False]).to_csv(
        out_dir / 'best_per_model_dataset.csv', index=False
    )
    overall = pick_best(best, ['data_name'])
    overall_cols = [
        'data_name',
        'model_name',
        'experiment',
        'adjacency_method',
        'node_sample_ratio',
        'sampling_method',
        mean_col,
        std_col,
        'study_name',
    ]
    if extra_cols:
        overall_cols.extend(
            c for c in extra_cols if c in overall.columns and c not in overall_cols
        )
    overall[overall_cols].sort_values('data_name').to_csv(
        out_dir / 'best_per_dataset.csv', index=False
    )


def render_sep24_factor_plots(
    best: pd.DataFrame,
    out_dir: Path,
    *,
    split: str = 'val',
) -> pd.DataFrame:
    """Write the standard Sep 24 factor plots for validation or test F1."""
    if split == 'test':
        ylabel = 'Test F1 macro (fold mean ± std)'
        overall_title = (
            'Best available setting per model — test F1\n'
            "(Optuna winner chosen on val; bars are that trial's test F1; "
            'error bars are 5-fold std; shards present in the input files only)'
        )
        readout_title = 'Readout ablation — best available cell per model × readout (test F1)'
        adj_title = 'Adjacency ablation — best available cell per model × adjacency (test F1)'
        sampling_title = 'Best test F1 by node sampling (any model / readout / adjacency)'
        overall_file = 'best_overall_test_f1'
        readout_file = 'readout_effect_test_f1'
        adj_file = 'adjacency_effect_test_f1'
        sampling_file = 'best_test_f1_by_ratio_and_method'
        mean_col, std_col = 'test_f1_mean', 'test_f1_std'
        extra_cols = ['n_folds', 'val_f1_mean']
    elif split == 'val':
        ylabel = 'Validation F1 macro (fold mean ± std)'
        overall_title = (
            'Best available setting per model — validation F1\n'
            '(7-trial Optuna budget; error bars are 5-fold std; shards present in the input CSVs only)'
        )
        readout_title = (
            'Readout ablation — best available cell per model × readout (validation F1)'
        )
        adj_title = (
            'Adjacency ablation — best available cell per model × adjacency (validation F1)'
        )
        sampling_title = 'Best validation F1 by node sampling (any model / readout / adjacency)'
        overall_file = 'best_overall_val_f1'
        readout_file = 'readout_effect_val_f1'
        adj_file = 'adjacency_effect_val_f1'
        sampling_file = 'best_val_f1_by_ratio_and_method'
        mean_col, std_col = 'val_f1_mean', 'val_f1_std'
        extra_cols = None
    else:
        raise ValueError(f'Unknown split: {split}')

    slice_best = plot_best_overall(
        best,
        out_dir,
        ylabel=ylabel,
        title=overall_title,
        out_name=overall_file,
    )
    plot_grouped_effect(
        best,
        factor='experiment',
        levels=['omics_readout', 'no_readout'],
        labels=READOUT_DISPLAY,
        hatches={'omics_readout': '', 'no_readout': '.'},
        title=readout_title,
        out_name=readout_file,
        out_dir=out_dir,
        exclude_mlp=True,
        ylabel=ylabel,
    )
    plot_grouped_effect(
        best,
        factor='adjacency_method',
        levels=['string', 'wgcna'],
        labels=ADJ_DISPLAY,
        hatches={'string': '', 'wgcna': '///'},
        title=adj_title,
        out_name=adj_file,
        out_dir=out_dir,
        exclude_mlp=False,
        ylabel=ylabel,
    )
    plot_ratio_and_method(
        best,
        out_dir,
        ylabel=ylabel,
        title=sampling_title,
        out_name=sampling_file,
    )
    write_summary(
        slice_best,
        best,
        out_dir,
        mean_col=mean_col,
        std_col=std_col,
        extra_cols=extra_cols,
    )
    return slice_best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--csv',
        action='append',
        type=Path,
        help='best_trials / live_best_trials CSV (repeat to concatenate servers)',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('plotting/plots_sep24_val'),
    )
    args = parser.parse_args()
    csvs = args.csv or [
        Path('/scratch/louisvl/ogbench/search_results/sep24_factorial_optuna/live_best_trials.csv')
    ]
    best = load_best_trials(csvs)
    print(
        f'Loaded {len(best)} complete studies from {len(csvs)} CSV(s); '
        f'datasets={sorted(best.data_name.unique())}; models={sorted(best.model_name.unique())}'
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_sep24_factor_plots(best, args.out_dir, split='val')
    print(f'Wrote figures and summary CSVs to {args.out_dir.resolve()}')


if __name__ == '__main__':
    main()

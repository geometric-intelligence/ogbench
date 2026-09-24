"""Sep 24 test-F1 plots + LaTeX ablation tables (with baselines).

Generates:
  1. best_overall_test_f1_with_baselines.{pdf,png}
       — best GNN per (model × dataset), baseline hlines
  2. readout_effect_test_f1.{pdf,png}
       — grouped bar: OmicsReadout vs NoReadout per (model × dataset)
  3. adjacency_effect_test_f1.{pdf,png}
       — grouped bar: PPI vs Co-expression per (model × dataset)
  4. best_performance_by_dataset_method_ratio.{pdf,png}
       — dense grid: datasets × methods × ratios, all models + baseline bars
  5. best_performance_by_dataset_method_ratio_collapsed.{pdf,png}
       — same grid, GNNs collapsed to the best-val graph model (4 bars)
  6. tables/readout_ablation_{dataset}.tex   (one per dataset)
  7. tables/adjacency_ablation_{dataset}.tex (one per dataset)

Server CSVs live in ``plotting/test_results/{hall,frank,parka}/``.
If no ``--folds-csv`` / ``--csv`` is given, those folders are auto-discovered.

    python plotting/plot_sep24_test_with_baselines.py \\
        --manifest tmp/wandb_ckpt_manifest.csv \\
        --baseline-csv plotting/baseline_results/baseline_best_trials.csv \\
        --out-dir plotting/plots_sep24_test_with_baselines

Or pass paths explicitly:

    python plotting/plot_sep24_test_with_baselines.py \\
        --folds-csv plotting/test_results/hall/live_test_folds.csv \\
        --folds-csv plotting/test_results/frank/live_test_folds.csv \\
        --manifest tmp/wandb_ckpt_manifest.csv \\
        --baseline-csv plotting/baseline_results/baseline_best_trials.csv

``--folds-csv`` paths override ``--csv`` for the same study when richer
metrics are available.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.lines as mlines  # noqa: F401
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global style — must be set before any figure is created
# ---------------------------------------------------------------------------
_CMU_FONT_DIR = Path(__file__).resolve().parent / 'fonts'


def _setup_cmu_serif() -> None:
    """Register project TTF copies of CMU Serif (OTF is dropped by the PDF backend)."""
    for ttf in (
        'CMUSerif-Roman.ttf',
        'CMUSerif-Bold.ttf',
        'CMUSerif-Italic.ttf',
        'CMUSerif-BoldItalic.ttf',
    ):
        path = _CMU_FONT_DIR / ttf
        if path.exists():
            fm.fontManager.addfont(str(path))
    fm.fontManager = fm.FontManager()
    # usetex = real Computer Modern, matching the older paper figures.
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'CMU Serif']
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['savefig.dpi'] = 600


_setup_cmu_serif()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_sep24_val_best import (  # noqa: E402
    ADJ_DISPLAY,
    CANONICAL_MODEL_ORDER,
    DATASET_DISPLAY,
    DATASET_ORDER,
    METHOD_DISPLAY,
    MODEL_DISPLAY,
    _annotate_cells,
    _ordered_datasets,
    _ordered_models,
    _save,
    _with_metric_columns,
    _ylim,
    pick_best,
)

# ---------------------------------------------------------------------------
# Baseline styling
# ---------------------------------------------------------------------------
BASELINE_MODEL_ORDER = ('calibrated_svm', 'elastic_net')
BASELINE_DISPLAY = {'calibrated_svm': 'SVM', 'elastic_net': 'Elastic Net'}
BASELINE_HLINE_COLOR = {'calibrated_svm': '#3A3A3A', 'elastic_net': '#777777'}
BASELINE_HLINE_STYLE = {'calibrated_svm': '-', 'elastic_net': ':'}
BASELINE_HLINE_LW = 2.2

# Override MODEL_COLORS with the exact palette from the reference figure
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

# Bar / figure geometry matching reference
BAR_WIDTH = 0.56
BAR_CAPSIZE = 4
BAR_LW = 1.5  # edgecolor linewidth
PANEL_W = 3.6
PANEL_H = 4.55
WSPACE = 0.175

# Font sizes matching reference
FS_TITLE = 20
FS_YLABEL = 21
FS_XTICK = 19
FS_YTICK = 16
FS_DSTITLE = 20  # per-dataset subtitle

METHOD_ORDER = ('correlation', 'distance_correlation', 'random', 'variance')
GRAPH_BAR_COLOR = '#D62828'

_B_TEST = 'test_f1_macro_mean'
_B_TEST_STD = 'test_f1_macro_std'
_B_VAL = 'val_f1_macro_mean'

# Metrics used in ablation tables (column header → (mean_col, std_col))
TABLE_METRICS = [
    (r'$F_{macro}$', 'test_f1_mean', 'test_f1_std'),
    (r'$F_{weighted}$', 'test_f1_weighted_mean', 'test_f1_weighted_std'),
    (r'Accuracy', 'test_accuracy_mean', 'test_accuracy_std'),
    (r'AUROC', 'test_auroc_mean', 'test_auroc_std'),
]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _annotate_best_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Add model_name / data_name / experiment / adjacency_method etc. columns.

    Works on both the live_test_best_trials.csv format (which already has 'model', 'dataset',
    'experiment' columns from the manifest merge) and any DataFrame that has those columns.
    """
    if 'model_name' not in df.columns and 'model' in df.columns:
        df = _annotate_cells(df)
    # Ensure metric_mean / metric_std exist (needed by pick_best / plots)
    for mean_col, std_col in [('test_f1_mean', 'test_f1_std'), ('val_f1_mean', 'val_f1_std')]:
        if mean_col in df.columns and 'metric_mean' not in df.columns:
            df = _with_metric_columns(df, mean_col, std_col)
            break
    return df


def _load_folds_csv(folds_paths: list[Path], manifest_path: Path | None) -> pd.DataFrame:
    """Aggregate fold-level CSVs into per-study metric summaries.

    Joins with the manifest to recover model / dataset / experiment / adjacency_method metadata
    that isn't stored in the folds CSV itself.
    """
    if not folds_paths:
        return pd.DataFrame()

    frames = []
    for p in folds_paths:
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f'  [skip] folds CSV not found: {p}')
    if not frames:
        return pd.DataFrame()
    folds = pd.concat(frames, ignore_index=True)

    success = folds.loc[folds['status'].astype(str) == 'success'].copy()
    if success.empty:
        print('  Warning: no successful folds in folds CSVs')
        return pd.DataFrame()

    latest = success.sort_values('attempt').drop_duplicates(
        ['study_name', 'trial_number', 'fold'], keep='last'
    )

    _fold_metrics = {
        'test_f1': ('test_f1_mean', 'test_f1_std'),
        'test_f1_weighted': ('test_f1_weighted_mean', 'test_f1_weighted_std'),
        'test_accuracy': ('test_accuracy_mean', 'test_accuracy_std'),
        'test_auroc': ('test_auroc_mean', 'test_auroc_std'),
        'val_f1': ('val_f1_mean', 'val_f1_std'),
    }
    agg_kwargs: dict = {'n_folds': ('fold', 'nunique')}
    for col, (out_mean, out_std) in _fold_metrics.items():
        if col in latest.columns and latest[col].notna().any():
            agg_kwargs[out_mean] = (col, 'mean')
            agg_kwargs[out_std] = (col, 'std')

    summary = latest.groupby(['study_name', 'trial_number'], as_index=False).agg(**agg_kwargs)

    if manifest_path is not None and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        # Keep only one row per study_name (the best-trial one is first after
        # select_best_trials, so drop_duplicates on study_name is safe)
        meta = manifest.drop_duplicates('study_name', keep='first')
        summary = summary.merge(
            meta[
                [
                    'study_name',
                    'trial_number',
                    'model',
                    'dataset',
                    'experiment',
                    'dataset.loader.parameters.adjacency_method',
                    'dataset.loader.parameters.node_sample_ratio',
                    'dataset.loader.parameters.method',
                    'val_f1_mean',
                ]
            ],
            on=['study_name', 'trial_number'],
            how='left',
            suffixes=('', '_manifest'),
        )
        # Fill missing val_f1_mean from manifest if not in folds
        if 'val_f1_mean_manifest' in summary.columns:
            summary['val_f1_mean'] = summary['val_f1_mean'].fillna(summary['val_f1_mean_manifest'])
            summary.drop(columns=['val_f1_mean_manifest'], inplace=True)
        # Add state='COMPLETE' so load_best_trials doesn't drop them
        summary['state'] = 'COMPLETE'
        # fold_mean / fold_std aliases expected by _annotate_cells path
        if 'test_f1_mean' in summary.columns:
            summary['fold_mean'] = summary['test_f1_mean']
            summary['fold_std'] = summary['test_f1_std'].fillna(0)
    return summary


def load_gnn_test(
    csv_paths: list[Path],
    folds_paths: list[Path],
    manifest_path: Path | None,
    min_folds: int,
    ledger_paths: list[Path] | None = None,
) -> pd.DataFrame:
    """Load GNN test results, merging CSV and folds inputs.

    Folds CSV is preferred when available because it carries all metrics. Ledger (SQLite) paths are
    treated exactly like folds CSVs.
    """
    import sqlite3

    frames: list[pd.DataFrame] = []

    # Collect folds DataFrames: from CSV paths + ledger sqlite3 files
    all_folds_paths = list(folds_paths)
    ledger_frames: list[pd.DataFrame] = []
    for lp in ledger_paths or []:
        if lp.exists():
            with sqlite3.connect(lp) as conn:
                ledger_frames.append(pd.read_sql_query('SELECT * FROM test_folds', conn))
        else:
            print(f'  [skip] ledger not found: {lp}')

    if ledger_frames:
        combined_ledger = pd.concat(ledger_frames, ignore_index=True)
        tmp = tempfile.NamedTemporaryFile(
            prefix='_ogbench_ledger_folds_', suffix='.csv', delete=False
        )
        _tmp = Path(tmp.name)
        tmp.close()
        combined_ledger.to_csv(_tmp, index=False)
        all_folds_paths.append(_tmp)

    # From folds CSV (richer)
    if all_folds_paths:
        folds_df = _load_folds_csv(all_folds_paths, manifest_path)
        if not folds_df.empty:
            frames.append(folds_df)
            # Don't double-count studies already in folds_df
            _folds_studies = set(folds_df['study_name'].dropna())
        else:
            _folds_studies = set()
    else:
        _folds_studies = set()

    # From pre-aggregated CSV
    for p in csv_paths:
        if not p.exists():
            print(f'  [skip] CSV not found: {p}')
            continue
        df = pd.read_csv(p)
        if _folds_studies:
            df = df[~df['study_name'].isin(_folds_studies)].copy()
        if df.empty:
            continue
        df['state'] = df.get('state', 'COMPLETE')
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = _annotate_best_trials(combined)

    if 'n_folds' in combined.columns:
        combined['n_folds'] = pd.to_numeric(combined['n_folds'], errors='coerce')
        combined = combined.loc[combined['n_folds'] >= min_folds].copy()

    if combined.empty:
        return combined

    # Best per study (highest test_f1_mean; ties broken by n_folds)
    combined = (
        combined.sort_values(['n_folds', 'metric_mean'], ascending=[False, False])
        .drop_duplicates('study_name', keep='first')
        .reset_index(drop=True)
    )
    return combined


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------


def load_baseline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['dataset'] = df['dataset'].astype(str).str.strip().str.lower()
    df['model_type'] = df['model_type'].astype(str).str.strip().str.lower()
    df['node_sample_ratio'] = pd.to_numeric(df['node_sample_ratio'], errors='coerce')
    for c in (_B_TEST, _B_VAL):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _best_baseline_per_dataset(
    df_b: pd.DataFrame,
) -> dict[str, dict[str, tuple[float | None, float | None]]]:
    """Return {dataset: {model_type: (mean, std)}} for the best-val baseline config.

    Prefers configs with more folds (so std is available), breaking ties by val F1.
    """
    result: dict[str, dict[str, tuple[float | None, float | None]]] = {}
    if df_b.empty or _B_VAL not in df_b.columns or _B_TEST not in df_b.columns:
        return result
    for dataset in df_b['dataset'].unique():
        result[dataset] = {}
        sub = df_b[df_b['dataset'] == dataset].dropna(subset=[_B_VAL, _B_TEST]).copy()
        if 'n_folds' in sub.columns:
            sub['n_folds'] = pd.to_numeric(sub['n_folds'], errors='coerce').fillna(1)
        for model_type in BASELINE_MODEL_ORDER:
            rows = sub[sub['model_type'] == model_type]
            if rows.empty:
                result[dataset][model_type] = (None, None)
            else:
                # Prefer more folds (std available), then best val F1
                sort_cols = ['n_folds', _B_VAL] if 'n_folds' in rows.columns else [_B_VAL]
                best_row = rows.sort_values(sort_cols, ascending=False).iloc[0]
                mean = float(best_row[_B_TEST])
                std = (
                    float(best_row[_B_TEST_STD])
                    if _B_TEST_STD in best_row.index and pd.notna(best_row[_B_TEST_STD])
                    else None
                )
                result[dataset][model_type] = (mean, std)
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _fit_suptitle(
    fig, axes_flat, title: str, *, gap: float = 0.018, wspace=None, hspace=None
) -> None:
    """Place suptitle just above the tallest rendered content, then re-apply subplots_adjust."""
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fh = fig.bbox.height
    tops = [
        ax.get_tightbbox(renderer).y1 / fh
        for ax in axes_flat
        if ax.get_visible() and ax.get_tightbbox(renderer) is not None
    ]
    content_top = max(tops) if tops else 0.85
    y = min(content_top + gap, 0.995)
    fig.suptitle(title, fontsize=FS_TITLE, fontweight='bold', y=y, va='bottom')
    kw = {}
    if wspace is not None:
        kw['wspace'] = wspace
    if hspace is not None:
        kw['hspace'] = hspace
    if kw:
        fig.subplots_adjust(**kw)


def plot_best_overall_with_baselines(
    best: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: Path,
    *,
    ylabel: str = 'F1 Macro Score',
    title: str = 'Best Overall Model Performance by Test F1 Macro',
    out_name: str = 'best_overall_test_f1_with_baselines',
) -> pd.DataFrame:
    baseline_lookup = _best_baseline_per_dataset(df_b)
    slice_best = pick_best(best, ['data_name', 'model_name'])
    models = _ordered_models(slice_best)
    datasets = _ordered_datasets(slice_best)

    # Baseline bars come first, then GNN bars
    b_labels = [BASELINE_DISPLAY[bk] for bk in BASELINE_MODEL_ORDER]
    all_labels = b_labels + [MODEL_DISPLAY.get(m, m) for m in models]
    n_baseline = len(BASELINE_MODEL_ORDER)
    x = np.arange(len(all_labels))

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(max(15.5, len(datasets) * PANEL_W), PANEL_H),
        sharey=False,
    )
    axes = np.atleast_1d(axes)

    for j, (ax, dataset) in enumerate(zip(axes, datasets, strict=True)):
        blines = baseline_lookup.get(dataset, {})
        all_means, all_stds, all_colors = [], [], []

        # Baseline bars (gray)
        for bk in BASELINE_MODEL_ORDER:
            entry = blines.get(bk, (None, None))
            y, s = entry if isinstance(entry, tuple) else (entry, None)
            all_means.append(y if (y is not None and np.isfinite(y)) else np.nan)
            all_stds.append(s if (s is not None and np.isfinite(s)) else 0.0)
            all_colors.append(BASELINE_HLINE_COLOR[bk])

        # GNN bars
        for model in models:
            row = slice_best[
                (slice_best['data_name'] == dataset) & (slice_best['model_name'] == model)
            ]
            all_means.append(float(row.iloc[0]['metric_mean']) if not row.empty else np.nan)
            all_stds.append(float(row.iloc[0]['metric_std']) if not row.empty else 0.0)
            all_colors.append(MODEL_COLORS.get(model, '#888888'))

        ax.bar(
            x,
            all_means,
            yerr=all_stds,
            width=BAR_WIDTH,
            color=all_colors,
            edgecolor='black',
            linewidth=BAR_LW,
            capsize=BAR_CAPSIZE,
            zorder=3,
            error_kw={'elinewidth': 1.2, 'capthick': 1.2},
        )

        # Vertical separator between baselines and GNN models
        ax.axvline(n_baseline - 0.5, color='#aaaaaa', linewidth=0.9, linestyle='--', zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=FS_XTICK)
        ax.tick_params(axis='y', labelsize=FS_YTICK)
        ax.set_title(
            DATASET_DISPLAY.get(dataset, dataset), fontsize=FS_DSTITLE, fontweight='bold', pad=5
        )
        ax.grid(axis='y', linestyle='--', alpha=0.30)
        ax.set_axisbelow(True)
        finite = [m - s for m, s in zip(all_means, all_stds, strict=True) if np.isfinite(m)]
        finite += [m + s for m, s in zip(all_means, all_stds, strict=True) if np.isfinite(m)]
        ax.set_ylim(_ylim(finite))
        if j == 0:
            ax.set_ylabel(ylabel, fontsize=FS_YLABEL)

    _fit_suptitle(fig, axes, title, wspace=WSPACE)
    _save(fig, out_dir / out_name)
    return slice_best


def plot_best_overall_2rows(
    best: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: Path,
    *,
    ncols: int = 3,
    ylabel: str = 'F1 Macro Score',
    title: str = 'Best Overall Model Performance by Test F1 Macro',
    out_name: str = 'best_overall_test_f1_with_baselines_2rows',
) -> None:
    """2-row grid version (ncols panels per row).

    X-tick labels are shown only on the bottom row to avoid repetition. The figure is kept as
    compact as possible vertically.
    """
    baseline_lookup = _best_baseline_per_dataset(df_b)
    slice_best = pick_best(best, ['data_name', 'model_name'])
    models = _ordered_models(slice_best)
    datasets = _ordered_datasets(slice_best)

    # Pad to a full grid if needed
    nrows = -(-len(datasets) // ncols)  # ceiling division
    while len(datasets) < nrows * ncols:
        datasets.append(None)  # placeholder for empty cells

    b_labels = [BASELINE_DISPLAY[bk] for bk in BASELINE_MODEL_ORDER]
    all_labels = b_labels + [MODEL_DISPLAY.get(m, m) for m in models]
    n_baseline = len(BASELINE_MODEL_ORDER)
    x = np.arange(len(all_labels))

    panel_h_compact = 3.2  # shorter panels so two rows stay compact
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(15.5, ncols * PANEL_W), panel_h_compact * nrows + 0.6),
        sharey=False,
    )
    axes = np.atleast_2d(axes)

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if dataset is None:
            ax.set_visible(False)
            continue

        blines = baseline_lookup.get(dataset, {})
        all_means, all_stds, all_colors = [], [], []

        for bk in BASELINE_MODEL_ORDER:
            entry = blines.get(bk, (None, None))
            y, s = entry if isinstance(entry, tuple) else (entry, None)
            all_means.append(y if (y is not None and np.isfinite(y)) else np.nan)
            all_stds.append(s if (s is not None and np.isfinite(s)) else 0.0)
            all_colors.append(BASELINE_HLINE_COLOR[bk])

        for model in models:
            r = slice_best[
                (slice_best['data_name'] == dataset) & (slice_best['model_name'] == model)
            ]
            all_means.append(float(r.iloc[0]['metric_mean']) if not r.empty else np.nan)
            all_stds.append(float(r.iloc[0]['metric_std']) if not r.empty else 0.0)
            all_colors.append(MODEL_COLORS.get(model, '#888888'))

        ax.bar(
            x,
            all_means,
            yerr=all_stds,
            width=BAR_WIDTH,
            color=all_colors,
            edgecolor='black',
            linewidth=BAR_LW,
            capsize=BAR_CAPSIZE,
            zorder=3,
            error_kw={'elinewidth': 1.2, 'capthick': 1.2},
        )

        ax.axvline(n_baseline - 0.5, color='#aaaaaa', linewidth=0.9, linestyle='--', zorder=4)

        ax.set_xticks(x)
        # Show x-labels only on the bottom row of each column
        bottom_row = (row == nrows - 1) or all(
            datasets[r2 * ncols + col] is None
            for r2 in range(row + 1, nrows)
            if r2 * ncols + col < len(datasets)
        )
        if bottom_row:
            ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=FS_XTICK)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0)

        ax.tick_params(axis='y', labelsize=FS_YTICK)
        ax.set_title(
            DATASET_DISPLAY.get(dataset, dataset), fontsize=FS_DSTITLE, fontweight='bold', pad=4
        )
        ax.grid(axis='y', linestyle='--', alpha=0.30)
        ax.set_axisbelow(True)
        finite = [m - s for m, s in zip(all_means, all_stds, strict=True) if np.isfinite(m)]
        finite += [m + s for m, s in zip(all_means, all_stds, strict=True) if np.isfinite(m)]
        ax.set_ylim(_ylim(finite))
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=FS_YLABEL)

    _fit_suptitle(fig, axes.flat, title, wspace=WSPACE, hspace=0.32)
    _save(fig, out_dir / out_name)


def plot_ablation_grouped(
    best: pd.DataFrame,
    *,
    factor: str,
    levels: list[str],
    labels: dict[str, str],
    hatches: dict[str, str],
    colors: dict[str, str],
    title: str,
    out_name: str,
    out_dir: Path,
    exclude_mlp: bool = False,
    ylabel: str = 'Test F1 macro (fold mean ± std)',
) -> None:
    """Grouped bar chart: two factor levels for each (model × dataset)."""
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
                    color=colors.get(level, MODEL_COLORS.get(model, '#888888')),
                    edgecolor='black',
                    linewidth=BAR_LW,
                    hatch=hatches.get(level, ''),
                    alpha=0.95 if r_idx == 0 else 0.7,
                    capsize=BAR_CAPSIZE,
                    zorder=3,
                    error_kw={'elinewidth': 1.2, 'capthick': 1.2},
                )
                ylim_vals.extend([mu - sig, mu + sig])

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY.get(m, m) for m in models], rotation=45, ha='right', fontsize=FS_XTICK
        )
        ax.tick_params(axis='y', labelsize=FS_YTICK)
        ax.set_title(
            DATASET_DISPLAY.get(dataset, dataset), fontsize=FS_DSTITLE, fontweight='bold', pad=5
        )
        ax.grid(axis='y', linestyle='--', alpha=0.30)
        ax.set_axisbelow(True)
        ax.set_ylim(_ylim(ylim_vals))
        if ax is axes[0]:
            ax.set_ylabel(ylabel, fontsize=FS_YLABEL)

    handles = [
        mpatches.Patch(
            facecolor=colors.get(level, '#aaaaaa'),
            edgecolor='black',
            hatch=hatches.get(level, ''),
            alpha=0.95 if i == 0 else 0.7,
            label=labels[level],
        )
        for i, level in enumerate(levels)
    ]
    axes[-1].legend(handles=handles, loc='lower right', fontsize=FS_YTICK)
    _fit_suptitle(fig, axes, title, wspace=WSPACE)
    _save(fig, out_dir / out_name)


def _plot_paired_gnn_factor(
    best: pd.DataFrame,
    out_dir: Path,
    *,
    factor: str,
    levels: list[str],
    hatches: dict[str, str],
    alphas: dict[str, str | float],
    legend_labels: dict[str, str],
    out_name: str,
    title: str,
) -> None:
    """GNN-only paired bars (no baselines): model color + hatch for the two factor levels."""
    if factor not in best.columns:
        print(f'  [skip {out_name}] no {factor!r} column')
        return

    frame = best.loc[best['model_name'] != 'mlp'].copy()
    frame = frame[frame[factor].isin(levels)]
    if frame.empty:
        print(f'  [skip {out_name}] no GNN rows for {factor}')
        return

    cells = _pick_best_val(frame, ['data_name', 'model_name', factor])
    models = [m for m in _ordered_models(cells) if m != 'mlp']
    datasets = _ordered_datasets(cells)
    bar_w = 0.35
    x = np.arange(len(models))

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(max(18, 4.5 * len(datasets)), 5.15),
        sharey=False,
    )
    axes = np.atleast_1d(axes)

    for j, (ax, dataset) in enumerate(zip(axes, datasets, strict=True)):
        ylim_vals: list[float] = []
        for m_idx, model in enumerate(models):
            present: list[tuple[str, float, float]] = []
            for level in levels:
                row = cells[
                    (cells['data_name'] == dataset)
                    & (cells['model_name'] == model)
                    & (cells[factor] == level)
                ]
                if row.empty:
                    continue
                present.append(
                    (
                        level,
                        float(row.iloc[0]['metric_mean']),
                        float(row.iloc[0]['metric_std']),
                    )
                )
            if not present:
                continue
            if len(present) == 2:
                for r_idx, (level, mu, sig) in enumerate(present):
                    ax.bar(
                        x[m_idx] + (r_idx - 0.5) * bar_w,
                        mu,
                        yerr=sig,
                        width=bar_w,
                        capsize=3,
                        color=MODEL_COLORS.get(model, '#888888'),
                        edgecolor='black',
                        linewidth=1.5,
                        hatch=hatches[level],
                        alpha=float(alphas[level]),
                        zorder=3,
                        error_kw={'elinewidth': 1, 'capthick': 1},
                    )
                    ylim_vals.extend([mu - sig, mu + sig])
            else:
                level, mu, sig = present[0]
                ax.bar(
                    x[m_idx],
                    mu,
                    yerr=sig,
                    width=bar_w * 1.5,
                    capsize=3,
                    color=MODEL_COLORS.get(model, '#888888'),
                    edgecolor='black',
                    linewidth=1.5,
                    hatch=hatches[level],
                    alpha=float(alphas[level]),
                    zorder=3,
                    error_kw={'elinewidth': 1, 'capthick': 1},
                )
                ylim_vals.extend([mu - sig, mu + sig])

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY.get(m, m) for m in models], rotation=45, ha='right', fontsize=FS_XTICK
        )
        ax.tick_params(axis='y', labelsize=FS_YTICK)
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=23, fontweight='bold', pad=5)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.grid(axis='y', linestyle='--', alpha=0.30)
        ax.set_axisbelow(True)
        ax.set_ylim(_ylim(ylim_vals))
        if j == 0:
            ax.set_ylabel('F1 Macro Score', fontsize=FS_YLABEL)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    cx = _axes_center_x(fig, axes)
    fig.suptitle(title, fontsize=23, fontweight='bold', x=cx, y=1.0)
    fig.legend(
        handles=[
            mpatches.Patch(
                facecolor='lightgray',
                edgecolor='black',
                hatch=hatches[level],
                alpha=float(alphas[level]),
                label=legend_labels[level],
            )
            for level in levels
        ],
        loc='upper center',
        bbox_to_anchor=(cx, 0.968),
        ncol=2,
        fontsize=18,
        frameon=True,
    )
    _save(fig, out_dir / out_name)


def plot_readout_effect(
    best: pd.DataFrame,
    out_dir: Path,
    *,
    out_name: str = 'readout_effect_test_f1',
    title: str = 'Effect of Readout on Model Performance',
) -> None:
    _plot_paired_gnn_factor(
        best,
        out_dir,
        factor='experiment',
        levels=['omics_readout', 'no_readout'],
        hatches={'omics_readout': '', 'no_readout': '.'},
        alphas={'omics_readout': 0.9, 'no_readout': 0.6},
        legend_labels={'omics_readout': 'MLP Readout', 'no_readout': 'Vanilla Readout'},
        out_name=out_name,
        title=title,
    )


def plot_edge_construction_effect(
    best: pd.DataFrame,
    out_dir: Path,
    *,
    out_name: str = 'adjacency_effect_test_f1',
    title: str = 'Effect of Edge Construction on Model Performance',
) -> None:
    _plot_paired_gnn_factor(
        best,
        out_dir,
        factor='adjacency_method',
        levels=['string', 'wgcna'],
        hatches={'string': '', 'wgcna': '///'},
        alphas={'string': 0.9, 'wgcna': 0.6},
        legend_labels={'string': 'PPI', 'wgcna': 'Co-expression'},
        out_name=out_name,
        title=title,
    )


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def _fmt_cell(mean: float, std: float, bold: bool, within_std: bool) -> str:
    """Format a table cell as ``0.xxx \\pm 0.xxx`` with optional markup."""
    body = f'{mean:.3f} $\\pm$ {std:.3f}'
    if bold:
        return f'\\textbf{{{body}}}'
    if within_std:
        return f'\\withinstd {body}'
    return body


def _ablation_table_latex(
    df: pd.DataFrame,
    *,
    dataset: str,
    factor: str,
    factor_labels: dict[str, str],
    level_order: list[str],
    caption: str,
    label: str,
) -> str:
    """Generate a LaTeX table for one dataset's ablation.

    ``df`` must already be annotated (has model_name, data_name, factor col,
    and all TABLE_METRICS mean/std columns).
    """
    sub = df[df['data_name'] == dataset].copy()
    if sub.empty:
        return ''

    # For each (model_name, level), pick the row with highest val F1
    slice_best = pick_best(sub, ['model_name', factor])

    models_present = [m for m in CANONICAL_MODEL_ORDER if m in slice_best['model_name'].values]

    header_metric_cols = ' & '.join(h for h, _, _ in TABLE_METRICS)
    lines: list[str] = [
        r'\begin{table*}[!h]',
        r'\centering',
        r'\small',
        r'\setlength{\tabcolsep}{5pt}',
        r'\renewcommand{\arraystretch}{1.15}',
        f'\\caption{{{caption}}}',
        r'\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{2.4cm} l l Y Y Y Y}',
        r'\toprule',
        f'Dataset & Model & {list(factor_labels.values())[0].split("/")[0].strip()} '
        f'& {header_metric_cols} \\\\',
        r'\midrule',
        f'\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{DATASET_DISPLAY.get(dataset, dataset)}}}}}\\\\[-0.6ex]',
    ]

    for model in models_present:
        mdisplay = MODEL_DISPLAY.get(model, model)
        rows_model: list[list[str]] = []
        for level in level_order:
            row = slice_best[(slice_best['model_name'] == model) & (slice_best[factor] == level)]
            if row.empty:
                rows_model.append(None)
            else:
                rows_model.append(row.iloc[0])

        # Build row strings: one pass per level
        for i, (level, r) in enumerate(zip(level_order, rows_model, strict=True)):
            model_label = mdisplay if i == 0 else ''
            level_label = factor_labels.get(level, level)
            cells: list[str] = []

            for _, mean_col, std_col in TABLE_METRICS:
                if r is None:
                    cells.append('---')
                    continue
                m_val = (
                    float(r[mean_col]) if mean_col in r.index and pd.notna(r[mean_col]) else np.nan
                )
                s_val = (
                    float(r[std_col]) if std_col in r.index and pd.notna(r[std_col]) else np.nan
                )

                # Recompute bold/within-std for this metric
                other_r = rows_model[1 - i] if len(rows_model) == 2 else None
                best_m = m_val
                win_std = s_val
                if other_r is not None and mean_col in other_r.index:
                    other_m = float(other_r[mean_col]) if pd.notna(other_r[mean_col]) else np.nan
                    if np.isfinite(other_m) and other_m > m_val:
                        best_m = other_m
                        other_s = (
                            float(other_r[std_col])
                            if std_col in other_r.index and pd.notna(other_r[std_col])
                            else np.nan
                        )
                        win_std = other_s

                bold_cell = np.isfinite(m_val) and m_val == best_m
                within = (
                    not bold_cell
                    and np.isfinite(m_val)
                    and np.isfinite(win_std)
                    and m_val >= best_m - win_std
                )
                cells.append(
                    _fmt_cell(m_val, s_val, bold_cell, within) if np.isfinite(m_val) else '---'
                )

            row_str = f' & {model_label} & {level_label} & ' + ' & '.join(cells) + r' \\'
            lines.append(row_str)

        lines.append(r'\midrule')

    # Remove trailing \midrule, replace with \bottomrule
    if lines[-1] == r'\midrule':
        lines[-1] = r'\bottomrule'

    lines += [
        r'\end{tabularx}',
        f'\\label{{{label}}}',
        r'\end{table*}',
    ]
    return '\n'.join(lines)


def write_latex_tables(
    df: pd.DataFrame,
    *,
    factor: str,
    factor_labels: dict[str, str],
    level_order: list[str],
    caption_template: str,  # use {dataset} placeholder
    label_template: str,  # use {dataset} placeholder
    out_dir: Path,
    filename_prefix: str,
) -> None:
    """Write one .tex file per dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if factor not in df.columns:
        print(f'  [skip tables] column {factor!r} not in DataFrame')
        return
    datasets = [d for d in DATASET_ORDER if d in df['data_name'].unique()]
    datasets += [d for d in sorted(df['data_name'].unique()) if d not in datasets]
    for dataset in datasets:
        tex = _ablation_table_latex(
            df,
            dataset=dataset,
            factor=factor,
            factor_labels=factor_labels,
            level_order=level_order,
            caption=caption_template.format(dataset=DATASET_DISPLAY.get(dataset, dataset)),
            label=label_template.format(dataset=dataset),
        )
        if tex:
            path = out_dir / f'{filename_prefix}_{dataset}.tex'
            path.write_text(tex)
            print(f'  wrote {path}')


# ---------------------------------------------------------------------------
# Mega ratio × method grid
# ---------------------------------------------------------------------------


def _ordered_methods(frame: pd.DataFrame) -> list[str]:
    present = {str(m) for m in frame['sampling_method'].dropna().unique()}
    return [m for m in METHOD_ORDER if m in present] + sorted(present - set(METHOD_ORDER))


def _ordered_ratios(frame: pd.DataFrame) -> list[float]:
    return sorted(
        pd.to_numeric(frame['node_sample_ratio'], errors='coerce').dropna().unique().tolist()
    )


def _pick_best_val(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Best remaining hyper/factor setting per group, selected on validation F1."""
    work = frame.copy()
    if 'val_f1_mean' in work.columns:
        work['_sel'] = pd.to_numeric(work['val_f1_mean'], errors='coerce')
    else:
        work['_sel'] = pd.to_numeric(work['metric_mean'], errors='coerce')
    return (
        work.sort_values('_sel', ascending=False)
        .drop_duplicates(group_cols, keep='first')
        .reset_index(drop=True)
    )


def _baseline_cell(
    df_b: pd.DataFrame,
    dataset: str,
    method: str,
    ratio: float,
    model_type: str,
) -> tuple[float, float]:
    if df_b.empty:
        return np.nan, np.nan
    rows = df_b[(df_b['dataset'] == dataset) & (df_b['model_type'] == model_type)]
    if 'feature_method' in rows.columns:
        rows = rows[rows['feature_method'].astype(str) == str(method)]
    if 'node_sample_ratio' in rows.columns:
        rows = rows[
            np.isclose(pd.to_numeric(rows['node_sample_ratio'], errors='coerce'), float(ratio))
        ]
    if rows.empty:
        return np.nan, np.nan
    row = rows.iloc[0]
    mu = float(row[_B_TEST]) if pd.notna(row.get(_B_TEST)) else np.nan
    sig = (
        float(row[_B_TEST_STD]) if _B_TEST_STD in row.index and pd.notna(row[_B_TEST_STD]) else 0.0
    )
    return mu, sig


def _axes_center_x(fig, axes) -> float:
    """Horizontal center of the visible axes grid (not the full figure)."""
    fig.canvas.draw()
    xs0, xs1 = [], []
    for ax in np.atleast_1d(axes).ravel():
        if not ax.get_visible():
            continue
        pos = ax.get_position()
        xs0.append(pos.x0)
        xs1.append(pos.x1)
    if not xs0:
        return 0.5
    return 0.5 * (min(xs0) + max(xs1))


def _format_ratio_tick(ratio: float) -> str:
    if float(ratio).is_integer() or abs(ratio - round(ratio)) < 1e-9:
        return f'{ratio:.1f}' if ratio != int(ratio) else f'{int(ratio)}.0'
    return f'{ratio:g}'


def _draw_mega_panel_frame(
    ax,
    *,
    row_i: int,
    col_j: int,
    n_rows: int,
    dataset: str,
    method: str,
    ratios: list[float],
    x_groups: np.ndarray,
    ylim: tuple[float, float],
) -> None:
    ax.set_xticks(x_groups)
    if row_i == n_rows - 1:
        ax.set_xticklabels([_format_ratio_tick(r) for r in ratios], fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(*ylim)
    if col_j > 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')
    if row_i == 0:
        ax.set_title(
            METHOD_DISPLAY.get(method, method.replace('_', ' ').title()),
            fontsize=21,
            fontweight='bold',
            pad=8,
        )
    if col_j == 0:
        ax.text(
            -0.30,
            0.5,
            DATASET_DISPLAY.get(dataset, dataset),
            transform=ax.transAxes,
            fontsize=21,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=90,
        )
        ax.text(
            -0.16,
            0.5,
            'F1 Macro Score',
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=90,
        )


def plot_mega_method_ratio(
    best: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: Path,
    *,
    out_name: str = 'best_performance_by_dataset_method_ratio',
    title: str = 'Best Model Performance by Dataset, Node Selection Method and Sampling Ratio',
) -> None:
    """Dense grid: rows=datasets, cols=methods, grouped bars=baselines + models × ratios."""
    if 'sampling_method' not in best.columns or 'node_sample_ratio' not in best.columns:
        print('  [skip mega] sampling_method / node_sample_ratio missing')
        return

    cells = _pick_best_val(
        best, ['data_name', 'model_name', 'sampling_method', 'node_sample_ratio']
    )
    models = _ordered_models(cells)
    datasets = _ordered_datasets(cells)
    methods = _ordered_methods(cells)
    ratios = _ordered_ratios(cells)
    if not methods or not ratios:
        print('  [skip mega] no methods or ratios')
        return

    plot_keys = list(BASELINE_MODEL_ORDER) + list(models)
    n_keys = max(len(plot_keys), 1)
    bar_w = 0.9 / n_keys
    group_sp = 1.1

    row_lims: dict[str, tuple[float, float]] = {}
    for dataset in datasets:
        vals: list[float] = []
        for method in methods:
            for ratio in ratios:
                for model in models:
                    row = cells[
                        (cells['data_name'] == dataset)
                        & (cells['model_name'] == model)
                        & (cells['sampling_method'] == method)
                        & np.isclose(cells['node_sample_ratio'], ratio)
                    ]
                    if not row.empty:
                        mu = float(row.iloc[0]['metric_mean'])
                        sig = float(row.iloc[0]['metric_std'])
                        vals.extend([mu - sig, mu + sig])
                for bk in BASELINE_MODEL_ORDER:
                    mu, sig = _baseline_cell(df_b, dataset, method, ratio, bk)
                    if np.isfinite(mu):
                        vals.extend([mu - (sig or 0.0), mu + (sig or 0.0)])
        row_lims[dataset] = _ylim(vals)

    fig, axes = plt.subplots(
        len(datasets),
        len(methods),
        figsize=(17.5, max(8.0, 2.55 * len(datasets) + 2.1)),
    )
    axes = np.atleast_2d(axes)

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            x_groups = np.arange(len(ratios)) * group_sp
            x_by_key = {
                k: x_groups + (ii - (n_keys - 1) / 2) * bar_w for ii, k in enumerate(plot_keys)
            }
            for bk in BASELINE_MODEL_ORDER:
                for ri, ratio in enumerate(ratios):
                    mu, sig = _baseline_cell(df_b, dataset, method, ratio, bk)
                    if not np.isfinite(mu):
                        continue
                    ax.bar(
                        x_by_key[bk][ri],
                        mu,
                        yerr=0.0 if not np.isfinite(sig) else sig,
                        width=bar_w,
                        capsize=2,
                        color=BASELINE_HLINE_COLOR[bk],
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.95,
                        zorder=3,
                        error_kw={'elinewidth': 1, 'capthick': 1},
                    )
            for model in models:
                for ri, ratio in enumerate(ratios):
                    row = cells[
                        (cells['data_name'] == dataset)
                        & (cells['model_name'] == model)
                        & (cells['sampling_method'] == method)
                        & np.isclose(cells['node_sample_ratio'], ratio)
                    ]
                    if row.empty:
                        continue
                    mu = float(row.iloc[0]['metric_mean'])
                    sig = float(row.iloc[0]['metric_std'])
                    ax.bar(
                        x_by_key[model][ri],
                        mu,
                        yerr=sig,
                        width=bar_w,
                        capsize=2,
                        color=MODEL_COLORS.get(model, '#888888'),
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.9,
                        zorder=3,
                        error_kw={'elinewidth': 1, 'capthick': 1},
                    )
            _draw_mega_panel_frame(
                ax,
                row_i=i,
                col_j=j,
                n_rows=len(datasets),
                dataset=dataset,
                method=method,
                ratios=ratios,
                x_groups=x_groups,
                ylim=row_lims[dataset],
            )

    handles = [
        mpatches.Patch(
            facecolor=BASELINE_HLINE_COLOR[bk],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.95,
            label=BASELINE_DISPLAY[bk],
        )
        for bk in BASELINE_MODEL_ORDER
    ] + [
        mpatches.Patch(
            facecolor=MODEL_COLORS.get(m, '#888888'),
            edgecolor='black',
            linewidth=0.5,
            alpha=0.9,
            label=MODEL_DISPLAY.get(m, m),
        )
        for m in models
    ]
    fig.tight_layout(rect=[0.04, 0.028, 1, 0.915])
    fig.subplots_adjust(hspace=0.12)
    cx = _axes_center_x(fig, axes)
    fig.suptitle(title, fontsize=FS_TITLE, fontweight='bold', x=cx, y=0.965)
    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(cx, 0.952),
        ncol=min(len(handles), 10),
        fontsize=16,
        frameon=True,
    )
    fig.supxlabel('Node Sampling Ratio', fontsize=21, y=0.018, x=cx)
    _save(fig, out_dir / out_name)


def plot_mega_method_ratio_collapsed(
    best: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: Path,
    *,
    out_name: str = 'best_performance_by_dataset_method_ratio_collapsed',
    title: str = 'Best Model Performance by Dataset, Node Selection Method and Sampling Ratio',
) -> None:
    """Same grid, but GNNs collapse to one bar: the best-val graph model in that cell."""
    if 'sampling_method' not in best.columns or 'node_sample_ratio' not in best.columns:
        print('  [skip collapsed mega] sampling_method / node_sample_ratio missing')
        return

    cells = _pick_best_val(
        best, ['data_name', 'model_name', 'sampling_method', 'node_sample_ratio']
    )
    mlp_cells = cells.loc[cells['model_name'] == 'mlp'].copy()
    graph_cells = _pick_best_val(
        cells.loc[cells['model_name'] != 'mlp'],
        ['data_name', 'sampling_method', 'node_sample_ratio'],
    )
    datasets = _ordered_datasets(cells)
    methods = _ordered_methods(cells)
    ratios = _ordered_ratios(cells)
    if not methods or not ratios:
        print('  [skip collapsed mega] no methods or ratios')
        return

    plot_keys = list(BASELINE_MODEL_ORDER) + ['mlp', 'graph_based']
    n_keys = len(plot_keys)
    bar_w = 0.72 / n_keys
    group_sp = 1.15
    key_colors = {
        'calibrated_svm': BASELINE_HLINE_COLOR['calibrated_svm'],
        'elastic_net': BASELINE_HLINE_COLOR['elastic_net'],
        'mlp': MODEL_COLORS['mlp'],
        'graph_based': GRAPH_BAR_COLOR,
    }

    def _lookup(
        frame: pd.DataFrame, dataset: str, method: str, ratio: float
    ) -> tuple[float, float]:
        if frame.empty:
            return np.nan, np.nan
        row = frame[
            (frame['data_name'] == dataset)
            & (frame['sampling_method'] == method)
            & np.isclose(frame['node_sample_ratio'], ratio)
        ]
        if row.empty:
            return np.nan, np.nan
        return float(row.iloc[0]['metric_mean']), float(row.iloc[0]['metric_std'])

    row_lims: dict[str, tuple[float, float]] = {}
    for dataset in datasets:
        vals: list[float] = []
        for method in methods:
            for ratio in ratios:
                for frame in (mlp_cells, graph_cells):
                    mu, sig = _lookup(frame, dataset, method, ratio)
                    if np.isfinite(mu):
                        vals.extend([mu - sig, mu + sig])
                for bk in BASELINE_MODEL_ORDER:
                    mu, sig = _baseline_cell(df_b, dataset, method, ratio, bk)
                    if np.isfinite(mu):
                        vals.extend([mu - (sig or 0.0), mu + (sig or 0.0)])
        row_lims[dataset] = _ylim(vals)

    fig, axes = plt.subplots(
        len(datasets),
        len(methods),
        figsize=(16.5, max(8.0, 2.55 * len(datasets) + 2.1)),
    )
    axes = np.atleast_2d(axes)

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            x_groups = np.arange(len(ratios)) * group_sp
            x_by_key = {
                k: x_groups + (ii - (n_keys - 1) / 2) * bar_w for ii, k in enumerate(plot_keys)
            }
            lookups = {
                'mlp': lambda r, d=dataset, m=method: _lookup(mlp_cells, d, m, r),
                'graph_based': lambda r, d=dataset, m=method: _lookup(graph_cells, d, m, r),
            }
            for key in plot_keys:
                for ri, ratio in enumerate(ratios):
                    if key in BASELINE_MODEL_ORDER:
                        mu, sig = _baseline_cell(df_b, dataset, method, ratio, key)
                    else:
                        mu, sig = lookups[key](ratio)
                    if not np.isfinite(mu):
                        continue
                    ax.bar(
                        x_by_key[key][ri],
                        mu,
                        yerr=0.0 if not np.isfinite(sig) else sig,
                        width=bar_w,
                        capsize=2,
                        color=key_colors[key],
                        edgecolor='black',
                        linewidth=0.6,
                        alpha=0.93,
                        zorder=3,
                        error_kw={'elinewidth': 1, 'capthick': 1},
                    )
            _draw_mega_panel_frame(
                ax,
                row_i=i,
                col_j=j,
                n_rows=len(datasets),
                dataset=dataset,
                method=method,
                ratios=ratios,
                x_groups=x_groups,
                ylim=row_lims[dataset],
            )

    handles = [
        mpatches.Patch(
            facecolor=BASELINE_HLINE_COLOR['calibrated_svm'],
            edgecolor='black',
            linewidth=0.5,
            label='SVM',
        ),
        mpatches.Patch(
            facecolor=BASELINE_HLINE_COLOR['elastic_net'],
            edgecolor='black',
            linewidth=0.5,
            label='Elastic Net',
        ),
        mpatches.Patch(
            facecolor=MODEL_COLORS['mlp'], edgecolor='black', linewidth=0.5, label='MLP'
        ),
        mpatches.Patch(
            facecolor=GRAPH_BAR_COLOR,
            edgecolor='black',
            linewidth=0.5,
            label='Best graph (val F1)',
        ),
    ]
    fig.tight_layout(rect=[0.04, 0.028, 1, 0.915])
    fig.subplots_adjust(hspace=0.12)
    cx = _axes_center_x(fig, axes)
    fig.suptitle(title, fontsize=FS_TITLE, fontweight='bold', x=cx, y=0.965)
    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(cx, 0.952),
        ncol=4,
        fontsize=16,
        frameon=True,
    )
    fig.supxlabel('Node Sampling Ratio', fontsize=21, y=0.018, x=cx)
    _save(fig, out_dir / out_name)


# ---------------------------------------------------------------------------
# Baseline-only preview
# ---------------------------------------------------------------------------


def plot_baseline_only(
    df_b: pd.DataFrame, out_dir: Path, *, out_name: str = 'baseline_only_test_f1'
) -> None:
    if df_b.empty:
        return
    baseline_lookup = _best_baseline_per_dataset(df_b)
    datasets = [d for d in DATASET_ORDER if d in baseline_lookup]
    datasets += [d for d in sorted(baseline_lookup) if d not in datasets]
    x = np.arange(len(BASELINE_MODEL_ORDER))
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(max(10, 2.8 * len(datasets)), 4.0), sharey=False
    )
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets, strict=True):
        blines = baseline_lookup.get(dataset, {})
        means = [blines.get(bk, (np.nan, None))[0] or np.nan for bk in BASELINE_MODEL_ORDER]
        stds = [blines.get(bk, (np.nan, None))[1] or 0.0 for bk in BASELINE_MODEL_ORDER]
        ax.bar(
            x,
            means,
            yerr=stds,
            color=[BASELINE_HLINE_COLOR[bk] for bk in BASELINE_MODEL_ORDER],
            edgecolor='black',
            linewidth=0.8,
            width=0.5,
            capsize=3,
            error_kw={'elinewidth': 1.2, 'capthick': 1.2},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [BASELINE_DISPLAY.get(bk, bk) for bk in BASELINE_MODEL_ORDER], rotation=30, ha='right'
        )
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)
        finite = [v for v in means if np.isfinite(v)]
        finite += [m + s for m, s in zip(means, stds, strict=True) if np.isfinite(m)]
        ax.set_ylim(_ylim(finite))
        if ax is axes[0]:
            ax.set_ylabel('Test F1 macro (best val config, mean across folds)')
    fig.suptitle(
        'Best baseline per dataset — test F1 (selected on val F1)', fontweight='bold', y=1.02
    )
    fig.tight_layout()
    _save(fig, out_dir / out_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--csv',
        action='append',
        type=Path,
        help='GNN live_test_best_trials.csv (repeat per server)',
    )
    parser.add_argument(
        '--folds-csv',
        action='append',
        type=Path,
        help='live_test_folds.csv (repeat per server); ' 'richer metrics, requires --manifest',
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('tmp/wandb_ckpt_manifest.csv'),
        help='wandb_ckpt_manifest.csv — provides metadata when ' 'loading from --folds-csv',
    )
    parser.add_argument(
        '--best-csv',
        action='append',
        type=Path,
        help='search live_best_trials.csv; required with --ledger',
    )
    parser.add_argument('--ledger', action='append', type=Path)
    parser.add_argument(
        '--baseline-csv', type=Path, help='baseline_best_trials.csv from wandb_baseline_best.py'
    )
    parser.add_argument('--min-folds', type=int, default=5)
    parser.add_argument(
        '--out-dir', type=Path, default=Path('plotting/plots_sep24_test_with_baselines')
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover copied server CSVs if the user did not pass any GNN inputs.
    if not args.csv and not args.folds_csv and not args.ledger:
        results_root = Path('plotting/test_results')
        auto_folds = sorted(results_root.glob('*/live_test_folds.csv'))
        if auto_folds:
            args.folds_csv = auto_folds
            print('Auto-discovered folds CSVs:')
            for p in auto_folds:
                print(f'  {p}')

    # ── Baselines ──────────────────────────────────────────────────────────
    df_b = pd.DataFrame()
    if args.baseline_csv and args.baseline_csv.exists():
        df_b = load_baseline(args.baseline_csv)
        print(
            f'Baseline: {len(df_b)} rows | '
            f'datasets={sorted(df_b["dataset"].unique())} | '
            f'model_types={sorted(df_b["model_type"].unique())}'
        )
        plot_baseline_only(df_b, args.out_dir)
        print('  → baseline_only_test_f1.{pdf,png}')

    # ── GNN test results ───────────────────────────────────────────────────
    best_gnn = load_gnn_test(
        csv_paths=list(args.csv or []),
        folds_paths=list(args.folds_csv or []),
        manifest_path=args.manifest if args.manifest and args.manifest.exists() else None,
        min_folds=args.min_folds,
        ledger_paths=list(args.ledger or []),
    )

    if best_gnn.empty:
        print('No GNN test data — only baseline plot produced.')
        return

    print(
        f'GNN: {len(best_gnn)} studies | '
        f'datasets={sorted(best_gnn.data_name.unique())} | '
        f'models={sorted(best_gnn.model_name.unique())}'
    )

    # ── Plot 1: best overall (single row) ─────────────────────────────────
    plot_best_overall_with_baselines(best_gnn, df_b, args.out_dir)
    print('  → best_overall_test_f1_with_baselines.{pdf,png}')

    # ── Plot 1b: best overall (2-row grid, shared x-ticks) ────────────────
    plot_best_overall_2rows(best_gnn, df_b, args.out_dir)
    print('  → best_overall_test_f1_with_baselines_2rows.{pdf,png}')

    # ── Plot 2: readout ablation (GNN only, no baselines) ─────────────────
    if 'experiment' in best_gnn.columns:
        plot_readout_effect(best_gnn, args.out_dir)
        print('  → readout_effect_test_f1.{pdf,png}')

    # ── Plot 3: edge construction (GNN only, no baselines) ─────────────────
    if 'adjacency_method' in best_gnn.columns:
        plot_edge_construction_effect(best_gnn, args.out_dir)
        print('  → adjacency_effect_test_f1.{pdf,png}')

    # ── Plot 4: mega method × ratio grid (all models) ──────────────────────
    if {'sampling_method', 'node_sample_ratio'} <= set(best_gnn.columns):
        plot_mega_method_ratio(best_gnn, df_b, args.out_dir)
        print('  → best_performance_by_dataset_method_ratio.{pdf,png}')
        plot_mega_method_ratio_collapsed(best_gnn, df_b, args.out_dir)
        print('  → best_performance_by_dataset_method_ratio_collapsed.{pdf,png}')

    # ── Tables ─────────────────────────────────────────────────────────────
    tables_dir = args.out_dir / 'tables'
    has_full_metrics = any(
        c in best_gnn.columns for c in ('test_f1_weighted_mean', 'test_auroc_mean')
    )

    if has_full_metrics:
        if 'adjacency_method' in best_gnn.columns:
            write_latex_tables(
                best_gnn,
                factor='adjacency_method',
                factor_labels=ADJ_DISPLAY,
                level_order=['string', 'wgcna'],
                caption_template=(
                    r'Adjacency ablation on \textit{{{dataset}}}. '
                    r'For each (model, adjacency), the highest val $F_{{macro}}$ configuration '
                    r'is selected; test metrics (mean $\pm$ std) are reported. '
                    r'Higher mean is bolded; within-one-std is shaded.'
                ),
                label_template='tab:{dataset}-adjacency-ppi-vs-coexpression',
                out_dir=tables_dir,
                filename_prefix='adjacency_ablation',
            )
        if 'experiment' in best_gnn.columns:
            write_latex_tables(
                best_gnn,
                factor='experiment',
                factor_labels={'omics_readout': 'OmicsReadOut', 'no_readout': 'NoReadOut'},
                level_order=['omics_readout', 'no_readout'],
                caption_template=(
                    r'Readout ablation on \textit{{{dataset}}}. '
                    r'For each (model, readout), the highest val $F_{{macro}}$ configuration '
                    r'is selected; test metrics (mean $\pm$ std) are reported. '
                    r'Higher mean is bolded; within-one-std is shaded.'
                ),
                label_template='tab:{dataset}-readout-vs-noreadout',
                out_dir=tables_dir,
                filename_prefix='readout_ablation',
            )
    else:
        print(
            '  [skip tables] test_f1_weighted/auroc columns missing — '
            'pass --folds-csv or re-run --export-only after Hall finishes'
        )

    print(f'\nAll outputs written to {args.out_dir.resolve()}')


if __name__ == '__main__':
    main()

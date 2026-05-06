#!/usr/bin/env python3
"""Transform aggregated CSV files to results.json for the webapp leaderboard.

Reads both the GNN results CSV and the baseline CSV, merges them, and for each (data_name,
model_name, adjacency_method, node_sample_ratio, sampling_method, readout_name) combination selects
the hyperparameter config with the best best_val_f1_macro_mean.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MODEL_NAME_MAP: dict[str, str] = {
    'gatv4': 'MLA-GNN',
    'gatv2': 'GATv2',
    'gcn': 'GCN',
    'gin': 'GIN',
    'sage': 'GraphSAGE',
    'sagn': 'SAGN',
    'mlp': 'MLP',
    'chebnet': 'ChebNet',
    'gps': 'GPS',
    'elastic_net': 'ElasticNet',
    'svm': 'SVM',
}

GROUP_COLS = [
    'data_name',
    'model_name',
    'adjacency_method',
    'node_sample_ratio',
    'sampling_method',
    'readout_name',
]

METRIC_PAIRS: list[tuple[str, str, str]] = [
    ('best_val_f1_macro_mean', 'best_val_f1_macro_std', 'val_f1_macro'),
    ('best_test_f1_macro_mean', 'best_test_f1_macro_std', 'test_f1_macro'),
    ('best_train_f1_macro_mean', 'best_train_f1_macro_std', 'train_f1_macro'),
    ('best_test_f1_weighted_mean', 'best_test_f1_weighted_std', 'test_f1_weighted'),
    ('best_test_accuracy_mean', 'best_test_accuracy_std', 'test_accuracy'),
    ('best_test_auroc_mean', 'best_test_auroc_std', 'test_auroc'),
]


def safe_float(val: object, default: float = 0.0) -> float:
    try:
        v = float(val)  # type: ignore[arg-type]
        return default if pd.isna(v) else v
    except (ValueError, TypeError):
        return default


def load_and_prepare_baseline(csv_path: Path) -> pd.DataFrame:
    """Load the baseline CSV and add missing columns to match GNN schema."""
    df = pd.read_csv(csv_path)
    df['adjacency_method'] = 'baseline'
    df['readout_name'] = 'baseline'
    df['node_sample_ratio'] = df['node_sample_ratio'].replace('full', '1.0')
    df['node_sample_ratio'] = df['node_sample_ratio'].astype(float)
    print(f'Loaded {len(df)} baseline rows from {csv_path}')
    return df


def load_and_prepare_gnn(csv_path: Path) -> pd.DataFrame:
    """Load the GNN results CSV."""
    df = pd.read_csv(csv_path)
    df['node_sample_ratio'] = pd.to_numeric(df['node_sample_ratio'], errors='coerce')
    print(f'Loaded {len(df)} GNN rows from {csv_path}')
    return df


def transform_to_json(df: pd.DataFrame, output_path: Path) -> dict:
    """Select best configs and write JSON."""
    print(f'  Datasets: {sorted(df["data_name"].unique())}')
    print(f'  Models: {sorted(df["model_name"].unique())}')

    idx_best = df.groupby(GROUP_COLS)['best_val_f1_macro_mean'].idxmax()
    best = df.loc[idx_best].copy()
    print(f'  Best configs selected: {len(best)}')

    results: dict[str, dict] = {}

    for _, row in best.iterrows():
        dataset = row['data_name']
        model_csv = row['model_name']
        model = MODEL_NAME_MAP.get(model_csv, model_csv)
        method = row['sampling_method']
        readout = row['readout_name']
        ratio = row['node_sample_ratio']
        adj_method = row['adjacency_method']

        graph_config = f'{dataset}|{ratio}|{method}|{readout}'
        if adj_method and adj_method != 'baseline':
            graph_config += f'|{adj_method}'
        key = f'{graph_config}|{model}'

        entry: dict[str, object] = {
            'graph_config': graph_config,
            'model': model,
            'dataset': dataset,
            'readout': readout,
            'node_sample_ratio': safe_float(ratio),
            'method': method,
            'adjacency_method': adj_method if adj_method else 'baseline',
        }

        for csv_mean, csv_std, out_name in METRIC_PAIRS:
            entry[out_name] = safe_float(row.get(csv_mean))
            entry[f'{out_name}_std'] = safe_float(row.get(csv_std))

        results[key] = entry

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        f.write('\n')

    n_baselines = sum(1 for e in results.values() if e['readout'] == 'baseline')
    print(f'Wrote {len(results)} entries to {output_path}')
    print(f'  - GNN entries: {len(results) - n_baselines}')
    print(f'  - Baseline entries: {n_baselines}')
    print(f'  - Models: {sorted({e["model"] for e in results.values()})}')
    print(f'  - Datasets: {sorted({e["dataset"] for e in results.values()})}')

    return results


def main() -> int:
    script_dir = Path(__file__).parent
    webapp_dir = script_dir.parent

    gnn_csv = webapp_dir / 'aggregated_final_results_neurips.csv'
    baseline_csv = webapp_dir / 'baseline_aggregated_gnn_features_neurips.csv'
    output_path = webapp_dir / 'public' / 'data' / 'results.json'

    if not gnn_csv.exists():
        print(f'Error: GNN CSV not found at {gnn_csv}')
        return 1
    if not baseline_csv.exists():
        print(f'Error: Baseline CSV not found at {baseline_csv}')
        return 1

    df_gnn = load_and_prepare_gnn(gnn_csv)
    df_baseline = load_and_prepare_baseline(baseline_csv)

    # Align columns before concat — baseline CSV may lack _bucket_key etc.
    common_cols = list(GROUP_COLS)
    metric_cols = []
    for csv_mean, csv_std, _ in METRIC_PAIRS:
        metric_cols.extend([csv_mean, csv_std])
    keep_cols = common_cols + ['n_runs_seeds'] + metric_cols

    df_gnn_slim = df_gnn[[c for c in keep_cols if c in df_gnn.columns]].copy()
    df_baseline_slim = df_baseline[[c for c in keep_cols if c in df_baseline.columns]].copy()

    df = pd.concat([df_gnn_slim, df_baseline_slim], ignore_index=True)
    print(f'\nMerged: {len(df)} total rows')

    transform_to_json(df, output_path)
    return 0


if __name__ == '__main__':
    exit(main())

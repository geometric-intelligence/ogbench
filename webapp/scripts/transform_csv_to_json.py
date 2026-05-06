#!/usr/bin/env python3
"""Transform aggregated_final_results_neurips.csv to results.json for the webapp leaderboard.

For each (data_name, model_name, adjacency_method, node_sample_ratio, sampling_method,
readout_name) combination, selects the hyperparameter config with the best best_val_f1_macro_mean.
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
}

# ML Baselines with fixed test F1 macro scores (don't depend on graph config)
BASELINES: dict[str, dict[str, dict[str, float]]] = {
    'parkinsons': {'ElasticNet': {'test_f1_macro': 0.64518}, 'SVM': {'test_f1_macro': 0.66422}},
    'motrpac': {'ElasticNet': {'test_f1_macro': 0.57419}, 'SVM': {'test_f1_macro': 0.54287}},
    'addneuromed': {'ElasticNet': {'test_f1_macro': 0.55762}, 'SVM': {'test_f1_macro': 0.46252}},
}

GROUP_COLS = [
    'data_name',
    'model_name',
    'adjacency_method',
    'node_sample_ratio',
    'sampling_method',
    'readout_name',
]


def transform_csv_to_json(csv_path: Path, output_path: Path) -> dict:
    """Transform aggregated CSV to JSON format for the webapp."""
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} rows from {csv_path}')
    print(f'  Datasets: {sorted(df["data_name"].unique())}')
    print(f'  Models: {sorted(df["model_name"].unique())}')

    # For each filter combination, keep the row with highest best_val_f1_macro_mean
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
        if adj_method:
            graph_config += f'|{adj_method}'
        key = f'{graph_config}|{model}'

        def safe_float(val: object, default: float = 0.0) -> float:
            try:
                v = float(val)  # type: ignore[arg-type]
                return default if pd.isna(v) else v
            except (ValueError, TypeError):
                return default

        entry = {
            'graph_config': graph_config,
            'model': model,
            'dataset': dataset,
            'readout': readout,
            'node_sample_ratio': safe_float(ratio),
            'method': method,
            'adjacency_method': adj_method if adj_method else 'string',
            'val_f1_macro': safe_float(row.get('best_val_f1_macro_mean')),
            'val_f1_macro_std': safe_float(row.get('best_val_f1_macro_std')),
            'test_f1_macro': safe_float(row.get('best_test_f1_macro_mean')),
            'test_f1_macro_std': safe_float(row.get('best_test_f1_macro_std')),
            'train_f1_macro': safe_float(row.get('best_train_f1_macro_mean')),
            'train_f1_macro_std': safe_float(row.get('best_train_f1_macro_std')),
        }

        results[key] = entry

    # Add baseline entries for each dataset that has them
    for dataset, models in BASELINES.items():
        for model, metrics in models.items():
            graph_config = f'{dataset}|baseline'
            key = f'{graph_config}|{model}'

            entry = {
                'graph_config': graph_config,
                'model': model,
                'dataset': dataset,
                'readout': 'baseline',
                'node_sample_ratio': 0.0,
                'method': 'baseline',
                'adjacency_method': 'baseline',
                'val_f1_macro': 0.0,
                'val_f1_macro_std': 0.0,
                'test_f1_macro': metrics['test_f1_macro'],
                'test_f1_macro_std': 0.0,
                'train_f1_macro': 0.0,
                'train_f1_macro_std': 0.0,
            }

            results[key] = entry

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    n_baselines = sum(len(m) for m in BASELINES.values())
    print(f'Wrote {len(results)} entries to {output_path}')
    print(f'  - CSV entries: {len(results) - n_baselines}')
    print(f'  - Baseline entries: {n_baselines}')
    print(f'  - Models: {sorted({e["model"] for e in results.values()})}')
    print(f'  - Datasets: {sorted({e["dataset"] for e in results.values()})}')

    return results


def main() -> int:
    script_dir = Path(__file__).parent
    webapp_dir = script_dir.parent

    csv_path = webapp_dir / 'aggregated_final_results_neurips.csv'
    output_path = webapp_dir / 'public' / 'data' / 'results.json'

    if not csv_path.exists():
        print(f'Error: CSV file not found at {csv_path}')
        return 1

    transform_csv_to_json(csv_path, output_path)
    return 0


if __name__ == '__main__':
    exit(main())

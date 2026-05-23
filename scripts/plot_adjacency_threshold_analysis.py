#!/usr/bin/env python3
"""Analyze and plot graph connectivity metrics as a function of adjacency threshold.

This script computes graph statistics (connectivity, degree distribution, etc.) for different
adjacency thresholds and visualizes how they change.
"""

import os
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from huggingface_hub import hf_hub_download
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from tqdm import tqdm

from ogbench.data.adjacency import get_adjacency_builder
from ogbench.data.selectors import get_selector

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / 'configs'


def load_dataset_config(dataset_name: str) -> dict[str, Any]:
    """Load dataset configuration from the YAML config file."""
    config_path = CONFIGS_DIR / 'dataset' / f'{dataset_name}.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    hf_config_path = CONFIGS_DIR / 'hf' / 'default.yaml'
    with open(hf_config_path) as f:
        hf_cfg = yaml.safe_load(f)

    return {
        'data_name': dataset_name,
        'revision': hf_cfg.get('revision', '83299150394717f0646b1bd44d6a55392ab789db'),
        'train_val_test_split': cfg['loader']['parameters']['train_val_test_split'],
    }


DATASET_NAMES = ['addneuromed', 'motrpac', 'parkinsons']
DATASETS = {name: load_dataset_config(name) for name in DATASET_NAMES}

ADJACENCY_METHOD = 'wgcna'
METHODS = ['variance', 'random', 'correlation', 'distance_correlation']
NODE_SAMPLE_RATIOS = [1.0, 0.8, 0.5, 0.3]

HF_REPO_ID = 'geometric-intelligence/ogbench'


def load_and_preprocess_dataset(
    dataset_name: str, config: dict[str, Any], method: str, node_sample_ratio: float
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load dataset and preprocess following the same steps as HFOmicsDataset.

    Returns training data only (after split and imputation).
    """
    print(f'\nLoading {dataset_name} (method={method}, ratio={node_sample_ratio})...')

    # Download data
    data_file = hf_hub_download(  # nosec
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        revision=config['revision'],
        filename=f'{dataset_name}_data.parquet',
    )
    targets_file = hf_hub_download(  # nosec
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        revision=config['revision'],
        filename=f'{dataset_name}_targets.parquet',
    )

    # Load data
    raw_data = pd.read_parquet(data_file)
    targets_df = pd.read_parquet(targets_file)

    if 'target' in raw_data.columns:
        raw_data = raw_data.drop('target', axis=1)

    targets = targets_df['target'].values

    print(f'  Total samples: {len(targets)}, features: {raw_data.shape[1]}')

    # Shuffle and split (same as HFOmicsDataset)
    raw_data, targets = shuffle(raw_data, targets, random_state=42)

    train_val_test_split = config['train_val_test_split']
    train_idx = int(len(targets) * train_val_test_split[0])

    # Get training data only
    train_data = raw_data.iloc[:train_idx]
    train_targets = targets[:train_idx]

    print(f'  Training samples: {len(train_targets)}')

    # Impute missing values (fit on training data)
    imputer = SimpleImputer(strategy='mean')
    train_data_imputed = imputer.fit_transform(train_data)
    train_data = pd.DataFrame(
        train_data_imputed, columns=train_data.columns, index=train_data.index
    )

    # Select nodes based on training data
    n_training_samples = len(train_targets)

    if node_sample_ratio == 'full':
        n_nodes = train_data.shape[1]
    elif isinstance(node_sample_ratio, float | int):
        n_nodes = int(n_training_samples / node_sample_ratio)
        if n_nodes > train_data.shape[1]:
            n_nodes = train_data.shape[1]

    # Select nodes
    selected_nodes = select_nodes(
        train_data.values, train_targets, n_selected=n_nodes, method=method
    )
    train_selected = train_data.iloc[:, selected_nodes]

    print(f'  Selected nodes: {len(selected_nodes)}')

    return train_selected, train_targets


def select_nodes(
    data: np.ndarray, targets: np.ndarray, n_selected: int = 10, method: str = 'variance'
) -> np.ndarray:
    """Select nodes using the same registry as ``HFOmicsDataset`` (variance, correlation, etc.)."""
    selector = get_selector(method)
    return selector.select(data, targets, n_selected)


def compute_continuous_adjacency(node_features: pd.DataFrame) -> np.ndarray:
    """Compute the continuous adjacency matrix (expensive, call once per dataset combo)."""
    builder = get_adjacency_builder(ADJACENCY_METHOD)
    adjacency = builder.build(node_features)
    return np.nan_to_num(adjacency, nan=0.0)


def binarize_adjacency(adjacency: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize a continuous adjacency matrix at the given threshold."""
    adj_matrix = np.where(adjacency > threshold, 1, 0)
    np.fill_diagonal(adj_matrix, 1)
    return adj_matrix


def compute_graph_metrics(adj_matrix: np.ndarray) -> dict[str, float]:
    """Compute various graph metrics from adjacency matrix."""
    n_nodes = adj_matrix.shape[0]

    adj_no_diag = adj_matrix.copy()
    np.fill_diagonal(adj_no_diag, 0)

    node_degrees = np.sum(adj_no_diag, axis=1)
    n_edges = np.sum(adj_no_diag) / 2
    max_edges = n_nodes * (n_nodes - 1) / 2
    connectivity = n_edges / max_edges if max_edges > 0 else 0

    return {
        'n_nodes': n_nodes,
        'n_edges': int(n_edges),
        'connectivity': connectivity,
        'mean_degree': np.mean(node_degrees),
        'median_degree': np.median(node_degrees),
        'std_degree': np.std(node_degrees),
        'min_degree': int(np.min(node_degrees)),
        'max_degree': int(np.max(node_degrees)),
        'n_isolated': int(np.sum(node_degrees == 0)),
    }


def analyze_threshold_sweep(
    dataset_name: str,
    config: dict[str, Any],
    method: str,
    node_sample_ratio: float,
    thresholds: list[float],
) -> pd.DataFrame:
    """Analyze graph metrics across different adjacency thresholds."""
    print(f"\n{'='*70}")
    print(f'Analyzing {dataset_name.upper()} | method={method} | ratio={node_sample_ratio}')
    print(f"{'='*70}")

    train_data, train_targets = load_and_preprocess_dataset(
        dataset_name, config, method, node_sample_ratio
    )

    print(f'\nComputing {ADJACENCY_METHOD} adjacency matrix (once)...')
    continuous_adj = compute_continuous_adjacency(train_data)

    results = []
    print(f'Binarizing at {len(thresholds)} thresholds...')
    for threshold in tqdm(thresholds, desc=f'{dataset_name}/{method}/{node_sample_ratio}'):
        adj_matrix = binarize_adjacency(continuous_adj, threshold)
        metrics = compute_graph_metrics(adj_matrix)
        metrics['threshold'] = threshold
        metrics['dataset'] = dataset_name
        metrics['method'] = method
        metrics['node_sample_ratio'] = node_sample_ratio
        results.append(metrics)

    return pd.DataFrame(results)


def interpolate_threshold_for_target(
    df: pd.DataFrame, target_connectivity: float
) -> dict[str, float]:
    """Interpolate to find precise threshold for target connectivity."""
    df_sorted = df.sort_values('connectivity').reset_index(drop=True)

    below = df_sorted[df_sorted['connectivity'] <= target_connectivity]
    above = df_sorted[df_sorted['connectivity'] >= target_connectivity]

    if len(below) == 0:
        row = df_sorted.iloc[0]
        return {
            'threshold': row['threshold'],
            'connectivity': row['connectivity'],
            'interpolated': False,
        }

    if len(above) == 0:
        row = df_sorted.iloc[-1]
        return {
            'threshold': row['threshold'],
            'connectivity': row['connectivity'],
            'interpolated': False,
        }

    point_below = below.iloc[-1]
    point_above = above.iloc[0]

    t1, c1 = point_below['threshold'], point_below['connectivity']
    t2, c2 = point_above['threshold'], point_above['connectivity']

    if abs(c2 - c1) < 1e-10:
        return {'threshold': t1, 'connectivity': c1, 'interpolated': False}

    interpolated_threshold = t1 + (t2 - t1) * (target_connectivity - c1) / (c2 - c1)
    return {
        'threshold': interpolated_threshold,
        'connectivity': target_connectivity,
        'interpolated': True,
    }


def save_results_csv(all_results: pd.DataFrame, output_dir: str = 'plots') -> None:
    """Save results and print summary table."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'adjacency_threshold_metrics.csv')
    all_results.to_csv(csv_path, index=False)
    print(f'\nSaved results to: {csv_path}')

    target_connectivity = 0.10
    summary_stats = []

    for (dataset, method, ratio), group in all_results.groupby(
        ['dataset', 'method', 'node_sample_ratio']
    ):
        result = interpolate_threshold_for_target(group, target_connectivity)
        closest_idx = (group['connectivity'] - target_connectivity).abs().idxmin()
        closest_row = group.loc[closest_idx]

        summary_stats.append(
            {
                'dataset': dataset,
                'method': method,
                'node_sample_ratio': ratio,
                'interpolated_threshold': result['threshold'],
                'closest_measured_threshold': closest_row['threshold'],
                'closest_measured_connectivity': closest_row['connectivity'],
            }
        )

    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, 'threshold_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved summary to: {summary_path}')

    print(f"\n{'='*70}")
    print('SUMMARY: Recommended Thresholds for ~10% Connectivity')
    print(f"{'='*70}")
    print(f"{'Dataset':<10} {'Method':<22} {'Ratio':<8} {'Threshold':>10} {'Connectivity':>12}")
    print('-' * 65)
    for _, row in summary_df.iterrows():
        print(
            f"{row['dataset']:<10} {row['method']:<22} {row['node_sample_ratio']:<8} "
            f"{row['interpolated_threshold']:>10.4f} "
            f"{row['closest_measured_connectivity']*100:>11.2f}%"
        )


def main() -> None:
    """Main analysis function."""
    print('=' * 70)
    print('ADJACENCY THRESHOLD ANALYSIS')
    print('=' * 70)

    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.01, 20),  # very fine resolution at ultra-low range
                np.linspace(0.01, 0.1, 20),  # fine resolution in low range
                np.linspace(0.1, 0.5, 10),  # medium resolution in mid range
                np.linspace(0.5, 0.9, 5),  # coarse resolution in high range
            ]
        )
    )

    print(f'\nThresholds: {len(thresholds)} values [{thresholds[0]:.2f}, {thresholds[-1]:.2f}]')
    print(f'Methods: {METHODS}')
    print(f'Ratios: {NODE_SAMPLE_RATIOS}')
    print(f'Total combinations: {len(DATASETS) * len(METHODS) * len(NODE_SAMPLE_RATIOS)}')

    all_results = []

    for dataset_name, config in DATASETS.items():
        for method, ratio in product(METHODS, NODE_SAMPLE_RATIOS):
            try:
                df = analyze_threshold_sweep(dataset_name, config, method, ratio, thresholds)
                all_results.append(df)
            except Exception as e:
                print(f'\nError analyzing {dataset_name}/{method}/{ratio}: {e}')
                import traceback

                traceback.print_exc()

    if not all_results:
        print('\nNo results!')
        return

    combined = pd.concat(all_results, ignore_index=True)
    save_results_csv(combined, output_dir='plots/threshold_analysis')

    print('\nAnalysis complete!')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Precompute graph statistics for all parameter combinations.

This script precomputes graph statistics for all combinations of:
- Node sample ratios: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
- Node selection methods: variance, correlation, random
- Adjacency thresholds: 0.02, 0.1, 0.2, 0.3, 0.4, 0.5
- Datasets: motrpac, addneuromed, parkinsons

Results are saved to a JSON file for fast loading by the app.
"""

import itertools
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

# Dataset configurations
DATASETS = {
    'motrpac': {
        'data_name': 'motrpac',
        'revision': '9f052d330ce130408a2c7c347b2ed197154da7c8',
        'train_val_test_split': [0.7, 0.2, 0.1],
    },
    'addneuromed': {
        'data_name': 'addneuromed',
        'revision': '9f052d330ce130408a2c7c347b2ed197154da7c8',
        'train_val_test_split': [0.7, 0.2, 0.1],
    },
    'parkinsons': {
        'data_name': 'parkinsons',
        'revision': '9f052d330ce130408a2c7c347b2ed197154da7c8',
        'train_val_test_split': [0.7, 0.15, 0.15],
    },
}

HF_REPO_ID = '<anonymous>/bgbench'

# Parameter grids
NODE_SAMPLE_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NODE_SELECTION_METHODS = ['variance', 'correlation', 'random']
ADJACENCY_THRESHOLDS = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5]

# Cache for loaded datasets
_data_cache = {}


def load_raw_data(dataset_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load raw dataset from HuggingFace (cached)."""
    if dataset_name in _data_cache:
        return _data_cache[dataset_name]

    print(f'  Loading {dataset_name} from HuggingFace...')
    config = DATASETS[dataset_name]

    data_file = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        revision=config['revision'],
        filename=f'{dataset_name}_data.parquet',
    )
    targets_file = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        revision=config['revision'],
        filename=f'{dataset_name}_targets.parquet',
    )

    raw_data = pd.read_parquet(data_file)
    targets_df = pd.read_parquet(targets_file)

    if 'target' in raw_data.columns:
        raw_data = raw_data.drop('target', axis=1)

    targets = targets_df['target'].values
    raw_data, targets = shuffle(raw_data, targets, random_state=42)

    train_val_test_split = config['train_val_test_split']
    train_idx = int(len(targets) * train_val_test_split[0])

    train_data = raw_data.iloc[:train_idx]
    train_targets = targets[:train_idx]

    imputer = SimpleImputer(strategy='mean')
    train_data_imputed = imputer.fit_transform(train_data)
    train_data = pd.DataFrame(
        train_data_imputed, columns=train_data.columns, index=train_data.index
    )

    print(f'    {dataset_name}: {train_data.shape[0]} samples, {train_data.shape[1]} features')

    _data_cache[dataset_name] = (train_data, train_targets)
    return train_data, train_targets


def select_nodes(
    data: np.ndarray, targets: np.ndarray, n_selected: int, method: str
) -> np.ndarray:
    """Select nodes based on feature importance."""
    np.random.seed(42)

    if method == 'variance':
        variances = np.std(data, axis=0)
        ranked_nodes = np.argsort(variances)[::-1]
    elif method == 'correlation':
        correlations = np.abs(
            np.array([np.corrcoef(data[:, i], targets)[0, 1] for i in range(data.shape[1])])
        )
        correlations = np.nan_to_num(correlations, nan=0.0)
        ranked_nodes = np.argsort(correlations)[::-1]
    elif method == 'random':
        ranked_nodes = np.random.permutation(data.shape[1])
    else:
        raise ValueError(f'Invalid method: {method}')

    return ranked_nodes[:n_selected]


def calculate_adjacency_matrix(
    node_features: pd.DataFrame, adjacency_threshold: float
) -> np.ndarray:
    """Calculate adjacency matrix using correlation-based approach."""
    corr_matrix = node_features.corr().values
    adjacency = np.abs(corr_matrix) ** 6
    adjacency = np.nan_to_num(adjacency, nan=0.0)
    adj_matrix = np.where(adjacency > adjacency_threshold, 1, 0)
    np.fill_diagonal(adj_matrix, 1)
    return adj_matrix


def compute_graph_metrics(adj_matrix: np.ndarray) -> dict[str, float]:
    """Compute various graph metrics from adjacency matrix."""
    n_nodes = adj_matrix.shape[0]

    adj_no_diag = adj_matrix.copy()
    np.fill_diagonal(adj_no_diag, 0)

    graph = nx.from_numpy_array(adj_no_diag)

    node_degrees = np.sum(adj_no_diag, axis=1)
    n_edges = np.sum(adj_no_diag) / 2
    max_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0
    mean_degree = np.mean(node_degrees) if n_nodes > 0 else 0
    std_degree = np.std(node_degrees) if n_nodes > 0 else 0

    n_components = nx.number_connected_components(graph)

    if n_nodes > 0 and n_components > 0:
        largest_cc = max(nx.connected_components(graph), key=len)
        largest_cc_ratio = len(largest_cc) / n_nodes * 100
    else:
        largest_cc_ratio = 0
        largest_cc = set()

    try:
        avg_clustering = nx.average_clustering(graph)
    except Exception:
        avg_clustering = 0

    try:
        if n_components > 0 and len(largest_cc) > 1:
            subgraph = graph.subgraph(largest_cc)
            if len(largest_cc) > 100:
                sample_nodes = list(largest_cc)[: min(50, len(largest_cc))]
                path_lengths = []
                for source in sample_nodes[:25]:
                    lengths = nx.single_source_shortest_path_length(subgraph, source)
                    path_lengths.extend(lengths.values())
                avg_path_length = np.mean(path_lengths) if path_lengths else 0
            else:
                avg_path_length = nx.average_shortest_path_length(subgraph)
        else:
            avg_path_length = 0
    except Exception:
        avg_path_length = 0

    return {
        'n_nodes': n_nodes,
        'n_edges': int(n_edges),
        'density': density * 100,
        'mean_degree': mean_degree,
        'std_degree': std_degree,
        'n_components': n_components,
        'largest_cc_ratio': largest_cc_ratio,
        'avg_clustering': avg_clustering,
        'avg_path_length': avg_path_length,
    }


def get_graph_stats_for_params(
    dataset_name: str, node_sample_ratio: float, method: str, adjacency_threshold: float
) -> dict[str, float]:
    """Compute graph statistics for given parameters."""
    train_data, train_targets = load_raw_data(dataset_name)

    n_training_samples = len(train_targets)
    if node_sample_ratio >= 1.0:
        n_nodes = min(train_data.shape[1], 1000)
    else:
        n_nodes = int(n_training_samples / node_sample_ratio)
        if n_nodes > train_data.shape[1]:
            n_nodes = train_data.shape[1]
        n_nodes = min(n_nodes, 1000)

    selected_nodes = select_nodes(
        train_data.values, train_targets, n_selected=n_nodes, method=method
    )
    train_selected = train_data.iloc[:, selected_nodes]

    adj_matrix = calculate_adjacency_matrix(train_selected, adjacency_threshold)
    metrics = compute_graph_metrics(adj_matrix)
    metrics['dataset'] = dataset_name

    return metrics


def main():
    """Precompute all graph statistics and save to file."""
    output_file = Path(__file__).parent / 'public' / 'data' / 'stats.json'

    print('=' * 60)
    print('Precomputing Graph Statistics')
    print('=' * 60)

    # Calculate total combinations
    total = (
        len(DATASETS)
        * len(NODE_SAMPLE_RATIOS)
        * len(NODE_SELECTION_METHODS)
        * len(ADJACENCY_THRESHOLDS)
    )
    print(f'\nTotal combinations to compute: {total}')
    print(f'  Datasets: {list(DATASETS.keys())}')
    print(f'  Node sample ratios: {NODE_SAMPLE_RATIOS}')
    print(f'  Selection methods: {NODE_SELECTION_METHODS}')
    print(f'  Adjacency thresholds: {ADJACENCY_THRESHOLDS}')

    # Preload all datasets
    print('\n--- Loading datasets ---')
    for dataset_name in DATASETS.keys():
        load_raw_data(dataset_name)

    # Compute all combinations
    print('\n--- Computing statistics ---')
    results = {}
    start_time = time.time()

    combinations = list(
        itertools.product(
            DATASETS.keys(), NODE_SAMPLE_RATIOS, NODE_SELECTION_METHODS, ADJACENCY_THRESHOLDS
        )
    )

    for i, (dataset, ratio, method, threshold) in enumerate(combinations, 1):
        # Create cache key as string for JSON
        cache_key = f'{dataset}|{ratio}|{method}|{threshold}'

        if i % 20 == 0 or i == 1:
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (total - i) if i > 0 else 0
            print(
                f'  [{i}/{total}] ETA: {eta:.0f}s - {dataset}, p={ratio}, {method}, τ={threshold}'
            )

        try:
            stats = get_graph_stats_for_params(dataset, ratio, method, threshold)
            results[cache_key] = stats
        except Exception as e:
            print(f'  ERROR: {dataset}, p={ratio}, {method}, τ={threshold}: {e}')

    # Save results
    print(f'\n--- Saving to {output_file} ---')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f'\nDone! Computed {len(results)} combinations in {elapsed:.1f}s')
    print(f'Results saved to: {output_file}')
    print('=' * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Analyze dataset statistics across different parameters. Generates plots for all combinations of
node_sample_ratio and sampling_method.

Graphs are the fold-0 training graphs from 5-fold CV (same node selection, corrections, grouping,
and split as the Optuna sweep). Website stats therefore describe one canonical fold, not an average
over folds.

Usage examples:
    python dataset_stats_analysis.py
    python dataset_stats_analysis.py --n-jobs 8
    python dataset_stats_analysis.py --datasets addneuromed parkinsons smoking tuberculosis
    python dataset_stats_analysis.py --adj-thresholds 21
    python dataset_stats_analysis.py --skip-plots
    python dataset_stats_analysis.py --node-ratios full 1.0 0.5 --methods variance random
"""

import csv
import itertools
import json
import os
import os.path as osp
import shutil
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
import pandas as pd

# Match configs/dataset/*.yaml so fold-0 caches align with the training pipeline.
K_FOLDS = 5
FOLD = 0
TRAINING_SEED = 42

SPECIES_BY_DATASET = {
    'tuberculosis': 83332,  # M. tuberculosis (STRING adjacency)
}

CORRECTIONS_BY_DATASET: dict[str, list[str]] = {
    'motrpac': ['covariate_adjust'],
    'addneuromed': ['combat'],
    'smoking': ['promoter_min_beta', 'median_center'],
}

GROUPING_BY_DATASET: dict[str, str] = {
    'parkinsons': 'batch',
}


def load_dataset(
    dataset_name: str,
    adj_thresh: float = 0.5,
    node_sample_ratio: str = 'full',
    method: str = 'variance',
    adjacency_method: str = 'wgcna',
    string_data_dir: str | None = None,
    cache_root: str = '/scratch/lcornelis/ogbench-1/run_data/omics',
) -> Any:
    """Load the fold-0 k-fold graph for the given parameters."""
    from ogbench.data.datasets.hf_omics import HFOmicsDataset

    # Pass 'full' as string, not None, because HFOmicsDataset checks for 'full' string
    ratio_value = 'full' if node_sample_ratio == 'full' else float(node_sample_ratio)
    np.random.seed(TRAINING_SEED)
    dataset = HFOmicsDataset(
        root=cache_root,
        data_name=dataset_name,
        method=method,
        adjacency_threshold=adj_thresh,
        node_sample_ratio=ratio_value,
        train_val_test_split=[0.7, 0.15, 0.15],
        imputation_method='mean',
        adjacency_method=adjacency_method,
        string_data_dir=string_data_dir,
        species=SPECIES_BY_DATASET.get(dataset_name, 9606),
        split_type='k-fold',
        k=K_FOLDS,
        fold=FOLD,
        corrections=CORRECTIONS_BY_DATASET.get(dataset_name, []),
        grouping=GROUPING_BY_DATASET.get(dataset_name),
    )

    return dataset


def _train_idx(dataset: Any) -> int:
    """Number of training samples in the reordered train|val|test cache."""
    split_path = osp.join(dataset.raw_dir, 'split_info.json')
    with open(split_path) as f:
        return int(json.load(f)['train_idx'])


def _compute_feature_homophily(graph: nx.Graph, node_features: np.ndarray) -> float:
    """Average cosine similarity of node features along graph edges."""
    if graph.number_of_edges() == 0 or node_features.shape[0] == 0:
        return 0.0

    norms = np.linalg.norm(node_features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = node_features / norms

    total_similarity = 0.0
    for u, v in graph.edges():
        total_similarity += float(np.dot(normalized[u], normalized[v]))
    return total_similarity / graph.number_of_edges()


def _compute_lcc_metrics(largest_cc_graph: nx.Graph) -> dict[str, float]:
    """Compute structural metrics on the largest connected component.

    Clustering uses random wedge sampling. Diameter uses NetworkX's 2-sweep approximation (O(n + m)
    lower bound; often exact in practice). Modularity uses Louvain community detection.
    """
    lcc_size = largest_cc_graph.number_of_nodes()
    if lcc_size == 0:
        return {
            'clustering_coefficient': 0.0,
            'diameter': float('nan'),
            'modularity': 0.0,
        }

    clustering_coefficient = nx.approximation.average_clustering(
        largest_cc_graph, trials=10_000, seed=0
    )

    if lcc_size > 1:
        try:
            # 2-sweep approximation: accepted efficient diameter lower bound
            diameter = float(nx.approximation.diameter(largest_cc_graph, seed=0))
        except (nx.NetworkXError, nx.NetworkXPointlessConcept):
            diameter = float('nan')
    else:
        diameter = 0.0

    try:
        # Louvain: standard efficient modularity maximization
        communities = nx.community.louvain_communities(largest_cc_graph, seed=0)
        modularity = float(nx.community.modularity(largest_cc_graph, communities))
    except (nx.NetworkXError, nx.NetworkXPointlessConcept, ValueError):
        modularity = float('nan')

    return {
        'clustering_coefficient': clustering_coefficient,
        'diameter': diameter,
        'modularity': modularity,
    }


def get_graph_stats(
    dataset: Any, node_features: np.ndarray | None = None
) -> dict[str, float]:
    """Get statistics of the graph from the dataset."""
    empty_stats = {
        'num_nodes': 0,
        'num_edges': 0,
        'avg_degree': 0.0,
        'density_pct': 0.0,
        'largest_cc_ratio_pct': 0.0,
        'num_connected_components': 0,
        'degree_std': 0.0,
        'clustering_coefficient': 0.0,
        'diameter': float('nan'),
        'modularity': 0.0,
        'homophily': float('nan'),
    }

    try:
        if hasattr(dataset, 'data') and dataset.data is not None:
            data = dataset[0]
            edge_index = data.edge_index
            num_nodes = data.x.shape[0]
            edge_list = edge_index.t().numpy()
            graph = nx.Graph()
            graph.add_nodes_from(range(num_nodes))
            graph.add_edges_from(edge_list)
        else:
            name = osp.join(dataset.raw_dir, 'adj_matrix.npy')
            try:
                adj_matrix = np.load(name)
                graph = nx.from_numpy_array(adj_matrix)
                graph.remove_edges_from(nx.selfloop_edges(graph))
            except FileNotFoundError:
                print(f'Warning: Adjacency matrix not found at {name}')
                return empty_stats

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        if num_nodes == 0:
            return empty_stats

        degrees = [d for _, d in graph.degree()]
        avg_degree = np.mean(degrees)
        degree_std = np.std(degrees)
        density_pct = nx.density(graph) * 100

        connected_components = list(nx.connected_components(graph))
        num_connected_components = len(connected_components)
        largest_cc = max(connected_components, key=len)
        largest_cc_ratio_pct = (len(largest_cc) / num_nodes) * 100
        largest_cc_graph = graph.subgraph(largest_cc)

        lcc_metrics = _compute_lcc_metrics(largest_cc_graph)
        homophily = (
            _compute_feature_homophily(graph, node_features)
            if node_features is not None
            else float('nan')
        )

        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'density_pct': density_pct,
            'largest_cc_ratio_pct': largest_cc_ratio_pct,
            'num_connected_components': num_connected_components,
            'degree_std': degree_std,
            'homophily': homophily,
            **lcc_metrics,
        }

    except Exception as e:
        print(f'Error getting graph stats: {e}')
        return empty_stats


def process_single_combination(
    args_tuple: tuple[str, str, str, float, str, str | None, str, bool],
) -> dict[str, Any]:
    """Process a single parameter combination for parallel processing."""
    (
        dataset_name,
        node_ratio,
        method,
        adj_thresh,
        adjacency_method,
        string_data_dir,
        cache_root,
        keep_cache,
    ) = args_tuple
    dataset = None
    dataset_dir = None

    try:
        dataset = load_dataset(
            dataset_name,
            adj_thresh,
            node_ratio,
            method,
            adjacency_method,
            string_data_dir=string_data_dir,
            cache_root=cache_root,
        )
        dataset_dir = dataset.get_data_dir()
        print(f'Dataset loaded: {dataset}, length: {len(dataset)}')

        node_features = None
        if len(dataset) > 0:
            n_train = _train_idx(dataset)
            sample_features = [dataset[i].x.numpy() for i in range(n_train)]
            node_features = np.mean(sample_features, axis=0)

        stats = get_graph_stats(dataset, node_features=node_features)
        stats.update({
            'dataset': dataset_name,
            'adj_thresh': adj_thresh,
            'node_sample_ratio': node_ratio,
            'method': method,
            'adjacency_method': adjacency_method,
        })
        return stats

    except Exception as e:
        return {
            'dataset': dataset_name,
            'adj_thresh': adj_thresh,
            'node_sample_ratio': node_ratio,
            'method': method,
            'adjacency_method': adjacency_method,
            'num_nodes': None,
            'num_edges': None,
            'avg_degree': None,
            'density_pct': None,
            'largest_cc_ratio_pct': None,
            'num_connected_components': None,
            'degree_std': None,
            'clustering_coefficient': None,
            'diameter': None,
            'modularity': None,
            'homophily': None,
            'error': str(e),
        }
    finally:
        if dataset_dir is not None and not keep_cache:
            # Keep the shared STRING cache at cache_root/string_cache, but remove
            # this combination's dense adjacency matrix and processed dataset.
            del dataset
            try:
                shutil.rmtree(dataset_dir)
                print(f'Removed temporary dataset cache: {dataset_dir}')
            except OSError as e:
                print(f'Warning: Could not remove temporary cache {dataset_dir}: {e}')


def compute_stats_for_combinations(
    dataset_name: str,
    node_sample_ratios: list[str],
    sampling_methods: list[str],
    adj_thresholds: list[float],
    adjacency_methods: list[str],
    n_jobs: int = -1,
    string_data_dir: str | None = None,
    cache_root: str = '/scratch/lcornelis/ogbench-1/run_data/omics',
    keep_cache: bool = False,
) -> list[dict[str, Any]]:
    """Compute statistics for all combinations of parameters using parallel processing."""
    combinations = [
        (
            dataset_name,
            node_ratio,
            method,
            adj_thresh,
            adj_m,
            string_data_dir,
            cache_root,
            keep_cache,
        )
        for node_ratio, method, adj_thresh, adj_m in itertools.product(
            node_sample_ratios, sampling_methods, adj_thresholds, adjacency_methods
        )
    ]

    print(f'Processing {len(combinations)} combinations for {dataset_name} using {n_jobs} jobs')

    # Process in parallel
    all_stats = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_single_combination)(args_tuple) for args_tuple in combinations
    )

    return all_stats


def save_stats_to_csv(all_stats: list[dict[str, Any]], output_file: str) -> None:
    """Save statistics to CSV file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fieldnames = [
        'dataset',
        'adj_thresh',
        'node_sample_ratio',
        'method',
        'adjacency_method',
        'num_nodes',
        'num_edges',
        'avg_degree',
        'density_pct',
        'largest_cc_ratio_pct',
        'num_connected_components',
        'degree_std',
        'clustering_coefficient',
        'diameter',
        'modularity',
        'homophily',
    ]

    if any('error' in stats for stats in all_stats):
        fieldnames.append('error')

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)

    print(f'Statistics saved to {output_file}')


def save_stats_to_json_for_webapp(
    all_stats: list[dict[str, Any]], output_file: str
) -> None:
    """Save statistics to JSON for webapp consumption.

    The JSON format uses keys like
    "dataset|node_sample_ratio|method|adj_thresh|adjacency_method"
    for fast lookup in the webapp.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    result = {}
    for stats in all_stats:
        # Skip entries with errors
        if 'error' in stats or stats.get('num_nodes') is None:
            continue

        key = (
            f"{stats['dataset']}|{stats['node_sample_ratio']}|{stats['method']}|"
            f"{stats['adj_thresh']}|{stats.get('adjacency_method', '')}"
        )
        result[key] = stats

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'Webapp JSON saved to {output_file}')


def create_plots_for_dataset(dataset_name: str, csv_file: str) -> None:
    """Create plots for a specific dataset."""
    try:
        df = pd.read_csv(csv_file)
        df = df[df['dataset'] == dataset_name]

        if df.empty:
            print(f'No data found for dataset {dataset_name}')
            return

        df = df.sort_values(by='adj_thresh', ascending=True)
        group_cols = ['node_sample_ratio', 'method']
        if 'adjacency_method' in df.columns:
            group_cols.append('adjacency_method')
        combinations = df[group_cols].drop_duplicates()

        for row in combinations.itertuples(index=False):
            node_ratio = row.node_sample_ratio
            method = row.method
            adj_m = getattr(row, 'adjacency_method', None)
            mask = (df['node_sample_ratio'] == node_ratio) & (df['method'] == method)
            if adj_m is not None:
                mask = mask & (df['adjacency_method'] == adj_m)
            subset_df = df[mask]
            if subset_df.empty:
                continue

            fig, axes = plt.subplots(4, 3, figsize=(16, 16))
            # Hide unused subplots (we have 11 metrics, grid has 12 cells)
            for ax in axes.flatten()[11:]:
                ax.set_visible(False)
            title_adj = f', Adjacency: {adj_m}' if adj_m is not None else ''
            fig.suptitle(
                f'{dataset_name} - Node Ratio: {node_ratio}, Method: {method}{title_adj}',
                fontsize=16,
                y=1.02,
            )

            plot_configs = [
                ('num_nodes', 'Number of Nodes', 'blue'),
                ('num_edges', 'Number of Edges', 'green'),
                ('avg_degree', 'Average Degree', 'orange'),
                ('density_pct', 'Density (%)', 'red'),
                ('largest_cc_ratio_pct', 'Largest CC / Total Nodes (%)', 'magenta'),
                ('num_connected_components', 'Connected Components', 'purple'),
                ('degree_std', 'Degree Std Dev', 'teal'),
                ('clustering_coefficient', 'Clustering Coefficient', 'brown'),
                ('diameter', 'Diameter', 'olive'),
                ('modularity', 'Modularity', 'navy'),
                ('homophily', 'Feature Homophily', 'crimson'),
            ]

            for ax, (col, title, color) in zip(axes.flatten(), plot_configs):
                ax.plot(subset_df['adj_thresh'], subset_df[col], color=color)
                ax.set_xlabel('Adjacency Threshold')
                ax.set_ylabel(title)
                ax.set_title(f'{title} vs. Adjacency Threshold')
                ax.grid(True)

            plt.tight_layout()

            plot_dir = f'./plots/{dataset_name}'
            os.makedirs(plot_dir, exist_ok=True)
            safe_adj = f'_adj_{adj_m}' if adj_m is not None else ''
            plot_filename = (
                f'{plot_dir}/{dataset_name}_node_ratio_{node_ratio}_method_{method}{safe_adj}.png'
            )
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f'Plot saved: {plot_filename}')

    except Exception as e:
        print(f'Error creating plots for {dataset_name}: {e}')


def main():
    """Run the analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze dataset statistics with parallel processing'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (-1 for all CPUs, default: -1)',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=[
            'motrpac',
            'tuberculosis',
            'parkinsons',
            'addneuromed',
            'smoking',
            'brca',
        ],
        help='List of datasets to process',
    )
    parser.add_argument(
        '--node-ratios',
        nargs='+',
        default=['full', '1.0', '0.5', '0.3'],
        help='List of node sample ratios',
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['variance', 'random', 'correlation', 'distance_correlation'],
        help='List of sampling methods',
    )
    parser.add_argument(
        '--adj-thresholds',
        type=int,
        default=10,
        help='Number of adjacency thresholds from 0.0 to 1.0 (default: 101)',
    )
    parser.add_argument(
        '--skip-plots', action='store_true', help='Skip plot generation (only compute statistics)'
    )

    parser.add_argument(
        '--adjacency-method',
        nargs='+',
        default=['wgcna'],
        help='One or more adjacency methods (e.g. string wgcna)',
    )
    parser.add_argument(
        '--string-data-dir',
        default=None,
        help='Path to pre-downloaded STRING bulk files (avoids downloading from stringdb-downloads.org)',
    )
    parser.add_argument(
        '--cache-root',
        default='/scratch/lcornelis/ogbench-1/run_data/omics',
        help=(
            'Working cache root. Combination-specific files are deleted after their '
            'statistics are computed; the shared STRING cache is retained.'
        ),
    )
    parser.add_argument(
        '--keep-cache',
        action='store_true',
        help='Retain generated dataset files instead of deleting each combination cache.',
    )

    args = parser.parse_args()
    os.makedirs(args.cache_root, exist_ok=True)

    # Define parameters
    datasets = args.datasets
    node_sample_ratios = args.node_ratios
    sampling_methods = args.methods
    adj_thresholds = [round(x, 2) for x in np.linspace(0.0, 1.0, args.adj_thresholds)]
    adjacency_methods = list(args.adjacency_method)
    n_jobs = args.n_jobs

    print(f'Processing {len(datasets)} datasets')
    print(f'Using 5-fold CV fold {FOLD} (training seed {TRAINING_SEED})')
    print(f'Node sample ratios: {node_sample_ratios}')
    print(f'Sampling methods: {sampling_methods}')
    print(f'Adjacency thresholds: {len(adj_thresholds)} values from 0.0 to 1.0')
    print(f'Adjacency methods: {adjacency_methods}')
    print(f'Parallel jobs: {n_jobs}')
    print(
        'Total combinations: '
        f'{len(datasets) * len(node_sample_ratios) * len(sampling_methods) * len(adj_thresholds) * len(adjacency_methods)}'
    )

    # Collect all stats across all datasets for webapp JSON export
    all_datasets_stats = []

    # Process each dataset
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f'Processing dataset: {dataset_name}')
        print(f"{'='*50}")

        # Compute statistics for all combinations
        all_stats = compute_stats_for_combinations(
            dataset_name,
            node_sample_ratios,
            sampling_methods,
            adj_thresholds,
            adjacency_methods,
            n_jobs,
            string_data_dir=args.string_data_dir,
            cache_root=args.cache_root,
            keep_cache=args.keep_cache,
        )

        # Collect for webapp JSON
        all_datasets_stats.extend(all_stats)

        # Save to CSV
        output_file = f'./stats/{dataset_name}/graph_stats_comprehensive.csv'
        save_stats_to_csv(all_stats, output_file)

        # Create plots (unless skipped)
        if not args.skip_plots:
            create_plots_for_dataset(dataset_name, output_file)

    # Save aggregated stats to webapp JSON
    webapp_json_path = os.path.join(
        os.path.dirname(__file__), '..', 'webapp', 'public', 'data', 'stats.json'
    )
    save_stats_to_json_for_webapp(all_datasets_stats, webapp_json_path)

    print('\nAnalysis complete!')
    print('Check the ./stats/ and ./plots/ directories for results.')

    # Print summary statistics
    total_combinations = (
        len(datasets)
        * len(node_sample_ratios)
        * len(sampling_methods)
        * len(adj_thresholds)
        * len(adjacency_methods)
    )
    print('\nSummary:')
    print(f'- Total parameter combinations processed: {total_combinations}')
    print(f'- Datasets: {len(datasets)}')
    print(f'- Node sample ratios: {len(node_sample_ratios)}')
    print(f'- Sampling methods: {len(sampling_methods)}')
    print(f'- Adjacency thresholds: {len(adj_thresholds)}')
    print(f'- Adjacency methods: {len(adjacency_methods)}')
    print(f'- Parallel jobs used: {n_jobs}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Analyze dataset statistics across different parameters. Generates plots for all combinations of
node_sample_ratio and sampling_method.

Usage examples:
    python dataset_stats_analysis.py
    python dataset_stats_analysis.py --n-jobs 8
    python dataset_stats_analysis.py --datasets addneuromed parkinsons
    python dataset_stats_analysis.py --adj-thresholds 21
    python dataset_stats_analysis.py --skip-plots
    python dataset_stats_analysis.py --node-ratios full 1.0 0.5 --methods variance random
"""

import csv
import itertools
import os
import os.path as osp
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def load_dataset(
    dataset_name: str,
    adj_thresh: float = 0.5,
    node_sample_ratio: str = 'full',
    method: str = 'variance',
) -> Any:
    """Load the dataset with specified parameters."""
    from omegaconf import OmegaConf

    from ogbench.data.datasets.hf_omics import HFOmicsDataset

    train_val_test_split = OmegaConf.create([0.7, 0.15, 0.15])

    # Pass 'full' as string, not None, because HFOmicsDataset checks for 'full' string
    ratio_value = 'full' if node_sample_ratio == 'full' else float(node_sample_ratio)
    dataset = HFOmicsDataset(
        root='/home/louisa/code/bgbench-1/run_data/omics',
        data_name=dataset_name,
        method=method,
        adjacency_threshold=adj_thresh,
        node_sample_ratio=ratio_value,
        train_val_test_split=train_val_test_split,
        imputation_method='mean',
    )

    return dataset


def get_graph_stats(dataset: Any) -> dict[str, float]:
    """Get statistics of the graph from the dataset."""
    empty_stats = {
        'num_nodes': 0,
        'num_edges': 0,
        'avg_degree': 0.0,
        'density_pct': 0.0,
        'avg_clustering_coeff': 0.0,
        'largest_cc_ratio_pct': 0.0,
        'avg_shortest_path_length': 0.0,
        'num_connected_components': 0,
        'degree_std': 0.0,
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
            root = '/home/louisa/code/bgbench-1/run_data/omics/'
            name = osp.join(
                root,
                f'{dataset.data_name}',
                f'adj_thresh_{dataset.adjacency_threshold}',
                f'{dataset.method}',
                f'p_{dataset.node_sample_ratio}',
                f'train_split_{dataset.train_val_test_split[0]}',
                'raw/adj_matrix.npy',
            )
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
        avg_clustering_coeff = nx.average_clustering(graph)

        connected_components = list(nx.connected_components(graph))
        num_connected_components = len(connected_components)
        largest_cc = max(connected_components, key=len)
        largest_cc_ratio_pct = (len(largest_cc) / num_nodes) * 100

        if len(largest_cc) > 1:
            largest_cc_subgraph = graph.subgraph(largest_cc)
            avg_shortest_path_length = nx.average_shortest_path_length(largest_cc_subgraph)
        else:
            avg_shortest_path_length = 0.0

        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'density_pct': density_pct,
            'avg_clustering_coeff': avg_clustering_coeff,
            'largest_cc_ratio_pct': largest_cc_ratio_pct,
            'avg_shortest_path_length': avg_shortest_path_length,
            'num_connected_components': num_connected_components,
            'degree_std': degree_std,
        }

    except Exception as e:
        print(f'Error getting graph stats: {e}')
        return empty_stats


def process_single_combination(args_tuple: tuple[str, str, str, float]) -> dict[str, Any]:
    """Process a single parameter combination for parallel processing."""
    dataset_name, node_ratio, method, adj_thresh = args_tuple

    try:
        dataset = load_dataset(dataset_name, adj_thresh, node_ratio, method)
        print(f'Dataset loaded: {dataset}, length: {len(dataset)}')

        stats = get_graph_stats(dataset)
        stats.update({
            'dataset': dataset_name,
            'adj_thresh': adj_thresh,
            'node_sample_ratio': node_ratio,
            'method': method,
        })
        return stats

    except Exception as e:
        return {
            'dataset': dataset_name,
            'adj_thresh': adj_thresh,
            'node_sample_ratio': node_ratio,
            'method': method,
            'num_nodes': None,
            'num_edges': None,
            'avg_degree': None,
            'density_pct': None,
            'avg_clustering_coeff': None,
            'largest_cc_ratio_pct': None,
            'avg_shortest_path_length': None,
            'num_connected_components': None,
            'degree_std': None,
            'error': str(e),
        }


def compute_stats_for_combinations(
    dataset_name: str,
    node_sample_ratios: list[str],
    sampling_methods: list[str],
    adj_thresholds: list[float],
    n_jobs: int = -1,
) -> list[dict[str, Any]]:
    """Compute statistics for all combinations of parameters using parallel processing."""
    combinations = [
        (dataset_name, node_ratio, method, adj_thresh)
        for node_ratio, method, adj_thresh in itertools.product(
            node_sample_ratios, sampling_methods, adj_thresholds
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
        'num_nodes',
        'num_edges',
        'avg_degree',
        'density_pct',
        'avg_clustering_coeff',
        'largest_cc_ratio_pct',
        'avg_shortest_path_length',
        'num_connected_components',
        'degree_std',
    ]

    if any('error' in stats for stats in all_stats):
        fieldnames.append('error')

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)

    print(f'Statistics saved to {output_file}')


def create_plots_for_dataset(dataset_name: str, csv_file: str) -> None:
    """Create plots for a specific dataset."""
    try:
        df = pd.read_csv(csv_file)
        df = df[df['dataset'] == dataset_name]

        if df.empty:
            print(f'No data found for dataset {dataset_name}')
            return

        df = df.sort_values(by='adj_thresh', ascending=True)
        combinations = df[['node_sample_ratio', 'method']].drop_duplicates()

        for _, (node_ratio, method) in combinations.iterrows():
            subset_df = df[(df['node_sample_ratio'] == node_ratio) & (df['method'] == method)]
            if subset_df.empty:
                continue

            fig, axes = plt.subplots(3, 3, figsize=(16, 12))
            fig.suptitle(
                f'{dataset_name} - Node Ratio: {node_ratio}, Method: {method}', fontsize=16, y=1.02
            )

            plot_configs = [
                ('num_nodes', 'Number of Nodes', 'blue'),
                ('num_edges', 'Number of Edges', 'green'),
                ('avg_degree', 'Average Degree', 'orange'),
                ('density_pct', 'Density (%)', 'red'),
                ('avg_clustering_coeff', 'Avg Clustering Coefficient', 'cyan'),
                ('largest_cc_ratio_pct', 'Largest CC / Total Nodes (%)', 'magenta'),
                ('avg_shortest_path_length', 'Avg Shortest Path Length', 'brown'),
                ('num_connected_components', 'Connected Components', 'purple'),
                ('degree_std', 'Degree Std Dev', 'teal'),
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
            plot_filename = f'{plot_dir}/{dataset_name}_node_ratio_{node_ratio}_method_{method}.png'
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
        default=8,
        help='Number of parallel jobs (-1 for all CPUs, default: -1)',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['addneuromed', 'parkinsons', 'motrpac'],
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

    args = parser.parse_args()

    # Define parameters
    datasets = args.datasets
    node_sample_ratios = args.node_ratios
    sampling_methods = args.methods
    adj_thresholds = [round(x, 2) for x in np.linspace(0.0, 1.0, args.adj_thresholds)]
    print(adj_thresholds)
    n_jobs = args.n_jobs

    print(f'Processing {len(datasets)} datasets')
    print(f'Node sample ratios: {node_sample_ratios}')
    print(f'Sampling methods: {sampling_methods}')
    print(f'Adjacency thresholds: {len(adj_thresholds)} values from 0.0 to 1.0')
    print(f'Parallel jobs: {n_jobs}')
    print(
        f'Total combinations: {len(datasets) * len(node_sample_ratios) * len(sampling_methods) * len(adj_thresholds)}'
    )

    # Process each dataset
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f'Processing dataset: {dataset_name}')
        print(f"{'='*50}")

        # Compute statistics for all combinations
        all_stats = compute_stats_for_combinations(
            dataset_name, node_sample_ratios, sampling_methods, adj_thresholds, n_jobs
        )

        # Save to CSV
        output_file = f'./stats/{dataset_name}/graph_stats_comprehensive.csv'
        save_stats_to_csv(all_stats, output_file)

        # Create plots (unless skipped)
        if not args.skip_plots:
            create_plots_for_dataset(dataset_name, output_file)

    print('\nAnalysis complete!')
    print('Check the ./stats/ and ./plots/ directories for results.')

    # Print summary statistics
    total_combinations = (
        len(datasets) * len(node_sample_ratios) * len(sampling_methods) * len(adj_thresholds)
    )
    print('\nSummary:')
    print(f'- Total parameter combinations processed: {total_combinations}')
    print(f'- Datasets: {len(datasets)}')
    print(f'- Node sample ratios: {len(node_sample_ratios)}')
    print(f'- Sampling methods: {len(sampling_methods)}')
    print(f'- Adjacency thresholds: {len(adj_thresholds)}')
    print(f'- Parallel jobs used: {n_jobs}')


if __name__ == '__main__':
    try:
        print('Testing dataset loading...')
        dataset = load_dataset('addneuromed', 0.5, '0.3', 'variance')
        print(f'Dataset loaded successfully: {dataset}')
        print(f'Dataset length: {len(dataset)}')
        if len(dataset) > 0:
            print(f'First data item: {dataset[0]}')
            stats = get_graph_stats(dataset)
            print(f'Graph stats: {stats}')
    except Exception as e:
        print(f'Error in test: {e}')
        import traceback
        traceback.print_exc()

    print('\n' + '=' * 50)
    print('Running main analysis...')
    print('=' * 50)
    main()

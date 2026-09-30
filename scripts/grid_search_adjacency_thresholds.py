#!/usr/bin/env python3
"""Grid search for optimal adjacency thresholds across datasets, node ratios, and methods.

This script performs a grid search over:
- Datasets: motrpac, parkinsons, addneuromed
- Node sample ratios: 1.0, 0.8
- Methods: variance, correlation, random

It finds optimal adjacency thresholds for ~10% connectivity for each combination
and saves results to structured JSON and CSV files.
"""

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Import reusable functions from existing script
from plot_adjacency_threshold_analysis import (
    DATASETS,
    calculate_adjacency_matrix_with_threshold,
    compute_graph_metrics,
    interpolate_threshold_for_target,
    load_and_preprocess_dataset,
)
from tqdm import tqdm


def run_grid_search(
    datasets: dict[str, Any],
    node_sample_ratios: list[float],
    methods: list[str],
    thresholds: list[float],
) -> pd.DataFrame:
    """Run grid search over all combinations of dataset, node_sample_ratio, and method.

    Args:
        datasets: Dictionary of dataset configurations
        node_sample_ratios: List of node sample ratios to test
        methods: List of node selection methods to test
        thresholds: List of adjacency thresholds to sweep

    Returns:
        DataFrame with all results including metadata columns
    """
    all_results = []
    total_combinations = len(datasets) * len(node_sample_ratios) * len(methods)

    print(f'\n{"="*80}')
    print('GRID SEARCH: ADJACENCY THRESHOLD ANALYSIS')
    print(f'{"="*80}')
    print(f'\nTotal combinations: {total_combinations}')
    print(f'  Datasets: {list(datasets.keys())}')
    print(f'  Node sample ratios: {node_sample_ratios}')
    print(f'  Methods: {methods}')
    print(f'  Thresholds per combination: {len(thresholds)}')
    print(f'\nTotal runs: {total_combinations * len(thresholds)}')

    combination_counter = 0

    for dataset_name, base_config in datasets.items():
        for node_sample_ratio in node_sample_ratios:
            for method in methods:
                combination_counter += 1

                print(f'\n{"-"*80}')
                print(
                    f'[{combination_counter}/{total_combinations}] {dataset_name.upper()} | '
                    f'ratio={node_sample_ratio} | method={method}'
                )
                print(f'{"-"*80}')

                # Create modified config for this combination
                config = base_config.copy()
                config['node_sample_ratio'] = node_sample_ratio
                config['method'] = method

                try:
                    # Load and preprocess data
                    train_data, train_targets = load_and_preprocess_dataset(dataset_name, config)

                    # Compute metrics for each threshold
                    print(f'Computing metrics for {len(thresholds)} thresholds...')
                    for threshold in tqdm(
                        thresholds, desc=f'{dataset_name}-{node_sample_ratio}-{method}'
                    ):
                        adj_matrix = calculate_adjacency_matrix_with_threshold(
                            train_data, threshold
                        )
                        metrics = compute_graph_metrics(adj_matrix)

                        # Add metadata
                        metrics['dataset'] = dataset_name
                        metrics['node_sample_ratio'] = node_sample_ratio
                        metrics['method'] = method
                        metrics['threshold'] = threshold

                        all_results.append(metrics)

                except Exception as e:
                    print(f'ERROR: Failed for {dataset_name}/{node_sample_ratio}/{method}: {e}')
                    import traceback

                    traceback.print_exc()

    if not all_results:
        raise RuntimeError('No results collected from grid search!')

    return pd.DataFrame(all_results)


def compute_optimal_thresholds(
    all_results: pd.DataFrame, target_connectivity: float = 0.10
) -> dict[str, Any]:
    """Extract optimal thresholds for each dataset/ratio/method combination.

    Args:
        all_results: DataFrame with all threshold sweep results
        target_connectivity: Target connectivity level (default 0.10 = 10%)

    Returns:
        Dictionary with metadata and list of optimal threshold results
    """
    optimal_results = []

    # Group by dataset, node_sample_ratio, method
    grouped = all_results.groupby(['dataset', 'node_sample_ratio', 'method'])

    for (dataset, ratio, method), group_df in grouped:
        # Use interpolation to find optimal threshold
        result = interpolate_threshold_for_target(group_df, target_connectivity)

        optimal_results.append(
            {
                'dataset': dataset,
                'node_sample_ratio': ratio,
                'method': method,
                'n_nodes': int(group_df['n_nodes'].iloc[0]),
                'optimal_threshold': float(result['threshold']),
                'connectivity': float(result['connectivity']),
                'mean_degree': float(result['mean_degree']),
                'interpolated': bool(result['interpolated']),
            }
        )

    return {
        'target_connectivity': target_connectivity,
        'results': optimal_results,
    }


def save_results_json(optimal_results: dict[str, Any], output_path: str) -> None:
    """Save optimal thresholds to JSON with pretty formatting.

    Args:
        optimal_results: Dictionary with optimal threshold data
        output_path: Path to output JSON file
    """
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'target_connectivity': optimal_results['target_connectivity'],
            'n_combinations': len(optimal_results['results']),
        },
        'results': optimal_results['results'],
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f'\nSaved optimal thresholds to: {output_path}')


def print_results_table(optimal_results: dict[str, Any]) -> None:
    """Print nicely formatted console output grouped by dataset.

    Args:
        optimal_results: Dictionary with optimal threshold data
    """
    results_df = pd.DataFrame(optimal_results['results'])
    target_conn = optimal_results['target_connectivity']

    print(f'\n{"="*80}')
    print(f'OPTIMAL THRESHOLDS FOR ~{target_conn*100:.0f}% CONNECTIVITY')
    print(
        f'Grid Search: {len(results_df["dataset"].unique())} datasets × '
        f'{len(results_df["node_sample_ratio"].unique())} ratios × '
        f'{len(results_df["method"].unique())} methods = '
        f'{len(results_df)} combinations'
    )
    print(f'{"="*80}')

    # Group by dataset for cleaner output
    for dataset in sorted(results_df['dataset'].unique()):
        dataset_df = results_df[results_df['dataset'] == dataset]

        print(f'\n{dataset.upper()}')
        print(
            f'  {"Node Ratio":<12} | {"Method":<12} | {"Nodes":>6} | {"Threshold":>10} | '
            f'{"Connectivity":>12} | {"Mean Degree":>12}'
        )
        print(f'  {"-"*12}-|-{"-"*12}-|-{"-"*6}-|-{"-"*10}-|-{"-"*12}-|-{"-"*12}')

        # Sort by ratio (descending) then method
        dataset_df = dataset_df.sort_values(
            ['node_sample_ratio', 'method'], ascending=[False, True]
        )

        for _, row in dataset_df.iterrows():
            print(
                f'  {row["node_sample_ratio"]:<12.1f} | {row["method"]:<12} | '
                f'{row["n_nodes"]:>6} | {row["optimal_threshold"]:>10.4f} | '
                f'{row["connectivity"]*100:>11.2f}% | {row["mean_degree"]:>12.1f}'
            )

    print(f'\n{"="*80}')


def main():
    """Main function orchestrating the grid search pipeline."""
    # Configuration
    NODE_SAMPLE_RATIOS = [1.0, 0.8]
    METHODS = ['variance', 'correlation', 'random']

    # Define threshold sweep (same as original script)
    thresholds = np.concatenate(
        [
            np.linspace(0.0, 0.1, 11),  # Fine resolution in low range
            np.linspace(0.15, 0.5, 8),  # Medium resolution in mid range
            np.linspace(0.6, 0.9, 4),  # Coarse resolution in high range
        ]
    )
    thresholds = np.unique(thresholds)

    # Run grid search
    all_results = run_grid_search(DATASETS, NODE_SAMPLE_RATIOS, METHODS, thresholds)

    # Compute optimal thresholds for target connectivity
    optimal_results = compute_optimal_thresholds(all_results, target_connectivity=0.10)

    # Create output directory
    output_dir = 'plots/threshold_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON
    json_path = os.path.join(output_dir, 'optimal_thresholds.json')
    save_results_json(optimal_results, json_path)

    # Save full results to CSV
    csv_path = os.path.join(output_dir, 'grid_search_results.csv')
    all_results.to_csv(csv_path, index=False)
    print(f'Saved full grid search results to: {csv_path}')

    # Print nice summary table
    print_results_table(optimal_results)

    print(f'\n{"="*80}')
    print('RESULTS SUMMARY')
    print(f'{"="*80}')
    print(f'JSON file: {json_path}')
    print(f'CSV file:  {csv_path}')
    print(f'\nTotal combinations analyzed: {len(optimal_results["results"])}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()

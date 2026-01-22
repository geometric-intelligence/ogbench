#!/usr/bin/env python3
"""Analyze and plot graph connectivity metrics as a function of adjacency threshold.

This script computes graph statistics (connectivity, degree distribution, etc.) for different
adjacency thresholds and visualizes how they change.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyWGCNA
import seaborn as sns
from huggingface_hub import hf_hub_download
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from tqdm import tqdm

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)

# Dataset configurations
DATASETS = {
    'motrpac': {
        'data_name': 'motrpac',
        'revision': '9f052d330ce130408a2c7c347b2ed197154da7c8',
        'train_val_test_split': [0.7, 0.2, 0.1],
        'node_sample_ratio': 0.5,
        'method': 'variance',
    },
    'parkinsons': {
        'data_name': 'parkinsons',
        'revision': '9f052d330ce130408a2c7c347b2ed197154da7c8',
        'train_val_test_split': [0.7, 0.15, 0.15],
        'node_sample_ratio': 0.5,
        'method': 'variance',
    },
    'addneuromed': {
        'data_name': 'addneuromed',
        'revision': '9f052d330ce130408a2c7c347b2ed197154da7c8',
        'train_val_test_split': [0.7, 0.2, 0.1],
        'node_sample_ratio': 0.5,
        'method': 'variance',
    },
}

HF_REPO_ID = 'geometric-intelligence/bgbench'


def load_and_preprocess_dataset(
    dataset_name: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load dataset and preprocess following the same steps as HFOmicsDataset.

    Returns training data only (after split and imputation).
    """
    print(f'\nLoading {dataset_name}...')

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
    node_sample_ratio = config['node_sample_ratio']
    n_training_samples = len(train_targets)

    if node_sample_ratio == 'full':
        n_nodes = train_data.shape[1]
    elif isinstance(node_sample_ratio, (float | int)):
        n_nodes = int(n_training_samples / node_sample_ratio)
        if n_nodes > train_data.shape[1]:
            n_nodes = train_data.shape[1]

    # Select nodes
    selected_nodes = select_nodes(
        train_data.values, train_targets, n_selected=n_nodes, method=config['method']
    )
    train_selected = train_data.iloc[:, selected_nodes]

    print(f'  Selected nodes: {len(selected_nodes)}')

    return train_selected, train_targets


def select_nodes(
    data: np.ndarray, targets: np.ndarray, n_selected: int = 10, method: str = 'variance'
) -> np.ndarray:
    """Select nodes based on feature importance."""
    if method == 'variance':
        variances = np.std(data, axis=0)
        ranked_nodes = np.argsort(variances)[::-1]
    elif method == 'correlation':
        correlations = np.abs(
            np.array([np.corrcoef(data[:, i], targets)[0, 1] for i in range(data.shape[1])])
        )
        ranked_nodes = np.argsort(correlations)[::-1]
    elif method == 'random':
        ranked_nodes = np.random.permutation(data.shape[1])
    else:
        raise ValueError(f'Invalid method: {method}')

    return ranked_nodes[:n_selected]


def calculate_adjacency_matrix_with_threshold(
    node_features: pd.DataFrame, adjacency_threshold: float
) -> np.ndarray:
    """Calculate adjacency matrix using WGCNA with given threshold."""
    # Use WGCNA to find optimal power
    soft_threshold = PyWGCNA.WGCNA.pickSoftThreshold(node_features)
    power = soft_threshold[0]

    # Apply soft-thresholding
    adjacency = PyWGCNA.WGCNA.adjacency(
        node_features,
        power=power,
        adjacencyType='signed hybrid',
    )

    # Binarize with given threshold
    adjacency = np.nan_to_num(adjacency, nan=0.0)
    adj_matrix = np.where(adjacency > adjacency_threshold, 1, 0)
    np.fill_diagonal(adj_matrix, 1)

    return adj_matrix


def compute_graph_metrics(adj_matrix: np.ndarray) -> dict[str, float]:
    """Compute various graph metrics from adjacency matrix."""
    n_nodes = adj_matrix.shape[0]

    # Remove self-loops for metric computation
    adj_no_diag = adj_matrix.copy()
    np.fill_diagonal(adj_no_diag, 0)

    # Node degrees (excluding self-loops)
    node_degrees = np.sum(adj_no_diag, axis=1)

    # Number of edges (undirected)
    n_edges = np.sum(adj_no_diag) / 2

    # Maximum possible edges (no self-loops)
    max_edges = n_nodes * (n_nodes - 1) / 2

    # Connectivity (density)
    connectivity = n_edges / max_edges if max_edges > 0 else 0

    # Degree statistics
    mean_degree = np.mean(node_degrees)
    median_degree = np.median(node_degrees)
    std_degree = np.std(node_degrees)
    min_degree = np.min(node_degrees)
    max_degree = np.max(node_degrees)

    # Isolated nodes (degree 0)
    n_isolated = np.sum(node_degrees == 0)

    return {
        'n_nodes': n_nodes,
        'n_edges': int(n_edges),
        'connectivity': connectivity,
        'mean_degree': mean_degree,
        'median_degree': median_degree,
        'std_degree': std_degree,
        'min_degree': int(min_degree),
        'max_degree': int(max_degree),
        'n_isolated': int(n_isolated),
    }


def analyze_threshold_sweep(
    dataset_name: str, config: dict[str, Any], thresholds: list[float]
) -> pd.DataFrame:
    """Analyze graph metrics across different adjacency thresholds."""
    print(f"\n{'='*70}")
    print(f'Analyzing {dataset_name.upper()}')
    print(f"{'='*70}")

    # Load and preprocess data
    train_data, train_targets = load_and_preprocess_dataset(dataset_name, config)

    results = []

    print(f'\nComputing graph metrics for {len(thresholds)} thresholds...')
    for threshold in tqdm(thresholds, desc=f'{dataset_name}'):
        adj_matrix = calculate_adjacency_matrix_with_threshold(train_data, threshold)
        metrics = compute_graph_metrics(adj_matrix)
        metrics['threshold'] = threshold
        metrics['dataset'] = dataset_name
        results.append(metrics)

    df = pd.DataFrame(results)
    print(f'\nResults summary for {dataset_name}:')
    print(f'  Threshold range: [{thresholds[0]:.3f}, {thresholds[-1]:.3f}]')
    print(
        f"  Connectivity range: [{df['connectivity'].min()*100:.2f}%, {df['connectivity'].max()*100:.2f}%]"
    )
    print(f"  Mean degree range: [{df['mean_degree'].min():.1f}, {df['mean_degree'].max():.1f}]")

    return df


def plot_results(all_results: pd.DataFrame, output_dir: str = 'plots'):
    """Create comprehensive plots of graph metrics vs adjacency threshold."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = all_results['dataset'].unique()

    # Color palette
    colors = sns.color_palette('husl', len(datasets))
    dataset_colors = {ds: colors[i] for i, ds in enumerate(datasets)}

    # Create comprehensive figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Graph Metrics vs Adjacency Threshold', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('connectivity', 'Connectivity (Graph Density)', True, True),  # log Y
        ('n_edges', 'Number of Edges', False, True),  # log Y
        ('mean_degree', 'Mean Node Degree', False, True),  # log Y
        ('median_degree', 'Median Node Degree', False, True),  # log Y
        ('std_degree', 'Std Node Degree', False, True),  # log Y
        ('n_isolated', 'Number of Isolated Nodes', False, False),  # linear (inverse relationship)
    ]

    for idx, (metric, title, as_percentage, use_log_y) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]

        for dataset in datasets:
            df_dataset = all_results[all_results['dataset'] == dataset]

            # Skip threshold=0.0 for log scale x-axis
            df_dataset = df_dataset[df_dataset['threshold'] > 0].copy()

            if as_percentage:
                y_values = df_dataset[metric] * 100
                ylabel = f'{title} (%)'
            else:
                y_values = df_dataset[metric]
                ylabel = title

            # Replace zeros with NaN for log scale
            if use_log_y:
                y_values = y_values.replace(0, np.nan)

            ax.plot(
                df_dataset['threshold'],
                y_values,
                marker='o',
                label=dataset,
                color=dataset_colors[dataset],
                linewidth=2,
                markersize=4,
            )

        ax.set_xlabel('Adjacency Threshold', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Apply log scales
        ax.set_xscale('log')
        if use_log_y:
            ax.set_yscale('log')

        ax.grid(True, alpha=0.3, which='both')
        ax.grid(True, alpha=0.15, which='minor')
        ax.legend(loc='best', fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'adjacency_threshold_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nSaved comprehensive plot to: {output_path}')
    plt.close()

    # Create individual plots for each dataset
    for dataset in datasets:
        df_dataset = all_results[all_results['dataset'] == dataset]

        # Skip threshold=0.0 for log scale x-axis
        df_dataset = df_dataset[df_dataset['threshold'] > 0].copy()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f'{dataset.upper()} - Graph Metrics vs Adjacency Threshold',
            fontsize=14,
            fontweight='bold',
        )

        # Plot 1: Connectivity (log-log)
        ax = axes[0, 0]
        connectivity_pct = df_dataset['connectivity'] * 100
        connectivity_pct = connectivity_pct.replace(0, np.nan)
        ax.plot(
            df_dataset['threshold'],
            connectivity_pct,
            marker='o',
            linewidth=2,
            color=dataset_colors[dataset],
        )
        ax.set_xlabel('Adjacency Threshold')
        ax.set_ylabel('Connectivity (%)')
        ax.set_title('Graph Connectivity (Density)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.grid(True, alpha=0.15, which='minor')

        # Plot 2: Number of edges (log-log)
        ax = axes[0, 1]
        n_edges = df_dataset['n_edges'].replace(0, np.nan)
        ax.plot(
            df_dataset['threshold'],
            n_edges,
            marker='o',
            linewidth=2,
            color=dataset_colors[dataset],
        )
        ax.set_xlabel('Adjacency Threshold')
        ax.set_ylabel('Number of Edges')
        ax.set_title('Total Edges')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.grid(True, alpha=0.15, which='minor')

        # Plot 3: Degree statistics (log-log)
        ax = axes[1, 0]
        mean_deg = df_dataset['mean_degree'].replace(0, np.nan)
        median_deg = df_dataset['median_degree'].replace(0, np.nan)
        ax.plot(
            df_dataset['threshold'],
            mean_deg,
            marker='o',
            linewidth=2,
            label='Mean',
            color=dataset_colors[dataset],
        )
        ax.plot(
            df_dataset['threshold'],
            median_deg,
            marker='s',
            linewidth=2,
            label='Median',
            color=dataset_colors[dataset],
            alpha=0.7,
        )
        # Note: fill_between doesn't work well with log scale, so we skip it
        ax.set_xlabel('Adjacency Threshold')
        ax.set_ylabel('Node Degree')
        ax.set_title('Degree Distribution (Mean & Median)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        ax.grid(True, alpha=0.15, which='minor')

        # Plot 4: Isolated nodes (semi-log X)
        ax = axes[1, 1]
        ax.plot(
            df_dataset['threshold'],
            df_dataset['n_isolated'],
            marker='o',
            linewidth=2,
            color=dataset_colors[dataset],
        )
        ax.set_xlabel('Adjacency Threshold')
        ax.set_ylabel('Number of Isolated Nodes')
        ax.set_title('Isolated Nodes (Degree = 0)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.grid(True, alpha=0.15, which='minor')

        plt.tight_layout()

        output_path = os.path.join(output_dir, f'{dataset}_threshold_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved {dataset} plot to: {output_path}')
        plt.close()


def interpolate_threshold_for_target(
    df_dataset: pd.DataFrame, target_connectivity: float
) -> dict[str, float]:
    """Interpolate to find precise threshold for target connectivity.

    Uses linear interpolation between the two adjacent data points.
    """
    # Sort by connectivity
    df_sorted = df_dataset.sort_values('connectivity').reset_index(drop=True)

    # Find the two points that bracket the target
    below = df_sorted[df_sorted['connectivity'] <= target_connectivity]
    above = df_sorted[df_sorted['connectivity'] >= target_connectivity]

    if len(below) == 0:
        # Target is below minimum connectivity
        row = df_sorted.iloc[0]
        return {
            'threshold': row['threshold'],
            'connectivity': row['connectivity'],
            'mean_degree': row['mean_degree'],
            'interpolated': False,
        }

    if len(above) == 0:
        # Target is above maximum connectivity
        row = df_sorted.iloc[-1]
        return {
            'threshold': row['threshold'],
            'connectivity': row['connectivity'],
            'mean_degree': row['mean_degree'],
            'interpolated': False,
        }

    # Get the bracketing points
    point_below = below.iloc[-1]
    point_above = above.iloc[0]

    # Check if we hit exactly
    if abs(point_below['connectivity'] - target_connectivity) < 1e-10:
        return {
            'threshold': point_below['threshold'],
            'connectivity': point_below['connectivity'],
            'mean_degree': point_below['mean_degree'],
            'interpolated': False,
        }

    if abs(point_above['connectivity'] - target_connectivity) < 1e-10:
        return {
            'threshold': point_above['threshold'],
            'connectivity': point_above['connectivity'],
            'mean_degree': point_above['mean_degree'],
            'interpolated': False,
        }

    # Linear interpolation
    # threshold = threshold1 + (threshold2 - threshold1) * (target - conn1) / (conn2 - conn1)
    t1, c1, d1 = point_below['threshold'], point_below['connectivity'], point_below['mean_degree']
    t2, c2, d2 = point_above['threshold'], point_above['connectivity'], point_above['mean_degree']

    # Interpolate threshold
    interpolated_threshold = t1 + (t2 - t1) * (target_connectivity - c1) / (c2 - c1)

    # Interpolate mean degree
    interpolated_degree = d1 + (d2 - d1) * (target_connectivity - c1) / (c2 - c1)

    return {
        'threshold': interpolated_threshold,
        'connectivity': target_connectivity,
        'mean_degree': interpolated_degree,
        'interpolated': True,
        'bracket_low': (t1, c1),
        'bracket_high': (t2, c2),
    }


def save_results_csv(all_results: pd.DataFrame, output_dir: str = 'plots'):
    """Save results to CSV for further analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Save combined results
    csv_path = os.path.join(output_dir, 'adjacency_threshold_metrics.csv')
    all_results.to_csv(csv_path, index=False)
    print(f'\nSaved results to: {csv_path}')

    # Save summary statistics with interpolation
    summary_stats = []
    target_connectivity = 0.10

    for dataset in all_results['dataset'].unique():
        df_dataset = all_results[all_results['dataset'] == dataset]

        # Use interpolation to find precise threshold
        result = interpolate_threshold_for_target(df_dataset, target_connectivity)

        # Also find closest measured point for comparison
        closest_idx = (df_dataset['connectivity'] - target_connectivity).abs().idxmin()
        closest_row = df_dataset.loc[closest_idx]

        summary_stats.append(
            {
                'dataset': dataset,
                'n_nodes': int(df_dataset['n_nodes'].iloc[0]),
                'interpolated_threshold': result['threshold'],
                'interpolated_connectivity': result['connectivity'],
                'interpolated_mean_degree': result['mean_degree'],
                'closest_measured_threshold': closest_row['threshold'],
                'closest_measured_connectivity': closest_row['connectivity'],
                'closest_measured_degree': closest_row['mean_degree'],
                'was_interpolated': result['interpolated'],
            }
        )

    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, 'threshold_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved summary to: {summary_path}')

    # Print summary table
    print(f"\n{'='*70}")
    print('SUMMARY: Recommended Thresholds for ~10% Connectivity (Interpolated)')
    print(f"{'='*70}")
    print('\nInterpolated values (precise estimates):')
    print(f"{'Dataset':<15} {'Threshold':>10} {'Connectivity':>12} {'Mean Degree':>12}")
    print(f"{'-'*50}")
    for _, row in summary_df.iterrows():
        print(
            f"{row['dataset']:<15} {row['interpolated_threshold']:>10.4f} "
            f"{row['interpolated_connectivity']*100:>11.2f}% {row['interpolated_mean_degree']:>12.1f}"
        )

    print('\nClosest measured points (for validation):')
    print(f"{'Dataset':<15} {'Threshold':>10} {'Connectivity':>12} {'Mean Degree':>12}")
    print(f"{'-'*50}")
    for _, row in summary_df.iterrows():
        print(
            f"{row['dataset']:<15} {row['closest_measured_threshold']:>10.4f} "
            f"{row['closest_measured_connectivity']*100:>11.2f}% {row['closest_measured_degree']:>12.1f}"
        )


def main():
    """Main analysis function."""
    print('=' * 70)
    print('ADJACENCY THRESHOLD ANALYSIS')
    print('Analyzing graph metrics as a function of adjacency threshold')
    print('=' * 70)

    # Define threshold range to sweep
    # Focus on range where most interesting behavior happens
    thresholds = np.concatenate(
        [
            np.linspace(0.0, 0.1, 11),  # Fine resolution in low range
            np.linspace(0.15, 0.5, 8),  # Medium resolution in mid range
            np.linspace(0.6, 0.9, 4),  # Coarse resolution in high range
        ]
    )
    thresholds = np.unique(thresholds)  # Remove duplicates

    print(f'\nAnalyzing {len(thresholds)} thresholds: [{thresholds[0]:.2f}, {thresholds[-1]:.2f}]')
    print(f'Datasets: {list(DATASETS.keys())}')

    # Run analysis for each dataset
    all_results = []
    for dataset_name, config in DATASETS.items():
        try:
            df = analyze_threshold_sweep(dataset_name, config, thresholds)
            all_results.append(df)
        except Exception as e:
            print(f'\nError analyzing {dataset_name}: {e}')
            import traceback

            traceback.print_exc()

    if not all_results:
        print('\nNo results to plot!')
        return

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Create plots
    print(f"\n{'='*70}")
    print('Creating plots...')
    print(f"{'='*70}")
    plot_results(combined_results, output_dir='plots/threshold_analysis')

    # Save results
    save_results_csv(combined_results, output_dir='plots/threshold_analysis')

    print(f"\n{'='*70}")
    print('Analysis complete!')
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

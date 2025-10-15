#!/usr/bin/env python3
"""Simple script to create aggregated plots from dataset statistics.

Aggregates data by node_sample_ratio and method combinations, showing how metrics vary with
adjacency threshold.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(csv_file: str) -> pd.DataFrame:
    """Load data from CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f'CSV file not found: {csv_file}')

    df = pd.read_csv(csv_file)
    print(f'Loaded {len(df)} rows from {csv_file}')
    return df


def create_aggregated_plots(
    df: pd.DataFrame, output_dir: str = './plots/aggregated', dataset_name: str = ''
) -> None:
    """Create aggregated plots showing metrics vs adjacency threshold.

    Each plot shows all combinations of node_sample_ratio and method as separate lines.
    All plots are organized as subplots in a single figure.
    Color families: each node_sample_ratio gets a color family, methods get different shades.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define metrics to plot
    metrics = {
        'num_nodes': 'Number of Nodes',
        'num_edges': 'Number of Edges',
        'avg_degree': 'Average Degree',
        'density': 'Density',
        'number_connected_components': 'Number of Connected Components',
    }

    # Get unique combinations of node_sample_ratio and method
    combinations = df[['node_sample_ratio', 'method']].drop_duplicates()
    print(f'Found {len(combinations)} unique combinations:')
    for _, (node_ratio, method) in combinations.iterrows():
        print(f'  - Node ratio: {node_ratio}, Method: {method}')

    # Get unique node ratios and methods
    unique_ratios = sorted(df['node_sample_ratio'].unique())
    unique_methods = sorted(df['method'].unique())

    print(f'Unique ratios: {unique_ratios}')
    print(f'Unique methods: {unique_methods}')

    # Define color families for each ratio (using valid matplotlib colormap names)
    color_families = {
        '0.125': 'Blues',
        '0.2': 'Greens',
        '0.5': 'Oranges',
        '1.0': 'YlOrBr',  # Yellow-Orange-Brown (closest to yellows)
        'full': 'Reds',
    }

    # Create a single figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()  # Flatten for easier indexing

    # Plot each metric in its own subplot
    for i, (metric, metric_name) in enumerate(metrics.items()):
        ax = axes[i]

        # Plot each combination as a separate line
        for ratio in unique_ratios:
            if ratio not in color_families:
                continue

            # Get color family for this ratio
            color_family = color_families[ratio]

            # Get methods for this ratio
            methods_for_ratio = df[df['node_sample_ratio'] == ratio]['method'].unique()

            # Create different shades for each method using the modern matplotlib approach
            if len(methods_for_ratio) == 1:
                # Single method, use middle shade
                colors = plt.colormaps[color_family](0.6)
            else:
                # Multiple methods, use different shades
                colors = plt.colormaps[color_family](np.linspace(0.3, 0.9, len(methods_for_ratio)))

            for j, method in enumerate(methods_for_ratio):
                subset_df = df[
                    (df['node_sample_ratio'] == ratio) & (df['method'] == method)
                ].sort_values('adj_thresh')

                if subset_df.empty:
                    continue

                # Get color (handle both single color and array of colors)
                if len(methods_for_ratio) == 1:
                    color = colors
                else:
                    color = colors[j]

                # Create label for this combination
                label = f'Ratio: {ratio}, Method: {method}'

                # Plot the line
                ax.plot(
                    subset_df['adj_thresh'],
                    subset_df[metric],
                    marker='o',
                    label=label,
                    linewidth=2,
                    markersize=4,
                    color=color,
                )

        # Customize the subplot
        ax.set_xlabel('Adjacency Threshold', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Set log scale for certain metrics if they span large ranges
        if metric in ['num_edges', 'avg_degree']:
            ax.set_yscale('log')

    # Remove the empty subplot (6th subplot)
    fig.delaxes(axes[5])

    # Create a single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.98, 0.02), fontsize=14)

    # Add overall title with dataset name
    title = (
        f'Graph Statistics vs Adjacency Threshold - {dataset_name}\n(All Parameter Combinations)'
    )
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.88, bottom=0.15)  # Make room for legend and titles

    # Save the plot
    output_file = os.path.join(output_dir, 'all_metrics_vs_adj_threshold.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved plot: {output_file}')


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create simple aggregated plots from dataset statistics'
    )
    parser.add_argument(
        '--csv-file',
        default='./stats/addneuromed/graph_stats_comprehensive.csv',
        help='CSV file to process',
    )
    parser.add_argument(
        '--output-dir', default='./plots/aggregated', help='Output directory for plots'
    )

    args = parser.parse_args()

    print('Loading data...')
    df = load_data(args.csv_file)

    print('\nData summary:')
    print(f'- Total rows: {len(df)}')
    print(f"- Node ratios: {df['node_sample_ratio'].unique()}")
    print(f"- Methods: {df['method'].unique()}")
    print(f"- Adjacency thresholds: {len(df['adj_thresh'].unique())} unique values")

    print('\nCreating aggregated plots...')

    # Extract dataset name from CSV file path
    csv_path = args.csv_file
    if '/stats/' in csv_path:
        dataset_name = csv_path.split('/stats/')[1].split('/')[0]
    else:
        dataset_name = 'Unknown Dataset'

    print(f'Dataset name: {dataset_name}')
    create_aggregated_plots(df, args.output_dir, dataset_name)

    print(f'\nAnalysis complete! Check {args.output_dir} for results.')


if __name__ == '__main__':
    main()

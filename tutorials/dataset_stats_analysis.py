#!/usr/bin/env python3
"""Standalone script to analyze dataset statistics across different parameters. Generates plots for
all combinations of node_sample_ratio and sampling_method.

This script uses joblib for parallel processing to significantly speed up computation.

Usage examples:
    # Run with default parameters (all CPUs, all datasets)
    python dataset_stats_analysis.py

    # Run with specific number of jobs
    python dataset_stats_analysis.py --n-jobs 8

    # Run only specific datasets
    python dataset_stats_analysis.py --datasets addneuromed parkinsons

    # Run with fewer adjacency thresholds for faster testing
    python dataset_stats_analysis.py --adj-thresholds 21

    # Skip plot generation (only compute statistics)
    python dataset_stats_analysis.py --skip-plots

    # Custom node ratios and methods
    python dataset_stats_analysis.py --node-ratios 1.0 0.5 --methods variance random
"""

import csv
import itertools
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def load_dataset(
    dataset_name: str,
    adj_thresh: float = 0.5,
    node_sample_ratio: str = "full",
    method: str = "variance",
) -> Any:
    """Load the dataset with specified parameters.

    Args:
        dataset_name: Name of the dataset
        adj_thresh: Adjacency threshold
        node_sample_ratio: Node sampling ratio ("full", "1.0", "0.5", "0.2", "0.125")
        method: Sampling method ("variance", "random", "correlation")

    Returns:
        Loaded dataset
    """
    from omegaconf import OmegaConf

    from ogbench.data.datasets.hf_omics import HFOmicsDataset

    # Convert node_sample_ratio to appropriate format
    if node_sample_ratio == "full":
        ratio_value = 1.0
    else:
        ratio_value = float(node_sample_ratio)

    # Convert train_val_test_split to OmegaConf object
    train_val_test_split = OmegaConf.create([0.7, 0.15, 0.15])

    # Create dataset directly without complex Hydra configuration
    dataset = HFOmicsDataset(
        root="/home/johmathe/bgbench/run_data/omics",
        data_name=dataset_name,
        method=method,
        adjacency_threshold=adj_thresh,
        node_sample_ratio=ratio_value,
        train_val_test_split=train_val_test_split,
        imputation_method="mean",
    )

    return dataset


def get_graph_stats(dataset: Any) -> Dict[str, float]:
    """Get statistics of the graph from the dataset.

    Args:
        dataset: Dataset object containing graph data

    Returns:
        Dictionary containing graph statistics
    """
    try:
        # First, try to access the processed data directly from the dataset
        if hasattr(dataset, "data") and dataset.data is not None:
            # Use the processed data from the dataset
            data = dataset[0]  # Get the first (and only) graph
            edge_index = data.edge_index
            num_nodes = data.x.shape[0]

            # Create graph from edge_index
            edge_list = edge_index.t().numpy()
            graph = nx.Graph()
            graph.add_nodes_from(range(num_nodes))
            graph.add_edges_from(edge_list)

        else:
            # Fallback: try to load from the raw directory
            root = "/home/johmathe/bgbench/run_data/omics/"
            name = osp.join(
                root,
                f"{dataset.data_name}",
                f"adj_thresh_{dataset.adjacency_threshold}",
                f"{dataset.method}",
                f"p_{dataset.node_sample_ratio}",
                f"train_split_{dataset.train_val_test_split[0]}",
                "raw/adj_matrix.npy",
            )

            try:
                adj_matrix = np.load(name)
                # Generate a graph from the adjacency matrix
                graph = nx.from_numpy_array(adj_matrix)
                graph.remove_edges_from(nx.selfloop_edges(graph))
            except FileNotFoundError:
                print(f"Warning: Adjacency matrix not found at {name}")
                return {
                    "num_nodes": 0,
                    "num_edges": 0,
                    "avg_degree": 0.0,
                    "density": 0.0,
                    "number_connected_components": 0,
                }

        # Calculate statistics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        avg_degree = np.mean([d for _, d in graph.degree()]) if num_nodes > 0 else 0.0
        density = nx.density(graph)
        number_connected_components = nx.number_connected_components(graph)

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "density": density,
            "number_connected_components": number_connected_components,
        }

    except Exception as e:
        print(f"Error getting graph stats: {e}")
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "avg_degree": 0.0,
            "density": 0.0,
            "number_connected_components": 0,
        }


def process_single_combination(args_tuple: Tuple[str, str, str, float]) -> Dict[str, Any]:
    """Process a single parameter combination. This function is designed to work with joblib's
    parallel processing.

    Args:
        args_tuple: Tuple containing (dataset_name, node_ratio, method, adj_thresh)

    Returns:
        Statistics dictionary
    """
    dataset_name, node_ratio, method, adj_thresh = args_tuple

    try:
        # Load the dataset directly without Hydra
        dataset = load_dataset(dataset_name, adj_thresh, node_ratio, method)

        # Debug: print dataset info
        print(f"Dataset loaded: {dataset}")
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            print(f"First data item: {dataset[0]}")

        # Get graph statistics
        stats = get_graph_stats(dataset)

        # Add metadata
        stats.update(
            {
                "dataset": dataset_name,
                "adj_thresh": adj_thresh,
                "node_sample_ratio": node_ratio,
                "method": method,
            }
        )

        return stats

    except Exception as e:
        # Return error entry
        return {
            "dataset": dataset_name,
            "adj_thresh": adj_thresh,
            "node_sample_ratio": node_ratio,
            "method": method,
            "num_nodes": None,
            "num_edges": None,
            "avg_degree": None,
            "density": None,
            "number_connected_components": None,
            "error": str(e),
        }


def compute_stats_for_combinations(
    dataset_name: str,
    node_sample_ratios: List[str],
    sampling_methods: List[str],
    adj_thresholds: List[float],
    n_jobs: int = -1,
) -> List[Dict[str, Any]]:
    """Compute statistics for all combinations of parameters using parallel processing.

    Args:
        dataset_name: Name of the dataset
        node_sample_ratios: List of node sample ratios
        sampling_methods: List of sampling methods
        adj_thresholds: List of adjacency thresholds
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        List of statistics dictionaries
    """
    # Create all parameter combinations as tuples for joblib
    combinations = [
        (dataset_name, node_ratio, method, adj_thresh)
        for node_ratio, method, adj_thresh in itertools.product(
            node_sample_ratios, sampling_methods, adj_thresholds
        )
    ]

    print(f"Processing {len(combinations)} combinations for {dataset_name} using {n_jobs} jobs")

    # Process in parallel
    all_stats = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_single_combination)(args_tuple) for args_tuple in combinations
    )

    return all_stats


def save_stats_to_csv(all_stats: List[Dict[str, Any]], output_file: str) -> None:
    """Save statistics to CSV file.

    Args:
        all_stats: List of statistics dictionaries
        output_file: Output CSV file path
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fieldnames = [
        "dataset",
        "adj_thresh",
        "node_sample_ratio",
        "method",
        "num_nodes",
        "num_edges",
        "avg_degree",
        "density",
        "number_connected_components",
    ]

    # Add error column if any errors exist
    if any("error" in stats for stats in all_stats):
        fieldnames.append("error")

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)

    print(f"Statistics saved to {output_file}")


def create_plots_for_dataset(dataset_name: str, csv_file: str) -> None:
    """Create plots for a specific dataset.

    Args:
        dataset_name: Name of the dataset
        csv_file: Path to CSV file with statistics
    """
    try:
        df = pd.read_csv(csv_file)
        # Filter for this dataset
        df = df[df["dataset"] == dataset_name]

        if df.empty:
            print(f"No data found for dataset {dataset_name}")
            return

        # Sort by adjacency threshold
        df = df.sort_values(by="adj_thresh", ascending=True)

        # Create plots for each combination of node_sample_ratio and method
        combinations = df[["node_sample_ratio", "method"]].drop_duplicates()

        for _, (node_ratio, method) in combinations.iterrows():
            subset_df = df[(df["node_sample_ratio"] == node_ratio) & (df["method"] == method)]

            if subset_df.empty:
                continue

            # Create the plot
            plt.figure(figsize=(14, 10))
            plt.suptitle(
                f"{dataset_name} - Node Ratio: {node_ratio}, Method: {method}", fontsize=16, y=1.02
            )

            # Plot number of edges
            plt.subplot(3, 2, 1)
            plt.plot(
                subset_df["adj_thresh"],
                subset_df["num_edges"],
                label="Number of Edges",
                color="green",
            )
            plt.xlabel("Adjacency Threshold")
            plt.ylabel("Number of Edges")
            plt.title("Number of Edges vs. Adjacency Threshold")
            plt.grid(True)

            # Plot average degree
            plt.subplot(3, 2, 2)
            plt.plot(
                subset_df["adj_thresh"],
                subset_df["avg_degree"],
                label="Average Degree",
                color="orange",
            )
            plt.xlabel("Adjacency Threshold")
            plt.ylabel("Average Degree")
            plt.title("Average Degree vs. Adjacency Threshold")
            plt.grid(True)

            # Plot density
            plt.subplot(3, 2, 3)
            plt.plot(subset_df["adj_thresh"], subset_df["density"], label="Density", color="red")
            plt.xlabel("Adjacency Threshold")
            plt.ylabel("Density")
            plt.title("Density vs. Adjacency Threshold")
            plt.grid(True)

            # Plot number of connected components
            plt.subplot(3, 2, 4)
            plt.plot(
                subset_df["adj_thresh"],
                subset_df["number_connected_components"],
                label="Connected Components",
                color="purple",
            )
            plt.xlabel("Adjacency Threshold")
            plt.ylabel("Number of Connected Components")
            plt.title("Connected Components vs. Adjacency Threshold")
            plt.grid(True)

            # Plot number of nodes
            plt.subplot(3, 2, 5)
            plt.plot(
                subset_df["adj_thresh"],
                subset_df["num_nodes"],
                label="Number of Nodes",
                color="blue",
            )
            plt.xlabel("Adjacency Threshold")
            plt.ylabel("Number of Nodes")
            plt.title("Number of Nodes vs. Adjacency Threshold")
            plt.grid(True)

            # Adjust layout and save the plot
            plt.tight_layout()

            # Create output directory for plots
            plot_dir = f"./plots/{dataset_name}"
            os.makedirs(plot_dir, exist_ok=True)

            plot_filename = (
                f"{plot_dir}/{dataset_name}_node_ratio_{node_ratio}_method_{method}.png"
            )
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Plot saved: {plot_filename}")

    except Exception as e:
        print(f"Error creating plots for {dataset_name}: {e}")


def main():
    """Main function to run the analysis."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze dataset statistics with parallel processing"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel jobs (-1 for all CPUs, default: -1)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["addneuromed", "parkinsons", "covidaki", "motrpac"],
        help="List of datasets to process",
    )
    parser.add_argument(
        "--node-ratios",
        nargs="+",
        default=["1.0", "0.5", "0.2", "0.125", "full"],
        help="List of node sample ratios",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["variance", "random", "correlation"],
        help="List of sampling methods",
    )
    parser.add_argument(
        "--adj-thresholds",
        type=int,
        default=10,
        help="Number of adjacency thresholds from 0.0 to 1.0 (default: 101)",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip plot generation (only compute statistics)"
    )

    args = parser.parse_args()

    # Define parameters
    datasets = args.datasets
    node_sample_ratios = args.node_ratios
    sampling_methods = args.methods
    adj_thresholds = [round(x, 2) for x in np.linspace(0.0, 1.0, args.adj_thresholds)]
    print(adj_thresholds)
    n_jobs = args.n_jobs

    print(f"Processing {len(datasets)} datasets")
    print(f"Node sample ratios: {node_sample_ratios}")
    print(f"Sampling methods: {sampling_methods}")
    print(f"Adjacency thresholds: {len(adj_thresholds)} values from 0.0 to 1.0")
    print(f"Parallel jobs: {n_jobs}")
    print(
        f"Total combinations: {len(datasets) * len(node_sample_ratios) * len(sampling_methods) * len(adj_thresholds)}"
    )

    # Process each dataset
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")

        # Compute statistics for all combinations
        all_stats = compute_stats_for_combinations(
            dataset_name, node_sample_ratios, sampling_methods, adj_thresholds, n_jobs
        )

        # Save to CSV
        output_file = f"./stats/{dataset_name}/graph_stats_comprehensive.csv"
        save_stats_to_csv(all_stats, output_file)

        # Create plots (unless skipped)
        if not args.skip_plots:
            create_plots_for_dataset(dataset_name, output_file)

    print("\nAnalysis complete!")
    print("Check the ./stats/ and ./plots/ directories for results.")

    # Print summary statistics
    total_combinations = (
        len(datasets) * len(node_sample_ratios) * len(sampling_methods) * len(adj_thresholds)
    )
    print("\nSummary:")
    print(f"- Total parameter combinations processed: {total_combinations}")
    print(f"- Datasets: {len(datasets)}")
    print(f"- Node sample ratios: {len(node_sample_ratios)}")
    print(f"- Sampling methods: {len(sampling_methods)}")
    print(f"- Adjacency thresholds: {len(adj_thresholds)}")
    print(f"- Parallel jobs used: {n_jobs}")


if __name__ == "__main__":
    # Quick test to see if dataset loading works
    try:
        print("Testing dataset loading...")
        dataset = load_dataset("addneuromed", 0.5, "0.2", "variance")
        print(f"Dataset loaded successfully: {dataset}")
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            print(f"First data item: {dataset[0]}")
            stats = get_graph_stats(dataset)
            print(f"Graph stats: {stats}")
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50)
    print("Running main analysis...")
    print("=" * 50)
    main()

import csv
import os

import networkx as nx
import numpy as np
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra  # Import GlobalHydra explicitly
from hydra.utils import instantiate
from topobench.utils.config_resolvers import (
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    infer_in_channels,
)

# Clear GlobalHydra instance if already initialized
if GlobalHydra().is_initialized():
    GlobalHydra().clear()

initialize(config_path="../configs", job_name="job")


def load_dataset(dataset_name, adjacency_threshold=0.5):
    """Load the FTD dataset with a specified adjacency threshold."""
    cfg = compose(
        config_name="train.yaml",
        overrides=[
            "model=graph/gat",
            f"dataset=graph/{dataset_name}",
            f"dataset.loader.parameters.method={adj_metric}",
            f"dataset.loader.parameters.adjacency_threshold={adjacency_threshold}",
        ],
        return_hydra_config=True,
    )
    loader = instantiate(cfg.dataset.loader)
    _, _ = loader.load()
    return loader.datasets[0]


def get_graph_stats(dataset):
    """Get statistics of the graph."""
    # Load the adjacency matrix
    adj_matrix = dataset.get_adjacency_matrix(
        dataset.adj_path, dataset.config.adjacency_threshold, dataset.config
    )
    # Generate a graph from the adjacency matrix
    graph = nx.from_numpy_matrix(adj_matrix.cpu().numpy())
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # Calculate statistics
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = np.mean([d for n, d in graph.degree()])
    density = nx.density(graph)
    number_connected_components = nx.number_connected_components(graph)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "density": density,
        "number_connected_components": number_connected_components,
    }


if __name__ == "__main__":
    # Define the output file and fieldnames
    adj_metric = "spearman_correlation"  # Change this to the desired adjacency metric
    directory = "./tutorials/stats/" + adj_metric + "/"
    output_file = directory + "/graph_stats.csv"
    fieldnames = [
        "adjacency_threshold",
        "num_nodes",
        "num_edges",
        "avg_degree",
        "density",
        "number_connected_components",
    ]

    for idx in range(23, -1, -1):
        adjacency_threshold = idx / 100.0
        print(
            f"Processing adj_metric={adj_metric} with adjacency_threshold={adjacency_threshold}..."
        )
        # Load the dataset
        dataset = load_dataset(adj_metric=adj_metric, adjacency_threshold=adjacency_threshold)
        # Get graph statistics
        stats = get_graph_stats(dataset)
        # Add the adjacency threshold to the stats
        stats["adjacency_threshold"] = adjacency_threshold

        # Append the stats to the CSV file
        with open(output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(stats)  # Write the current row

        print(f"Saved stats for adjacency_threshold={adjacency_threshold} to {output_file}")

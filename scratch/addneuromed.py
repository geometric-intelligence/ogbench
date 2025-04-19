import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from sklearn.feature_selection import mutual_info_classif
import PyWGCNA


def select_nodes(node_features, graph_label, n_selected_nodes=100):
    """Select nodes based on graph label."""
    """
    Compute feature importance scores between node features and graph labels using mutual information.
    Since graph labels are discrete (0,1,2), mutual information is more appropriate than correlation.
    
    Args:
        node_features: numpy array of shape (n_samples, n_features)
        graph_label: numpy array of shape (n_samples,) containing discrete labels
        
    Returns:
        selected_features: indices of most informative features
    """
    
    # Compute mutual information between each feature and the graph label
    mi_scores = mutual_info_classif(node_features, graph_label)
    
    # Sort features by mutual information score
    ranked_features = np.argsort(mi_scores)[::-1]
    
    # Select top features (can adjust threshold as needed)
    n_select = min(n_selected_nodes, len(ranked_features))  # Select top 100 or all if less
    selected_features = ranked_features[:n_select]
    
    return selected_features

def calculate_adjacency_matrix(node_features, save_to):
    """Calculate and save adjacency matrix."""
    node_features_df = pd.DataFrame(node_features)
    softThreshold = PyWGCNA.WGCNA.pickSoftThreshold(node_features_df)
    print("Soft threshold:", softThreshold[0])
    adjacency = PyWGCNA.WGCNA.adjacency(
        node_features, power=softThreshold[0], adjacencyType="signed hybrid"
    )

    adjacency_df = pd.DataFrame(adjacency)
    print(f"Saving adjacency matrix to: {save_to}...")
    adjacency_df.to_csv(save_to, header=None, index=False)

def create_graph_data(node_features, graph_label, adj_matrix):
    """Create Data object for each graph.

    Compute attributes x, edge_index, and y for each graph.
    """
    x = node_features  # what is on the nodes
    adj_tensor = torch.tensor(adj_matrix)
    # Find the indices where the matrix has non-zero elements
    pairs_indices = torch.nonzero(adj_tensor, as_tuple=False)
    # Extract the pairs of connected nodes
    edge_index = torch.tensor(pairs_indices.tolist())
    edge_index = torch.transpose(edge_index, 0, 1)  # reshape(edge_index, (2, -1))
    return Data(x=x, edge_index=edge_index, y=graph_label)
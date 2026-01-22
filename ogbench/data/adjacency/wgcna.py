"""WGCNA-based adjacency matrix builder."""

import numpy as np
import pandas as pd
import PyWGCNA

from ogbench.data.adjacency.base import AbstractAdjacencyBuilder


class WGCNAAdjacencyBuilder(AbstractAdjacencyBuilder):
    """Build adjacency matrix using WGCNA with soft-thresholding.

    Uses WGCNA to find optimal power for scale-free topology and applies soft-thresholding to
    create a signed hybrid adjacency matrix.
    """

    def build(self, node_features: pd.DataFrame) -> np.ndarray:
        """Build adjacency matrix using WGCNA."""
        # Use WGCNA to find optimal power for scale-free topology
        try:
            soft_threshold = PyWGCNA.WGCNA.pickSoftThreshold(node_features)
            power = soft_threshold[0]
        except Exception as e:
            raise ValueError(f'Error finding optimal power for WGCNA: {e}') from e

        # Apply soft-thresholding
        adjacency = PyWGCNA.WGCNA.adjacency(
            node_features,
            power=power,
            adjacencyType='signed hybrid',
        )

        return adjacency

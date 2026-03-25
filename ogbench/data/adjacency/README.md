# Adjacency Matrix Builders Module

This module provides an interchangeable adjacency matrix construction system with a standardized interface. Each builder takes node features as input and returns a continuous adjacency matrix as output.

## Architecture

- **AbstractAdjacencyBuilder**: Abstract base class defining the interface
- **Concrete Builders**: Individual implementations (WGCNAAdjacencyBuilder, etc.)
- **Registry Pattern**: `ADJACENCY_BUILDER_REGISTRY` maps string names to builder classes
- **Factory Function**: `get_adjacency_builder()` instantiates builders by name

## Adding a New Adjacency Builder

To add a new adjacency construction strategy:

1. Create a new file in `ogbench/data/adjacency/` (e.g., `correlation.py`)
2. Inherit from `AbstractAdjacencyBuilder` and implement the `build()` method:

```python
import numpy as np
import pandas as pd
from ogbench.data.adjacency.base import AbstractAdjacencyBuilder

class CorrelationAdjacencyBuilder(AbstractAdjacencyBuilder):
    def build(self, node_features: pd.DataFrame) -> np.ndarray:
        # Compute correlation matrix
        corr_matrix = node_features.corr().values
        # Apply power transformation
        power = 6
        adjacency = np.power(np.abs(corr_matrix), power)
        # Apply sign
        adjacency = np.sign(corr_matrix) * adjacency
        return adjacency
```

3. Register it in `ogbench/data/adjacency/__init__.py`:

```python
from ogbench.data.adjacency.correlation import CorrelationAdjacencyBuilder

ADJACENCY_BUILDER_REGISTRY['correlation'] = CorrelationAdjacencyBuilder
```

4. Use it by name: `adjacency_method='correlation'`

## Interface

- **Input**: `node_features: pd.DataFrame` (samples x features)
- **Output**: `np.ndarray` (continuous adjacency matrix, features x features)

The continuous adjacency matrix is then binarized in `calculate_adjacency_matrix()` based on `adjacency_threshold`.

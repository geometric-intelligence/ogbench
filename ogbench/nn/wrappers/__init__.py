"""Wrappers implemented for OGBench."""

from ogbench.nn.wrappers.base import AbstractWrapper
from ogbench.nn.wrappers.cell import (
    WRAPPER_CLASSES as CELL_WRAPPER_CLASSES,
)
from ogbench.nn.wrappers.cell import (
    CANWrapper,
    CCCNWrapper,
    CCXNWrapper,
    CWNWrapper,
)
from ogbench.nn.wrappers.combinatorial import (
    WRAPPER_CLASSES as COMBINATORIAL_WRAPPER_CLASSES,
)
from ogbench.nn.wrappers.combinatorial import (
    TuneWrapper,
)
from ogbench.nn.wrappers.graph import (
    WRAPPER_CLASSES as GRAPH_WRAPPER_CLASSES,
)
from ogbench.nn.wrappers.graph import (
    GATv4Wrapper,
    GNNWrapper,
    GraphMLPWrapper,
    SAGNWrapper,
)
from ogbench.nn.wrappers.hypergraph import (
    WRAPPER_CLASSES as HYPERGRAPH_WRAPPER_CLASSES,
)
from ogbench.nn.wrappers.hypergraph import (
    HypergraphWrapper,
)

# Aggregate all wrapper classes from subpackages
WRAPPER_CLASSES: dict[str, type] = {
    'AbstractWrapper': AbstractWrapper,
    **CELL_WRAPPER_CLASSES,
    **COMBINATORIAL_WRAPPER_CLASSES,
    **GRAPH_WRAPPER_CLASSES,
    **HYPERGRAPH_WRAPPER_CLASSES,
}

# Generate __all__
__all__ = [
    'WRAPPER_CLASSES',
    'AbstractWrapper',
    'CANWrapper',
    'CCCNWrapper',
    'CCXNWrapper',
    'CWNWrapper',
    'TuneWrapper',
    'GATv4Wrapper',
    'GNNWrapper',
    'GraphMLPWrapper',
    'SAGNWrapper',
    'HypergraphWrapper',
]

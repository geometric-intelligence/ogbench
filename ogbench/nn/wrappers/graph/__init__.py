"""Wrappers for graph neural networks."""

from ogbench.nn.wrappers.graph.gatv4_wrapper import GATv4Wrapper
from ogbench.nn.wrappers.graph.gnn_wrapper import GNNWrapper
from ogbench.nn.wrappers.graph.graph_mlp_wrapper import GraphMLPWrapper
from ogbench.nn.wrappers.graph.sagn_wrapper import SAGNWrapper

# Create dictionary of all wrapper classes
WRAPPER_CLASSES: dict[str, type] = {
    'GATv4Wrapper': GATv4Wrapper,
    'GNNWrapper': GNNWrapper,
    'GraphMLPWrapper': GraphMLPWrapper,
    'SAGNWrapper': SAGNWrapper,
}

# Generate __all__
__all__ = [
    'WRAPPER_CLASSES',
    'GATv4Wrapper',
    'GNNWrapper',
    'GraphMLPWrapper',
    'SAGNWrapper',
]

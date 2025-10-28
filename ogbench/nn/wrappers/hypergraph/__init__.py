"""Wrappers for hypergraph neural networks."""

from ogbench.nn.wrappers.hypergraph.hypergraph_wrapper import HypergraphWrapper

# Create dictionary of all wrapper classes
WRAPPER_CLASSES: dict[str, type] = {
    'HypergraphWrapper': HypergraphWrapper,
}

# Generate __all__
__all__ = [
    'WRAPPER_CLASSES',
    'HypergraphWrapper',
]

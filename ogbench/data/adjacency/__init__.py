"""Adjacency matrix construction modules."""

from ogbench.data.adjacency.base import AbstractAdjacencyBuilder
from ogbench.data.adjacency.string import STRINGAdjacencyBuilder
from ogbench.data.adjacency.wgcna import WGCNAAdjacencyBuilder

__all__ = [
    'AbstractAdjacencyBuilder',
    'WGCNAAdjacencyBuilder',
    'get_adjacency_builder',
    'ADJACENCY_BUILDER_REGISTRY',
    'STRINGAdjacencyBuilder',
]

# Registry mapping string names to adjacency builder classes
ADJACENCY_BUILDER_REGISTRY = {
    'wgcna': WGCNAAdjacencyBuilder,
    'string': STRINGAdjacencyBuilder,
}


def get_adjacency_builder(method: str) -> AbstractAdjacencyBuilder:
    """Get an adjacency builder instance by method name."""
    if method not in ADJACENCY_BUILDER_REGISTRY:
        raise ValueError(
            f'Invalid method: {method}. Available methods: {list(ADJACENCY_BUILDER_REGISTRY.keys())}'
        )
    return ADJACENCY_BUILDER_REGISTRY[method]()

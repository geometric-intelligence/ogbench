"""Node selection modules for feature selection."""

import zlib

from ogbench.data.selectors.base import AbstractNodeSelector
from ogbench.data.selectors.correlation import CorrelationSelector
from ogbench.data.selectors.distance_correlation import DistanceCorrelationSelector
from ogbench.data.selectors.random import RandomSelector
from ogbench.data.selectors.variance import VarianceSelector

__all__ = [
    'AbstractNodeSelector',
    'VarianceSelector',
    'CorrelationSelector',
    'DistanceCorrelationSelector',
    'RandomSelector',
    'get_selector',
    'selection_seed',
    'SELECTOR_REGISTRY',
]

# Registry mapping string names to selector classes
SELECTOR_REGISTRY = {
    'variance': VarianceSelector,
    'correlation': CorrelationSelector,
    'distance_correlation': DistanceCorrelationSelector,
    'random': RandomSelector,
}


def selection_seed(data_name: str, k: int, fold: int, node_sample_ratio: float | str) -> int:
    """Return the seed for ``method=random`` on one dataset, fold, and node budget.

    The GNN cache and the sklearn baselines both call this, so they select the same nodes.
    """
    ratio = node_sample_ratio if node_sample_ratio == 'full' else f'{float(node_sample_ratio):g}'
    return zlib.crc32(f'{data_name}|k={int(k)}|fold={int(fold)}|ratio={ratio}'.encode())


def get_selector(method: str, seed: int | None = None) -> AbstractNodeSelector:
    """Get a node selector instance by method name.

    ``seed`` is used only by the random selector.
    """
    if method not in SELECTOR_REGISTRY:
        raise ValueError(
            f'Invalid method: {method}. Available methods: {list(SELECTOR_REGISTRY.keys())}'
        )
    if method == 'random':
        return RandomSelector(seed=seed)
    return SELECTOR_REGISTRY[method]()

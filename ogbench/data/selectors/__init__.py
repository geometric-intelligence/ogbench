"""Node selection modules for feature selection."""

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
    'SELECTOR_REGISTRY',
]

# Registry mapping string names to selector classes
SELECTOR_REGISTRY = {
    'variance': VarianceSelector,
    'correlation': CorrelationSelector,
    'distance_correlation': DistanceCorrelationSelector,
    'random': RandomSelector,
}


def get_selector(method: str) -> AbstractNodeSelector:
    """Get a node selector instance by method name."""
    if method not in SELECTOR_REGISTRY:
        raise ValueError(
            f'Invalid method: {method}. Available methods: {list(SELECTOR_REGISTRY.keys())}'
        )
    return SELECTOR_REGISTRY[method]()

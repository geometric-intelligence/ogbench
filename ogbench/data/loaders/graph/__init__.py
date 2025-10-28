"""Init file for graph load module."""

from ogbench.data.loaders.graph.modecule_datasets import MoleculeDatasetLoader
from ogbench.data.loaders.graph.omics_datasets import OmicsDatasetLoader
from ogbench.data.loaders.graph.planetoid_datasets import PlanetoidDatasetLoader
from ogbench.data.loaders.graph.tu_datasets import TUDatasetLoader

# Create dictionary of all graph loaders
GRAPH_LOADERS: dict[str, type] = {
    'MoleculeDatasetLoader': MoleculeDatasetLoader,
    'OmicsDatasetLoader': OmicsDatasetLoader,
    'PlanetoidDatasetLoader': PlanetoidDatasetLoader,
    'TUDatasetLoader': TUDatasetLoader,
}

GRAPH_LOADERS_list: list[str] = list(GRAPH_LOADERS.keys())

# Generate __all__
__all__ = [
    'GRAPH_LOADERS',
    'GRAPH_LOADERS_list',
    'MoleculeDatasetLoader',
    'OmicsDatasetLoader',
    'PlanetoidDatasetLoader',
    'TUDatasetLoader',
]

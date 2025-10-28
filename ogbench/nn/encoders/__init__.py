"""Init file for encoder module."""

from ogbench.nn.encoders.all_cell_encoder import AllCellFeatureEncoder
from ogbench.nn.encoders.dgm_encoder import DGMStructureFeatureEncoder
from ogbench.nn.encoders.flat_encoder import FlatEncoder

# Create dictionary of all feature encoders
FEATURE_ENCODERS: dict[str, type] = {
    'AllCellFeatureEncoder': AllCellFeatureEncoder,
    'DGMStructureFeatureEncoder': DGMStructureFeatureEncoder,
    'FlatEncoder': FlatEncoder,
}

FEATURE_ENCODERS_list: list[str] = list(FEATURE_ENCODERS.keys())

# Generate __all__
__all__ = [
    'FEATURE_ENCODERS',
    'FEATURE_ENCODERS_list',
    'AllCellFeatureEncoder',
    'DGMStructureFeatureEncoder',
    'FlatEncoder',
]

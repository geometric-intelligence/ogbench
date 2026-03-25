"""Some models implemented for OGBenchX."""

from ogbench.nn.backbones.graph import (
    BACKBONE_CLASSES as GRAPH_BACKBONE_CLASSES,
)
from ogbench.nn.backbones.graph import (
    MLP,
    SAGN,
    ChebNet,
    CustomGATConv,
    GATv4,
    GroupMLP,
    IdentityGAT,
    IdentityGCN,
    IdentityGIN,
    IdentitySAGE,
    MultiHeadBatchNorm,
    MultiHeadLinear,
)

# Aggregate all model classes from subpackages
MODEL_CLASSES: dict[str, type] = {
    **GRAPH_BACKBONE_CLASSES,
}

# Generate __all__
__all__ = [
    'MODEL_CLASSES',
    'ChebNet',
    'CustomGATConv',
    'GATv4',
    'GroupMLP',
    'IdentityGAT',
    'IdentityGCN',
    'IdentityGIN',
    'IdentitySAGE',
    'MLP',
    'MultiHeadBatchNorm',
    'MultiHeadLinear',
    'SAGN',
]

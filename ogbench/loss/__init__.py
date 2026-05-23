"""This module implements the loss functions for the ogbench package."""

from ogbench.loss.base import AbstractLoss
from ogbench.loss.dataset import LOSSES as DATASET_LOSSES
from ogbench.loss.dataset import DatasetLoss
from ogbench.loss.loss import TBLoss
from ogbench.loss.model import LOSSES as MODEL_LOSSES
from ogbench.loss.model import DGMLoss, GraphMLPLoss

# Aggregate all losses from subpackages
LOSSES: dict[str, type] = {
    **DATASET_LOSSES,
    **MODEL_LOSSES,
}

LOSSES_list: list[str] = list(LOSSES.keys())

# Generate __all__
__all__ = [
    'LOSSES',
    'LOSSES_list',
    'AbstractLoss',
    'DatasetLoss',
    'TBLoss',
    'DGMLoss',
    'GraphMLPLoss',
]

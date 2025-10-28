"""This module implements the loss functions for the ogbench package."""

from ogbench.loss.dataset.DatasetLoss import DatasetLoss

# Create dictionary of all dataset losses
LOSSES: dict[str, type] = {
    'DatasetLoss': DatasetLoss,
}

LOSSES_list: list[str] = list(LOSSES.keys())

# Generate __all__
__all__ = [
    'LOSSES',
    'LOSSES_list',
    'DatasetLoss',
]

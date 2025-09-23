"""Denormalized RMSE metric for regression tasks."""

from typing import Any

import torch
from torchmetrics import Metric
from torchmetrics.functional.regression.mse import (
    _mean_squared_error_compute,
    _mean_squared_error_update,
)


class DenormalizedRMSE(Metric):
    r"""Denormalized Root Mean Squared Error metric.

    This metric computes RMSE on the original scale by denormalizing predictions
    and targets using the target normalizer statistics.

    Parameters
    ----------
    target_mean : float
        Mean of the target values used for normalization.
    target_std : float
        Standard deviation of the target values used for normalization.
    num_outputs : int
        The number of outputs (default: 1).
    **kwargs : Any
        Additional keyword arguments.
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    sum_squared_error: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        target_mean: float,
        target_std: float,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(num_outputs, int) or num_outputs <= 0:
            raise ValueError(
                f"Expected num_outputs to be a positive integer but got {num_outputs}"
            )
        self.num_outputs = num_outputs
        self.target_mean = target_mean
        self.target_std = target_std

        self.add_state(
            "sum_squared_error",
            default=torch.zeros(num_outputs),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predictions from model (in normalized space).
        target : torch.Tensor
            Ground truth values (in normalized space).
        """
        # Denormalize predictions and targets
        preds_denorm = preds * self.target_std + self.target_mean
        target_denorm = target * self.target_std + self.target_mean

        sum_squared_error, num_obs = _mean_squared_error_update(
            preds_denorm, target_denorm, num_outputs=self.num_outputs
        )

        self.sum_squared_error += sum_squared_error
        self.total += num_obs

    def compute(self) -> torch.Tensor:
        """Compute the denormalized RMSE.

        Returns
        -------
        torch.Tensor
            Denormalized root mean squared error.
        """
        return _mean_squared_error_compute(self.sum_squared_error, self.total, squared=False)

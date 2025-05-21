"""Additional pytorch-lightning learning rate schedulers."""

import math
import warnings
from typing import Final, List

from torch.optim import Optimizer, lr_scheduler


class LinearWarmupCosineAnnealingLR(lr_scheduler.LRScheduler):
    """Learning rate scheduler with linear warmup followed by cosine annealing.

        Sets the learning rate of each parameter group to follow a linear warmup schedule
        between warmup_start_lr and base_lr followed by a cosine annealing schedule between
        base_lr and eta_min.

        .. warning::
            It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
            after each iteration as calling it after each epoch will keep the starting lr at
            warmup_start_lr for the first epoch which is 0 in most cases.

        .. warning::
            passing epoch to :func:`.step()` is being deprecated and comes with an
            EPOCH_DEPRECATION_WARNING. It calls the :func:`_get_closed_form_lr()` method for
            this scheduler instead of :func:`get_lr()`. Though this does not change the behavior
            of the scheduler, when passing epoch param to :func:`.step()`, the user should call
            the :func:`.step()` function before calling train and validation methods.

        Examples
        --------
    import atmo.atmonet.modules.unet    >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> #
        >>> layer = atmo.atmonet.modules.unet.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(
        ...     optimizer,
        ...     warmup_epochs=10,
        ...     max_epochs=40,
        ... )
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    DEFAULT_WARMUP_START_LR: Final[float] = 1.0e-7

    # Should be declared in the base LRScheduler class.
    _get_lr_called_within_step: bool

    warmup_epochs: int
    max_epochs: int
    warmup_start_lr: float
    eta_min: float

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = DEFAULT_WARMUP_START_LR,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """LinearWarmupCosineAnnealingLR constructor.

        :param optimizer: Wrapped optimizer.
        :param warmup_epochs: Maximum number of iterations for linear warmup
        :param max_epochs: Maximum number of iterations
        :param warmup_start_lr: Learning rate to start the linear warmup. Default: 0.
        :param eta_min: Minimum learning rate. Default: 0.
        :param last_epoch: The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        # run the base class constructor *after* initializing the new attributes,
        # because the base class constructor calls `get_lr` which uses the new attributes.
        super().__init__(optimizer, last_epoch)

    # LRScheduler type annotation is wrong.
    def get_lr(self) -> List[float]:  # type: ignore
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                'To get the last learning rate computed by the scheduler; '
                'please use `get_last_lr()`.',
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group['lr'] + (base_lr - self.warmup_start_lr) / self.warmup_epochs
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group['lr']
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group['lr'] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    # no @overrides decorator as it is not declared in the base class;
    # though it is used in the base class.
    def _get_closed_form_lr(self) -> List[float]:
        """Get the closed form learning rate.

        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]
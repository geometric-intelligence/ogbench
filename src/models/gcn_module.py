from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression import MeanSquaredError, R2Score


class GCNLitModule(LightningModule):
    """Example of a `LightningModule` for GNN regression."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `GCNLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        self.criterion = torch.nn.MSELoss()
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor, adj_t: torch.sparse.Tensor, batch_vector: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of features.
        :return: A tensor of predictions.
        """
        return self.net(x, adj_t, batch_vector)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_mse.reset()
        self.val_r2.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of features and target values.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target values.
        """
        x, adj_t, y, batch_vector = batch.x, batch.adj_t, batch.y, batch.batch
        predictions = self.forward(x, adj_t, batch_vector)
        loss = self.criterion(predictions, y.reshape(-1, 1))
        return loss, predictions, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of features and target values.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.train_mse(preds, targets.reshape(-1, 1))
        self.train_r2(preds, targets.reshape(-1, 1))
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True)

        return loss

                
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of features and target values.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_mse(preds, targets.reshape(-1, 1))
        self.val_r2(preds, targets.reshape(-1, 1))
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of features and target values.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.test_loss(loss)
        self.test_mse(preds, targets.reshape(-1, 1))
        self.test_r2(preds, targets.reshape(-1, 1))
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use."""
        # optimizer = self.hparams.optimizer(params=self.trainer.model.parameters()) # Old way
        optimizer = self.optimizer_partial(params=self.parameters()) # Use self.parameters()

        if self.scheduler_partial is not None:
            scheduler = self.scheduler_partial(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss", # Ensure you have validation
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
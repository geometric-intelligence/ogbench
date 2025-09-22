"""Test the TBEvaluator class."""

import pytest
import torch
import torch_geometric

from ogbench.loss.dataset import DatasetLoss


class TestDatasetLoss:
    """Test the TBEvaluator class."""

    def setup_method(self):
        """Setup the test."""
        dataset_loss = {"task": "classification", "loss_type": "cross_entropy"}
        self.dataset1 = DatasetLoss(dataset_loss)
        dataset_loss = {"task": "regression", "loss_type": "mse"}
        self.dataset2 = DatasetLoss(dataset_loss)
        dataset_loss = {"task": "regression", "loss_type": "mae"}
        self.dataset3 = DatasetLoss(dataset_loss)
        dataset_loss = {
            "task": "multilabel classification",
            "loss_type": "BCE",
        }
        self.dataset4 = DatasetLoss(dataset_loss)

        dataset_loss = {"task": "wrong", "loss_type": "wrong"}
        with pytest.raises(Exception):
            DatasetLoss(dataset_loss)

        dataset_loss = {"task": "classification", "loss_type": "cross_entropy"}
        self.dataset5 = DatasetLoss(dataset_loss)

        # Test with class weights
        dataset_loss = {
            "task": "classification",
            "loss_type": "cross_entropy",
            "class_weights": [1.0, 2.0, 1.5],
        }
        self.dataset6 = DatasetLoss(dataset_loss)

        repr = self.dataset1.__repr__()
        assert repr == "DatasetLoss(task=classification, loss_type=cross_entropy)"

        repr = self.dataset6.__repr__()
        assert "class_weights" in repr

    def test_forward(self):
        """Test the forward method."""
        batch = torch_geometric.data.Data()

        model_out = {
            "logits": torch.tensor([0.1, 0.2, 0.3]),
            "labels": torch.tensor([0.1, 0.2, 0.3]),
        }
        out = self.dataset1.forward(model_out, batch)
        assert out.item() >= 0

        model_out = {
            "logits": torch.tensor([0.1, 0.2, 0.3]),
            "labels": torch.tensor([0.1, 0.2, 0.3]),
        }
        out = self.dataset3.forward(model_out, batch)
        assert out.item() >= 0

        model_out = {
            "logits": torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
            "labels": torch.tensor([[0.1, float("nan"), 0.3], [0.1, 0.2, float("nan")]]),
        }
        out = self.dataset4.forward(model_out, batch)
        assert out.item() >= 0

        self.dataset5.task = "not defined"
        with pytest.raises(Exception):
            self.dataset5(model_out, batch)

    def test_class_weights(self):
        """Test the class weights functionality."""
        batch = torch_geometric.data.Data()

        # Test with class weights
        model_out = {
            "logits": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),  # Predictions
            "labels": torch.tensor([0, 1]),  # True labels
        }

        # Test that the loss with weights is different from without weights
        out_with_weights = self.dataset6.forward(model_out, batch)
        out_without_weights = self.dataset1.forward(model_out, batch)

        # The losses should be different due to different class weights
        assert not torch.allclose(out_with_weights, out_without_weights)

        # Test that class_weights tensor is properly created
        assert self.dataset6.class_weights is not None
        assert torch.allclose(self.dataset6.class_weights, torch.tensor([1.0, 2.0, 1.5]))

        # Test that criterion has the correct weight
        assert self.dataset6.criterion.weight is not None
        assert torch.allclose(self.dataset6.criterion.weight, torch.tensor([1.0, 2.0, 1.5]))

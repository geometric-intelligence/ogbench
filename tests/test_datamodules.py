from pathlib import Path

import pytest
import torch
import tempfile
import torch_geometric
from src.data import omics_datamodule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_pancancer_datamodule(batch_size: int) -> None:
    """Basic functionality test for ``PanCancerDataModule``.

    The test ensures that the datamodule can prepare the dataset (creating the
    expected files) and that a batch returned from the dataloader has the correct
    type and dtype.  Only minimal assertions are made to keep the test fast and
    independent of network connectivity.
    """

    with tempfile.TemporaryDirectory() as data_dir:
        dm = omics_datamodule.PanCancerDataModule(data_dir=data_dir, batch_size=batch_size)
        dm.prepare_data()

        assert Path(dm.selected_data_path).exists()
        assert Path(dm.targets_path).exists()
        assert Path(dm.adj_matrix_path).exists()

        dm.setup()
        assert dm.train_graph_data_list and dm.val_graph_data_list and dm.test_graph_data_list
        assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, torch_geometric.data.Batch)
        assert batch.x.dtype == torch.float32
        assert batch.y.dtype == torch.float32
        assert batch.num_graphs <= batch_size


@pytest.mark.parametrize("batch_size", [32])
def test_motrpac_datamodule(batch_size: int) -> None:
    """Similar smoke test for ``MotrPacDataModule``."""
    with tempfile.TemporaryDirectory() as data_dir:
        dm = omics_datamodule.MotrPacDataModule(data_dir=data_dir, batch_size=batch_size)
        dm.prepare_data()

        assert Path(dm.selected_data_path).exists()
        assert Path(dm.targets_path).exists()
        assert Path(dm.adj_matrix_path).exists()

        dm.setup()
        assert dm.train_graph_data_list and dm.val_graph_data_list and dm.test_graph_data_list
        assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, torch_geometric.data.Batch)
        assert batch.x.dtype == torch.float32
        assert batch.y.dtype == torch.float32        assert batch.num_graphs <= batch_size
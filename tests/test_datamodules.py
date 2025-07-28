import tempfile
from pathlib import Path

import pytest
import torch
import torch_geometric

from src.data import omics_datamodule


@pytest.mark.parametrize("batch_size", [32])
def test_motrpac_datamodule(batch_size: int) -> None:
    with tempfile.TemporaryDirectory() as data_dir:
        dm = omics_datamodule.MotrPacDataModule(data_dir=data_dir, batch_size=batch_size)
        dm.prepare_data()

        # Updated file checks
        assert Path(data_dir, "mortrpac_selected_data.parquet").exists()
        assert Path(data_dir, "mortrpac_targets.npy").exists()
        assert Path(data_dir, "mortrpac_adj_matrix.npy").exists()

        dm.setup()
        assert dm.data_train and dm.data_val and dm.data_test
        assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

        num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
        assert num_datapoints > 0

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, torch_geometric.data.Batch)
        assert batch.x.dtype == torch.float32
        assert batch.y.dtype == torch.float32
        assert batch.num_graphs <= batch_size

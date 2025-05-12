from pathlib import Path

import pytest
import torch
import tempfile

from src.data import omics_datamodule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_omics_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = omics_datamodule.PanCancerDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32])
def test_omics_datamodule(batch_size: int) -> None:
    """Tests `MortrPacDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    with tempfile.TemporaryDirectory() as data_dir:
        dm = omics_datamodule.MortrPacDataModule(data_dir=data_dir, batch_size=batch_size)
        dm.prepare_data()

        assert not dm.data_train and not dm.data_val and not dm.data_test
        assert Path(data_dir, "mortrpac").exists()
        assert Path(data_dir, "mortrpac", "raw").exists()

        dm.setup()
        assert dm.data_train and dm.data_val and dm.data_test
        assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

        num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
        assert num_datapoints == 70_000

        batch = next(iter(dm.train_dataloader()))
        x, y = batch
        assert len(x) == batch_size
        assert len(y) == batch_size
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64
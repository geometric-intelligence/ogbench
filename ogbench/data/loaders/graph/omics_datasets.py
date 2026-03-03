"""Loaders for Omics datasets."""


import numpy as np
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset

from ogbench.data.datasets import HFOmicsDataset
from ogbench.data.loaders.base import AbstractLoader


class OmicsDatasetLoader(AbstractLoader):
    """Load OMICS datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "cocitation")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load Omics dataset.

        Returns
        -------
        Dataset
            The loaded Omics dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        dataset = HFOmicsDataset(
            root=str(self.root_data_dir),
            **self.parameters,
        )
        dataset.split_idx = self._prepare_split_idx(len(dataset))
        return dataset

    def _prepare_split_idx(self, dataset_length: int) -> dict[str, np.ndarray]:
        """Prepare the split indices for the dataset.

        Parameters
        ----------
        dataset_length : int
            The length of the dataset.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping split names to index arrays.
        """
        split_sizes = [int(x * dataset_length) for x in self.parameters['train_val_test_split']]
        assert all(s > 0 for s in split_sizes), (
            f'All split sizes must be > 0, got {split_sizes} for dataset_length={dataset_length} '
            f'and splits={self.parameters["train_val_test_split"]}'
        )
        split_idx = {'train': np.arange(split_sizes[0])}
        split_idx['valid'] = np.arange(
            split_sizes[0],
            split_sizes[0] + split_sizes[1],
        )
        split_idx['test'] = np.arange(
            split_sizes[0] + split_sizes[1],
            dataset_length,
        )
        return split_idx

    def load(self, **kwargs) -> tuple[Data, str]:
        """Load data.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[torch_geometric.data.Data, str]
            Tuple containing the loaded data and the data directory.
        """
        dataset = self.load_dataset(**kwargs)
        data_dir = dataset.get_data_dir()

        return dataset, data_dir

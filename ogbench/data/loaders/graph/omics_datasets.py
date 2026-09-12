"""Loaders for Omics datasets."""


import json
import logging
import os.path as osp

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, Dataset

from ogbench.data.datasets import HFOmicsDataset
from ogbench.data.loaders.base import AbstractLoader

logger = logging.getLogger(__name__)


class OmicsDatasetLoader(AbstractLoader):
    """Load OMICS datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "cocitation")
            - split_type / k / fold: forwarded to HFOmicsDataset for fold-aware caches
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
        params = OmegaConf.to_container(self.parameters, resolve=True)
        assert isinstance(params, dict)
        dataset = HFOmicsDataset(
            root=str(self.root_data_dir),
            **params,
        )
        dataset.uses_precomputed_split = True
        dataset.split_idx = self._prepare_split_idx(dataset)
        return dataset

    def _prepare_split_idx(self, dataset: HFOmicsDataset) -> dict[str, np.ndarray]:
        """Build contiguous split indices from saved split_info.json cut points.

        Parameters
        ----------
        dataset : HFOmicsDataset
            Loaded omics dataset (artifacts already on disk).

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping split names to index arrays into the
            reordered train|val|test dataset.
        """
        split_info_path = osp.join(dataset.raw_dir, 'split_info.json')
        if osp.exists(split_info_path):
            with open(split_info_path) as f:
                split_info = json.load(f)
            train_end = int(split_info['train_idx'])
            val_end = int(split_info['val_idx'])
            total = int(split_info.get('total_samples', len(dataset)))
            if total != len(dataset):
                logger.warning(
                    'split_info total_samples=%s != len(dataset)=%s; using dataset length',
                    total,
                    len(dataset),
                )
                total = len(dataset)
            assert 0 < train_end < val_end <= total, (
                f'Invalid split cut points train_idx={train_end}, val_idx={val_end}, '
                f'total={total}'
            )
            split_idx = {
                'train': np.arange(train_end),
                'valid': np.arange(train_end, val_end),
                'test': np.arange(val_end, total),
            }
            logger.info(
                'Using split_info cut points: train=%s, valid=%s, test=%s',
                len(split_idx['train']),
                len(split_idx['valid']),
                len(split_idx['test']),
            )
            return split_idx

        dataset_length = len(dataset)
        split_sizes = [int(x * dataset_length) for x in self.parameters['train_val_test_split']]
        assert all(s > 0 for s in split_sizes), (
            f'All split sizes must be > 0, got {split_sizes} for dataset_length={dataset_length} '
            f'and splits={self.parameters["train_val_test_split"]}'
        )
        logger.warning(
            'split_info.json not found at %s; falling back to proportion cuts',
            split_info_path,
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

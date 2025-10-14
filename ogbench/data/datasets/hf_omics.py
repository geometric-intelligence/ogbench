"""HuggingFace datamodule for omics datasets."""


import abc
import logging
import os
from typing import Any, Dict, Final, Optional, Tuple, Union

import numpy as np
import pandas as pd
import PyWGCNA
import torch
import torch_geometric.data
import torch_geometric.transforms as T
from huggingface_hub import hf_hub_download
from sklearn.impute import SimpleImputer
from tqdm import tqdm

# Avoid circular import for baseline runner
try:
    from ogbench.data.utils import MeanStdNormalizer
except (ImportError, ModuleNotFoundError):
    # Inline definition for standalone use
    class MeanStdNormalizer:
        """Normalize features using mean and standard deviation."""

        def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> None:
            self.mean = mean
            self.std = std

        def fit(self, data: np.ndarray) -> None:
            """Compute mean and standard deviation from data."""
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)

        def transform(self, x: np.ndarray) -> np.ndarray:
            """Normalize input array."""
            if self.mean is None or self.std is None:
                raise ValueError("Normalizer must be fitted before use")
            return (x - self.mean) / self.std


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os.path as osp

from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs


class AddEdgeIndex(T.BaseTransform):
    """Transform that adds a fixed edge_index to each graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        The edge index to add to each graph.
    """

    def __init__(self, edge_index: torch.Tensor) -> None:
        self.edge_index = edge_index

    def forward(self, data: Data) -> Data:
        """Add the fixed edge_index to the data object.

        Returns
        -------
        Data
            The graph data object with the added edge_index.
        """
        data.edge_index = self.edge_index
        return data


class HFOmicsDataset(InMemoryDataset):
    """`InMemoryDataset` for omics datasets loaded from HuggingFace."""

    revision: Final[str] = "e1631e8"
    classification_datasets: Final[list[str]] = [
        "covidaki",
        "addneuromed",
    ]

    def __init__(
        self,
        root: str,
        data_name: str,
        method: str = "variance",
        imputation_method: str = "mean",
        adjacency_threshold: float = 0.3,
        node_sample_ratio: Union[float, str] = 1.0,
        train_val_test_split: list[float] = [0.7, 0.15, 0.15],
        hf_repo_id: str = "geometric-intelligence/bgbench",
        **kwargs: Any,
    ) -> None:
        """Initialize a `HFOmicsDataModule`.

        Args:
            root: The local data directory for caching
            data_name: The name of the dataset
            method: Method for node selection ("variance", "correlation", "random")
            imputation_method: Method for handling missing values
            adjacency_threshold: Threshold for adjacency matrix binarization
            node_sample_ratio: Ratio of nodes to sample
            hf_repo_id: HuggingFace repository ID
            **kwargs: Additional keyword arguments
        """
        self.data_name = data_name
        self.adjacency_threshold = adjacency_threshold
        self.node_sample_ratio = node_sample_ratio
        self.method = method
        self.train_val_test_split = train_val_test_split
        self.hf_repo_id = hf_repo_id
        self.imputer = SimpleImputer(strategy=imputation_method)
        self.feature_normalizer = MeanStdNormalizer()
        self.target_normalizer = MeanStdNormalizer()

        self.name = osp.join(
            f"{self.data_name}",
            f"adj_thresh_{self.adjacency_threshold}",
            f"{self.method}",
            f"p_{self.node_sample_ratio}",
            f"train_split_{self.train_val_test_split[0]}",
        )
        self.normalize_targets = False if data_name in self.classification_datasets else True

        super().__init__(root)
        data, self.slices, self.sizes, data_cls, self.edge_index = fs.torch_load(
            self.processed_paths[0]
        )
        self.data = data_cls.from_dict(data)
        assert isinstance(self._data, Data)
        self.transform = T.Compose([AddEdgeIndex(self.edge_index)])

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(
            self.root,
            self.name,
            "raw",
        )

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        self.processed_root = osp.join(
            self.root,
            self.name,
            "processed",
        )
        return self.processed_root

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return ["selected_data.parquet", "targets.npy", "adj_matrix.npy"]

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def get_data_dir(self) -> str:
        """Return the path to the data directory.

        Returns
        -------
        str
            Path to the data directory.
        """
        return osp.join(self.root, self.name)

    def download(self) -> None:
        r"""Download the dataset from HuggingFace and saves it to the raw directory."""
        logger.info(f"Downloading raw data for {self.data_name} from HuggingFace...")

        # Download parquet files directly from HuggingFace
        data_file = hf_hub_download(  # nosec
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            revision=self.revision,
            filename=f"{self.data_name}_data.parquet",
        )
        targets_file = hf_hub_download(  # nosec
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            revision=self.revision,
            filename=f"{self.data_name}_targets.parquet",
        )

        # Load data and targets using pandas
        raw_data = pd.read_parquet(data_file)
        targets_df = pd.read_parquet(targets_file)

        # Convert to proper format - data should be features only
        if "target" in raw_data.columns:
            raw_data = raw_data.drop("target", axis=1)

        targets = targets_df["target"].values

        logger.info(f"Downloaded {len(targets)} samples with {raw_data.shape[1]} features")
        np.save(os.path.join(self.raw_dir, "targets.npy"), targets)

        # Calculate number of nodes to select
        n_training_samples = int(raw_data.shape[0] * self.train_val_test_split[0])
        if self.node_sample_ratio == "full":
            print("Using full node sample ratio")
            n_nodes = raw_data.shape[1]
        elif isinstance(self.node_sample_ratio, float):
            n_nodes = int(n_training_samples / self.node_sample_ratio)
            if n_nodes > raw_data.shape[1]:
                n_nodes = raw_data.shape[1]
        logger.info(
            f"Training samples: {n_training_samples}, node_sample_ratio: {self.node_sample_ratio}, n_nodes: {n_nodes}"
        )

        # Select nodes
        logger.info("Selecting nodes...")
        selected_nodes = self.select_nodes(
            raw_data.values, targets, n_selected=n_nodes, method=self.method
        )
        selected_data = raw_data.iloc[:, selected_nodes]
        selected_data.to_parquet(osp.join(self.raw_dir, "selected_data.parquet"))

        # Calculate adjacency matrix
        logger.info("Calculating adjacency matrix...")
        adj_matrix = self.calculate_adjacency_matrix(selected_data)
        np.save(osp.join(self.raw_dir, "adj_matrix.npy"), adj_matrix)

        # Log statistics
        node_degrees = np.sum(adj_matrix, axis=1)
        logger.info("Node degrees statistics:")
        logger.info(f"Mean degree: {np.mean(node_degrees):.2f}")
        logger.info(f"Median degree: {np.median(node_degrees):.2f}")
        logger.info(f"Min degree: {np.min(node_degrees):.2f}")
        logger.info(f"Max degree: {np.max(node_degrees):.2f}")
        logger.info(f"Total edges: {np.sum(node_degrees)/2:.0f}")

    def select_nodes(
        self, data: np.ndarray, targets: np.ndarray, n_selected: int = 10, method: str = "variance"
    ) -> np.ndarray:
        """Select nodes based on feature importance or randomly."""
        if method == "variance":
            # Variance-based filtering
            variances = np.std(data, axis=0)
            ranked_nodes = np.argsort(variances)[::-1]
        elif method == "correlation":
            # Correlation-based filtering
            correlations = np.abs(
                np.array([np.corrcoef(data[:, i], targets)[0, 1] for i in range(data.shape[1])])
            )
            ranked_nodes = np.argsort(correlations)[::-1]
        elif method == "random":
            # Random selection
            ranked_nodes = np.random.permutation(data.shape[1])
        else:
            raise ValueError(f"Invalid method: {method}")

        return ranked_nodes[:n_selected]

    def calculate_adjacency_matrix(self, node_features: pd.DataFrame) -> np.ndarray:
        """Calculate adjacency matrix using WGCNA with soft-thresholding and binarization."""
        # Use WGCNA to find optimal power for scale-free topology
        soft_threshold = PyWGCNA.WGCNA.pickSoftThreshold(node_features)
        power = soft_threshold[0]

        # Apply soft-thresholding
        adjacency = PyWGCNA.WGCNA.adjacency(
            node_features,
            power=power,
            adjacencyType="signed hybrid",
        )

        # Binarize adjacency matrix
        adjacency = np.nan_to_num(adjacency, nan=0.0)
        adj_matrix = np.where(adjacency > self.adjacency_threshold, 1, 0)
        np.fill_diagonal(adj_matrix, 1)

        assert not np.isnan(adj_matrix).any(), "Adjacency matrix has nan values"
        return adj_matrix

    def create_graph_data(
        self, subject_data: np.ndarray, subject_target: float, adj_matrix: np.ndarray
    ) -> torch_geometric.data.Data:
        """Create graph data object."""
        node_features_raw = subject_data
        node_features_normalized = torch.from_numpy(
            self.feature_normalizer.transform(node_features_raw)
        ).to(torch.float32)

        y_tensor = np.array([subject_target])
        if self.normalize_targets:
            y_normalized = torch.from_numpy(self.target_normalizer.transform(y_tensor)).to(
                torch.float32
            )
        else:
            y_normalized = torch.from_numpy(y_tensor)

        graph = torch_geometric.data.Data(x=node_features_normalized.unsqueeze(1), y=y_normalized)
        return graph
        # transform = T.ToSparseTensor()
        # return transform(graph)

    def process(self) -> None:
        r"""Handle the data for the dataset."""
        logger.info("Loading data...")
        selected_data = pd.read_parquet(osp.join(self.raw_dir, "selected_data.parquet"))
        targets = np.load(osp.join(self.raw_dir, "targets.npy"))
        adj_matrix = np.load(osp.join(self.raw_dir, "adj_matrix.npy"))
        edge_index = torch.nonzero(torch.tensor(adj_matrix)).t().contiguous()

        # Shuffle selected_data and targets in unison
        from sklearn.utils import shuffle

        selected_data, targets = shuffle(selected_data, targets, random_state=42)

        # Fit normalizers on training data
        train_idx = int(len(selected_data) * self.train_val_test_split[0])
        train_data = selected_data.iloc[:train_idx]
        train_targets = targets[:train_idx]
        logger.info("Fitting normalizers")
        self.feature_normalizer.fit(train_data.values)
        if self.normalize_targets:
            self.target_normalizer.fit(train_targets)
        # Save normalizer statistics to JSON
        import json

        # Convert train_val_test_split to list if it's a ListConfig
        train_val_test_split = OmegaConf.to_object(self.train_val_test_split)
        normalizer_stats = {
            "train_val_test_split": train_val_test_split,
            "train_idx": train_idx,
            "feature_normalizer": {
                "mean": list(self.feature_normalizer.mean),
                "std": list(self.feature_normalizer.std),
            },
            "target_normalizer": {
                "mean": self.target_normalizer.mean,
                "std": self.target_normalizer.std,
            },
        }
        normalizers_stats_path = os.path.join(self.processed_dir, "processing_stats.json")
        with open(normalizers_stats_path, "w") as f:
            json.dump(normalizer_stats, f, indent=4)
        logger.info(f"Saved processing and normalizer stats to {normalizers_stats_path}")
        logger.info("Creating graph data...")

        graph_data_list = []
        for (_, subject_data), subject_target in tqdm(
            zip(selected_data.iterrows(), targets),
            total=len(selected_data),
            desc="Creating graphs",
        ):
            graph_data_list.append(
                self.create_graph_data(subject_data.values, subject_target, adj_matrix)
            )
        logger.info(f"Graph data list length: {len(graph_data_list)}")
        self.n_graphs = len(graph_data_list)

        self.data, self.slices = self.collate(graph_data_list)
        self.graph_list = []  # Reset cache.
        self._data_list = None  # Reset cache.
        self.edge_index = edge_index
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__, self.edge_index),
            self.processed_paths[0],
        )

    def load_raw_data_for_baselines(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
        """Load and preprocess raw data for sklearn baselines.

        Returns data, targets, and split indices following the same preprocessing
        as used for graph data (imputation, same shuffling, same splits).

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray, dict[str, int]]
            - Raw feature data (after selection but before normalization)
            - Target values
            - Dictionary with 'train_idx' and 'val_idx' split indices
        """
        import json

        from sklearn.utils import shuffle

        # Load raw data
        logger.info("Loading raw data for baseline...")
        selected_data = pd.read_parquet(osp.join(self.raw_dir, "selected_data.parquet"))
        targets = np.load(osp.join(self.raw_dir, "targets.npy"))

        # Apply same shuffling as in process()
        selected_data, targets = shuffle(selected_data, targets, random_state=42)

        # Calculate split indices (same as in process())
        train_idx = int(len(selected_data) * self.train_val_test_split[0])
        val_idx = int(
            len(selected_data) * (self.train_val_test_split[0] + self.train_val_test_split[1])
        )

        split_indices = {
            "train_idx": train_idx,
            "val_idx": val_idx,
        }

        logger.info(f"Loaded {len(selected_data)} samples with {selected_data.shape[1]} features")
        logger.info(
            f"Train samples: {train_idx}, Val samples: {val_idx - train_idx}, Test samples: {len(selected_data) - val_idx}"
        )

        return selected_data, targets, split_indices

    def __repr__(self) -> str:
        return (
            f"HFOmicsDataset(data_name={self.data_name}, "
            f"adjacency_threshold={self.adjacency_threshold}, "
            f"node_sample_ratio={self.node_sample_ratio}, "
            f"method={self.method}, "
            f"train_val_test_split={self.train_val_test_split}, "
            f"normalize_targets={self.normalize_targets})"
        )

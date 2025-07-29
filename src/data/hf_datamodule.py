"""HuggingFace datamodule for omics datasets."""

import abc
import logging
import os
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np
import pandas as pd
import PyWGCNA
import torch
import torch_geometric.data
import torch_geometric.transforms as T
from huggingface_hub import hf_hub_download
from lightning import LightningDataModule
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, Dataset

from src.data import transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFOmicsDataModule(LightningDataModule, abc.ABC):
    """`LightningDataModule` for omics datasets loaded from HuggingFace."""

    revision: Final[str] = "a64a861"

    def __init__(
        self,
        data_dir: str = "data/hf_omics/",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        method: str = "variance",
        imputation_method: str = "mean",
        adjacency_threshold: float = 0.01,
        node_sample_ratio: float = 1.0,
        hf_repo_id: str = "geometric-intelligence/bgbench",
    ) -> None:
        """Initialize a `HFOmicsDataModule`.

        Args:
            data_dir: The local data directory for caching
            train_val_test_split: The train, validation and test split ratios
            batch_size: The batch size
            num_workers: The number of workers
            pin_memory: Whether to pin memory
            method: Method for node selection ("variance", "correlation", "random")
            imputation_method: Method for handling missing values
            adjacency_threshold: Threshold for adjacency matrix binarization
            node_sample_ratio: Ratio of nodes to sample
            hf_repo_id: HuggingFace repository ID
        """
        super().__init__()
        self.adjacency_threshold = adjacency_threshold
        self.node_sample_ratio = node_sample_ratio
        self.method = method
        self.hf_repo_id = hf_repo_id
        self.save_hyperparameters(logger=False)
        self.imputer = SimpleImputer(strategy=imputation_method)
        self.feature_normalizer = transforms.MeanStdNormalizer()
        self.target_normalizer = transforms.MeanStdNormalizer()

        # Cache paths for processed data
        self.cache_dir = os.path.join(data_dir, self.dataset_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.selected_data_path: str = os.path.join(
            self.cache_dir,
            f"{self.dataset_name}_{method}_{adjacency_threshold}_{node_sample_ratio}_selected_data.parquet",
        )
        self.targets_path: str = os.path.join(
            self.cache_dir,
            f"{self.dataset_name}_{method}_{adjacency_threshold}_{node_sample_ratio}_targets.npy",
        )
        self.adj_matrix_path: str = os.path.join(
            self.cache_dir,
            f"{self.dataset_name}_{method}_{adjacency_threshold}_{node_sample_ratio}_adj_matrix.npy",
        )

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device: int = batch_size
        self.train_val_test_split: Tuple[float, float, float] = train_val_test_split

    def prepare_data(self) -> None:
        """Download raw data from HuggingFace and prepare processed data."""
        if not os.path.exists(self.hparams.data_dir):
            os.makedirs(self.hparams.data_dir)

        # Check if processed data already exists
        if (
            os.path.exists(self.selected_data_path)
            and os.path.exists(self.targets_path)
            and os.path.exists(self.adj_matrix_path)
        ):
            logger.info(f"Processed data found in cache for {self.dataset_name}")
            return

        logger.info(f"Processing dataset {self.dataset_name}...")

        # Download raw data from HuggingFace
        raw_data, targets = self._download_raw_data()

        # Process data (node selection, adjacency matrix calculation)
        self._process_data(raw_data, targets)

    def _download_raw_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Download raw data from HuggingFace."""
        logger.info(f"Downloading raw data for {self.dataset_name} from HuggingFace...")

        # Download parquet files directly from HuggingFace
        data_file = hf_hub_download(  # nosec
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            revision=self.revision,
            filename=f"{self.dataset_name}_data.parquet",
        )
        targets_file = hf_hub_download(  # nosec
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            revision=self.revision,
            filename=f"{self.dataset_name}_targets.parquet",
        )

        # Load data and targets using pandas
        raw_data = pd.read_parquet(data_file)
        targets_df = pd.read_parquet(targets_file)

        # Convert to proper format - data should be features only
        if "target" in raw_data.columns:
            raw_data = raw_data.drop("target", axis=1)

        targets = targets_df["target"].values

        logger.info(f"Downloaded {len(targets)} samples with {raw_data.shape[1]} features")
        return raw_data, targets

    def _process_data(self, raw_data: pd.DataFrame, targets: np.ndarray) -> None:
        """Process raw data (node selection, adjacency matrix calculation)."""
        # Save raw targets
        np.save(self.targets_path, targets)

        # Calculate number of nodes to select
        n_training_samples = int(raw_data.shape[0] * self.train_val_test_split[0])
        n_nodes = int(n_training_samples / self.node_sample_ratio)
        logger.info(
            f"Training samples: {n_training_samples}, node_sample_ratio: {self.node_sample_ratio}, n_nodes: {n_nodes}"
        )

        # Select nodes
        logger.info("Selecting nodes...")
        selected_nodes = self.select_nodes(
            raw_data.values, targets, n_selected=n_nodes, method=self.method
        )
        selected_data = raw_data.iloc[:, selected_nodes]
        selected_data.to_parquet(self.selected_data_path)

        # Calculate adjacency matrix
        logger.info("Calculating adjacency matrix...")
        adj_matrix = self.calculate_adjacency_matrix(selected_data)
        np.save(self.adj_matrix_path, adj_matrix)

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
        edge_index = torch.nonzero(torch.tensor(adj_matrix)).t().contiguous()

        y_tensor = np.array([subject_target])
        y_normalized = torch.from_numpy(self.target_normalizer.transform(y_tensor)).to(
            torch.float32
        )

        graph = torch_geometric.data.Data(
            x=node_features_normalized.unsqueeze(1), edge_index=edge_index, y=y_normalized
        )
        transform = T.ToSparseTensor()
        return transform(graph)

    def log_stats(self, log_callback) -> None:
        """Log dataset statistics."""
        adj_matrix = np.load(self.adj_matrix_path)
        node_degrees = np.sum(adj_matrix, axis=1)
        log_callback("mean_degree", np.mean(node_degrees))
        log_callback("median_degree", np.median(node_degrees))
        log_callback("min_degree", np.min(node_degrees))
        log_callback("max_degree", np.max(node_degrees))
        log_callback("total_edges", np.sum(node_degrees) / 2)
        log_callback("num_nodes", adj_matrix.shape[0])

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and create train/val/test splits."""
        logger.info("Loading data...")
        selected_data = pd.read_parquet(self.selected_data_path)
        targets = np.load(self.targets_path)
        adj_matrix = np.load(self.adj_matrix_path)

        # Fit normalizers on training data
        train_idx = int(len(selected_data) * self.train_val_test_split[0])
        train_data = selected_data.iloc[:train_idx]
        train_targets = targets[:train_idx]
        logger.info("Fitting normalizers")
        self.feature_normalizer.fit(train_data.values)
        self.target_normalizer.fit(train_targets)

        logger.info("Normalizers fitted")
        graph_data_list = []
        for (_, subject_data), subject_target in zip(selected_data.iterrows(), targets):
            graph_data_list.append(
                self.create_graph_data(subject_data.values, subject_target, adj_matrix)
            )
        logger.info(f"Graph data list length: {len(graph_data_list)}")
        logger.info(f"Graph data list created, shape: {graph_data_list[0].x.shape}")
        self.n_graphs = len(graph_data_list)
        i_train = int(self.n_graphs * self.train_val_test_split[0])
        i_val = int(self.n_graphs * (self.train_val_test_split[0] + self.train_val_test_split[1]))

        self.train_graph_data_list = graph_data_list[:i_train]
        self.val_graph_data_list = graph_data_list[i_train:i_val]
        self.test_graph_data_list = graph_data_list[i_val:]

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return torch_geometric.loader.DataLoader(
            dataset=self.train_graph_data_list,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return torch_geometric.loader.DataLoader(
            dataset=self.val_graph_data_list,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return torch_geometric.loader.DataLoader(
            dataset=self.test_graph_data_list,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Return datamodule state."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load datamodule state."""
        pass


class HFMotrPacDataModule(HFOmicsDataModule):
    """`LightningDataModule` for MotrPac proteomics datasets from HuggingFace."""

    dataset_name: Final[str] = "motrpac"
    revision: Final[str] = "a64a861"

    def __init__(self, data_dir: str = "data/hf_omics/", *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)


class HFPanCancerDataModule(HFOmicsDataModule):
    """`LightningDataModule` for PanCancer proteomics datasets from HuggingFace."""

    dataset_name: Final[str] = "pancancer"
    revision: Final[str] = "a64a861"

    def __init__(self, data_dir: str = "data/hf_omics/", *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)


class HFAddNeuroMedOmicsDataModule(HFOmicsDataModule):
    """`LightningDataModule` for AddNeuroMed-style omics datasets from HuggingFace."""

    dataset_name: Final[str] = "addneuromed"
    revision: Final[str] = "a64a861"

    def __init__(self, data_dir: str = "data/hf_omics/", *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)


class HFCovidAKIOmicsDataModule(HFOmicsDataModule):
    """`LightningDataModule` for COVID AKI omics datasets from HuggingFace."""

    dataset_name: Final[str] = "covidaki"
    revision: Final[str] = "a64a861"

    def __init__(self, data_dir: str = "data/hf_omics/", *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)


class HFParkinsonsOmicsDataModule(HFOmicsDataModule):
    """LightningDataModule for GSE99039 (Parkinson's Disease) gene expression data from
    HuggingFace."""

    dataset_name: Final[str] = "parkinsons"
    revision: Final[str] = "a64a861"

    def __init__(self, data_dir: str = "data/hf_omics/", *args, **kwargs) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)

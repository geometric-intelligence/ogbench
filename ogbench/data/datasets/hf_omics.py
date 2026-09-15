"""HuggingFace datamodule for omics datasets."""


import json
import logging
import os
import os.path as osp
from typing import Any, Final

import numpy as np
import pandas as pd
import torch
import torch_geometric.data
import torch_geometric.transforms as T
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from tqdm import tqdm

from ogbench.data.adjacency import binarize_to_target_connectivity, get_adjacency_builder
from ogbench.data.selectors import get_selector
from ogbench.data.utils import MeanStdNormalizer
from ogbench.data.utils.split_utils import (
    build_omics_cache_relative_name,
    compute_omics_split_indices,
)


def _infer_batch_column(meta: pd.DataFrame) -> str | None:
    """Return a sample-meta column that looks like a hybridization batch label."""
    preferred = ('batch', 'hybridization', 'hyb_batch', 'scan_date', 'chip')
    lower = {str(col).lower(): col for col in meta.columns}
    for key in preferred:
        if key in lower:
            return lower[key]
    for lowered, original in lower.items():
        if 'batch' in lowered:
            return original
    return None


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    classification_datasets: Final[list[str]] = [
        'addneuromed',
        'parkinsons',
        'motrpac',
        'brca',
    ]

    def __init__(
        self,
        root: str,
        data_name: str,
        method: str = 'correlation',
        imputation_method: str = 'mean',
        adjacency_threshold: float = 0.3,
        adjacency_method: str = 'string',
        adjacency_target_connectivity: float | None = None,
        node_sample_ratio: float | str = 1.0,
        train_val_test_split: list[float] | None = None,
        hf_repo_id: str = 'geometric-intelligence/ogbench',
        revision: str = '056dfdc4f434fd35355ffbe5f7b63910d785a97a',
        string_data_dir: str | None = None,
        species: int = 9606,
        split_type: str = 'fixed',
        k: int = 5,
        fold: int = 0,
        corrections: list[str] | None = None,
        grouping: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `HFOmicsDataModule`.

        Args:
            root: The local data directory for caching
            data_name: The name of the dataset
            method: Method for node selection ("variance", "correlation", "distance_correlation", "random")
            imputation_method: Method for handling missing values
            adjacency_threshold: Threshold for adjacency matrix binarization
            adjacency_method: Method for adjacency matrix construction (default: "string")
            adjacency_target_connectivity: For WGCNA, retain the strongest edges nearest
                to this train-fold connectivity instead of using ``adjacency_threshold``
            node_sample_ratio: Ratio of nodes to sample
            hf_repo_id: HuggingFace repository ID
            revision: HuggingFace dataset revision/commit hash
            string_data_dir: Optional path to pre-downloaded STRING bulk files
            species: NCBI taxonomy ID for STRING adjacency (9606 = human, 83332 = M. tuberculosis)
            split_type: ``fixed`` (default) or ``k-fold``
            k: Number of CV folds when ``split_type='k-fold'``
            fold: Test-fold index when ``split_type='k-fold'`` (usually ``data_seed``)
            corrections: Optional train-only corrections (``covariate_adjust``, ``combat``)
            grouping: Optional group key for k-fold (``batch``) using a sidecar file
            **kwargs: Additional keyword arguments
        """
        self.data_name = data_name
        self.adjacency_threshold = adjacency_threshold
        self.adjacency_method = adjacency_method
        self.adjacency_target_connectivity = adjacency_target_connectivity
        self.string_data_dir = string_data_dir
        self.species = species
        self.node_sample_ratio = node_sample_ratio
        self.method = method
        self.train_val_test_split = train_val_test_split or [0.7, 0.15, 0.15]
        self.hf_repo_id = hf_repo_id
        self.revision = revision
        self.split_type = split_type
        self.k = int(k)
        self.fold = int(fold)
        self.corrections = list(corrections or [])
        grouping_value = grouping if grouping not in (None, 'null', '') else None
        self.grouping = grouping_value
        self.imputer = SimpleImputer(strategy=imputation_method)
        self.feature_normalizer = MeanStdNormalizer()

        self.name = build_omics_cache_relative_name(
            data_name=self.data_name,
            adjacency_threshold=self.adjacency_threshold,
            adjacency_method=self.adjacency_method,
            method=self.method,
            node_sample_ratio=self.node_sample_ratio,
            train_split=self.train_val_test_split[0],
            split_type=self.split_type,
            k=self.k,
            fold=self.fold,
            corrections=self.corrections,
            grouping=self.grouping,
            adjacency_target_connectivity=self.adjacency_target_connectivity,
        )

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
            'raw',
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
            'processed',
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
        return ['selected_data.parquet', 'targets.npy', 'adj_matrix.npy', 'split_info.json']

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return 'data.pt'

    def get_data_dir(self) -> str:
        """Return the path to the data directory.

        Returns
        -------
        str
            Path to the data directory.
        """
        return osp.join(self.root, self.name)

    def _hf_download(self, filename: str) -> str:
        """Download a file from the configured HuggingFace dataset revision."""
        return hf_hub_download(  # nosec
            repo_id=self.hf_repo_id,
            repo_type='dataset',
            revision=self.revision,
            filename=filename,
        )

    def _download_optional_parquet(self, filename: str) -> pd.DataFrame | None:
        """Return a parquet sidecar or None if it is not on the Hub."""
        try:
            path = self._hf_download(filename)
        except Exception as exc:
            logger.info('Optional sidecar %s not found: %s', filename, exc)
            return None
        return pd.read_parquet(path)

    def _load_batch_labels(self) -> np.ndarray | None:
        """Load per-sample batch ids from batches.parquet or sample_meta.parquet."""
        batches_df = self._download_optional_parquet(f'{self.data_name}_batches.parquet')
        if batches_df is not None:
            column = 'batch' if 'batch' in batches_df.columns else batches_df.columns[0]
            return batches_df[column].to_numpy()
        meta = self._download_optional_parquet(f'{self.data_name}_sample_meta.parquet')
        if meta is None:
            return None
        batch_col = _infer_batch_column(meta)
        if batch_col is None:
            return None
        return meta[batch_col].to_numpy()

    def _apply_corrections(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        *,
        train_ids: np.ndarray,
        valid_ids: np.ndarray,
        test_ids: np.ndarray,
        train_targets: np.ndarray,
        covariates_df: pd.DataFrame | None,
        batches: np.ndarray | None,
        probe_map_df: pd.DataFrame | None = None,
        names: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit configured corrections on train and transform val/test."""
        from ogbench.data.corrections import (
            CombatCorrector,
            CovariateAdjuster,
            MedianCenterer,
            PromoterMinBetaSelector,
        )

        for name in names if names is not None else self.corrections:
            if name == 'covariate_adjust':
                if covariates_df is None:
                    raise FileNotFoundError(
                        f'corrections includes covariate_adjust but '
                        f'{self.data_name}_covariates.parquet was not found'
                    )
                adjuster = CovariateAdjuster()
                adjuster.fit(train_data, covariates_df.iloc[train_ids])
                train_data = adjuster.transform(train_data, covariates_df.iloc[train_ids])
                val_data = adjuster.transform(val_data, covariates_df.iloc[valid_ids])
                test_data = adjuster.transform(test_data, covariates_df.iloc[test_ids])
                logger.info('Applied train-only covariate adjustment')
            elif name == 'combat':
                if batches is None:
                    raise FileNotFoundError(
                        f'corrections includes combat but no batch labels were found for '
                        f'{self.data_name}'
                    )
                corrector = CombatCorrector()
                corrector.fit(train_data, batches[train_ids])
                train_data = corrector.transform(train_data, batches[train_ids])
                val_data = corrector.transform(val_data, batches[valid_ids])
                test_data = corrector.transform(test_data, batches[test_ids])
                logger.info('Applied train-only ComBat correction')
            elif name == 'promoter_min_beta':
                if probe_map_df is None:
                    raise FileNotFoundError(
                        f'corrections includes promoter_min_beta but '
                        f'{self.data_name}_probe_map.parquet was not found'
                    )
                selector = PromoterMinBetaSelector()
                selector.fit(train_data, train_targets, probe_map_df)
                train_data = selector.transform(train_data)
                val_data = selector.transform(val_data)
                test_data = selector.transform(test_data)
                logger.info(
                    'Applied train-only promoter min-beta probe selection '
                    f'({train_data.shape[1]} genes)'
                )
            elif name == 'median_center':
                centerer = MedianCenterer()
                centerer.fit(train_data)
                train_data = centerer.transform(train_data)
                val_data = centerer.transform(val_data)
                test_data = centerer.transform(test_data)
                logger.info('Applied train-only median centering')
            else:
                raise ValueError(
                    f"Unknown correction {name!r}; use 'covariate_adjust', 'combat', "
                    f"'promoter_min_beta', or 'median_center'"
                )
        return train_data, val_data, test_data

    def download(self) -> None:
        r"""Download the dataset from HuggingFace and saves it to the raw directory."""
        logger.info(f'Downloading raw data for {self.data_name} from HuggingFace...')

        # Download parquet files directly from HuggingFace
        data_file = self._hf_download(f'{self.data_name}_data.parquet')
        targets_file = self._hf_download(f'{self.data_name}_targets.parquet')
        map_df = None
        if self.adjacency_method == 'string':
            map_file = self._hf_download(f'{self.data_name}_map.parquet')
            map_df = pd.read_parquet(map_file)

        # Load data and targets with pandas
        raw_data = pd.read_parquet(data_file)
        targets_df = pd.read_parquet(targets_file)

        # Convert to proper format - data should be features only
        if 'target' in raw_data.columns:
            raw_data = raw_data.drop('target', axis=1)

        targets = targets_df['target'].values

        logger.info(f'Downloaded {len(targets)} samples with {raw_data.shape[1]} features')

        covariates_df = self._download_optional_parquet(f'{self.data_name}_covariates.parquet')
        probe_map_df = self._download_optional_parquet(f'{self.data_name}_probe_map.parquet')
        batches = self._load_batch_labels()
        groups = None
        if self.grouping == 'batch':
            if batches is None:
                raise FileNotFoundError(
                    f'grouping=batch requires {self.data_name}_batches.parquet or '
                    f'{self.data_name}_sample_meta.parquet with a batch column'
                )
            if self.split_type == 'k-fold':
                groups = batches
            else:
                logger.warning(
                    'grouping=%s only applies to split_type=k-fold; ignoring it for '
                    'split_type=%s',
                    self.grouping,
                    self.split_type,
                )

        # IMPORTANT: Split data BEFORE any feature engineering to avoid data leakage.
        # Indices refer to the original sample order; we then reorder to train|val|test.
        if OmegaConf.is_config(self.train_val_test_split):
            train_val_test_split = list(OmegaConf.to_object(self.train_val_test_split))
        else:
            train_val_test_split = list(self.train_val_test_split)

        split_arrays = compute_omics_split_indices(
            targets,
            split_type=self.split_type,
            k=self.k,
            fold=self.fold,
            train_val_test_split=train_val_test_split,
            random_state=42,
            groups=groups,
        )
        train_ids = split_arrays['train']
        valid_ids = split_arrays['valid']
        test_ids = split_arrays['test']

        train_data = raw_data.iloc[train_ids]
        train_targets = targets[train_ids]
        val_data = raw_data.iloc[valid_ids]
        val_targets = targets[valid_ids]
        test_data = raw_data.iloc[test_ids]
        test_targets = targets[test_ids]

        n_samples = len(targets)
        train_idx = len(train_targets)
        val_idx = train_idx + len(val_targets)

        logger.info(
            f'Split ({self.split_type}'
            + (f', fold={self.fold}/{self.k}' if self.split_type == 'k-fold' else '')
            + f'): Train={len(train_targets)}, Val={len(val_targets)}, Test={len(test_targets)}'
        )

        pre_impute = [name for name in self.corrections if name != 'median_center']
        post_impute = [name for name in self.corrections if name == 'median_center']
        if pre_impute:
            train_data, val_data, test_data = self._apply_corrections(
                train_data,
                val_data,
                test_data,
                train_ids=train_ids,
                valid_ids=valid_ids,
                test_ids=test_ids,
                train_targets=train_targets,
                covariates_df=covariates_df,
                batches=batches,
                probe_map_df=probe_map_df,
                names=pre_impute,
            )

        # Impute missing values - FIT on training data only, TRANSFORM on all splits
        nan_count = train_data.isna().sum().sum()
        if nan_count > 0 or raw_data.isna().sum().sum() > 0:
            logger.info(f'Training data has {nan_count} NaN values')
            logger.info(f'Fitting imputer on training data using {self.imputer.strategy} strategy')
            self.imputer.fit(train_data)

            # Transform all splits
            train_data_imputed = self.imputer.transform(train_data)
            train_data = pd.DataFrame(
                train_data_imputed, columns=train_data.columns, index=train_data.index
            )

            if val_data.shape[0] > 0:
                val_data_imputed = self.imputer.transform(val_data)
                val_data = pd.DataFrame(
                    val_data_imputed, columns=val_data.columns, index=val_data.index
                )

            if test_data.shape[0] > 0:
                test_data_imputed = self.imputer.transform(test_data)
                test_data = pd.DataFrame(
                    test_data_imputed, columns=test_data.columns, index=test_data.index
                )

            logger.info(
                f'After imputation: Train NaN={train_data.isna().sum().sum()}, '
                f'Val NaN={val_data.isna().sum().sum()}, Test NaN={test_data.isna().sum().sum()}'
            )

        if post_impute:
            train_data, val_data, test_data = self._apply_corrections(
                train_data,
                val_data,
                test_data,
                train_ids=train_ids,
                valid_ids=valid_ids,
                test_ids=test_ids,
                train_targets=train_targets,
                covariates_df=covariates_df,
                batches=batches,
                probe_map_df=probe_map_df,
                names=post_impute,
            )

        # Calculate number of nodes to select based on TRAINING data only
        n_training_samples = len(train_targets)
        if self.node_sample_ratio == 'full':
            logger.info('Using full node sample ratio')
            n_nodes = train_data.shape[1]
        elif isinstance(self.node_sample_ratio, float):
            n_nodes = int(n_training_samples / self.node_sample_ratio)
            if n_nodes > train_data.shape[1]:
                n_nodes = train_data.shape[1]
        else:
            raise ValueError(
                f'node_sample_ratio must be "full" or numeric, got {self.node_sample_ratio!r}'
            )
        logger.info(
            f'Training samples: {n_training_samples}, node_sample_ratio: {self.node_sample_ratio}, n_nodes: {n_nodes}'
        )

        # Select nodes based ONLY on training data
        logger.info('Selecting nodes based on training data only...')
        selected_node_indices = self.select_nodes(
            train_data.values, train_targets, n_selected=n_nodes, method=self.method
        )

        # Apply same node selection to all splits
        train_selected = train_data.iloc[:, selected_node_indices]
        val_selected = val_data.iloc[:, selected_node_indices]
        test_selected = test_data.iloc[:, selected_node_indices]

        # Concatenate back for saving (maintaining split order)
        selected_data = pd.concat([train_selected, val_selected, test_selected], axis=0)
        selected_data.to_parquet(osp.join(self.raw_dir, 'selected_data.parquet'))

        # Save all targets (maintaining split order)
        all_targets = np.concatenate([train_targets, val_targets, test_targets])
        np.save(os.path.join(self.raw_dir, 'targets.npy'), all_targets)

        # train_idx/val_idx remain cut points into the reordered train|val|test arrays.
        split_info = {
            'split_type': self.split_type,
            'k': self.k,
            'fold': self.fold,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'total_samples': n_samples,
            'train_indices': train_ids.tolist(),
            'valid_indices': valid_ids.tolist(),
            'test_indices': test_ids.tolist(),
        }

        with open(os.path.join(self.raw_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=4)
        logger.info(
            f'Saved split info: type={self.split_type}, '
            f'train_idx={train_idx}, val_idx={val_idx}, total={n_samples}'
        )

        # Calculate adjacency matrix based ONLY on training data
        logger.info('Calculating adjacency matrix based on training data only...')
        adj_matrix = self.calculate_adjacency_matrix(train_selected, map_df=map_df)
        np.save(osp.join(self.raw_dir, 'adj_matrix.npy'), adj_matrix)
        if hasattr(self, 'adjacency_achieved_connectivity'):
            split_info.update(
                {
                    'adjacency_target_connectivity': self.adjacency_target_connectivity,
                    'adjacency_achieved_connectivity': self.adjacency_achieved_connectivity,
                    'adjacency_effective_cutoff': self.adjacency_effective_cutoff,
                }
            )
            with open(os.path.join(self.raw_dir, 'split_info.json'), 'w') as f:
                json.dump(split_info, f, indent=4)

        # Log statistics
        node_degrees = np.sum(adj_matrix, axis=1)
        logger.info('Node degrees statistics (from training data):')
        logger.info(f'Mean degree: {np.mean(node_degrees):.2f}')
        logger.info(f'Median degree: {np.median(node_degrees):.2f}')
        logger.info(f'Min degree: {np.min(node_degrees):.2f}')
        logger.info(f'Max degree: {np.max(node_degrees):.2f}')
        logger.info(f'Total edges: {np.sum(node_degrees)/2:.0f}')

    def select_nodes(
        self, data: np.ndarray, targets: np.ndarray, n_selected: int = 10, method: str = 'variance'
    ) -> np.ndarray:
        """Select nodes using a modular selector system."""
        selector = get_selector(method)
        return selector.select(data, targets, n_selected)

    def calculate_adjacency_matrix(
        self, node_features: pd.DataFrame, map_df: pd.DataFrame | None = None
    ) -> np.ndarray:
        """Calculate adjacency matrix using a modular adjacency builder system."""
        # Build continuous adjacency matrix using modular builder
        builder_kwargs = {}
        if self.adjacency_method == 'string':
            builder_kwargs['cache_dir'] = osp.join(self.root, 'string_cache')
            builder_kwargs['species'] = self.species
            if self.string_data_dir:
                builder_kwargs['string_data_dir'] = self.string_data_dir
        adjacency_builder = get_adjacency_builder(self.adjacency_method, **builder_kwargs)
        adjacency = adjacency_builder.build(node_features, map_df)

        # Binarize adjacency matrix
        adjacency = np.nan_to_num(adjacency, nan=0.0)
        if self.adjacency_method == 'wgcna' and self.adjacency_target_connectivity is not None:
            adj_matrix, cutoff, achieved = binarize_to_target_connectivity(
                adjacency, self.adjacency_target_connectivity
            )
            self.adjacency_effective_cutoff = cutoff
            self.adjacency_achieved_connectivity = achieved
            logger.info(
                'WGCNA train-fold graph target connectivity=%.6f, achieved=%.6f, '
                'weakest retained weight=%.8g',
                self.adjacency_target_connectivity,
                achieved,
                cutoff,
            )
        else:
            adj_matrix = np.where(adjacency > self.adjacency_threshold, 1, 0)
            np.fill_diagonal(adj_matrix, 1)

        assert not np.isnan(adj_matrix).any(), 'Adjacency matrix has nan values'
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
        y_normalized = torch.from_numpy(y_tensor)

        graph = torch_geometric.data.Data(x=node_features_normalized.unsqueeze(1), y=y_normalized)
        return graph
        # transform = T.ToSparseTensor()
        # return transform(graph)

    def process(self) -> None:
        r"""Handle the data for the dataset."""
        logger.info('Loading data...')
        selected_data = pd.read_parquet(osp.join(self.raw_dir, 'selected_data.parquet'))
        targets = np.load(osp.join(self.raw_dir, 'targets.npy'))
        adj_matrix = np.load(osp.join(self.raw_dir, 'adj_matrix.npy'))
        edge_index = torch.nonzero(torch.tensor(adj_matrix)).t().contiguous()

        # Load split info - data is already ordered train|val|test in download()
        with open(os.path.join(self.raw_dir, 'split_info.json')) as f:
            split_info = json.load(f)
        train_idx = split_info['train_idx']
        logger.info(f'Loaded split info: train_idx={train_idx}')

        # Fit normalizers on training data
        train_data = selected_data.iloc[:train_idx]
        logger.info('Fitting normalizers on training data')
        self.feature_normalizer.fit(train_data.values)

        # Convert train_val_test_split to list if it's a ListConfig
        if OmegaConf.is_config(self.train_val_test_split):
            train_val_test_split = OmegaConf.to_object(self.train_val_test_split)
        else:
            train_val_test_split = list(self.train_val_test_split)
        normalizer_stats = {
            'train_val_test_split': train_val_test_split,
            'split_type': split_info.get('split_type', self.split_type),
            'k': split_info.get('k', self.k),
            'fold': split_info.get('fold', self.fold),
            'train_idx': train_idx,
            'feature_normalizer': {
                'mean': list(self.feature_normalizer.mean),
                'std': list(self.feature_normalizer.std),
            },
        }
        normalizers_stats_path = os.path.join(self.processed_dir, 'processing_stats.json')
        with open(normalizers_stats_path, 'w') as f:
            json.dump(normalizer_stats, f, indent=4)
        logger.info(f'Saved processing and normalizer stats to {normalizers_stats_path}')
        logger.info('Creating graph data...')

        graph_data_list = []
        for (_, subject_data), subject_target in tqdm(
            zip(selected_data.iterrows(), targets, strict=True),
            total=len(selected_data),
            desc='Creating graphs',
        ):
            graph_data_list.append(
                self.create_graph_data(subject_data.values, subject_target, adj_matrix)
            )
        logger.info(f'Graph data list length: {len(graph_data_list)}')
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
        # Load raw data
        logger.info('Loading raw data for baseline...')
        selected_data = pd.read_parquet(osp.join(self.raw_dir, 'selected_data.parquet'))
        targets = np.load(osp.join(self.raw_dir, 'targets.npy'))

        # Load split indices from the same file used in download() and process()
        with open(os.path.join(self.raw_dir, 'split_info.json')) as f:
            split_info = json.load(f)

        train_idx = split_info['train_idx']
        val_idx = split_info['val_idx']

        split_indices = {
            'train_idx': train_idx,
            'val_idx': val_idx,
        }

        logger.info(f'Loaded {len(selected_data)} samples with {selected_data.shape[1]} features')
        logger.info(
            f'Train samples: {train_idx}, Val samples: {val_idx - train_idx}, Test samples: {len(selected_data) - val_idx}'
        )

        return selected_data, targets, split_indices

    def __repr__(self) -> str:
        return (
            f'HFOmicsDataset(data_name={self.data_name}, '
            f'adjacency_threshold={self.adjacency_threshold}, '
            f'adjacency_target_connectivity={self.adjacency_target_connectivity}, '
            f'adjacency_method={self.adjacency_method}, '
            f'node_sample_ratio={self.node_sample_ratio}, '
            f'method={self.method}, '
            f'train_val_test_split={self.train_val_test_split}, '
            f'split_type={self.split_type}, k={self.k}, fold={self.fold})'
        )

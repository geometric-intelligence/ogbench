import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch_geometric
from src.data import hf_datamodule


def test_hf_datamodule_initialization() -> None:
    """Test HF datamodule initialization and configuration."""
    with tempfile.TemporaryDirectory() as data_dir:
        dm = hf_datamodule.HFMotrPacDataModule(
            data_dir=data_dir,
            batch_size=32,
            method="variance",
            adjacency_threshold=0.01,
            node_sample_ratio=0.5,
            imputation_method="mean",
        )

        # Test initialization
        assert dm.dataset_name == "motrpac"
        assert dm.method == "variance"
        assert dm.adjacency_threshold == 0.01
        assert dm.node_sample_ratio == 0.5
        assert dm.batch_size_per_device == 32
        assert dm.train_val_test_split == (0.7, 0.15, 0.15)

        # Test cache paths
        expected_cache_dir = Path(data_dir) / "motrpac"
        assert dm.cache_dir == str(expected_cache_dir)
        assert dm.selected_data_path.endswith("motrpac_variance_0.01_0.5_selected_data.parquet")
        assert dm.targets_path.endswith("motrpac_variance_0.01_0.5_targets.npy")
        assert dm.adj_matrix_path.endswith("motrpac_variance_0.01_0.5_adj_matrix.npy")


def test_hf_datamodule_node_selection_methods() -> None:
    """Test different node selection methods."""
    # Create mock data
    data = np.random.randn(100, 50)  # 100 samples, 50 features
    targets = np.random.randn(100)

    dm = hf_datamodule.HFMotrPacDataModule()

    # Test variance method
    selected_variance = dm.select_nodes(data, targets, n_selected=10, method="variance")
    assert len(selected_variance) == 10
    assert len(set(selected_variance)) == 10  # No duplicates

    # Test correlation method
    selected_correlation = dm.select_nodes(data, targets, n_selected=10, method="correlation")
    assert len(selected_correlation) == 10
    assert len(set(selected_correlation)) == 10

    # Test random method
    selected_random = dm.select_nodes(data, targets, n_selected=10, method="random")
    assert len(selected_random) == 10
    assert len(set(selected_random)) == 10


def test_hf_datamodule_invalid_method() -> None:
    """Test that invalid method raises ValueError."""
    data = np.random.randn(100, 50)
    targets = np.random.randn(100)

    dm = hf_datamodule.HFMotrPacDataModule()

    with pytest.raises(ValueError, match="Invalid method: invalid_method"):
        dm.select_nodes(data, targets, n_selected=10, method="invalid_method")


@patch("src.data.hf_datamodule.PyWGCNA.WGCNA.pickSoftThreshold")
@patch("src.data.hf_datamodule.PyWGCNA.WGCNA.adjacency")
def test_hf_datamodule_adjacency_matrix_calculation(mock_adjacency, mock_pick_threshold) -> None:
    """Test adjacency matrix calculation with mocked WGCNA."""
    # Mock WGCNA responses
    mock_pick_threshold.return_value = (2, None)  # power=2
    mock_adjacency.return_value = np.random.rand(10, 10) * 0.5  # Random adjacency matrix

    dm = hf_datamodule.HFMotrPacDataModule(adjacency_threshold=0.3)

    # Create mock node features
    node_features = pd.DataFrame(np.random.randn(10, 5))

    # Calculate adjacency matrix
    adj_matrix = dm.calculate_adjacency_matrix(node_features)

    # Check properties
    assert adj_matrix.shape == (10, 10)
    assert adj_matrix.dtype == np.int64 or adj_matrix.dtype == np.int32
    assert np.all(np.diag(adj_matrix) == 1)  # Self-loops
    assert not np.isnan(adj_matrix).any()  # No NaN values

    # Verify WGCNA was called
    mock_pick_threshold.assert_called_once()
    mock_adjacency.assert_called_once()


def test_hf_datamodule_create_graph_data() -> None:
    """Test graph data creation."""
    dm = hf_datamodule.HFMotrPacDataModule()

    # Create mock data
    subject_data = np.random.randn(10)  # 10 features
    subject_target = 1.5
    adj_matrix = np.random.randint(0, 2, (10, 10))  # Binary adjacency matrix
    np.fill_diagonal(adj_matrix, 1)  # Add self-loops

    # Fit normalizers with mock data
    mock_features = np.random.randn(100, 10)  # 100 samples, 10 features
    mock_targets = np.random.randn(100)
    dm.feature_normalizer.fit(mock_features)
    dm.target_normalizer.fit(mock_targets)

    # Create graph data
    graph = dm.create_graph_data(subject_data, subject_target, adj_matrix)

    # Check graph properties
    assert isinstance(graph, torch_geometric.data.Data)
    assert hasattr(graph, "x")  # Node features
    assert hasattr(graph, "edge_index")  # Edge indices
    assert hasattr(graph, "y")  # Target values

    # Check data types
    assert graph.x.dtype == torch.float32
    assert graph.y.dtype == torch.float32

    # Check shapes
    assert graph.x.shape[0] == 10  # Number of nodes
    assert graph.x.shape[1] == 1  # Feature dimension
    assert graph.y.shape == (1,)  # Single target value (1D tensor)

    # Check that graph has the expected attributes (edge_index might be None after ToSparseTensor)
    assert hasattr(graph, "x")
    assert hasattr(graph, "y")
    # Note: edge_index might be None after ToSparseTensor transform, which is normal


def test_hf_datamodule_dataloader_creation() -> None:
    """Test dataloader creation with mock data."""
    with tempfile.TemporaryDirectory() as data_dir:
        dm = hf_datamodule.HFMotrPacDataModule(
            data_dir=data_dir, batch_size=16, num_workers=0, pin_memory=False
        )

        # Mock the setup method to avoid expensive operations
        with patch.object(dm, "setup") as mock_setup:
            # Mock graph data lists
            dm.train_graph_data_list = [
                torch_geometric.data.Data(
                    x=torch.randn(10, 1, 5),
                    edge_index=torch.randint(0, 10, (2, 20)),
                    y=torch.randn(1, 1),
                )
                for _ in range(50)
            ]
            dm.val_graph_data_list = [
                torch_geometric.data.Data(
                    x=torch.randn(10, 1, 5),
                    edge_index=torch.randint(0, 10, (2, 20)),
                    y=torch.randn(1, 1),
                )
                for _ in range(20)
            ]
            dm.test_graph_data_list = [
                torch_geometric.data.Data(
                    x=torch.randn(10, 1, 5),
                    edge_index=torch.randint(0, 10, (2, 20)),
                    y=torch.randn(1, 1),
                )
                for _ in range(20)
            ]

            # Test dataloader creation
            train_loader = dm.train_dataloader()
            val_loader = dm.val_dataloader()
            test_loader = dm.test_dataloader()

            # Check dataloader properties
            assert train_loader.batch_size == 16
            assert train_loader.num_workers == 0
            assert train_loader.pin_memory is False

            assert val_loader.batch_size == 16
            assert val_loader.num_workers == 0
            assert val_loader.pin_memory is False

            assert test_loader.batch_size == 16
            assert test_loader.num_workers == 0
            assert test_loader.pin_memory is False

            # Test batch iteration
            batch = next(iter(train_loader))
            assert isinstance(batch, torch_geometric.data.Batch)
            assert batch.x.dtype == torch.float32
            assert batch.y.dtype == torch.float32
            assert batch.num_graphs <= 16


def test_hf_datamodule_file_paths() -> None:
    """Test that file paths are constructed correctly for different datasets."""
    with tempfile.TemporaryDirectory() as data_dir:
        # Test MotrPac
        dm_motrpac = hf_datamodule.HFMotrPacDataModule(data_dir=data_dir)
        assert "motrpac" in dm_motrpac.selected_data_path
        assert "motrpac" in dm_motrpac.targets_path
        assert "motrpac" in dm_motrpac.adj_matrix_path

        # Test PanCancer
        dm_pancancer = hf_datamodule.HFPanCancerDataModule(data_dir=data_dir)
        assert "pancancer" in dm_pancancer.selected_data_path
        assert "pancancer" in dm_pancancer.targets_path
        assert "pancancer" in dm_pancancer.adj_matrix_path

        # Test AddNeuroMed
        dm_addneuromed = hf_datamodule.HFAddNeuroMedOmicsDataModule(data_dir=data_dir)
        assert "addneuromed" in dm_addneuromed.selected_data_path
        assert "addneuromed" in dm_addneuromed.targets_path
        assert "addneuromed" in dm_addneuromed.adj_matrix_path

        # Test COVID AKI
        dm_covidaki = hf_datamodule.HFCovidAKIOmicsDataModule(data_dir=data_dir)
        assert "covidaki" in dm_covidaki.selected_data_path
        assert "covidaki" in dm_covidaki.targets_path
        assert "covidaki" in dm_covidaki.adj_matrix_path

        # Test Parkinsons
        dm_parkinsons = hf_datamodule.HFParkinsonsOmicsDataModule(data_dir=data_dir)
        assert "parkinsons" in dm_parkinsons.selected_data_path
        assert "parkinsons" in dm_parkinsons.targets_path
        assert "parkinsons" in dm_parkinsons.adj_matrix_path


def test_hf_datamodule_parameter_combinations() -> None:
    """Test different parameter combinations for initialization."""
    with tempfile.TemporaryDirectory() as data_dir:
        # Test different methods
        for method in ["variance", "correlation", "random"]:
            dm = hf_datamodule.HFMotrPacDataModule(data_dir=data_dir, method=method)
            assert dm.method == method

        # Test different adjacency thresholds
        for threshold in [0.01, 0.05, 0.1]:
            dm = hf_datamodule.HFMotrPacDataModule(
                data_dir=data_dir, adjacency_threshold=threshold
            )
            assert dm.adjacency_threshold == threshold

        # Test different node sample ratios
        for ratio in [0.5, 0.8, 1.0]:
            dm = hf_datamodule.HFMotrPacDataModule(data_dir=data_dir, node_sample_ratio=ratio)
            assert dm.node_sample_ratio == ratio

        # Test different imputation methods
        for imputation in ["mean", "median"]:
            dm = hf_datamodule.HFMotrPacDataModule(data_dir=data_dir, imputation_method=imputation)
            assert dm.imputer.strategy == imputation

        # Test different train/val/test splits
        for split in [(0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (0.7, 0.15, 0.15)]:
            dm = hf_datamodule.HFMotrPacDataModule(data_dir=data_dir, train_val_test_split=split)
            assert dm.train_val_test_split == split


def test_hf_datamodule_state_dict() -> None:
    """Test state dict methods."""
    dm = hf_datamodule.HFMotrPacDataModule()

    # Test state dict
    state_dict = dm.state_dict()
    assert isinstance(state_dict, dict)

    # Test load state dict
    dm.load_state_dict(state_dict)  # Should not raise any error


def test_hf_datamodule_teardown() -> None:
    """Test teardown method."""
    dm = hf_datamodule.HFMotrPacDataModule()

    # Should not raise any error
    dm.teardown()


def test_hf_datamodule_log_stats() -> None:
    """Test log stats method with mocked data."""
    dm = hf_datamodule.HFMotrPacDataModule()

    # Mock adjacency matrix
    adj_matrix = np.random.randint(0, 2, (10, 10))
    np.fill_diagonal(adj_matrix, 1)

    # Mock the log_callback
    log_callback = MagicMock()

    # Mock np.load to return our test adjacency matrix
    with patch("numpy.load", return_value=adj_matrix):
        dm.log_stats(log_callback)

    # Check that log_callback was called with expected keys
    expected_calls = [
        ("mean_degree", pytest.approx(np.mean(np.sum(adj_matrix, axis=1)))),
        (
            "median_degree",
            pytest.approx(np.median(np.sum(adj_matrix, axis=1))),
        ),
        ("min_degree", pytest.approx(np.min(np.sum(adj_matrix, axis=1)))),
        ("max_degree", pytest.approx(np.max(np.sum(adj_matrix, axis=1)))),
        ("total_edges", pytest.approx(np.sum(np.sum(adj_matrix, axis=1)) / 2)),
        ("num_nodes", 10),
    ]

    for call in expected_calls:
        log_callback.assert_any_call(call[0], call[1])

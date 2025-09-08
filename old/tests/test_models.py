import pytest
import torch
from src.models.gnn_models import MLAGNN, NRGNN, RTGNN, SAGN, EnGCN
from src.models.gnn_module import GNNLitModule
from src.models.naive_models import MLP4
from torch_geometric.data import Data


@pytest.mark.parametrize("model_class", [NRGNN, RTGNN, EnGCN, SAGN, MLAGNN])
def test_lit_module_forward_pass(model_class):
    num_nodes = 8
    num_edges = 16
    num_features = 32
    out_channels = 1

    # Dummy graph
    x = torch.randn((num_nodes, num_features))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randn((1,))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y, batch=batch)
    data.adj_t = edge_index

    model = model_class(
        in_channels=num_features,
        hidden_channels=32,
        out_channels=out_channels,
        num_layers=2,
        dropout=0.1,
    )

    lit_model = GNNLitModule(
        net=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        scheduler=None,
        compile=False,
        adjacency_aware=True,
    )

    lit_model.eval()
    with torch.no_grad():
        loss, preds, targets = lit_model.model_step(data)

    assert preds.shape == torch.Size(
        [1, 1]
    ), f"Expected graph-level output shape (1,) but got {preds.shape}"
    assert torch.isfinite(preds).all(), "Predictions contain NaNs or Infs"
    assert torch.isfinite(loss), "Loss is NaN or Inf"
    assert preds.dtype == torch.float32, "Expected float32 predictions"


@pytest.mark.parametrize("model_class", [MLP4])
def test_lit_module_forward_pass_non_adjacency_aware(model_class):
    num_nodes = 8
    num_edges = 16
    num_features = 32
    out_channels = 1

    # Dummy graph
    x = torch.randn((num_nodes, num_features))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randn((1,))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y, batch=batch)
    data.adj_t = edge_index

    model = model_class(
        in_channels=num_features,
        hidden_channels=32,
        out_channels=out_channels,
        num_layers=2,
        dropout=0.1,
    )

    lit_model = GNNLitModule(
        net=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        scheduler=None,
        compile=False,
        adjacency_aware=False,
    )

    lit_model.eval()
    with torch.no_grad():
        loss, preds, targets = lit_model.model_step(data)

    assert preds.shape == torch.Size(
        [1, 1]
    ), f"Expected graph-level output shape (1,) but got {preds.shape}"
    assert torch.isfinite(preds).all(), "Predictions contain NaNs or Infs"
    assert torch.isfinite(loss), "Loss is NaN or Inf"
    assert preds.dtype == torch.float32, "Expected float32 predictions"

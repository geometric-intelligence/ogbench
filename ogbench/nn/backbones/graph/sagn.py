"""SAGN backbone inspired by https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking/blob/main/Precomputing/SAGN.py."""

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import Adj
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_sparse import SparseTensor


def normalize_adj(
    edge_index: torch.Tensor, num_nodes: int, add_loops: bool = True
) -> SparseTensor:
    r"""Build sym-normalized adjacency \hat{A} = D^{-1/2} (A + I) D^{-1/2} as SparseTensor."""

    row, col = edge_index
    if add_loops:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index

    # undirected
    row, col = to_undirected(row, col, num_nodes=num_nodes)
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv_sqrt = (deg + 1e-12).pow(-0.5)
    val = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    A = SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes))
    return A


@torch.no_grad()
def precompute_khop_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_hops: int,
    *,
    add_loops: bool = True,
    keep_x0: bool = True,
) -> List[torch.Tensor]:
    """Returns [X^0, A X^0, A^2 X^0, ...] length = num_hops (+1 if keep_x0)."""
    N = x.size(0)
    A = normalize_adj(edge_index, N, add_loops=add_loops)
    feats = []
    h = x

    if keep_x0:
        feats.append(h.clone())

    for _ in range(1 if keep_x0 else 0, num_hops + (0 if keep_x0 else 1)):
        h = A @ h
        feats.append(h.clone())

    return feats


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0, bn=True):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                if bn:
                    layers += [nn.BatchNorm1d(dims[i + 1])]
                layers += [nn.ReLU()]
                if dropout > 0:
                    layers += [nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HopAttention(nn.Module):
    """Multi-head attention over the list of hop-encoded features.

    Treat hops as the 'sequence'. Simple additive attention per head.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.proj = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
                for _ in range(num_heads)
            ]
        )

    def forward(self, hop_feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hop_feats: list of [N, D]
        returns:
          H: [num_heads, N, D] attended head embeddings
          attn: [num_heads, N, L] attention weights
        """
        L = len(hop_feats)
        N, D = hop_feats[0].shape
        stack = torch.stack(hop_feats, dim=1)  # [N, L, D]

        heads_out = []
        attn_ws = []
        for head in self.proj:
            # scores: [N, L, 1] -> [N, L]
            scores = head(stack).squeeze(-1)
            alpha = F.softmax(scores, dim=1)
            # weighted sum over hops
            h = (alpha.unsqueeze(-1) * stack).sum(dim=1)  # [N, D]
            heads_out.append(h)
            attn_ws.append(alpha)

        H = torch.stack(heads_out, dim=0)  # [H, N, D]
        A = torch.stack(attn_ws, dim=0)  # [H, N, L]
        return H, A


# ---------- SAGN core ----------


class SAGNEncoder(nn.Module):
    """
    SAGN-style encoder:
      1) Precompute K-hop feature list externally (or on the fly once).
      2) Encode each hop with (shared or per-hop) MLP.
      3) Attend over hops with multi-head attention.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_hops: int,
        *,
        mlp_layers: int = 2,
        mlp_dropout: float = 0.2,
        share_mlp_across_hops: bool = True,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.share = share_mlp_across_hops

        if share_mlp_across_hops:
            self.hop_mlps = nn.ModuleList(
                [MLP(in_dim, hidden_dim, embed_dim, num_layers=mlp_layers, dropout=mlp_dropout)]
            )
        else:
            self.hop_mlps = nn.ModuleList(
                [
                    MLP(in_dim, hidden_dim, embed_dim, num_layers=mlp_layers, dropout=mlp_dropout)
                    for _ in range(num_hops + 1)  # including X^0
                ]
            )

        self.attn = HopAttention(embed_dim, num_heads)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(self, hop_feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hop_feats: list length L = num_hops+1, each [N, Fin]
        returns:
          H: [H, N, D] (head-wise embeddings) and attention weights [H, N, L]
        """
        L = len(hop_feats)
        assert L == self.num_hops + 1, f"expected {self.num_hops+1} hops, got {L}"

        encs = []
        if self.share:
            mlp = self.hop_mlps[0]
            for h in hop_feats:
                encs.append(mlp(h))
        else:
            for mlp, h in zip(self.hop_mlps, hop_feats):
                encs.append(mlp(h))

        H, A = self.attn(encs)  # [H, N, D], [H, N, L]
        return H, A


class SAGNHeadClassifier(nn.Module):
    """Maps per-head embeddings to logits, then aggregates heads.

    For node-level: inputs [H, N, D] -> [N, C]
    For graph-level: pool nodes first, then heads -> [B, C]
    """

    def __init__(self, embed_dim: int, num_classes: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.per_head = nn.ModuleList(
            [
                nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
                for _ in range(num_heads)
            ]
        )

    def forward_node(self, H: torch.Tensor) -> torch.Tensor:
        # H: [H, N, D] -> list of [N, C]
        logits = [head(H[i]) for i, head in enumerate(self.per_head)]
        # aggregate heads
        return torch.stack(logits, dim=0).mean(dim=0)  # [N, C]

    def forward_graph(self, H_graph: torch.Tensor) -> torch.Tensor:
        # H_graph: [H, B, D]
        logits = [head(H_graph[i]) for i, head in enumerate(self.per_head)]
        return torch.stack(logits, dim=0).mean(dim=0)  # [B, C]


class SAGN(nn.Module):
    """Full model. You can:

    - Use node-level classification (default)
    - Or set task='graph' to pool over node dim before classification
    - Or set `return_embeddings=True` to skip classifier and use as backbone
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_classes: int,
        num_hops: int,
        *,
        num_heads: int = 4,
        task: Literal["node", "graph"] = "node",
        graph_pool: Literal["mean", "sum", "max"] = "mean",
        mlp_layers: int = 2,
        mlp_dropout: float = 0.2,
        share_mlp_across_hops: bool = True,
        head_dropout: float = 0.0,
        return_embeddings: bool = False,
    ):
        super().__init__()
        self.encoder = SAGNEncoder(
            in_dim,
            hidden_dim,
            embed_dim,
            num_hops,
            mlp_layers=mlp_layers,
            mlp_dropout=mlp_dropout,
            share_mlp_across_hops=share_mlp_across_hops,
            num_heads=num_heads,
        )
        self.task = task
        self.return_embeddings = return_embeddings

        if not return_embeddings:
            self.classifier = SAGNHeadClassifier(embed_dim, num_classes, num_heads, head_dropout)

        # simple pooling
        if task == "graph":
            if graph_pool == "mean":
                self.pool = lambda x, batch: torch.scatter_reduce(
                    batch, x, reduce="mean"
                )  # placeholder; replaced below
            # We'll implement using torch_scatter if available:
            try:
                from torch_scatter import scatter

                def _pool(x, batch, reduce):
                    return scatter(x, batch, dim=0, reduce=reduce)

                if graph_pool == "mean":
                    self._pool = lambda x, batch: _pool(x, batch, "mean")
                elif graph_pool == "sum":
                    self._pool = lambda x, batch: _pool(x, batch, "sum")
                elif graph_pool == "max":
                    self._pool = lambda x, batch: _pool(x, batch, "max")
                else:
                    raise ValueError("graph_pool must be one of ['mean','sum','max']")
            except Exception:
                raise ImportError("Install torch-scatter for graph-level pooling.")

    def forward(self, hop_feats: List[torch.Tensor], batch: Optional[torch.Tensor] = None):
        """
        hop_feats: list of [N, Fin] (precomputed)
        batch: node->graph assignment [N] if task='graph'
        """
        H, attn = self.encoder(hop_feats)  # [H, N, D], [H, N, L]
        if self.return_embeddings:
            if self.task == "node":
                # Return per-head node embeddings and attention weights
                return H, attn
            else:
                assert batch is not None, "batch is required for graph-level."
                # pool per head -> [H, B, D]
                B = int(batch.max()) + 1
                pooled = []
                for i in range(H.size(0)):
                    pooled.append(self._pool(H[i], batch))  # [B, D]
                H_graph = torch.stack(pooled, dim=0)  # [H, B, D]
                return H_graph, attn

        # classification
        if self.task == "node":
            logits = self.classifier.forward_node(H)  # [N, C]
            return logits, attn
        else:
            assert batch is not None, "batch is required for graph-level."
            pooled = []
            for i in range(H.size(0)):
                pooled.append(self._pool(H[i], batch))  # [B, D]
            H_graph = torch.stack(pooled, dim=0)  # [H, B, D]
            logits = self.classifier.forward_graph(H_graph)  # [B, C]
            return logits, attn


# How to use it
# 1) Precompute once (offline or on first epoch)

# python
# Copy code
# # x: [N, Fin]; edge_index: [2, E]; K: number of hops (e.g., 4)
# hop_feats = precompute_khop_features(x, edge_index, num_hops=4, add_loops=True, keep_x0=True)
# # hop_feats = [X^0, A X^0, A^2 X^0, A^3 X^0, A^4 X^0]
# 2) Node classification

# python
# Copy code
# model = SAGN(
#     in_dim=x.size(1),
#     hidden_dim=256,
#     embed_dim=256,
#     num_classes=C,
#     num_hops=4,
#     num_heads=4,
#     task="node",
#     return_embeddings=False,
# )

# logits, attn = model(hop_feats)   # logits: [N, C]
# loss = F.cross_entropy(logits[idx_train], y[idx_train])

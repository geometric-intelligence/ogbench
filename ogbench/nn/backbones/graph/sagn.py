"""SAGN-style GNN (PyTorch Geometric) with proper edge-weight handling and a fast SparseTensor
path.

Notes:
- Adjacency orientation: for SparseTensor inputs we ensure **adj_t** (col->row)
  before normalization (see `.t()` in code). Coordinate format inputs are normalized then
  converted to adj_t explicitly.
- This is **SAGN-style**, not the full canonical recipe: it computes k-hop
  features on-the-fly (no precompute cache) and omits label-enhanced inputs /
  self-training and logit-level head ensembling used in some SAGN pipelines.
- We avoid MessagePassing.propagate and use SparseTensor.matmul for stability
  and speed.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
from torch_sparse import SparseTensor


class HopAttention(nn.Module):
    """
    Multi-head per-hop attention:
      - Input: list of K+1 tensors [N, D].
      - Output: mixed features [N, D] and attention weights [N, H, K+1].
    This computes one scalar score per (node, head, hop), softmax over hops.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.score = nn.Linear(self.head_dim, 1)  # scalar per hop per head
        self.dropout = nn.Dropout(dropout)

    def forward(self, hop_features: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # Stack hops: [N, K+1, D]
        hop_stack = torch.stack(hop_features, dim=1)
        N, Kp1, D = hop_stack.shape

        x = self.proj(hop_stack)  # [N, K+1, D]
        x = x.view(N, Kp1, self.num_heads, self.head_dim).transpose(1, 2)  # [N, H, K+1, d]

        scores = self.score(x).squeeze(-1)  # [N, H, K+1]
        attn = F.softmax(scores, dim=-1)  # [N, H, K+1]
        attn = self.dropout(attn)

        out = (attn.unsqueeze(-1) * x).sum(dim=2)  # [N, H, d]
        out = out.reshape(N, D)  # [N, D]
        return out, attn  # attn = [N, H, K+1]


class HopConv(nn.Module):
    """Computes 0..K hop features with learnable per-hop linear maps.

    Uses SparseTensor matmul for both coordinate tensor and SparseTensor inputs to avoid
    PyG MessagePassing API quirks with edge_weight kwargs.

    Normalization:
      Uses GCN normalization (D^{-1/2} A D^{-1/2}) via `gcn_norm`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_hops: int = 1,
        add_self_loops: bool = True,
        make_undirected: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_hops = num_hops
        self.add_self_loops = add_self_loops
        self.make_undirected = make_undirected

        # One linear per hop (including 0-hop)
        self.hop_lin = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=False) for _ in range(num_hops + 1)]
        )

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.hop_lin:
            lin.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Returns:
            out_sum: [N, D_out]
            hop_features: list of [N, D_out] for hops 0..K
        """
        N = x.size(0)

        # 1) Build a normalized SparseTensor adjacency A_norm (adj_t orientation)
        if isinstance(edge_index, SparseTensor):
            A = edge_index
            if not A.is_t():  # ensure adj_t expected by gcn_norm
                A = A.t()
            if self.make_undirected:
                A = A.to_symmetric()
            # gcn_norm on SparseTensor returns adj_t
            A_norm = gcn_norm(A, add_self_loops=self.add_self_loops, improved=False)
        else:
            # edge_index is a coordinate tensor
            ei = edge_index
            ew = edge_weight
            if self.make_undirected:
                ei, ew = to_undirected(ei, ew, num_nodes=N)
            ei_norm, ew_norm = gcn_norm(
                ei,
                edge_weight=ew,
                num_nodes=N,
                add_self_loops=self.add_self_loops,
                improved=False,
                dtype=x.dtype,
            )
            # Build adj_t (col->row) so that A_norm.matmul(x) aggregates INTO rows
            row, col = ei_norm
            A_norm = SparseTensor(row=col, col=row, value=ew_norm, sparse_sizes=(N, N))

        # 2) Build hop features via repeated matmul
        hop_features: List[Tensor] = []
        # 0-hop
        hop_features.append(self.hop_lin[0](x))
        current = x
        for k in range(1, self.num_hops + 1):
            # Multiply once per hop
            current = A_norm.matmul(current)  # [N, Din]
            hop_features.append(self.hop_lin[k](current))

        out = torch.stack(hop_features, dim=0).sum(dim=0)
        if self.bias is not None:
            out = out + self.bias
        return out, hop_features


class SAGNEncoder(nn.Module):
    """
    SAGN-style encoder:
      InputProj -> HopConv (builds 0..K hop features) -> HopAttention -> OutputProj
    Returns node-level embeddings and hop attention weights.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_hops: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        add_self_loops: bool = True,
        make_undirected: bool = True,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.num_heads = num_heads

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.hop_conv = HopConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_hops=num_hops,
            add_self_loops=add_self_loops,
            make_undirected=make_undirected,
        )

        self.hop_attention = HopAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [N, Fin]
            edge_index: coordinate tensor [2,E] or SparseTensor
            edge_weight: [E] if coordinate tensor

        Returns:
            node_embeddings: [N, Fout]
            attn_weights:    [N, H, K+1]
        """
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)

        _, hop_features = self.hop_conv(x, edge_index, edge_weight)
        mixed, attn = self.hop_attention(hop_features)

        out = self.output_proj(mixed)
        return out, attn


class SAGN(nn.Module):
    """
    Backbone version: returns graph-level embeddings for readout compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_hops: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        add_self_loops: bool = True,
        make_undirected: bool = True,
        pooling_type: str = "sum",
        **kwargs,
    ):
        super().__init__()
        self.encoder = SAGNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout,
            add_self_loops=add_self_loops,
            make_undirected=make_undirected,
        )
        self.pooling_type = pooling_type

    def forward(
        self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, batch: OptTensor = None
    ) -> Tensor:
        node_embeddings, _ = self.encoder(x, edge_index, edge_weight)

        # If batch is provided, pool to graph-level embeddings
        if batch is not None:
            graph_embeddings = scatter(node_embeddings, batch, dim=0, reduce=self.pooling_type)
            return graph_embeddings

        # Otherwise return node embeddings (for backward compatibility)
        return node_embeddings

from typing import Final

from torch_geometric.nn.conv import ChebConv, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN


class ChebNet(BasicGNN):
    """Chebyshev GNN using `~torch_geometric.nn.conv.ChebConv` operator for message passing."""

    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return ChebConv(in_channels, out_channels, **kwargs)

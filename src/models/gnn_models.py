from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_self_loops, degree


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATv2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphSage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class NRGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        alpha: float = 0.1,
        beta: float = 0.1,
        gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.alpha = alpha  # Node feature importance
        self.beta = beta    # Edge importance
        self.gamma = gamma  # Structure importance
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # Node feature transformation
        x = x * self.alpha
        
        # Edge importance
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device) * self.beta
        
        # Structure importance
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x * self.gamma  # Structure importance scaling
        x = self.convs[-1](x, edge_index, edge_weight)
        return x


class RTGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        alpha: float = 0.1,
        beta: float = 0.1,
        gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.alpha = alpha  # Temporal importance
        self.beta = beta    # Spatial importance
        self.gamma = gamma  # Feature importance
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # Feature importance
        x = x * self.gamma
        
        # Spatial importance
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device) * self.beta
        
        # Temporal importance
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x * self.alpha  # Temporal importance scaling
        x = self.convs[-1](x, edge_index, edge_weight)
        return x 


class EnGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_propagations: int = 20,
        aggregation_ratio: float = 0.75,
        adj_norm: str = "AD",
        auto_scale: bool = True,
        num_mlp_layers: int = 2,
        diffusion_type: str = "residual",
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.num_propagations = num_propagations
        self.aggregation_ratio = aggregation_ratio
        self.adj_norm = adj_norm
        self.auto_scale = auto_scale
        self.diffusion_type = diffusion_type
        self.dropout = dropout
        
        # MLP layers for feature transformation
        self.mlp = nn.ModuleList()
        for _ in range(num_mlp_layers):
            self.mlp.append(nn.Linear(hidden_channels, hidden_channels))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout))

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # Initial feature transformation
        for layer in self.mlp:
            x = layer(x)
        
        # Multiple propagation steps
        for _ in range(self.num_propagations):
            if self.diffusion_type == "residual":
                x_new = x
            else:  # zeros
                x_new = torch.zeros_like(x)
            
            # Apply GCN layers
            for conv in self.convs[:-1]:
                x_new = conv(x_new, edge_index)
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Aggregate with previous features
            x = self.aggregation_ratio * x_new + (1 - self.aggregation_ratio) * x
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        return x


class SAGN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_propagations: int = 20,
        aggregation_ratio: float = 0.75,
        adj_norm: str = "AD",
        auto_scale: bool = True,
        num_mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.num_propagations = num_propagations
        self.aggregation_ratio = aggregation_ratio
        self.adj_norm = adj_norm
        self.auto_scale = auto_scale
        self.dropout = dropout
        
        # MLP layers for feature transformation
        self.mlp = nn.ModuleList()
        for _ in range(num_mlp_layers):
            self.mlp.append(nn.Linear(hidden_channels, hidden_channels))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout))

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # Initial feature transformation
        for layer in self.mlp:
            x = layer(x)
        
        # Multiple propagation steps
        for _ in range(self.num_propagations):
            x_new = torch.zeros_like(x)
            
            # Apply SAGE layers
            for conv in self.convs[:-1]:
                x_new = conv(x_new, edge_index)
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Aggregate with previous features
            x = self.aggregation_ratio * x_new + (1 - self.aggregation_ratio) * x
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        return x


class MLAGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_propagations: int = 20,
        aggregation_ratio: float = 0.75,
        adj_norm: str = "AD",
        auto_scale: bool = True,
        num_mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.num_propagations = num_propagations
        self.aggregation_ratio = aggregation_ratio
        self.adj_norm = adj_norm
        self.auto_scale = auto_scale
        self.dropout = dropout
        
        # Multiple MLP layers for feature transformation
        self.mlps = nn.ModuleList()
        for _ in range(num_mlp_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.mlps.append(mlp)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # Initial feature transformation through multiple MLPs
        for mlp in self.mlps:
            x = mlp(x)
        
        # Multiple propagation steps
        for _ in range(self.num_propagations):
            x_new = torch.zeros_like(x)
            
            # Apply GCN layers
            for conv in self.convs[:-1]:
                x_new = conv(x_new, edge_index)
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Aggregate with previous features
            x = self.aggregation_ratio * x_new + (1 - self.aggregation_ratio) * x
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        return x 
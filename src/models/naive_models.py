"""Various GNN models."""

import torch.nn as nn
from torch import Tensor
import torch

class MLP4(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 4, hidden_channels: int = 1024, num_nodes: int = 100) -> None:
        self.num_nodes = num_nodes
        super().__init__()
        self.out_channels = out_channels
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([nn.Linear(in_channels, hidden_channels), nn.ReLU()])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_channels, out_channels))
            else:
                layers.extend([nn.Linear(hidden_channels, hidden_channels), nn.ReLU()])
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        x = torch.reshape(x, (-1, self.num_nodes))
        x = self.model(x)
        return torch.reshape(x, (-1,))
        
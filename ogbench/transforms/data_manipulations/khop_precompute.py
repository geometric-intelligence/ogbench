"""K-hop precompute transform."""

import torch_geometric
from torch_geometric.transforms import SIGN


class KHopPrecompute(torch_geometric.transforms.BaseTransform):
    """Precompute k-hop features using SIGN and attach them to `data`.

    Produces:
      - data.x, data.x1, ..., data.xK   (SIGN outputs)
      - data.xs : list[Tensor] = [x, x1, ..., xK]

    Args:
        num_layers (int): K in SIGN (number of hops).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.parameters = kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}(parameters={self.parameters})'

    def forward(self, data: torch_geometric.data.Data):
        # SIGN adds x1..xK to `data` (and updates x if needed)
        num_layers = self.parameters['num_layers']
        data = SIGN(num_layers)(data)
        # Collect as a convenient list like your PrecomputingBase
        data.xs = [data.x] + [data[f'x{i}'] for i in range(1, num_layers + 1)]
        return data

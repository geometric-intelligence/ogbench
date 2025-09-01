from typing import Final
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import (
    ChebConv,
    MessagePassing,
)
class ChebNet(BasicGNN):
    r"""Chebyshev GNN using :class:`~torch_geometric.nn.conv.ChebConv` 
    operator for message passing.

    Parameters
    ----------
    in_channels : int
        Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    hidden_channels : int
        Size of each hidden sample.
    num_layers : int
        Number of message passing layers.
    out_channels : int, optional
        Size of each output sample. If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    K : int
        Chebyshev filter size :math:`K`.
    dropout : float, optional
        Dropout probability. (default: :obj:`0.`)
    act : str or Callable, optional
        The non-linear activation function to use. (default: :obj:`"relu"`)
    act_first : bool, optional
        If set to :obj:`True`, activation is applied before normalization. (default: :obj:`False`)
    act_kwargs : Dict[str, Any], optional
        Arguments passed to the respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm : str or Callable, optional
        The normalization function to use. (default: :obj:`None`)
    norm_kwargs : Dict[str, Any], optional
        Arguments passed to the respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk : str, optional
        The Jumping Knowledge mode. If specified, the model will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.ChebConv`.
    """
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return ChebConv(in_channels, out_channels, **kwargs)
    
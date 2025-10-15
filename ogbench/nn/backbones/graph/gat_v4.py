import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch


# Overriding parameter initialization
class CustomGATConv(GATv2Conv):
    def __init__(
        self,
        in_channels: int | tuple[int, int],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: int | None = None,
        fill_value: float | Tensor | str = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        weight_initializer=None,  # The custom weight initializer
    ):
        # Save custom weight initializer
        self.weight_init = weight_initializer

        # Call the base GATv2Conv initializer
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias,
            share_weights=share_weights,
            residual=residual,
        )

        # Apply custom weight initialization if provided
        if weight_initializer is not None:
            self.apply_custom_initializers()

    def apply_custom_initializers(self):
        """Apply custom weight initializers to the GATv2Conv layers."""
        if self.lin_l is not None:
            self.weight_init(self.lin_l.weight)
        if self.lin_r is not None and not self.share_weights:
            self.weight_init(self.lin_r.weight)
        if self.lin_edge is not None:
            self.weight_init(self.lin_edge.weight)
        if self.res is not None:
            self.weight_init(self.res.weight)

        # Initialize the attention coefficient
        self.weight_init(self.att)

        # Initialize the bias if it exists.
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters(self):
        """Override reset_parameters to ensure custom initialization."""
        # Call the original reset_parameters
        super().reset_parameters()

        # Re-apply custom initialization after reset
        if self.weight_init is not None:
            self.apply_custom_initializers()


class GATv4(nn.Module):
    ACT_MAP = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
    }
    INIT_MAP = {
        'uniform': nn.init.uniform_,
        'xavier': nn.init.xavier_uniform_,
        'kaiming': nn.init.kaiming_uniform_,
        'orthogonal': nn.init.orthogonal_,
        'truncated_normal': torch.nn.init.trunc_normal_,
    }

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads,
        dropout,
        act,
        which_layer,
        use_layer_norm,
        num_nodes,
        weight_initializer,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.act = act
        self.which_layer = which_layer
        self.use_layer_norm = use_layer_norm
        self.fc_input_dim = num_nodes * len(which_layer)
        self.weight_initializer = self.INIT_MAP[weight_initializer]
        self.num_nodes = num_nodes

        # GAT layers
        self.convs = nn.ModuleList()
        self.build_gat_layers()

        # Pooling layers
        self.pools = nn.ModuleList()
        self.build_pooling_layers()

        # Layer normalization
        self.layer_norm = LayerNorm(self.num_nodes)

        # Initialize weights
        self.reset_parameters()

    def build_gat_layers(self):
        input_dim = self.in_channels
        for hidden_dim, num_heads in zip(self.hidden_channels, self.heads, strict=True):
            self.convs.append(
                GATv2Conv(
                    in_channels=input_dim,
                    out_channels=hidden_dim,  # dim of each node at the end
                    heads=num_heads,
                    dropout=self.dropout,
                    concat=True,
                )
            )
            input_dim = hidden_dim * num_heads

    def build_pooling_layers(self):
        for hidden_dim, num_heads in zip(self.hidden_channels, self.heads, strict=True):
            self.pools.append(nn.Linear(hidden_dim * num_heads, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for pool in self.pools:
            self.weight_initializer(pool.weight)
            if pool.bias is not None:
                pool.bias.data.fill_(0)

    def forward(self, x, edge_index=None, data=None, return_attention_weights=False):
        if not isinstance(data, Batch):
            data = Batch().from_data_list([data])

        batch = data.batch_0
        edge_index = data.edge_index

        # Initial operations before GAT layers
        x = x.requires_grad_()  # [bs*nodes, in_channels]
        # This does mean on features, per subgraph and per node.
        # [bs, nodes], converts it to be broken up by batches
        x0, _ = to_dense_batch(torch.mean(x, dim=-1), batch=batch)

        # Apply first GAT layer and pooling
        if return_attention_weights:
            x, att1 = self.convs[0](x, edge_index, return_attention_weights=True)
        else:
            x = self.convs[0](x, edge_index)

        # [bs*nodes, hidden_channels[0]*heads[0]], Apply the gatconv layer
        x = self.ACT_MAP[self.act](
            x
        )  # [bs*nodes, hidden_channels[0]*heads[0]], Apply elu activation function
        x1 = self.pools[0](x)  # [bs*nodes, 1]
        x1 = x1.squeeze(-1)  # [bs*nodes]
        x1, _ = to_dense_batch(x1, batch=batch)  # [bs, nodes]

        # Apply second GAT layer and pooling
        x = F.dropout(
            x, p=self.dropout, training=self.training
        )  # apply dropout if we are training
        if return_attention_weights:
            x, att2 = self.convs[1](x, edge_index, return_attention_weights=True)
        else:
            x = self.convs[1](x, edge_index)

        x = self.ACT_MAP[self.act](x)  # [bs*nodes, hidden_channels[1]*heads[1]]
        x2 = self.pools[1](x)  # [bs*nodes, 1]
        x2 = x2.squeeze(-1)  # [bs*nodes]
        x2, _ = to_dense_batch(x2, batch=batch)  # [bs, nodes]

        # Apply layer normalization to each individual graph to have mean 0, std 1
        if self.use_layer_norm:
            x0 = self.layer_norm(x0)  # [bs, nodes]
            x1 = self.layer_norm(x1)  # [bs, nodes]
            x2 = self.layer_norm(x2)  # [bs, nodes]

        # Concatenate multiscale features - results in [bs, 3*nodes]
        multiscale_features = {'layer1': x0, 'layer2': x1, 'layer3': x2}
        multiscale_features = torch.cat(
            [multiscale_features[layer] for layer in self.which_layer],
            dim=1,
        )

        aux = [x0, x1, x2, multiscale_features]

        if return_attention_weights:
            return multiscale_features, aux, att1, att2
        return multiscale_features, aux

import torch.nn as nn

from ogbench.nn.readouts.base import AbstractZeroCellReadOut


class OmicsReadOut(AbstractZeroCellReadOut):
    ACT_MAP = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
    }

    def __init__(
        self,
        num_nodes,
        hidden_dim,
        which_layer,
        fc_dim=None,
        fc_dropout=None,
        fc_act=None,
        out_channels=None,
        graph_encoder_dim=None,  # 256
        **kwargs,
    ):
        super().__init__(out_channels=out_channels, hidden_dim=hidden_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.graph_encoder_dim = (
            [graph_encoder_dim] if isinstance(graph_encoder_dim, int) else list(graph_encoder_dim)
        )
        self.which_layer = which_layer
        self.fc_dim = fc_dim
        self.fc_dropout = fc_dropout
        self.fc_act = fc_act
        self.fc_input_dim = self.hidden_dim
        self.out_channels = out_channels  # 1
        self.readout_layers = self.build_readout_layers()
        # self.graph_encoder = self.build_graph_encoder()

    # def build_graph_encoder(self):
    #     channel_list = [self.hidden_dim] + self.graph_encoder_dim
    #     return MLP(channel_list, dropout=self.fc_dropout, act=self.ACT_MAP[self.fc_act])

    def build_readout_layers(self):
        layers = [nn.LayerNorm(self.hidden_dim)]
        fc_layer_input_dim = self.fc_input_dim
        for fc_dim in self.fc_dim:
            layers.append(
                nn.Sequential(
                    nn.Linear(fc_layer_input_dim, fc_dim),
                    self.ACT_MAP[self.fc_act],
                    nn.AlphaDropout(p=self.fc_dropout, inplace=True),
                )
            )
            fc_layer_input_dim = fc_dim
        layers.append(nn.Linear(fc_dim, self.out_channels))
        return nn.Sequential(*layers)

    def forward(self, model_out, batch):
        flattened_features = model_out['x_0'].view(batch.batch_size, -1)
        # encoded_graph = self.graph_encoder(flattened_features)
        model_out['x_0'] = self.readout_layers(flattened_features)
        return model_out

    def __call__(self, model_out, batch) -> dict:
        """Readout logic based on model_output.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        model_out = self.forward(model_out, batch)
        model_out['logits'] = model_out['x_0']

        return model_out

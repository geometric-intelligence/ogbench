"""Wrapper for the GATv4 models."""

from ogbench.nn.wrappers.base import AbstractWrapper


class GATv4Wrapper(AbstractWrapper):
    r"""Wrapper for the GATv4 models.

    This wrapper defines the forward pass of the model. The GNN models return the embeddings of the
    cells of rank 0.
    """

    def forward(self, batch):
        r"""Forward pass for the GNN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        x_0, _ = self.backbone(
            batch.x_0,
            batch.edge_index,
            batch,
        )

        model_out = {'labels': batch.y, 'batch_0': batch.batch_0}
        # Use actual total number of nodes from the batch instead of assuming
        # a fixed num_nodes per graph, which would break if graphs vary in size.
        model_out['x_0'] = x_0.view(batch.x_0.size(0), -1)

        return model_out

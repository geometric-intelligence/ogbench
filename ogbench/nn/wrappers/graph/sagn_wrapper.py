"""Wrapper for the SAGN models."""

from ogbench.nn.wrappers.base import AbstractWrapper


class SAGNWrapper(AbstractWrapper):
    r"""Wrapper for the SAGN models.

    This wrapper defines the forward pass of the model. The SAGN models return the embeddings of
    the cells of rank 0 using PyTorch Geometric message passing framework.
    """

    def forward(self, batch):
        r"""Forward pass for the SAGN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        x_0 = self.backbone(
            batch.x_0,
            batch.edge_index,
            edge_weight=batch.get("edge_weight", None),
            batch=batch.batch_0,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0

        return model_out

"""Combined Positional and Structural Encodings Transform."""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# Supported Positional and Structural Encodings
PSE_ENCODINGS = {'LapPE', 'RWSE', 'ElectrostaticPE', 'HKdiagSE'}


class CombinedPSEs(BaseTransform):
    r"""Combined PSEs transform.

    Applies one or more pre-defined positional or structural encoding transforms
    (LapPE, RWSE) to a graph, storing their outputs and optionally
    concatenating them to `data.x`.

    When ``shared_graph=True``, the encodings are computed only once (on the
    first sample) and reused for all subsequent samples. This is useful when
    all samples share the same graph structure (e.g. omics datasets), since
    all PSE encodings depend only on ``edge_index`` and ``num_nodes``, not on
    the per-sample node features ``data.x``.

    Parameters
    ----------
    encodings : list of str
        List of structural encodings to apply. Supported values are
        "LapPE" for Laplacian Positional Encoding and "RWSE" for
        Random Walk Structural Encoding.
    parameters : dict, optional
        Additional parameters for the encoding transforms.
    shared_graph : bool, optional
        If True, compute PSEs once and cache them for reuse across samples
        that share the same graph structure. Default is False.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        encodings: list[str],
        parameters: dict | None = None,
        shared_graph: bool = False,
        **kwargs,
    ):
        self.encodings = encodings
        self.parameters = parameters if parameters is not None else {}
        self.shared_graph = shared_graph
        self._pse_cache: dict | None = None

    def forward(self, data: Data) -> Data:
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data with added structural encodings.
        """
        if self.shared_graph and self._pse_cache is not None:
            return self._apply_cached_pses(data)

        from ogbench.transforms.data_manipulations import (
            RWSE,
            ElectrostaticPE,
            HKdiagSE,
            LapPE,
        )

        encoding_classes = {
            'LapPE': LapPE,
            'RWSE': RWSE,
            'ElectrostaticPE': ElectrostaticPE,
            'HKdiagSE': HKdiagSE,
        }

        # Validate encoding_classes matches PSE_ENCODINGS
        if set(encoding_classes.keys()) != PSE_ENCODINGS:
            missing_in_classes = PSE_ENCODINGS - set(encoding_classes.keys())
            missing_in_set = set(encoding_classes.keys()) - PSE_ENCODINGS
            raise RuntimeError(
                f'encoding_classes and PSE_ENCODINGS are out of sync. '
                f'Missing in encoding_classes: {missing_in_classes}. '
                f'Missing in PSE_ENCODINGS: {missing_in_set}.'
            )

        # Record original feature dimension to extract PSE columns later
        original_x_dim = data.x.shape[-1] if data.x is not None else 0

        for enc in self.encodings:
            if enc not in encoding_classes:
                raise ValueError(f'Unsupported encoding type: {enc}')

            encoder = encoding_classes[enc](**self.parameters.get(enc, {}))
            data = encoder(data)

        if self.shared_graph:
            self._build_cache(data, original_x_dim)

        return data

    def _build_cache(self, data: Data, original_x_dim: int) -> None:
        """Extract and cache the PSE tensors from the first processed sample.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The processed data (after all encodings have been applied).
        original_x_dim : int
            The feature dimension of ``data.x`` before encodings were applied.
        """
        cache: dict = {}

        # Cache concatenated PSE columns (from concat_to_x=True encodings)
        if data.x is not None and data.x.shape[-1] > original_x_dim:
            cache['_concat_pse'] = data.x[:, original_x_dim:].clone()

        # Cache attribute-based PSEs (from concat_to_x=False encodings)
        for enc in self.encodings:
            if hasattr(data, enc):
                cache[enc] = getattr(data, enc).clone()

        self._pse_cache = cache

    def _apply_cached_pses(self, data: Data) -> Data:
        """Apply previously cached PSE tensors to a new sample.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data (with per-sample node features but no PSEs yet).

        Returns
        -------
        torch_geometric.data.Data
            The data with cached PSEs applied.
        """
        cache = self._pse_cache

        # Apply concatenated PSE columns
        if '_concat_pse' in cache:
            if data.x is None:
                data.x = cache['_concat_pse']
            else:
                data.x = torch.cat([data.x, cache['_concat_pse']], dim=-1)

        # Apply attribute-based PSEs
        for enc in self.encodings:
            if enc in cache:
                setattr(data, enc, cache[enc])

        return data


class SelectDestinationPSEs(BaseTransform):
    r"""Select Destination Positional and Structural Encodings (PSEs) transform.

    Selects and retains only the PSEs corresponding to the destination nodes
    of edges in `data.edge_index`.

    Parameters
    ----------
    encoding_key : str
        Key in `data` where the PSEs are stored (e.g., 'LapPE', 'RWSE').
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(self, encodings, **kwargs):
        self.encodings = encodings

    def forward(self, data: Data, n_dst_nodes: int) -> Data:
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        n_dst_nodes : int
            Number of destination nodes.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data with selected PSEs.
        """
        new_data = {}
        new_data['x'] = data.x[:n_dst_nodes, :] if data.x is not None else None
        for encoding_key in self.encodings:
            if hasattr(data, encoding_key):
                pe = getattr(data, encoding_key)
                selected_pe = pe[:n_dst_nodes, :]
                new_data[encoding_key] = selected_pe
            else:
                raise ValueError(f"Encoding key '{encoding_key}' not found in data.")
        return Data(**new_data)

    def __call__(self, data: Data, n_dst_nodes: int) -> Data:
        return self.forward(data, n_dst_nodes)

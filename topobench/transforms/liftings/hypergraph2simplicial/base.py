"""Abstract class for lifting hypergraphs to simplicial complexes."""

import torch
from toponetx.classes import SimplicialComplex

from topobench.data.utils.utils import get_complex_connectivity
from topobench.transforms.liftings.liftings import HypergraphLifting


class Hypergraph2SimplicialLifting(HypergraphLifting):
    r"""Abstract class for lifting hypergraphs to simplicial complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "hypergraph2simplicial"

    def _get_lifted_topology(
        self, simplicial_complex: SimplicialComplex
    ) -> dict:
        r"""Return the lifted topology.

        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(
            simplicial_complex,
            self.complex_dim,
            neighborhoods=self.neighborhoods,
        )

        return lifted_topology

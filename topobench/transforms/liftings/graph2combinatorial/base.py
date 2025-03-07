"""Abstract class for lifting graphs to combinatorial complexes."""

from collections.abc import Iterable
import torch_geometric
from topobenchmark.transforms.liftings.liftings import (
    AbstractLifting,
    GraphLifting,
)

class Graph2CombinatorialLifting(GraphLifting):
    r"""Abstract class for lifting graphs to combinatorial complexes.

    Parameters
    ----------
    *liftings : optional
        The topological liftings needed to go from a graph to a combinatorial complex.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, *liftings, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2combinatorial"
        self.liftings: Iterable[AbstractLifting] = liftings

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a graph to a combinatorial complex. This is modified so that we can define multiple liftings.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        if not self.liftings:
            raise NotImplementedError
        for lifting in self.liftings:
            data = lifting.lift_topology(data)
        return data
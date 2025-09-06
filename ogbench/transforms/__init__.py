"""This module contains the transforms for the ogbench package."""

from typing import Any

from ogbench.transforms.data_manipulations import DATA_MANIPULATIONS
from ogbench.transforms.feature_liftings import FEATURE_LIFTINGS
from ogbench.transforms.liftings.graph2cell import GRAPH2CELL_LIFTINGS
from ogbench.transforms.liftings.graph2hypergraph import GRAPH2HYPERGRAPH_LIFTINGS
from ogbench.transforms.liftings.graph2simplicial import GRAPH2SIMPLICIAL_LIFTINGS

LIFTINGS = {
    **GRAPH2CELL_LIFTINGS,
    **GRAPH2HYPERGRAPH_LIFTINGS,
    **GRAPH2SIMPLICIAL_LIFTINGS,
}

TRANSFORMS: dict[Any, Any] = {
    **LIFTINGS,
    **FEATURE_LIFTINGS,
    **DATA_MANIPULATIONS,
}

__all__ = [
    "DATA_MANIPULATIONS",
    "FEATURE_LIFTINGS",
    "LIFTINGS",
    "TRANSFORMS",
]

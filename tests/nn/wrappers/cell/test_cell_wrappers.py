"""Unit tests for cell model wrappers."""

from unittest.mock import MagicMock

import torch
from topomodelx.nn.cell.can import CAN
from topomodelx.nn.cell.ccxn import CCXN
from topomodelx.nn.cell.cwn import CWN
from torch_geometric.utils import get_laplacian

from ogbench.nn.backbones.cell.cccn import CCCN
from ogbench.nn.wrappers import (
    AbstractWrapper,
    CANWrapper,
    CCCNWrapper,
    CCXNWrapper,
    CWNWrapper,
)

from ...._utils.flow_mocker import FlowMocker
from ...._utils.nn_module_auto_test import NNModuleAutoTest


class TestCellWrappers:
    """Test cell model wrappers."""

    def test_CCCNWrapper(self, sg1_clique_lifted):
        """Test CCCNWrapper.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with CellLifting.
        """
        data = sg1_clique_lifted
        out_channels = data.x_1.shape[1]
        num_cell_dimensions = 2

        wrapper = CCCNWrapper(
            CCCN(data.x_1.shape[1]),
            out_channels=out_channels,
            num_cell_dimensions=num_cell_dimensions,
        )
        out = wrapper(data)

        for key in ["labels", "batch_0", "x_0", "x_1"]:
            assert key in out

    def test_CCXNWrapper(self, sg1_cell_lifted):
        """Test CCXNWrapper.

        Parameters
        ----------
        sg1_cell_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with CellLifting.
        """
        data = sg1_cell_lifted
        out_channels = data.x_1.shape[1]
        num_cell_dimensions = 2

        wrapper = CCXNWrapper(
            CCXN(data.x_0.shape[1], data.x_1.shape[1], out_channels),
            out_channels=out_channels,
            num_cell_dimensions=num_cell_dimensions,
        )
        out = wrapper(data)

        for key in ["labels", "batch_0", "x_0", "x_1"]:
            assert key in out

    def test_CWNWrapper(self, sg1_cell_lifted):
        """Test CWNWrapper.

        Parameters
        ----------
        sg1_cell_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with CellLifting.
        """
        data = sg1_cell_lifted
        out_channels = data.x_1.shape[1]
        hid_channels = data.x_1.shape[1]
        num_cell_dimensions = 2

        wrapper = CWNWrapper(
            CWN(
                data.x_0.shape[1],
                data.x_1.shape[1],
                data.x_2.shape[1],
                hid_channels,
                2,
            ),
            out_channels=out_channels,
            num_cell_dimensions=num_cell_dimensions,
        )
        out = wrapper(data)

        for key in ["labels", "batch_0", "x_0", "x_1", "x_2"]:
            assert key in out

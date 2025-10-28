"""Wrappers for cell neural networks."""

from ogbench.nn.wrappers.cell.can_wrapper import CANWrapper
from ogbench.nn.wrappers.cell.cccn_wrapper import CCCNWrapper
from ogbench.nn.wrappers.cell.ccxn_wrapper import CCXNWrapper
from ogbench.nn.wrappers.cell.cwn_wrapper import CWNWrapper

# Create dictionary of all wrapper classes
WRAPPER_CLASSES: dict[str, type] = {
    'CANWrapper': CANWrapper,
    'CCCNWrapper': CCCNWrapper,
    'CCXNWrapper': CCXNWrapper,
    'CWNWrapper': CWNWrapper,
}

# Generate __all__
__all__ = [
    'WRAPPER_CLASSES',
    'CANWrapper',
    'CCCNWrapper',
    'CCXNWrapper',
    'CWNWrapper',
]

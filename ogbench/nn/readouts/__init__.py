"""This module contains the readout classes that are used by the library."""

from ogbench.nn.readouts.identical import NoReadOut
from ogbench.nn.readouts.omics_readout import OmicsReadOut
from ogbench.nn.readouts.propagate_signal_down import PropagateSignalDown

# Create dictionary of all readout classes
READOUT_CLASSES: dict[str, type] = {
    'NoReadOut': NoReadOut,
    'OmicsReadOut': OmicsReadOut,
    'PropagateSignalDown': PropagateSignalDown,
}

# Generate __all__
__all__ = [
    'READOUT_CLASSES',
    'NoReadOut',
    'OmicsReadOut',
    'PropagateSignalDown',
]

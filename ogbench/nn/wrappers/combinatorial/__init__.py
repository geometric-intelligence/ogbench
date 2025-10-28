"""Wrappers for combinatorial neural networks."""

from ogbench.nn.wrappers.combinatorial.tune_wrapper import TuneWrapper

# Create dictionary of all wrapper classes
WRAPPER_CLASSES: dict[str, type] = {
    'TuneWrapper': TuneWrapper,
}

# Generate __all__
__all__ = [
    'WRAPPER_CLASSES',
    'TuneWrapper',
]

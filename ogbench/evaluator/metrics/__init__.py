"""Init file for custom metrics in evaluator module."""

from ogbench.evaluator.metrics.denormalized_rmse import DenormalizedRMSE
from ogbench.evaluator.metrics.example import ExampleRegressionMetric

# Create dictionary of all custom metrics
CUSTOM_METRICS: dict[str, type] = {
    'DenormalizedRMSE': DenormalizedRMSE,
    'ExampleRegressionMetric': ExampleRegressionMetric,
}

CUSTOM_METRICS_list: list[str] = list(CUSTOM_METRICS.keys())

# Generate __all__
__all__ = [
    'CUSTOM_METRICS',
    'CUSTOM_METRICS_list',
    'DenormalizedRMSE',
    'ExampleRegressionMetric',
]

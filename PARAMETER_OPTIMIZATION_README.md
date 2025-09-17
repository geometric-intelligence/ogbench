# Parameter Optimization Scripts

This directory contains scripts for finding optimal hyperparameter combinations to achieve target parameter counts for your models.

## Overview

The scripts provide multiple optimization strategies:

1. **Bisection Search**: Efficiently finds hyperparameters using binary search
2. **Grid Search**: Explores combinations of secondary parameters
3. **Multi-target Optimization**: Optimizes for multiple parameter count targets
4. **Parameter Sweep Analysis**: Analyzes parameter space across different configurations

## Files

- `find_target_params.py`: Basic parameter optimization with bisection search
- `advanced_param_optimizer.py`: Advanced optimization with multiple strategies and visualization
- `example_usage.py`: Examples demonstrating how to use both scripts
- `PARAMETER_OPTIMIZATION_README.md`: This documentation

## Quick Start

### Basic Usage

```bash
# Find configurations for 100k parameters
python find_target_params.py --target-params 100000

# Optimize specific models only
python find_target_params.py --target-params 100000 --models gat gcn gin

# Set custom tolerance (default: 0.1 = 10%)
python find_target_params.py --target-params 100000 --tolerance 0.05
```

### Advanced Usage

```bash
# Multi-target optimization with visualization
python advanced_param_optimizer.py --targets 50000 100000 200000 --visualize

# Grid search strategy
python advanced_param_optimizer.py --targets 100000 --strategy grid

# Custom output directory
python advanced_param_optimizer.py --targets 100000 --output-dir ./my_results
```

### Run Examples

```bash
python example_usage.py
```

## Supported Models

The scripts support the following models:

- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network
- **GATv2**: Improved Graph Attention Network
- **GIN**: Graph Isomorphism Network
- **GraphSAGE**: Graph Sample and Aggregate
- **ChebNet**: Chebyshev Spectral Graph Convolution
- **MLP**: Multi-Layer Perceptron
- **SAGN**: Structure-Aware Graph Network
- **GATv4**: Latest Graph Attention Network variant

## Optimization Strategies

### Bisection Search

Uses binary search on the primary parameter (usually `num_layers` or `hidden_channels`) to efficiently find configurations close to the target parameter count.

**Advantages:**

- Fast convergence (typically 10-20 iterations)
- Guaranteed to find a solution if one exists
- Works well for single parameter optimization

**Best for:** Quick optimization when you have a single target parameter count.

### Grid Search

Explores combinations of secondary parameters (like `heads`, `out_channels`, `dropout`) while using bisection for the primary parameter.

**Advantages:**

- Finds better configurations by exploring parameter combinations
- Can discover non-obvious optimal settings
- More thorough exploration of parameter space

**Best for:** When you want the best possible configuration and can afford longer computation time.

## Output Files

### Basic Script Output

- `optimization_summary.csv`: Summary with model, target, actual parameters, and error
- `optimal_configurations.csv`: Detailed hyperparameter configurations

### Advanced Script Output

- `optimization_summary.csv`: Summary of all optimizations
- `detailed_results.csv`: Complete results with configurations
- `results.json`: Machine-readable results
- `parameter_optimization_results.png`: Visualization of target vs actual parameters
- `convergence_analysis.png`: Analysis of optimization convergence

## Configuration

### Model-Specific Parameters

Each model has different parameters that affect parameter count:

**GCN/GAT/GIN/GraphSAGE:**

- Primary: `model.backbone.num_layers` (1-8)
- Secondary: `model.feature_encoder.out_channels` (32, 64, 128, 256)

**MLP:**

- Primary: `model.backbone.hidden_channels` (32-2048)
- Creates decreasing sequence: [val, val/2, val/4]

**SAGN:**

- Primary: `model.backbone.hidden_channels` (64-512)
- Secondary: `model.backbone.num_layers`, `model.backbone.dropout`

**ChebNet:**

- Primary: `model.backbone.num_layers` (1-6)
- Secondary: `model.backbone.K` (2, 3, 4)

### Base Configuration

All models use these base settings:

```yaml
optimizer.parameters.lr: 0.001
optimizer.parameters.weight_decay: 0.0004
model.readout.pooling_type: mean
model.backbone.act: relu
```

## Examples

### Example 1: Find 100k Parameter Configurations

```python
from find_target_params import ParameterOptimizer

optimizer = ParameterOptimizer()
results = optimizer.optimize_all_models(target_params=100000, tolerance=0.1)
print(results[['model', 'actual_params', 'error']])
```

### Example 2: Advanced Multi-Target Optimization

```python
from advanced_param_optimizer import AdvancedParameterOptimizer

optimizer = AdvancedParameterOptimizer()
results = optimizer.multi_target_optimization(
    target_params_list=[50000, 100000, 200000],
    tolerance=0.1
)

# Access results
for model_name, model_results in results.items():
    for result in model_results:
        print(f"{model_name}: {result.actual_params} params (error: {result.error:.3f})")
```

### Example 3: Parameter Sweep Analysis

```python
from advanced_param_optimizer import AdvancedParameterOptimizer

optimizer = AdvancedParameterOptimizer()
sweep_df = optimizer.parameter_sweep_analysis(
    model_name="gat",
    param_ranges={
        "model.backbone.num_layers": [1, 2, 3, 4, 5],
        "model.backbone.heads": [2, 4, 8],
        "model.feature_encoder.out_channels": [32, 64, 128, 256]
    }
)
print(sweep_df.groupby('param_count').size())
```

## Tips for Best Results

1. **Start with bisection search** for quick results, then use grid search for refinement
2. **Use appropriate tolerance**: 0.1 (10%) for quick results, 0.05 (5%) for precision
3. **Consider multiple targets** to understand parameter scaling across your models
4. **Use visualization** to understand optimization patterns and convergence
5. **Check the parameter sweep** to understand the parameter space before optimization

## Troubleshooting

### Common Issues

**"Failed to create model"**: Check that your Hydra configuration is correct and all required parameters are set.

**"Search converged with high error"**: Try increasing the parameter range or using grid search with secondary parameters.

**"No valid configurations found"**: The target parameter count might be outside the achievable range for the model.

### Debug Mode

Enable debug logging to see detailed optimization progress:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Performance Notes

- **Bisection search**: ~10-20 iterations per model
- **Grid search**: Depends on number of secondary parameter combinations
- **Multi-target**: Scales linearly with number of targets
- **Parameter sweep**: Can be slow for large parameter spaces

The scripts are designed to run without GPU (CPU-only) for parameter counting, making them suitable for any machine with sufficient RAM.

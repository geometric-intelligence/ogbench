#!/usr/bin/env python3
"""Example usage of the parameter optimization scripts.

This script demonstrates how to use both the basic and advanced parameter optimizers.
"""

import os
import sys

from advanced_param_optimizer import AdvancedParameterOptimizer
from find_target_params import ParameterOptimizer


def example_basic_optimization():
    """Example of basic parameter optimization."""
    print("=" * 60)
    print("BASIC PARAMETER OPTIMIZATION EXAMPLE")
    print("=" * 60)

    # Initialize optimizer
    optimizer = ParameterOptimizer()

    # Find optimal config for 100k parameters
    target_params = 100000
    tolerance = 0.1

    print(f"Finding configurations for {target_params:,} parameters (tolerance: {tolerance})")

    # Optimize all models
    results_df = optimizer.optimize_all_models(target_params, tolerance)

    # Display results
    print("\nResults:")
    print(results_df[["model", "target_params", "actual_params", "error"]].to_string(index=False))

    # Save results
    optimizer.save_results(results_df, "./example_results")

    return results_df


def example_advanced_optimization():
    """Example of advanced parameter optimization."""
    print("\n" + "=" * 60)
    print("ADVANCED PARAMETER OPTIMIZATION EXAMPLE")
    print("=" * 60)

    # Initialize advanced optimizer
    optimizer = AdvancedParameterOptimizer()

    # Multiple target parameter counts
    targets = [50000, 100000, 200000]
    tolerance = 0.1

    print(f"Finding configurations for targets: {targets}")

    # Run multi-target optimization
    results = optimizer.multi_target_optimization(targets, tolerance)

    # Display results
    print("\nResults:")
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        for result in model_results:
            print(
                f"  Target: {result.target_params:,} → Actual: {result.actual_params:,} "
                f"(error: {result.error:.3f})"
            )

    # Save results and create visualizations
    optimizer.save_results(results, "./advanced_example_results")
    optimizer.create_visualizations(results, "./advanced_example_results")

    return results


def example_single_model_optimization():
    """Example of optimizing a single model with detailed analysis."""
    print("\n" + "=" * 60)
    print("SINGLE MODEL OPTIMIZATION EXAMPLE")
    print("=" * 60)

    optimizer = AdvancedParameterOptimizer()

    # Focus on GAT model
    model_name = "gat"
    target_params = 150000

    print(f"Detailed optimization for {model_name} targeting {target_params:,} parameters")

    # Use grid search for better optimization
    results = optimizer.grid_search(model_name, target_params, tolerance=0.05)

    if results:
        result = results[0]
        print("\nBest configuration:")
        print(f"  Actual parameters: {result.actual_params:,}")
        print(f"  Error: {result.error:.3f}")
        print("  Configuration:")
        for key, value in result.config.items():
            if key not in ["actual_params", "error"]:
                print(f"    {key}: {value}")

    return results


def example_parameter_sweep():
    """Example of parameter sweep analysis."""
    print("\n" + "=" * 60)
    print("PARAMETER SWEEP ANALYSIS EXAMPLE")
    print("=" * 60)

    optimizer = AdvancedParameterOptimizer()

    # Analyze GCN parameter space
    model_name = "gcn"
    param_ranges = {
        "model.backbone.num_layers": [1, 2, 3, 4, 5, 6],
        "model.feature_encoder.out_channels": [32, 64, 128, 256],
    }

    print(f"Parameter sweep for {model_name}")
    print("Parameter ranges:")
    for param, values in param_ranges.items():
        print(f"  {param}: {values}")

    # Run sweep
    sweep_df = optimizer.parameter_sweep_analysis(model_name, param_ranges)

    print(f"\nSweep results ({len(sweep_df)} configurations):")
    print(
        sweep_df[
            ["param_count", "model.backbone.num_layers", "model.feature_encoder.out_channels"]
        ].head(10)
    )

    # Save sweep results
    os.makedirs("./sweep_results", exist_ok=True)
    sweep_df.to_csv("./sweep_results/gcn_parameter_sweep.csv", index=False)

    return sweep_df


def main():
    """Run all examples."""
    print("Parameter Optimization Examples")
    print("=" * 80)

    try:
        # Example 1: Basic optimization
        basic_results = example_basic_optimization()

        # Example 2: Advanced optimization
        advanced_results = example_advanced_optimization()

        # Example 3: Single model optimization
        single_results = example_single_model_optimization()

        # Example 4: Parameter sweep
        sweep_results = example_parameter_sweep()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Check the following directories for results:")
        print("  - ./example_results/")
        print("  - ./advanced_example_results/")
        print("  - ./sweep_results/")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

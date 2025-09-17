#!/usr/bin/env python3
"""Demo script showing the key features of parameter optimization.

This script demonstrates the main capabilities without requiring full dataset setup.
"""

import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def demo_basic_optimization():
    """Demonstrate basic parameter optimization."""
    print("=" * 60)
    print("DEMO: Basic Parameter Optimization")
    print("=" * 60)

    try:
        from find_target_params import ParameterOptimizer

        # Initialize optimizer
        optimizer = ParameterOptimizer()

        print("✅ Basic optimizer initialized successfully")
        print(f"📊 Supported models: {list(optimizer.models.keys())}")
        print(f"🎯 Base configuration parameters: {len(optimizer.base_overrides)}")

        # Show model configurations
        print("\n📋 Model configurations:")
        for model_name, config in optimizer.models.items():
            print(f"  {model_name}: {config['param_key']} in range {config['range']}")

        print("\n✨ Basic optimization features:")
        print("  • Bisection search for efficient parameter finding")
        print("  • Support for 9 different model architectures")
        print("  • Automatic error handling and convergence detection")
        print("  • CSV output for results analysis")

        return True

    except Exception as e:
        print(f"❌ Error in basic optimization demo: {e}")
        return False


def demo_advanced_optimization():
    """Demonstrate advanced parameter optimization."""
    print("\n" + "=" * 60)
    print("DEMO: Advanced Parameter Optimization")
    print("=" * 60)

    try:
        from advanced_param_optimizer import (
            AdvancedParameterOptimizer,
            OptimizationResult,
        )

        # Initialize advanced optimizer
        optimizer = AdvancedParameterOptimizer()

        print("✅ Advanced optimizer initialized successfully")
        print(f"📊 Enhanced model configurations: {len(optimizer.models)}")

        # Show advanced features
        print("\n🚀 Advanced optimization features:")
        print("  • Multi-target optimization (multiple parameter counts)")
        print("  • Grid search with secondary parameters")
        print("  • Parameter sweep analysis")
        print("  • Visualization with matplotlib/seaborn")
        print("  • JSON output for programmatic access")
        print("  • Convergence analysis and iteration tracking")

        # Show model-specific secondary parameters
        print("\n🔧 Model-specific secondary parameters:")
        for model_name, config in optimizer.models.items():
            secondary = config.get("secondary_params", {})
            if secondary:
                print(f"  {model_name}: {list(secondary.keys())}")
            else:
                print(f"  {model_name}: Primary parameter only")

        # Demo OptimizationResult
        result = OptimizationResult(
            model_name="demo_model",
            target_params=100000,
            actual_params=95000,
            error=0.05,
            config={"param1": 5, "param2": "relu"},
            iterations=12,
        )

        print("\n📈 Example optimization result:")
        print(f"  Model: {result.model_name}")
        print(f"  Target: {result.target_params:,} → Actual: {result.actual_params:,}")
        print(f"  Error: {result.error:.3f} ({result.error*100:.1f}%)")
        print(f"  Iterations: {result.iterations}")

        return True

    except Exception as e:
        print(f"❌ Error in advanced optimization demo: {e}")
        return False


def demo_usage_examples():
    """Show usage examples."""
    print("\n" + "=" * 60)
    print("DEMO: Usage Examples")
    print("=" * 60)

    print("🔧 Command Line Usage:")
    print("\n1. Basic optimization:")
    print("   python find_target_params.py --target-params 100000")
    print("   python find_target_params.py --target-params 100000 --models gat gcn")
    print("   python find_target_params.py --target-params 100000 --tolerance 0.05")

    print("\n2. Advanced optimization:")
    print("   python advanced_param_optimizer.py --targets 50000 100000 200000")
    print("   python advanced_param_optimizer.py --targets 100000 --strategy grid")
    print("   python advanced_param_optimizer.py --targets 100000 --visualize")

    print("\n📝 Python API Usage:")
    print(
        """
# Basic optimization
from find_target_params import ParameterOptimizer
optimizer = ParameterOptimizer()
results = optimizer.optimize_all_models(target_params=100000, tolerance=0.1)

# Advanced optimization
from advanced_param_optimizer import AdvancedParameterOptimizer
optimizer = AdvancedParameterOptimizer()
results = optimizer.multi_target_optimization([50000, 100000, 200000])
"""
    )

    print("📊 Output Files:")
    print("  • optimization_summary.csv - Summary of results")
    print("  • detailed_results.csv - Complete configurations")
    print("  • results.json - Machine-readable format")
    print("  • *.png - Visualization plots (advanced only)")


def demo_optimization_strategies():
    """Explain optimization strategies."""
    print("\n" + "=" * 60)
    print("DEMO: Optimization Strategies")
    print("=" * 60)

    print("🎯 Bisection Search:")
    print("  • Uses binary search on primary parameter")
    print("  • Fast convergence (10-20 iterations)")
    print("  • Best for: Single target, quick results")
    print("  • Example: Find num_layers for 100k parameters")

    print("\n🔍 Grid Search:")
    print("  • Explores combinations of secondary parameters")
    print("  • Uses bisection for primary parameter")
    print("  • Best for: Finding optimal configurations")
    print("  • Example: Try different heads + out_channels combinations")

    print("\n📈 Multi-target Optimization:")
    print("  • Optimizes for multiple parameter counts")
    print("  • Shows parameter scaling across models")
    print("  • Best for: Understanding model capacity")
    print("  • Example: Find configs for 50k, 100k, 200k parameters")

    print("\n📊 Parameter Sweep Analysis:")
    print("  • Analyzes entire parameter space")
    print("  • Shows parameter count distributions")
    print("  • Best for: Understanding parameter scaling")
    print("  • Example: How does num_layers affect parameter count?")


def demo_model_specifics():
    """Show model-specific information."""
    print("\n" + "=" * 60)
    print("DEMO: Model-Specific Configurations")
    print("=" * 60)

    models_info = {
        "GCN": "Graph Convolutional Network - Primary: num_layers",
        "GAT": "Graph Attention Network - Primary: num_layers, Secondary: heads",
        "GATv2": "Improved GAT - Primary: num_layers, Secondary: heads",
        "GIN": "Graph Isomorphism Network - Primary: num_layers",
        "GraphSAGE": "Sample and Aggregate - Primary: num_layers",
        "ChebNet": "Chebyshev Spectral - Primary: num_layers, Secondary: K",
        "MLP": "Multi-Layer Perceptron - Primary: hidden_channels",
        "SAGN": "Structure-Aware Graph - Primary: hidden_channels, Secondary: layers/dropout",
        "GATv4": "Latest GAT variant - Primary: hidden_channels, Secondary: heads",
    }

    for model, description in models_info.items():
        print(f"  {model}: {description}")

    print("\n🎛️ Parameter Types:")
    print("  • num_layers: Number of GNN layers (1-8)")
    print("  • hidden_channels: Hidden dimension size (32-2048)")
    print("  • heads: Number of attention heads (2-8)")
    print("  • out_channels: Feature encoder output (32-256)")
    print("  • dropout: Dropout rate (0.1-0.5)")
    print("  • K: Chebyshev polynomial order (2-4)")


def main():
    """Run all demos."""
    print("🚀 Parameter Optimization Demo")
    print("=" * 80)

    success_count = 0
    total_demos = 5

    # Run demos
    if demo_basic_optimization():
        success_count += 1

    if demo_advanced_optimization():
        success_count += 1

    demo_usage_examples()
    success_count += 1

    demo_optimization_strategies()
    success_count += 1

    demo_model_specifics()
    success_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    print(f"✅ Successful demos: {success_count}/{total_demos}")

    if success_count == total_demos:
        print("🎉 All demos completed successfully!")
        print("\n📚 Next steps:")
        print("  1. Run: python example_usage.py")
        print("  2. Try: python find_target_params.py --target-params 100000")
        print("  3. Explore: python advanced_param_optimizer.py --help")
        print("  4. Read: PARAMETER_OPTIMIZATION_README.md")
    else:
        print("⚠️  Some demos had issues. Check the error messages above.")

    print("\n🔗 Files created:")
    print("  • find_target_params.py - Basic optimization")
    print("  • advanced_param_optimizer.py - Advanced optimization")
    print("  • example_usage.py - Usage examples")
    print("  • test_param_optimization.py - Unit tests")
    print("  • PARAMETER_OPTIMIZATION_README.md - Documentation")


if __name__ == "__main__":
    main()

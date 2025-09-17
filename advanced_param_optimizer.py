#!/usr/bin/env python3
"""Advanced hyperparameter optimization for target parameter counts.

This script provides multiple optimization strategies:
1. Bisection search for single target
2. Multi-target optimization
3. Parameter sweep analysis
4. Visualization of parameter vs performance trade-offs
"""

import argparse
import itertools
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


# Define custom resolvers inline to avoid dependency issues
def infer_in_channels(dataset, transforms):
    """Simple resolver for infer_in_channels - returns a default value."""
    return 32  # Default input channels


def get_default_transform(dataset, model):
    """Simple resolver for get_default_transform - returns no_transform."""
    return "no_transform"


def get_monitor_metric(task):
    """Simple resolver for get_monitor_metric - returns accuracy."""
    return "accuracy"


def get_monitor_mode(task):
    """Simple resolver for get_monitor_mode - returns max."""
    return "max"


def infer_num_cell_dimensions(selected_dimensions, in_channels):
    """Simple resolver for infer_num_cell_dimensions - returns default value."""
    return 1  # Default cell dimensions


def get_default_metrics(*args):
    """Simple resolver for get_default_metrics - returns default metrics."""
    return ["accuracy"]


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Stay off GPU for parameter counting
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_grad_enabled(False)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    model_name: str
    target_params: int
    actual_params: int
    error: float
    config: Dict[str, Any]
    iterations: int


class AdvancedParameterOptimizer:
    """Advanced parameter optimizer with multiple strategies."""

    def __init__(self, config_path: str = "./configs", project_config: str = "train.yaml"):
        self.config_path = config_path
        self.project_config = project_config
        self._initialize_hydra()

        # Model configurations with parameter ranges and search strategies
        self.models = {
            "gcn": {
                "primary_param": "model.backbone.num_layers",
                "range": [1, 8],
                "secondary_params": {"model.feature_encoder.out_channels": [32, 64, 128, 256]},
            },
            "gat": {
                "primary_param": "model.backbone.num_layers",
                "range": [1, 6],
                "secondary_params": {
                    "model.backbone.heads": [2, 4, 8],
                    "model.feature_encoder.out_channels": [32, 64, 128, 256],
                },
            },
            "gatv2": {
                "primary_param": "model.backbone.num_layers",
                "range": [1, 6],
                "secondary_params": {
                    "model.backbone.heads": [2, 4, 8],
                    "model.feature_encoder.out_channels": [32, 64, 128, 256],
                },
            },
            "gin": {
                "primary_param": "model.backbone.num_layers",
                "range": [1, 8],
                "secondary_params": {"model.feature_encoder.out_channels": [32, 64, 128, 256]},
            },
            "graph_sage": {
                "primary_param": "model.backbone.num_layers",
                "range": [1, 8],
                "secondary_params": {"model.feature_encoder.out_channels": [32, 64, 128, 256]},
            },
            "chebnet": {
                "primary_param": "model.backbone.num_layers",
                "range": [1, 6],
                "secondary_params": {
                    "model.backbone.K": [2, 3, 4],
                    "model.feature_encoder.out_channels": [32, 64, 128, 256],
                },
            },
            "mlp": {
                "primary_param": "model.backbone.hidden_channels",
                "range": [32, 2048],
                "secondary_params": {},
            },
            "sagn": {
                "primary_param": "model.backbone.hidden_channels",
                "range": [64, 512],
                "secondary_params": {
                    "model.backbone.num_layers": [2, 4, 6],
                    "model.backbone.dropout": [0.1, 0.3, 0.5],
                },
            },
            "gatv4": {
                "primary_param": "model.backbone.hidden_channels",
                "range": [8, 256],
                "secondary_params": {"model.backbone.heads": [[2, 2], [3, 3], [4, 4]]},
            },
        }

        # Base configuration
        self.base_overrides = {
            "dataset": "motrpac",  # Default dataset
            "transforms": "no_transform",  # No transforms for parameter counting
            "optimizer.parameters.lr": 0.001,
            "optimizer.parameters.weight_decay": 0.0004,
            "model.readout.pooling_type": "mean",
            "model.backbone.act": "relu",
        }

        # Model-specific overrides
        self.model_overrides = {
            "gat": {"model.backbone.v2": False},
            "gatv2": {"model.backbone.v2": True},
            "chebnet": {"model.backbone.K": 3},
            "sagn": {"model.backbone.alpha": 0.7},
        }

    def _initialize_hydra(self) -> None:
        """Initialize Hydra configuration."""
        if GlobalHydra().is_initialized():
            GlobalHydra().clear()

        # Register custom resolvers
        OmegaConf.register_new_resolver("infer_in_channels", infer_in_channels, replace=True)
        OmegaConf.register_new_resolver(
            "get_default_transform", get_default_transform, replace=True
        )
        OmegaConf.register_new_resolver("get_monitor_metric", get_monitor_metric, replace=True)
        OmegaConf.register_new_resolver("get_monitor_mode", get_monitor_mode, replace=True)
        OmegaConf.register_new_resolver(
            "infer_num_cell_dimensions", infer_num_cell_dimensions, replace=True
        )
        OmegaConf.register_new_resolver("get_default_metrics", get_default_metrics, replace=True)

        initialize(config_path=self.config_path, job_name="advanced_param_optimization")

    def count_trainable_params(self, model: torch.nn.Module) -> int:
        """Count trainable parameters in a model."""
        return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)

    def create_model(
        self, model_name: str, overrides: Dict[str, Any]
    ) -> Optional[torch.nn.Module]:
        """Create a model instance with given overrides."""
        try:
            override_list = [f"model={model_name}"]
            override_list.extend([f"{k}={v}" for k, v in overrides.items()])

            cfg = compose(
                config_name=self.project_config,
                overrides=override_list,
                return_hydra_config=True,
            )

            model = instantiate(
                cfg.model, evaluator=cfg.evaluator, optimizer=cfg.optimizer, loss=cfg.loss
            ).cpu()

            return model
        except Exception as e:
            logger.debug(f"Failed to create model {model_name}: {e}")
            return None

    def bisection_search(
        self, model_name: str, target_params: int, tolerance: float = 0.1, max_iterations: int = 20
    ) -> Optional[OptimizationResult]:
        """Bisection search for single target parameter count."""
        if model_name not in self.models:
            logger.error(f"Unknown model: {model_name}")
            return None

        model_config = self.models[model_name]
        param_key = model_config["primary_param"]
        min_val, max_val = model_config["range"]

        logger.info(f"Bisection search for {model_name} targeting {target_params} parameters")

        best_config = None
        best_error = float("inf")
        iterations = 0

        for iteration in range(max_iterations):
            mid_val = (min_val + max_val) / 2

            # Create overrides
            overrides = dict(self.base_overrides)
            overrides.update(self.model_overrides.get(model_name, {}))

            # Set primary parameter
            if param_key == "model.backbone.hidden_channels":
                if model_name == "mlp":
                    overrides[param_key] = [int(mid_val), int(mid_val / 2), int(mid_val / 4)]
                elif model_name == "sagn":
                    overrides[param_key] = int(mid_val)
                elif model_name == "gatv4":
                    overrides[param_key] = [int(mid_val), int(mid_val * 2)]
            else:
                overrides[param_key] = int(mid_val)

            # Create model and count parameters
            model = self.create_model(model_name, overrides)
            if model is None:
                continue

            actual_params = self.count_trainable_params(model)
            error = abs(actual_params - target_params) / target_params

            logger.debug(
                f"Iteration {iteration}: {param_key}={mid_val:.1f}, "
                f"params={actual_params}, error={error:.3f}"
            )

            # Track best configuration
            if error < best_error:
                best_error = error
                best_config = dict(overrides)
                best_config["actual_params"] = actual_params
                best_config["error"] = error

            # Check convergence
            if error <= tolerance:
                logger.info(f"Converged: {actual_params} params (error: {error:.3f})")
                return OptimizationResult(
                    model_name=model_name,
                    target_params=target_params,
                    actual_params=actual_params,
                    error=error,
                    config=best_config,
                    iterations=iteration + 1,
                )

            # Update search range
            if actual_params < target_params:
                min_val = mid_val
            else:
                max_val = mid_val

            iterations = iteration + 1

            # Check for numerical convergence
            if abs(max_val - min_val) < 1:
                break

        if best_config:
            logger.info(
                f"Best result: {best_config['actual_params']} params (error: {best_error:.3f})"
            )
            return OptimizationResult(
                model_name=model_name,
                target_params=target_params,
                actual_params=best_config["actual_params"],
                error=best_error,
                config=best_config,
                iterations=iterations,
            )
        else:
            logger.error(f"Failed to find any valid configuration for {model_name}")
            return None

    def grid_search(
        self, model_name: str, target_params: int, tolerance: float = 0.1
    ) -> List[OptimizationResult]:
        """Grid search over secondary parameters for better optimization."""
        if model_name not in self.models:
            return []

        model_config = self.models[model_name]
        secondary_params = model_config.get("secondary_params", {})

        if not secondary_params:
            # No secondary parameters, use bisection only
            result = self.bisection_search(model_name, target_params, tolerance)
            return [result] if result else []

        logger.info(f"Grid search for {model_name} with secondary parameters")

        # Generate all combinations of secondary parameters
        param_names = list(secondary_params.keys())
        param_values = list(secondary_params.values())
        combinations = list(itertools.product(*param_values))

        best_results = []

        for combo in combinations:
            # Create overrides with secondary parameters
            overrides = dict(self.base_overrides)
            overrides.update(self.model_overrides.get(model_name, {}))

            for name, value in zip(param_names, combo):
                overrides[name] = value

            # Use bisection for primary parameter
            result = self.bisection_search(model_name, target_params, tolerance)
            if result:
                # Update config with secondary parameters
                result.config.update({name: value for name, value in zip(param_names, combo)})
                best_results.append(result)

        # Return best result
        if best_results:
            best = min(best_results, key=lambda x: x.error)
            logger.info(
                f"Best grid search result: {best.actual_params} params (error: {best.error:.3f})"
            )
            return [best]

        return []

    def multi_target_optimization(
        self, target_params_list: List[int], tolerance: float = 0.1
    ) -> Dict[str, List[OptimizationResult]]:
        """Optimize all models for multiple target parameter counts."""
        results = {}

        for model_name in self.models.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Multi-target optimization for {model_name}")
            logger.info(f"{'='*60}")

            model_results = []
            for target in target_params_list:
                logger.info(f"Target: {target} parameters")
                result = self.bisection_search(model_name, target, tolerance)
                if result:
                    model_results.append(result)

            results[model_name] = model_results

        return results

    def parameter_sweep_analysis(
        self, model_name: str, param_ranges: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """Analyze parameter count across different hyperparameter combinations."""
        logger.info(f"Parameter sweep analysis for {model_name}")

        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(itertools.product(*param_values))

        results = []
        for combo in combinations:
            overrides = dict(self.base_overrides)
            overrides.update(self.model_overrides.get(model_name, {}))

            for name, value in zip(param_names, combo):
                overrides[name] = value

            model = self.create_model(model_name, overrides)
            if model:
                param_count = self.count_trainable_params(model)
                result = {
                    "model": model_name,
                    "param_count": param_count,
                    **{name: value for name, value in zip(param_names, combo)},
                }
                results.append(result)

        return pd.DataFrame(results)

    def create_visualizations(
        self, results: Dict[str, List[OptimizationResult]], output_dir: str
    ) -> None:
        """Create visualization plots for optimization results."""
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for plotting
        plot_data = []
        for model_name, model_results in results.items():
            for result in model_results:
                plot_data.append(
                    {
                        "model": model_name,
                        "target_params": result.target_params,
                        "actual_params": result.actual_params,
                        "error": result.error,
                        "iterations": result.iterations,
                    }
                )

        df = pd.DataFrame(plot_data)

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Parameter count comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Target vs actual parameters
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            ax1.scatter(
                model_data["target_params"],
                model_data["actual_params"],
                label=model,
                alpha=0.7,
                s=60,
            )

        ax1.plot(
            [df["target_params"].min(), df["target_params"].max()],
            [df["target_params"].min(), df["target_params"].max()],
            "k--",
            alpha=0.5,
            label="Perfect match",
        )
        ax1.set_xlabel("Target Parameters")
        ax1.set_ylabel("Actual Parameters")
        ax1.set_title("Target vs Actual Parameter Counts")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Error distribution
        sns.boxplot(data=df, x="model", y="error", ax=ax2)
        ax2.set_title("Optimization Error by Model")
        ax2.set_ylabel("Relative Error")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "parameter_optimization_results.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Iterations vs error
        fig, ax = plt.subplots(figsize=(10, 6))
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            ax.scatter(model_data["iterations"], model_data["error"], label=model, alpha=0.7, s=60)

        ax.set_xlabel("Iterations to Convergence")
        ax.set_ylabel("Final Error")
        ax.set_title("Convergence Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "convergence_analysis.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}")

    def save_results(self, results: Dict[str, List[OptimizationResult]], output_dir: str) -> None:
        """Save optimization results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Convert to DataFrame
        all_results = []
        for model_name, model_results in results.items():
            for result in model_results:
                all_results.append(
                    {
                        "model": result.model_name,
                        "target_params": result.target_params,
                        "actual_params": result.actual_params,
                        "error": result.error,
                        "iterations": result.iterations,
                        "config": json.dumps(result.config, indent=2),
                    }
                )

        df = pd.DataFrame(all_results)

        # Save summary
        summary_path = os.path.join(output_dir, "optimization_summary.csv")
        df[["model", "target_params", "actual_params", "error", "iterations"]].to_csv(
            summary_path, index=False
        )

        # Save detailed results
        detailed_path = os.path.join(output_dir, "detailed_results.csv")
        df.to_csv(detailed_path, index=False)

        # Save JSON for programmatic access
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump({k: [r.__dict__ for r in v] for k, v in results.items()}, f, indent=2)

        logger.info("Results saved to:")
        logger.info(f"  Summary: {summary_path}")
        logger.info(f"  Detailed: {detailed_path}")
        logger.info(f"  JSON: {json_path}")


def main():
    """Main function for advanced parameter optimization."""
    parser = argparse.ArgumentParser(description="Advanced parameter optimization")
    parser.add_argument(
        "--targets",
        nargs="+",
        type=int,
        default=[50000, 100000, 200000],
        help="Target parameter counts",
    )
    parser.add_argument("--tolerance", type=float, default=0.1, help="Acceptable relative error")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./advanced_optimization_results",
        help="Output directory",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to optimize")
    parser.add_argument(
        "--strategy",
        choices=["bisection", "grid"],
        default="bisection",
        help="Optimization strategy",
    )
    parser.add_argument(
        "--config-path", type=str, default="./configs", help="Path to Hydra configs"
    )
    parser.add_argument("--visualize", action="store_true", help="Create visualization plots")

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = AdvancedParameterOptimizer(config_path=args.config_path)

    # Filter models if specified
    if args.models:
        optimizer.models = {k: v for k, v in optimizer.models.items() if k in args.models}

    logger.info("Advanced parameter optimization")
    logger.info(f"Targets: {args.targets}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Models: {list(optimizer.models.keys())}")

    # Run optimization
    if args.strategy == "bisection":
        results = optimizer.multi_target_optimization(args.targets, args.tolerance)
    else:
        # Grid search for each target
        results = {}
        for model_name in optimizer.models.keys():
            model_results = []
            for target in args.targets:
                model_results.extend(optimizer.grid_search(model_name, target, args.tolerance))
            results[model_name] = model_results

    # Display results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        for result in model_results:
            print(
                f"  Target: {result.target_params:,} → Actual: {result.actual_params:,} "
                f"(error: {result.error:.3f}, iterations: {result.iterations})"
            )

    # Save results
    optimizer.save_results(results, args.output_dir)

    # Create visualizations
    if args.visualize:
        optimizer.create_visualizations(results, args.output_dir)

    logger.info("Advanced parameter optimization completed!")


if __name__ == "__main__":
    main()

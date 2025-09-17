#!/usr/bin/env python3
"""Hyperparameter optimization script to find configurations that achieve target parameter counts.

This script uses bisection search to efficiently find hyperparameter combinations that result in
models with approximately the desired number of parameters.
"""

import argparse
import itertools
import logging
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


# Define minimal resolvers to avoid dependency issues
def infer_in_channels(dataset, transforms):
    """Minimal resolver for infer_in_channels - returns default value as list for AllCellFeatureEncoder."""
    return [32]  # Default input channels as list


def get_default_transform(dataset, model):
    """Minimal resolver for get_default_transform - returns no_transform."""
    return "no_transform"


def get_monitor_metric(task):
    """Minimal resolver for get_monitor_metric - returns accuracy."""
    return "accuracy"


def get_monitor_mode(task):
    """Minimal resolver for get_monitor_mode - returns max."""
    return "max"


def infer_num_cell_dimensions(selected_dimensions, in_channels):
    """Minimal resolver for infer_num_cell_dimensions - returns default value."""
    return 1  # Default cell dimensions


def get_default_metrics(task, metrics=None):
    """Minimal resolver for get_default_metrics - returns default metrics."""
    if task == "classification":
        return ["accuracy", "precision", "recall", "auroc"]
    elif task == "regression":
        return ["mae", "mse"]
    else:
        return ["accuracy"]


def get_dataset_task(dataset):
    """Minimal resolver for dataset task - returns regression for motrpac."""
    return "regression"


def get_dataset_task_level(dataset):
    """Minimal resolver for dataset task level - returns graph."""
    return "graph"


def get_dataset_num_classes(dataset):
    """Minimal resolver for dataset num classes - returns 1 for regression."""
    return 1


def calculate_num_nodes(num_samples, train_val_test_split, node_sample_ratio, full_num_nodes):
    """Minimal resolver for calculate_num_nodes - returns default."""
    return 1000  # Default number of nodes


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Stay off GPU for parameter counting
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_grad_enabled(False)


class ParameterOptimizer:
    """Finds hyperparameter combinations to achieve target parameter counts."""

    def __init__(self, config_path: str = "./configs", project_config: str = "train.yaml"):
        self.config_path = config_path
        self.project_config = project_config
        self._initialize_hydra()

        # Model configurations
        self.models = {
            "gcn": {"param_key": "model.backbone.num_layers", "range": [1, 8]},
            "gat": {"param_key": "model.backbone.num_layers", "range": [1, 6]},
            "gatv2": {"param_key": "model.backbone.num_layers", "range": [1, 6]},
            "gin": {"param_key": "model.backbone.num_layers", "range": [1, 8]},
            "graph_sage": {"param_key": "model.backbone.num_layers", "range": [1, 20]},
            "chebnet": {"param_key": "model.backbone.num_layers", "range": [1, 20]},
            "mlp": {"param_key": "model.backbone.hidden_channels", "range": [4, 2048]},
            "sagn": {"param_key": "model.backbone.hidden_channels", "range": [32, 512]},
            "gatv4": {"param_key": "model.backbone.hidden_channels", "range": [8, 256]},
        }

        # Base configuration overrides
        self.base_overrides = {
            "dataset": "motrpac",  # Default dataset
            "transforms": "no_transform",  # No transforms for parameter counting
            "optimizer.parameters.lr": 0.001,
            "optimizer.parameters.weight_decay": 0.0004,
            "model.readout.pooling_type": "mean",
        }

        # Model-specific overrides
        self.model_overrides = {
            "gat": {"model.backbone.heads": 4, "model.backbone.v2": False},
            "gatv2": {"model.backbone.heads": 4, "model.backbone.v2": True},
            "gatv4": {"model.backbone.heads": [3, 3]},
            "chebnet": {"model.backbone.K": 3},
            "mlp": {"model.backbone.hidden_channels": [128, 64, 32]},
            "sagn": {"model.backbone.dropout": 0.3, "model.backbone.alpha": 0.7},
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
        OmegaConf.register_new_resolver("dataset.parameters.task", get_dataset_task, replace=True)
        OmegaConf.register_new_resolver(
            "dataset.parameters.task_level", get_dataset_task_level, replace=True
        )
        OmegaConf.register_new_resolver(
            "dataset.parameters.num_classes", get_dataset_num_classes, replace=True
        )
        OmegaConf.register_new_resolver("calculate_num_nodes", calculate_num_nodes, replace=True)

        initialize(config_path=self.config_path, job_name="param_optimization")

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
            logger.warning(f"Failed to create model {model_name} with overrides {overrides}: {e}")
            return None

    def bisection_search(
        self, model_name: str, target_params: int, tolerance: float = 0.1, max_iterations: int = 20
    ) -> Optional[Dict[str, Any]]:
        """Use bisection search to find hyperparameters that achieve target parameter count.

        Args:
            model_name: Name of the model to optimize
            target_params: Target number of parameters
            tolerance: Acceptable relative error (e.g., 0.1 = 10%)
            max_iterations: Maximum number of bisection iterations

        Returns:
            Dictionary with optimal hyperparameters or None if not found
        """
        if model_name not in self.models:
            logger.error(f"Unknown model: {model_name}")
            return None

        model_config = self.models[model_name]
        param_key = model_config["param_key"]
        min_val, max_val = model_config["range"]

        logger.info(f"Optimizing {model_name} for target {target_params} parameters")
        logger.info(f"Searching parameter: {param_key} in range [{min_val}, {max_val}]")

        best_config = None
        best_error = float("inf")

        for iteration in range(max_iterations):
            mid_val = (min_val + max_val) / 2

            # Create overrides
            overrides = dict(self.base_overrides)
            overrides.update(self.model_overrides.get(model_name, {}))

            # Handle different parameter types
            if param_key == "model.backbone.hidden_channels":
                if model_name == "mlp":
                    # For MLP, create a decreasing sequence
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
                logger.warning(f"Failed to create model at iteration {iteration}")
                continue

            actual_params = self.count_trainable_params(model)
            error = abs(actual_params - target_params) / target_params

            logger.info(
                f"Iteration {iteration}: {param_key}={mid_val:.1f}, "
                f"params={actual_params}, error={error:.3f}"
            )

            # Check if this is the best configuration so far
            if error < best_error:
                best_error = error
                best_config = dict(overrides)
                best_config["actual_params"] = actual_params
                best_config["error"] = error

            # Check if we've reached the target within tolerance
            if error <= tolerance:
                logger.info(
                    f"Found optimal configuration with {actual_params} parameters "
                    f"(error: {error:.3f})"
                )
                return best_config

            # Update search range
            if actual_params < target_params:
                min_val = mid_val
            else:
                max_val = mid_val

            # Check for convergence
            if abs(max_val - min_val) < 1:
                logger.info(f"Search converged. Best error: {best_error:.3f}")
                break

        if best_config:
            logger.info(
                f"Best configuration found with {best_config['actual_params']} parameters "
                f"(error: {best_error:.3f})"
            )
            return best_config
        else:
            logger.error(f"Failed to find any valid configuration for {model_name}")
            return None

    def optimize_all_models(self, target_params: int, tolerance: float = 0.1) -> pd.DataFrame:
        """Find optimal hyperparameters for all models to achieve target parameter count.

        Args:
            target_params: Target number of parameters
            tolerance: Acceptable relative error

        Returns:
            DataFrame with results for all models
        """
        results = []

        for model_name in self.models.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Optimizing {model_name}")
            logger.info(f"{'='*50}")

            config = self.bisection_search(model_name, target_params, tolerance)

            if config:
                result = {
                    "model": model_name,
                    "target_params": target_params,
                    "actual_params": config["actual_params"],
                    "error": config["error"],
                    "optimal_config": config,
                }
                results.append(result)
            else:
                logger.error(f"Failed to optimize {model_name}")

        return pd.DataFrame(results)

    def save_results(
        self, df: pd.DataFrame, output_dir: str = "./param_optimization_results"
    ) -> None:
        """Save optimization results to files."""
        os.makedirs(output_dir, exist_ok=True)

        if df.empty:
            logger.warning("No results to save - DataFrame is empty")
            return

        # Save summary
        summary_df = df[["model", "target_params", "actual_params", "error"]].copy()
        summary_path = os.path.join(output_dir, "optimization_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        # Save detailed configurations
        configs = []
        for _, row in df.iterrows():
            config = row["optimal_config"].copy()
            config["model"] = row["model"]
            configs.append(config)

        configs_df = pd.DataFrame(configs)
        configs_path = os.path.join(output_dir, "optimal_configurations.csv")
        configs_df.to_csv(configs_path, index=False)

        logger.info("Results saved to:")
        logger.info(f"  Summary: {summary_path}")
        logger.info(f"  Configurations: {configs_path}")


def main():
    """Main function to run parameter optimization."""
    parser = argparse.ArgumentParser(
        description="Find hyperparameters for target parameter counts"
    )
    parser.add_argument(
        "--target-params",
        type=int,
        default=100000,
        help="Target number of parameters (default: 100000)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.1, help="Acceptable relative error (default: 0.1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./param_optimization_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="Specific models to optimize (default: all)"
    )
    parser.add_argument(
        "--config-path", type=str, default="./configs", help="Path to Hydra configs"
    )

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = ParameterOptimizer(config_path=args.config_path)

    # Filter models if specified
    if args.models:
        optimizer.models = {k: v for k, v in optimizer.models.items() if k in args.models}

    # Run optimization
    logger.info(f"Starting parameter optimization for target: {args.target_params}")
    logger.info(f"Tolerance: {args.tolerance}")
    logger.info(f"Models: {list(optimizer.models.keys())}")

    results_df = optimizer.optimize_all_models(args.target_params, args.tolerance)

    # Display results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    if not results_df.empty:
        print(
            results_df[["model", "target_params", "actual_params", "error"]].to_string(index=False)
        )
    else:
        print("No successful optimizations found. Check the error messages above.")

    # Save results
    optimizer.save_results(results_df, args.output_dir)

    logger.info("Parameter optimization completed!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hyperparameter search driver script.

This script performs grid search over hyperparameters by calling run.py via subprocess. Supports
both normal execution and dry-run mode for parameter counting.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Import resolvers
from ogbench.utils.config_resolvers import (
    calculate_num_nodes,
    get_default_metrics,
    get_default_trainer,
    get_default_transform,
    get_flattened_channels,
    get_gatv4_output_dim,
    get_monitor_metric,
    get_monitor_mode,
    get_non_relational_out_channels,
    get_required_lifting,
    infer_in_channels,
    infer_num_cell_dimensions,
)


def register_resolvers():
    """Register all custom OmegaConf resolvers."""
    OmegaConf.register_new_resolver("calculate_num_nodes", calculate_num_nodes, replace=True)
    OmegaConf.register_new_resolver("get_default_metrics", get_default_metrics, replace=True)
    OmegaConf.register_new_resolver("get_default_trainer", get_default_trainer, replace=True)
    OmegaConf.register_new_resolver("get_default_transform", get_default_transform, replace=True)
    OmegaConf.register_new_resolver("get_flattened_channels", get_flattened_channels, replace=True)
    OmegaConf.register_new_resolver("get_required_lifting", get_required_lifting, replace=True)
    OmegaConf.register_new_resolver("get_monitor_metric", get_monitor_metric, replace=True)
    OmegaConf.register_new_resolver("get_monitor_mode", get_monitor_mode, replace=True)
    OmegaConf.register_new_resolver("get_gatv4_output_dim", get_gatv4_output_dim, replace=True)
    OmegaConf.register_new_resolver(
        "get_non_relational_out_channels", get_non_relational_out_channels, replace=True
    )
    OmegaConf.register_new_resolver("infer_in_channels", infer_in_channels, replace=True)
    OmegaConf.register_new_resolver(
        "infer_num_cell_dimensions", infer_num_cell_dimensions, replace=True
    )


# Register resolvers immediately when module is imported
register_resolvers()


class HyperparameterSearch:
    """Driver for hyperparameter search via subprocess calls to run.py."""

    def __init__(self, config_path: str = "configs", project_cfg: str = "train.yaml"):
        self.config_path = config_path
        self.project_cfg = project_cfg
        self.results: List[Dict[str, Any]] = []

        # Stay off GPU for dry runs
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_grad_enabled(False)

    def product_dict(self, grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations from a parameter grid."""
        keys = list(grid.keys())
        vals = [list(v) for v in grid.values()]
        return [{k: v for k, v in zip(keys, tup)} for tup in itertools.product(*vals)]

    def to_override(self, k: str, v: Any) -> str:
        """Convert parameter to Hydra override string."""
        if isinstance(v, bool):
            return f"{k}={'true' if v else 'false'}"
        if isinstance(v, str):
            return f"{k}={v}"
        if isinstance(v, (list, tuple)):
            inner = ",".join(str(x) for x in v)
            return f"{k}=[{inner}]"
        return f"{k}={v}"

    def build_overrides(self, base_overrides: List[str], hp_dict: Dict[str, Any]) -> List[str]:
        """Build complete override list from base + hyperparameters."""
        return base_overrides + [self.to_override(k, v) for k, v in hp_dict.items()]

    def count_trainable_params(self, model: torch.nn.Module) -> int:
        """Count trainable parameters in a model."""
        return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)

    def dry_run_config(self, overrides: List[str]) -> Tuple[Optional[int], Optional[str]]:
        """Evaluate a configuration without running training (dry run)"""
        try:
            # Clear GlobalHydra instance if already initialized
            if GlobalHydra().is_initialized():
                GlobalHydra().clear()

            # Register resolvers before initializing Hydra
            register_resolvers()

            initialize(config_path=self.config_path, job_name="dry_run")

            # Parse overrides to extract model and dataset
            model_name = None
            dataset_name = None
            for override in overrides:
                if override.startswith("model="):
                    model_name = override.split("=", 1)[1]
                elif override.startswith("dataset="):
                    dataset_name = override.split("=", 1)[1]

            # If we have model/dataset, add them to overrides for proper interpolation
            final_overrides = overrides.copy()
            if model_name and not any(o.startswith("dataset=") for o in overrides):
                # Use default dataset if not specified
                final_overrides.append("dataset=motrpac")

            cfg = compose(
                config_name=self.project_cfg,
                overrides=final_overrides,
                return_hydra_config=True,
            )

            # Instantiate model only (no training)
            model = instantiate(
                cfg.model, evaluator=cfg.evaluator, optimizer=cfg.optimizer, loss=cfg.loss
            ).cpu()

            n_params = self.count_trainable_params(model)
            return n_params, None

        except Exception as e:
            return None, str(e)

    def run_config(
        self, overrides: List[str], timeout: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Run a configuration via subprocess call to run.py."""
        # Parse overrides to ensure dataset is specified for proper interpolation
        model_name = None
        dataset_name = None
        for override in overrides:
            if override.startswith("model="):
                model_name = override.split("=", 1)[1]
            elif override.startswith("dataset="):
                dataset_name = override.split("=", 1)[1]

        # If we have model but no dataset, add default dataset
        final_overrides = overrides.copy()
        if model_name and not any(o.startswith("dataset=") for o in overrides):
            final_overrides.append("dataset=motrpac")

        cmd = [sys.executable, "ogbench/run.py"] + final_overrides

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=Path(__file__).parent
            )

            if result.returncode == 0:
                # Try to parse metrics from stdout if available
                metrics = None
                try:
                    # Look for JSON-like output in stdout
                    lines = result.stdout.strip().split("\n")
                    for line in lines:
                        if line.startswith("{") and line.endswith("}"):
                            metrics = json.loads(line)
                            break
                except:  # noqa: E722
                    pass

                return True, None, metrics
            else:
                error_msg = f"Return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                return False, error_msg, None

        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout} seconds", None
        except Exception as e:
            return False, str(e), None

    def search(
        self,
        models: List[str],
        shared_grid: Dict[str, Iterable[Any]],
        per_model_grid: Dict[str, Dict[str, Iterable[Any]]],
        dry_run: bool = False,
        timeout: Optional[int] = None,
        output_dir: str = "./search_results",
    ) -> pd.DataFrame:
        """Perform hyperparameter search.

        Parameters
        ----------
        models : List[str]
            List of model names to search over
        shared_grid : Dict[str, Iterable[Any]]
            Shared hyperparameters for all models
        per_model_grid : Dict[str, Dict[str, Iterable[Any]]]
            Model-specific hyperparameters
        dry_run : bool
            If True, only count parameters without training
        timeout : Optional[int]
            Timeout in seconds for each run (None for no timeout)
        output_dir : str
            Directory to save results

        Returns
        -------
        pd.DataFrame
            Results dataframe
        """
        os.makedirs(output_dir, exist_ok=True)

        total_combinations = 0
        for model_key in models:
            model_specific = per_model_grid.get(model_key, {})
            full_grid = dict(shared_grid)
            full_grid.update(model_specific)
            total_combinations += len(self.product_dict(full_grid))

        print("Starting hyperparameter search...")
        print(f"Models: {models}")
        print(f"Total combinations: {total_combinations}")
        print(f"Mode: {'DRY RUN' if dry_run else 'TRAINING'}")
        print(f"Output directory: {output_dir}")
        print("-" * 50)

        current_run = 0

        for model_key in models:
            model_specific = per_model_grid.get(model_key, {})
            full_grid = dict(shared_grid)
            full_grid.update(model_specific)

            for hp in self.product_dict(full_grid):
                current_run += 1
                overrides = [f"model={model_key}"]
                overrides = self.build_overrides(overrides, hp)

                print(f"[{current_run}/{total_combinations}] Model: {model_key}")
                print(f"Overrides: {' '.join(overrides)}")

                start_time = time.time()

                if dry_run:
                    n_params, error = self.dry_run_config(overrides)
                    success = n_params is not None
                    metrics = {"params": n_params} if success else None
                    error_msg = error
                else:
                    success, error_msg, metrics = self.run_config(overrides, timeout)
                    n_params = None

                elapsed = time.time() - start_time

                result = {
                    "run_id": current_run,
                    "model": model_key,
                    "success": success,
                    "elapsed_time": elapsed,
                    "overrides": " ".join(overrides),
                    "error": error_msg,
                    **hp,
                }

                if dry_run:
                    result["params"] = n_params
                elif metrics:
                    result.update(metrics)

                self.results.append(result)

                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"{status} ({elapsed:.1f}s)")
                if not success:
                    print(f"Error: {error_msg}")
                print()

        # Save results
        df = pd.DataFrame(self.results)

        # Save detailed results
        detailed_csv = os.path.join(output_dir, "search_results_detailed.csv")
        df.to_csv(detailed_csv, index=False)

        # Create summary
        summary_rows = []
        for model, group in df.groupby("model"):
            total_runs = len(group)
            successful_runs = group["success"].sum()
            failed_runs = total_runs - successful_runs

            summary_row = {
                "model": model,
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
                "avg_time": group["elapsed_time"].mean(),
            }

            if dry_run:
                valid_params = group.dropna(subset=["params"])
                if not valid_params.empty:
                    summary_row.update(
                        {
                            "min_params": valid_params["params"].min(),
                            "max_params": valid_params["params"].max(),
                            "avg_params": valid_params["params"].mean(),
                        }
                    )

            summary_rows.append(summary_row)

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(output_dir, "search_summary.csv")
        summary_df.to_csv(summary_csv, index=False)

        print("Search completed!")
        print("Results saved to:")
        print(f"  - {detailed_csv}")
        print(f"  - {summary_csv}")

        return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hyperparameter search driver")
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode (count parameters only)"
    )
    parser.add_argument("--timeout", type=int, help="Timeout in seconds for each run")
    parser.add_argument("--output-dir", default="./search_results", help="Output directory")
    parser.add_argument("--config-path", default="configs", help="Hydra config path")
    parser.add_argument("--models", nargs="+", help="Models to search (default: all)")

    args = parser.parse_args()

    # Define hyperparameter grids (from your notebook)
    DATASETS = ["covidaki", "motrpac", "addneuromed", "parkinsons"]
    ADJ_THRESHOLDS = [0.8, 0.85]
    DATALOADER_BATCH_SIZES = [8, 16]
    NODE_SAMPLE_RATIOS = [1.0, 0.5, 0.2, 0.125, "full"]
    SAMPLE_METHODS = ["variance", "random", "correlation"]

    OPT_LRS = [0.001]
    OPT_WD = [0.0004]

    FE_OUT = [64, 128, 256]
    BB_NUM_LAYERS = [2, 4]
    BB_DROPOUT = [0.2, 0.4]
    BB_ACT = ["relu"]

    READOUT_POOL = ["mean"]  # "sum"

    # Models
    MODEL_KEYS = ["sagn", "chebnet", "mlp", "gin", "gatv4", "gcn", "gatv2", "graph_sage"]

    # Model-specific grids
    PER_MODEL_GRID = {
        "gcn": {"model.backbone.num_layers": [4]},
        "gin": {"model.backbone.num_layers": [8]},
        "gatv2": {
            "model.backbone.v2": [True],
            "model.backbone.heads": [8],
            "model.backbone.num_layers": [8],
        },
        "gatv4": {
            "model.backbone.hidden_channels": [[384, 64, 32]],
            "model.backbone.heads": [[6]],
        },
        "graph_sage": {"model.backbone.num_layers": [8]},
        "chebnet": {
            "model.backbone.K": [2],
            "model.backbone.num_layers": [2],
        },
        "mlp": {
            "model.backbone.hidden_channels": [
                [8, 16, 4],
            ],
        },
        "sagn": {
            "model.backbone.hidden_channels": [32],
            "model.backbone.dropout": [0.2],
            "model.backbone.num_layers": [4],
            "model.backbone.alpha": [0.5],
        },
    }

    # Shared grid
    SHARED_GRID = {
        "dataset": ["motrpac"],  # Default dataset to avoid interpolation issues
        "optimizer.parameters.lr": OPT_LRS,
        "optimizer.parameters.weight_decay": OPT_WD,
        "model.readout.pooling_type": READOUT_POOL,
    }

    # Filter models if specified
    models_to_search = args.models if args.models else MODEL_KEYS

    # Initialize search
    search = HyperparameterSearch(config_path=args.config_path)

    # Run search
    results_df = search.search(
        models=models_to_search,
        shared_grid=SHARED_GRID,
        per_model_grid=PER_MODEL_GRID,
        dry_run=args.dry_run,
        timeout=args.timeout,
        output_dir=args.output_dir,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("SEARCH SUMMARY")
    print("=" * 50)
    print(results_df.groupby("model")["success"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()

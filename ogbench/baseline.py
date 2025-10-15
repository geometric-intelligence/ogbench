"""Sklearn baseline runner for omics classification tasks."""

import importlib
import logging
import os.path as osp
from typing import Any

import hydra
import numpy as np
import pandas as pd
import rootutils
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import wandb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def task_wrapper(task_func):
    """Simplified task wrapper that handles wandb cleanup."""

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            logger.exception("Task failed with exception")
            # Always close wandb run (even if exception occurs)
            if wandb.run:
                logger.info("Closing wandb!")
                wandb.finish()
            raise ex
        finally:
            # Note: we handle wandb finish inside run_baseline for each baseline
            pass
        return metric_dict, object_dict

    return wrap


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, float]:
    """Compute classification metrics.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param y_proba: Predicted probabilities (for ROC-AUC and PR-AUC)
    :return: Dictionary of metric names and values
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    if y_proba is not None and len(np.unique(y_true)) == 2:  # Binary classification
        metrics["auroc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)

    return metrics


def prepare_param_grid(baseline_config: DictConfig) -> dict[str, list]:
    """Prepare parameter grid from config.

    :param baseline_config: Baseline configuration containing param_grid
    :return: Parameter grid for hyperparameter search
    """
    if "param_grid" not in baseline_config:
        return {}

    # Convert OmegaConf to dict and handle nested _target_ instantiations
    param_grid = OmegaConf.to_object(baseline_config.param_grid)

    # Handle any _target_ instantiations in param values
    for key, values in param_grid.items():
        processed_values = []
        for val in values:
            if isinstance(val, dict) and "_target_" in val:
                # Import the function/class directly
                target_path = val["_target_"]
                module_path, obj_name = target_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                processed_values.append(getattr(module, obj_name))
            else:
                processed_values.append(val)
        param_grid[key] = processed_values

    return param_grid


def load_and_prepare_data(
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare data for baseline evaluation.

    :param cfg: Configuration composed by Hydra
    :return: Tuple of (X_train, y_train, X_val, y_val, X_combined, y_combined)
    """
    data_name = cfg.dataset.loader.parameters.data_name

    # Check for local temp_data first
    local_data_dir = "temp_data"
    local_data_file = osp.join(local_data_dir, data_name, f"{data_name}_data.parquet")
    target_filename = f"{data_name}_target.parquet"
    local_targets_file = osp.join(local_data_dir, data_name, target_filename)

    if osp.exists(local_data_file) and osp.exists(local_targets_file):
        logger.info("Loading from local temp_data...")
        data_file = local_data_file
        targets_file = local_targets_file
    else:
        # Download from HuggingFace
        logger.info("Downloading from HuggingFace...")

        hf_repo_id = "geometric-intelligence/bgbench"
        revision = cfg.dataset.loader.parameters.get("revision", "e1631e8")

        data_file = hf_hub_download(  # nosec
            repo_id=hf_repo_id,
            repo_type="dataset",
            revision=revision,
            filename=f"{data_name}_data.parquet",
        )
        targets_file = hf_hub_download(  # nosec
            repo_id=hf_repo_id,
            repo_type="dataset",
            revision=revision,
            filename=target_filename,
        )

    # Load data
    data = pd.read_parquet(data_file)
    targets_df = pd.read_parquet(targets_file)

    if "target" in data.columns:
        data = data.drop("target", axis=1)

    targets = targets_df["target"].values

    logger.info(f"Loaded {len(targets)} samples with {data.shape[1]} features")

    # Apply shuffling (same random_state as in hf_omics.py process())
    data, targets = shuffle(data, targets, random_state=42)

    # Calculate split indices
    train_val_test_split = cfg.dataset.loader.parameters.train_val_test_split
    train_idx = int(len(data) * train_val_test_split[0])
    val_idx = int(len(data) * (train_val_test_split[0] + train_val_test_split[1]))

    # Split data
    X_train = data.iloc[:train_idx].values
    y_train = targets[:train_idx]
    X_val = data.iloc[train_idx:val_idx].values
    y_val = targets[train_idx:val_idx]

    logger.info(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    logger.info(
        f"Class distribution - Train: {dict(zip(unique_train, counts_train))}, Val: {dict(zip(unique_val, counts_val))}"
    )

    # Impute missing values (using mean strategy like in HFOmicsDataset)
    imputer = SimpleImputer(strategy=cfg.dataset.loader.parameters.imputation_method)
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    # Combine train and val for GridSearchCV with custom split
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    return X_train, y_train, X_val, y_val, X_combined, y_combined


def evaluate_and_log_metrics(
    pipeline: Pipeline,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: dict[str, Any],
    best_score: float | None,
) -> dict[str, float]:
    """Evaluate pipeline and log metrics to wandb.

    :param pipeline: Trained sklearn pipeline
    :param X_val: Validation features
    :param y_val: Validation targets
    :param best_params: Best hyperparameters from grid search
    :param best_score: Best CV score from grid search
    :return: Dictionary of validation metrics
    """
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    y_val_pred = pipeline.predict(X_val)

    # Get probabilities if available
    y_val_proba = None
    if hasattr(pipeline, "predict_proba"):
        y_val_proba_full = pipeline.predict_proba(X_val)
        if y_val_proba_full.shape[1] == 2:  # Binary classification
            y_val_proba = y_val_proba_full[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_val_proba = pipeline.decision_function(X_val)

    # Compute metrics
    val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba)

    logger.info("Validation Metrics:")
    for metric_name, metric_value in val_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    # Log to wandb
    wandb_metrics = {f"val/{k}": v for k, v in val_metrics.items()}

    if best_score is not None:
        wandb_metrics["train/best_cv_score"] = best_score

    if best_params:
        # Log hyperparameters
        for param_name, param_value in best_params.items():
            wandb_metrics[f"hyperparams/{param_name}"] = str(param_value)

    wandb.log(wandb_metrics)

    return val_metrics


def build_pipeline(baseline_config: DictConfig, seed: int) -> Pipeline:
    """Build sklearn pipeline from config.

    :param baseline_config: Baseline configuration containing pipeline steps
    :param seed: Random seed for reproducibility
    :return: Constructed sklearn pipeline
    """
    steps = []

    for step_config in baseline_config.pipeline:
        step_name = step_config.name

        # Create a copy without the 'name' field, resolving interpolations
        step_config_copy = OmegaConf.to_container(step_config, resolve=True)
        step_config_copy.pop("name", None)

        # Replace ${seed} with actual seed value if present
        def replace_seed(obj):
            if isinstance(obj, dict):
                return {k: replace_seed(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_seed(item) for item in obj]
            elif obj == "${seed}":
                return seed
            else:
                return obj

        step_config_copy = replace_seed(step_config_copy)

        # Handle nested _target_ (like score_func in SelectKBest)
        if "score_func" in step_config_copy and isinstance(step_config_copy["score_func"], dict):
            if "_target_" in step_config_copy["score_func"]:
                # Import the function directly
                func_path = step_config_copy["score_func"]["_target_"]
                module_path, func_name = func_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                step_config_copy["score_func"] = getattr(module, func_name)

        # Handle estimator (nested pipeline like in CalibratedClassifierCV)
        if "estimator" in step_config_copy and isinstance(step_config_copy["estimator"], dict):
            if "_target_" in step_config_copy["estimator"]:
                step_config_copy["estimator"] = hydra.utils.instantiate(
                    step_config_copy["estimator"]
                )

        # Instantiate the step
        step_obj = hydra.utils.instantiate(step_config_copy, _convert_="partial")
        steps.append((step_name, step_obj))

    return Pipeline(steps)


@task_wrapper
def run_baseline(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run sklearn baseline classification.

    :param cfg: Configuration composed by Hydra
    :return: Tuple with metrics and object dict
    """
    logger.info("Starting Sklearn Baseline Runner")
    logger.info(f"Loading dataset: {cfg.dataset.loader.parameters.data_name}")

    # Load and prepare data
    X_train, y_train, X_val, y_val, X_combined, y_combined = load_and_prepare_data(cfg)

    # Run baselines
    all_results = {}

    if "baselines" not in cfg.dataset:
        raise ValueError("No baselines defined in dataset config")

    for baseline_name, baseline_config in cfg.dataset.baselines.items():
        logger.info(f"Running baseline: {baseline_name}")

        # Initialize wandb run for this baseline
        run_name = f"baseline_{baseline_name}_{cfg.dataset.loader.parameters.data_name}"

        # Prepare config for wandb (avoid resolving custom resolvers)
        wandb_config = {
            "baseline_name": baseline_name,
            "dataset": cfg.dataset.loader.parameters.data_name,
            "seed": cfg.seed,
            "train_val_test_split": list(cfg.dataset.loader.parameters.train_val_test_split),
            "task": cfg.dataset.parameters.task,
            "monitor_metric": cfg.dataset.parameters.get("monitor_metric", "f1_weighted"),
        }

        wandb_run = wandb.init(
            project=cfg.logger.wandb.project,
            name=run_name,
            config=wandb_config,
            tags=cfg.get("tags", []) + ["baseline", baseline_name],
            group=cfg.logger.wandb.get("group", ""),
            save_code=True,
        )

        # Build pipeline
        logger.info("Building pipeline...")
        pipeline = build_pipeline(baseline_config, cfg.seed)
        logger.info(f"Pipeline: {pipeline}")

        # Prepare parameter grid
        param_grid = prepare_param_grid(baseline_config)

        if param_grid:
            logger.info(f"Parameter grid: {param_grid}")

            # Create custom CV split using train/val indices
            # CV expects indices relative to X_combined
            train_indices = np.arange(len(X_train))
            val_indices = np.arange(len(X_train), len(X_combined))
            cv_split = [(train_indices, val_indices)]

            logger.info(
                f"Using custom CV split: train={len(train_indices)}, val={len(val_indices)}"
            )

            # Set up GridSearchCV with custom split
            scoring = baseline_config.get("scoring", "f1_weighted")

            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv_split,
                scoring=scoring,
                n_jobs=baseline_config.get("n_jobs", -1),
                verbose=1,
                refit=True,
            )

            # Train with hyperparameter search
            logger.info("Training with grid search...")
            search.fit(X_combined, y_combined)

            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score: {search.best_score_:.4f}")

            # Use best estimator
            best_pipeline = search.best_estimator_
            best_params = search.best_params_
            best_score = search.best_score_
        else:
            logger.info("No parameter grid provided, training with default parameters...")
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_params = {}
            best_score = None

        # Evaluate and log metrics
        val_metrics = evaluate_and_log_metrics(
            best_pipeline, X_val, y_val, best_params, best_score
        )

        # Store results
        all_results[baseline_name] = {
            "val_metrics": val_metrics,
            "best_params": best_params,
            "best_cv_score": best_score,
        }

        # Finish wandb run for this baseline
        wandb.finish()

        logger.info(f"Completed baseline: {baseline_name}")

    # Create summary
    logger.info("BASELINE RESULTS SUMMARY")

    monitor_metric = cfg.dataset.parameters.get("monitor_metric", "f1_weighted")

    # Sort baselines by monitor metric
    sorted_baselines = sorted(
        all_results.items(),
        key=lambda x: x[1]["val_metrics"].get(monitor_metric, 0),
        reverse=True,
    )

    logger.info(f"Ranking by {monitor_metric}:")
    for rank, (name, results) in enumerate(sorted_baselines, 1):
        metric_val = results["val_metrics"].get(monitor_metric, 0)
        logger.info(f"{rank}. {name:30s} {metric_val:.4f}")

    # Return best baseline results
    best_baseline_name, best_results = sorted_baselines[0]
    logger.info(f"Best baseline: {best_baseline_name}")

    metric_dict = {f"val/{k}": v for k, v in best_results["val_metrics"].items()}
    object_dict = {
        "cfg": cfg,
        "all_results": all_results,
        "best_baseline": best_baseline_name,
    }

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="baseline.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for baseline runner.

    :param cfg: Configuration composed by Hydra
    :return: Optimized metric value
    """
    # Run baselines
    metric_dict, _ = run_baseline(cfg)

    # Return monitor metric for optimization
    monitor_metric = cfg.dataset.parameters.get("monitor_metric", "f1_weighted")
    return metric_dict.get(f"val/{monitor_metric}")


if __name__ == "__main__":
    main()

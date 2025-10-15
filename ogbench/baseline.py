"""Sklearn baseline runner for omics classification tasks."""

import importlib
import logging
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import seaborn as sns
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
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

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (12, 8)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetContainer:
    """Container for all dataset variants at different preprocessing stages."""

    # Raw data (before any preprocessing)
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray

    # Processed data (after imputation but before scaling - pipeline handles scaling)
    X_train_processed: np.ndarray
    X_val_processed: np.ndarray

    # Combined data for GridSearchCV
    X_combined: np.ndarray
    y_combined: np.ndarray

    # Dataset metadata
    dataset_name: str
    n_features: int
    n_samples: int
    class_distribution: dict

    def get_features_at_stage(self, stage: str) -> tuple[np.ndarray, np.ndarray]:
        """Get features at a specific preprocessing stage.

        :param stage: 'raw' (before any preprocessing), 'processed' (after imputation but before
            scaling)
        :return: Tuple of (X_train, X_val) at the specified stage
        """
        if stage == "raw":
            return self.X_train_raw, self.X_val_raw
        elif stage == "processed":
            return self.X_train_processed, self.X_val_processed
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def get_class_info(self) -> dict:
        """Get class distribution information."""
        unique_train, counts_train = np.unique(self.y_train, return_counts=True)
        unique_val, counts_val = np.unique(self.y_val, return_counts=True)

        return {
            "train": dict(zip(unique_train, counts_train)),
            "val": dict(zip(unique_val, counts_val)),
            "n_classes": len(np.unique(self.y_train)),
            "is_binary": len(np.unique(self.y_train)) == 2,
        }


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


def load_and_prepare_data(cfg: DictConfig) -> DatasetContainer:
    """Load and prepare data for baseline evaluation.

    :param cfg: Configuration composed by Hydra
    :return: DatasetContainer with all data variants
    """
    data_name = cfg.dataset.loader.parameters.data_name

    # Check for local temp_data first
    local_data_dir = "temp_data"
    local_data_file = osp.join(local_data_dir, data_name, f"{data_name}_data.parquet")
    target_filename = f"{data_name}_targets.parquet"
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

    # Keep raw data for plotting
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()

    # Impute missing values (using mean strategy like in HFOmicsDataset)
    imputer = SimpleImputer(strategy=cfg.dataset.loader.parameters.imputation_method)
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    # Combine train and val for GridSearchCV with custom split (use imputed but unscaled)
    X_combined = np.vstack([X_train_imputed, X_val_imputed])
    y_combined = np.concatenate([y_train, y_val])

    # Get class distribution info
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    class_distribution = {
        "train": dict(zip(unique_train, counts_train)),
        "val": dict(zip(unique_val, counts_val)),
    }

    return DatasetContainer(
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        y_train=y_train,
        y_val=y_val,
        X_train_processed=X_train_imputed,  # Now contains imputed but unscaled data
        X_val_processed=X_val_imputed,  # Now contains imputed but unscaled data
        X_combined=X_combined,
        y_combined=y_combined,
        dataset_name=data_name,
        n_features=data.shape[1],
        n_samples=len(targets),
        class_distribution=class_distribution,
    )


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


def generate_comprehensive_plots(
    pipeline: Pipeline,
    dataset: DatasetContainer,
    baseline_name: str,
    output_dir: Path,
) -> None:
    """Generate comprehensive evaluation plots in a single figure.

    :param pipeline: Trained sklearn pipeline
    :param dataset: DatasetContainer with all data variants
    :param baseline_name: Name of the baseline model
    :param output_dir: Directory to save plots
    """
    logger.info(f"Generating comprehensive plots for best baseline: {baseline_name}")

    # Create output directory
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Get imputed but unscaled features (let pipeline handle all transformations)
    X_train_imputed, X_val_imputed = dataset.get_features_at_stage("processed")

    # Apply pipeline up to feature selection to get selected features
    if "feature_selection" in [step[0] for step in pipeline.steps]:
        X_train_before_selection = X_train_imputed.copy()
        X_val_before_selection = X_val_imputed.copy()
        X_train_after_selection = X_train_imputed.copy()
        X_val_after_selection = X_val_imputed.copy()

        for step_name, step in pipeline.steps[:-1]:  # Exclude classifier
            X_train_after_selection = step.transform(X_train_after_selection)
            X_val_after_selection = step.transform(X_val_after_selection)
    else:
        X_train_before_selection = X_train_imputed
        X_val_before_selection = X_val_imputed
        X_train_after_selection = X_train_imputed
        X_val_after_selection = X_val_imputed

    # Generate predictions (pipeline will handle all preprocessing internally)
    y_train_pred = pipeline.predict(X_train_imputed)
    y_val_pred = pipeline.predict(X_val_imputed)

    # Get probabilities if available
    y_train_proba = None
    y_val_proba = None
    if hasattr(pipeline, "predict_proba"):
        y_train_proba_full = pipeline.predict_proba(X_train_imputed)
        y_val_proba_full = pipeline.predict_proba(X_val_imputed)
        if y_train_proba_full.shape[1] == 2:  # Binary classification
            y_train_proba = y_train_proba_full[:, 1]
            y_val_proba = y_val_proba_full[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_train_proba = pipeline.decision_function(X_train_imputed)
        y_val_proba = pipeline.decision_function(X_val_imputed)

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 24))

    # Define grid layout: 4 rows, 3 columns
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Class Balance (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    class_info = dataset.get_class_info()
    unique_train = list(class_info["train"].keys())
    counts_train = list(class_info["train"].values())
    unique_val = list(class_info["val"].keys())
    counts_val = list(class_info["val"].values())

    x_pos = np.arange(len(unique_train))
    width = 0.35

    ax1.bar(x_pos - width / 2, counts_train, width, label="Train", alpha=0.7, edgecolor="black")
    ax1.bar(x_pos + width / 2, counts_val, width, label="Val", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.set_title("Class Distribution")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(unique_train)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Add count labels
    for i, (train_count, val_count) in enumerate(zip(counts_train, counts_val)):
        ax1.text(i - width / 2, train_count, str(train_count), ha="center", va="bottom")
        ax1.text(i + width / 2, val_count, str(val_count), ha="center", va="bottom")

    # 2. Raw Features Distribution (top row, right column)
    ax2 = fig.add_subplot(gs[0, 2])
    X_train_raw, X_val_raw = dataset.get_features_at_stage("raw")
    n_features_to_plot = min(100, X_train_raw.shape[1])
    feature_indices = np.random.RandomState(42).choice(
        X_train_raw.shape[1], n_features_to_plot, replace=False
    )

    data_to_plot = X_train_raw[:, feature_indices].flatten()
    data_to_plot = data_to_plot[~np.isnan(data_to_plot)]

    # Create histogram with better visualization
    counts, bins, patches = ax2.hist(
        data_to_plot, bins=50, alpha=0.7, edgecolor="black", color="skyblue"
    )
    ax2.set_xlabel("Feature Value")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"Raw Features Distribution\n({n_features_to_plot} sampled from {X_train_raw.shape[1]:,} features)"
    )
    ax2.grid(True, alpha=0.3, axis="y")

    # Add statistics text
    mean_val = np.mean(data_to_plot)
    std_val = np.std(data_to_plot)
    ax2.text(
        0.02,
        0.98,
        f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 3. Feature Variance Before Selection (second row, left)
    ax3 = fig.add_subplot(gs[1, 0])
    feature_vars = np.var(X_train_before_selection, axis=0)
    sorted_vars = np.sort(feature_vars)[::-1][:500]  # Show top 500 for better visualization

    ax3.plot(sorted_vars, marker=".", linestyle="-", alpha=0.7, color="darkgreen", markersize=2)
    ax3.set_xlabel("Feature Rank (by variance)")
    ax3.set_ylabel("Variance")
    ax3.set_title(f"Feature Variance Ranking\n(Top 500 of {len(feature_vars):,} features)")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")

    # Add percentile markers
    p50_idx = int(0.5 * len(sorted_vars))
    p90_idx = int(0.9 * len(sorted_vars))
    ax3.axvline(p50_idx, color="red", linestyle="--", alpha=0.7, label="50th percentile")
    ax3.axvline(p90_idx, color="orange", linestyle="--", alpha=0.7, label="90th percentile")
    ax3.legend(fontsize=8)

    # 4. Selected Features Distribution (second row, middle)
    ax4 = fig.add_subplot(gs[1, 1])
    n_sel_to_plot = min(100, X_train_after_selection.shape[1])
    sel_indices = np.random.RandomState(42).choice(
        X_train_after_selection.shape[1], n_sel_to_plot, replace=False
    )
    data_to_plot = X_train_after_selection[:, sel_indices].flatten()

    # Create histogram with better visualization
    counts, bins, patches = ax4.hist(
        data_to_plot, bins=50, alpha=0.7, edgecolor="black", color="orange"
    )
    ax4.set_xlabel("Feature Value")
    ax4.set_ylabel("Count")
    ax4.set_title(
        f"Selected Features Distribution\n({n_sel_to_plot} sampled from {X_train_after_selection.shape[1]} selected)"
    )
    ax4.grid(True, alpha=0.3, axis="y")

    # Add statistics text
    mean_val = np.mean(data_to_plot)
    std_val = np.std(data_to_plot)
    ax4.text(
        0.02,
        0.98,
        f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}",
        transform=ax4.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 5. PCA Visualization Train (second row, right)
    ax5 = fig.add_subplot(gs[1, 2])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train_after_selection)
    scatter = ax5.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=dataset.y_train,
        cmap="viridis",
        alpha=0.6,
        edgecolors="k",
        linewidth=0.5,
    )
    ax5.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    ax5.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    ax5.set_title(f"PCA - Train Set\n({X_train_after_selection.shape[1]} features)")
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label="Class")

    # 6. PCA Visualization Val (third row, left)
    ax6 = fig.add_subplot(gs[2, 0])
    X_pca_val = pca.transform(X_val_after_selection)
    scatter = ax6.scatter(
        X_pca_val[:, 0],
        X_pca_val[:, 1],
        c=dataset.y_val,
        cmap="viridis",
        alpha=0.6,
        edgecolors="k",
        linewidth=0.5,
    )
    ax6.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    ax6.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    ax6.set_title(f"PCA - Val Set\n({X_val_after_selection.shape[1]} features)")
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label="Class")

    # 7. Confusion Matrix Train (third row, middle)
    ax7 = fig.add_subplot(gs[2, 1])
    ConfusionMatrixDisplay.from_predictions(
        dataset.y_train, y_train_pred, ax=ax7, cmap="Blues", values_format="d"
    )
    ax7.set_title("Confusion Matrix - Train")

    # 8. Confusion Matrix Val (third row, right)
    ax8 = fig.add_subplot(gs[2, 2])
    ConfusionMatrixDisplay.from_predictions(
        dataset.y_val, y_val_pred, ax=ax8, cmap="Blues", values_format="d"
    )
    ax8.set_title("Confusion Matrix - Val")

    # Add threshold info for binary classification
    if y_val_proba is not None and dataset.get_class_info()["is_binary"]:
        ax8.text(
            0.02,
            0.98,
            "Threshold: 0.5",
            transform=ax8.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            fontsize=10,
        )

    # 9. ROC Curves (bottom row, left and middle) - only for binary classification
    if y_train_proba is not None and dataset.get_class_info()["is_binary"]:
        ax9 = fig.add_subplot(gs[3, 0])
        RocCurveDisplay.from_predictions(dataset.y_train, y_train_proba, ax=ax9, name="Train")
        ax9.plot([0, 1], [0, 1], "k--", label="Random")
        ax9.axvline(0.5, color="red", linestyle=":", alpha=0.7, label="Threshold: 0.5")
        ax9.set_title("ROC Curve - Train")
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        ax10 = fig.add_subplot(gs[3, 1])
        RocCurveDisplay.from_predictions(dataset.y_val, y_val_proba, ax=ax10, name="Val")
        ax10.plot([0, 1], [0, 1], "k--", label="Random")
        ax10.axvline(0.5, color="red", linestyle=":", alpha=0.7, label="Threshold: 0.5")
        ax10.set_title("ROC Curve - Val")
        ax10.legend()
        ax10.grid(True, alpha=0.3)

        # 10. Precision-Recall Curves (bottom row, right)
        ax11 = fig.add_subplot(gs[3, 2])
        PrecisionRecallDisplay.from_predictions(dataset.y_val, y_val_proba, ax=ax11, name="Val")
        ax11.axhline(0.5, color="red", linestyle=":", alpha=0.7, label="Threshold: 0.5")
        ax11.set_title("Precision-Recall Curve - Val")
        ax11.legend()
        ax11.grid(True, alpha=0.3)
    else:
        # If not binary classification, show additional feature analysis
        ax9 = fig.add_subplot(gs[3, :])
        class_info = dataset.get_class_info()
        ax9.text(
            0.5,
            0.5,
            f'Multiclass Classification\n({class_info["n_classes"]} classes)\nROC/PR curves not shown',
            ha="center",
            va="center",
            transform=ax9.transAxes,
            fontsize=14,
        )
        ax9.set_title("Classification Type")
        ax9.axis("off")

    # Add overall title with dataset name (much bigger)
    fig.suptitle(
        f"{dataset.dataset_name.upper()} Dataset - Comprehensive Evaluation Report - {baseline_name}",
        fontsize=24,
        y=0.98,
        fontweight="bold",
    )

    # Save the comprehensive plot
    output_path = plot_dir / f"{baseline_name}_comprehensive_report.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    logger.info(f"Saved comprehensive report to {output_path}")

    # Log to wandb
    if wandb.run is not None:
        wandb.log({"plots/comprehensive_report": wandb.Image(fig)})

    plt.close(fig)


@task_wrapper
def run_baseline(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run sklearn baseline classification.

    :param cfg: Configuration composed by Hydra
    :return: Tuple with metrics and object dict
    """
    logger.info("Starting Sklearn Baseline Runner")
    logger.info(f"Loading dataset: {cfg.dataset.loader.parameters.data_name}")

    # Load and prepare data
    dataset = load_and_prepare_data(cfg)

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
            train_indices = np.arange(len(dataset.y_train))
            val_indices = np.arange(len(dataset.y_train), len(dataset.y_combined))
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
                refit=False,
            )

            # Train with hyperparameter search
            logger.info("Training with grid search...")
            search.fit(dataset.X_combined, dataset.y_combined)

            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score: {search.best_score_:.4f}")

            # Manually refit the best pipeline on training data only
            logger.info("Refitting best pipeline on training data only...")
            best_pipeline = build_pipeline(baseline_config, cfg.seed)

            # Set the best parameters
            for param_name, param_value in search.best_params_.items():
                best_pipeline.set_params(**{param_name: param_value})

            # Fit only on training data (imputed but unscaled)
            best_pipeline.fit(dataset.X_train_processed, dataset.y_train)

            best_params = search.best_params_
            best_score = search.best_score_
        else:
            logger.info("No parameter grid provided, training with default parameters...")
            pipeline.fit(dataset.X_train_processed, dataset.y_train)  # imputed but unscaled
            best_pipeline = pipeline
            best_params = {}
            best_score = None

        # Evaluate and log metrics
        val_metrics = evaluate_and_log_metrics(
            best_pipeline, dataset.X_val_processed, dataset.y_val, best_params, best_score
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

    # Generate comprehensive plots for the best baseline
    logger.info("Generating comprehensive plots for the best baseline...")

    # Rebuild the best pipeline with the best hyperparameters
    best_baseline_config = cfg.dataset.baselines[best_baseline_name]
    best_pipeline = build_pipeline(best_baseline_config, cfg.seed)

    # Set hyperparameters if they were searched
    if best_results["best_params"]:
        best_pipeline.set_params(**best_results["best_params"])

    # Retrain on train set only (not combined) for proper evaluation (imputed but unscaled)
    best_pipeline.fit(dataset.X_train_processed, dataset.y_train)

    # Generate comprehensive plots
    output_dir = Path(cfg.get("output_dir", "outputs"))
    generate_comprehensive_plots(
        pipeline=best_pipeline,
        dataset=dataset,
        baseline_name=best_baseline_name,
        output_dir=output_dir,
    )

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

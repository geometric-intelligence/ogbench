"""Feature selection using MMHC (Max-Min Hill Climbing) Bayesian Network structure learning."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pgmpy.estimators import MmhcEstimator
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

# Load data
data = pd.read_parquet("temp_data/motrpac/motrpac_data.parquet")
targets = pd.read_parquet("temp_data/motrpac/motrpac_targets_vo2_rel.parquet")

print(f"Data shape: {data.shape}")
print(f"Targets shape: {targets.shape}")

# Handle NaNs
data_clean = data.fillna(data.mean())

# Train/test split (split before discretization to avoid leakage)
X_train, X_test, y_train, y_test = train_test_split(
    data_clean.values, targets["target"].values, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Discretize features for MMHC (using training data only)
print("\nDiscretizing features for MMHC...")
n_bins = 3  # Low, Medium, High
discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
X_train_disc = discretizer.fit_transform(X_train)

# Discretize target
y_discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
y_train_disc = y_discretizer.fit_transform(y_train.reshape(-1, 1)).ravel()

# Create DataFrame for MMHC with feature names
feature_names = list(data_clean.columns)
df_train_disc = pd.DataFrame(X_train_disc, columns=feature_names)
df_train_disc["target"] = y_train_disc.astype(int)

# Convert all columns to int for pgmpy
for col in df_train_disc.columns:
    df_train_disc[col] = df_train_disc[col].astype(int)

print(f"Discretized data shape: {df_train_disc.shape}")
print(f"Sample discretized values:\n{df_train_disc.head()}")

# Learn Bayesian Network structure using MMHC
print("\nLearning Bayesian Network structure with MMHC...")
mmhc = MmhcEstimator(df_train_disc)
model = mmhc.estimate(significance_level=0.01, tabu_length=10)

print(f"Learned model with {len(model.nodes())} nodes and {len(model.edges())} edges")

# Extract features connected to target
target_parents = list(model.predecessors("target"))
target_children = list(model.successors("target"))
selected_features = list(set(target_parents + target_children))

print(f"\nFeatures directly connected to target: {len(selected_features)}")
print(f"  - Parents (features → target): {len(target_parents)}")
print(f"  - Children (target → features): {len(target_children)}")

if selected_features:
    print(f"\nSelected features: {selected_features[:10]}...")

    # Get indices of selected features
    feature_indices = [feature_names.index(f) for f in selected_features if f != "target"]

    # Use selected features for regression
    X_train_selected = X_train[:, feature_indices]
    X_test_selected = X_test[:, feature_indices]

    print(f"\nTraining ElasticNet with {X_train_selected.shape[1]} selected features...")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # ElasticNet regression
    model_reg = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99], cv=5, random_state=42)
    model_reg.fit(X_train_scaled, y_train)

    print(f"Best alpha: {model_reg.alpha_:.4f}, Best l1_ratio: {model_reg.l1_ratio_:.2f}")

    # Predictions
    y_pred_train = model_reg.predict(X_train_scaled)
    y_pred_test = model_reg.predict(X_test_scaled)

    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_corr, test_pval = pearsonr(y_test, y_pred_test)
    train_corr, train_pval = pearsonr(y_train, y_pred_train)

    print(f"\nTrain R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test Correlation: {test_corr:.3f} (p={test_pval:.2e})")

    # Correlation plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Train plot
    axes[0].scatter(y_train, y_pred_train, alpha=0.5, edgecolors="k", linewidth=0.5)
    axes[0].plot(
        [y_train.min(), y_train.max()],
        [y_train.min(), y_train.max()],
        "r--",
        lw=2,
        label="Perfect prediction",
    )
    axes[0].set_xlabel("Actual VO₂max Response (relative)", fontsize=12)
    axes[0].set_ylabel("Predicted VO₂max Response", fontsize=12)
    axes[0].set_title(
        f"Train Set (n={len(y_train)}, {len(selected_features)} features)\nR² = {train_r2:.3f}, Corr = {train_corr:.3f}",
        fontsize=13,
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Test plot
    axes[1].scatter(y_test, y_pred_test, alpha=0.6, edgecolors="k", linewidth=0.5)
    axes[1].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Perfect prediction",
    )
    axes[1].set_xlabel("Actual VO₂max Response (relative)", fontsize=12)
    axes[1].set_ylabel("Predicted VO₂max Response", fontsize=12)
    axes[1].set_title(
        f"Test Set (n={len(y_test)}, {len(selected_features)} features)\nR² = {test_r2:.3f}, Corr = {test_corr:.3f}",
        fontsize=13,
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("motrpac_mmhc_correlation.png", dpi=150)
    print("\nPlot saved to: motrpac_mmhc_correlation.png")

else:
    print("\nNo features were selected by MMHC!")
    print("Try adjusting significance_level or n_bins parameters.")

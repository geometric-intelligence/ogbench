"""Minimal script to test MotrPac data with PCA + ElasticNet regression."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_parquet("temp_data/motrpac/motrpac_data.parquet")
targets = pd.read_parquet("temp_data/motrpac/motrpac_targets_vo2_rel.parquet")

print(f"Data shape: {data.shape}")
print(f"Targets shape: {targets.shape}")

# Handle NaNs
data_clean = data.fillna(data.mean())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data_clean.values, targets["target"].values, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# ElasticNet regression
model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99], cv=5, random_state=42)
model.fit(X_train_pca, y_train)

print(f"\nBest alpha: {model.alpha_:.4f}, Best l1_ratio: {model.l1_ratio_:.2f}")

# Predictions
y_pred_train = model.predict(X_train_pca)
y_pred_test = model.predict(X_test_pca)

# Metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
test_corr, test_pval = pearsonr(y_test, y_pred_test)

print(f"\nTrain R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test Correlation: {test_corr:.3f} (p={test_pval:.2e})")

# Correlation plots
train_corr, train_pval = pearsonr(y_train, y_pred_train)

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
    f"Train Set (n={len(y_train)})\nR² = {train_r2:.3f}, Corr = {train_corr:.3f}", fontsize=13
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
    f"Test Set (n={len(y_test)})\nR² = {test_r2:.3f}, Corr = {test_corr:.3f}", fontsize=13
)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("motrpac_correlation.png", dpi=150)
print("\nPlot saved to: motrpac_correlation.png")

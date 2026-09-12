"""Train-only linear covariate adjustment (MoTrPAC-style)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class CovariateAdjuster:
    """Fit ``feature ~ covariates`` on train and apply the same coefficients.

    Categorical covariates in ``{sex, race}`` are one-hot encoded with
    ``drop_first=True``. Encoding columns and train covariate means are frozen
    at ``fit`` time so val/test cannot change the model.
    """

    def __init__(self, covariate_names: list[str] | None = None) -> None:
        self.covariate_names = list(covariate_names or ['age', 'sex', 'bmi', 'race'])
        self.feature_names_: list[str] | None = None
        self.encoded_columns_: list[str] | None = None
        self.coef_: dict[str, np.ndarray] = {}
        self.intercept_: dict[str, float] = {}
        self.cov_mean_: np.ndarray | None = None

    def fit(self, data: pd.DataFrame, covariates: pd.DataFrame) -> CovariateAdjuster:
        """Fit one linear model per feature on training rows."""
        data = data.reset_index(drop=True)
        covariates = covariates.reset_index(drop=True)
        if len(data) != len(covariates):
            raise ValueError('data and covariates must have the same number of rows')

        x_cov = self._encode_covariates(covariates, fit=True)
        valid_cov_mask = x_cov.notna().all(axis=1)
        if int(valid_cov_mask.sum()) == 0:
            raise ValueError('No training samples with complete covariates')

        self.cov_mean_ = x_cov.loc[valid_cov_mask].mean(axis=0).to_numpy(dtype=float)
        self.feature_names_ = [str(c) for c in data.columns]
        self.coef_.clear()
        self.intercept_.clear()

        x_cov_values = x_cov.to_numpy(dtype=float)
        for col in self.feature_names_:
            y = data[col].to_numpy(dtype=float)
            valid = valid_cov_mask.to_numpy() & ~np.isnan(y)
            if int(valid.sum()) < 10:
                continue
            model = LinearRegression()
            model.fit(x_cov_values[valid], y[valid])
            self.coef_[col] = model.coef_.astype(float)
            self.intercept_[col] = float(model.intercept_)
        return self

    def transform(self, data: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
        """Apply train-fitted adjustment; leave incomplete-covariate rows unchanged."""
        if self.encoded_columns_ is None or self.cov_mean_ is None:
            raise RuntimeError('CovariateAdjuster must be fit before transform')
        data = data.reset_index(drop=True)
        covariates = covariates.reset_index(drop=True)
        if len(data) != len(covariates):
            raise ValueError('data and covariates must have the same number of rows')

        x_cov = self._encode_covariates(covariates, fit=False)
        valid_cov_mask = x_cov.notna().all(axis=1).to_numpy()
        x_valid = x_cov.to_numpy(dtype=float)[valid_cov_mask]
        adjusted = data.copy()

        for col, coef in self.coef_.items():
            if col not in adjusted.columns:
                continue
            values = np.array(adjusted[col].to_numpy(dtype=float), copy=True)
            predicted = x_valid @ coef + self.intercept_[col]
            predicted_mean = float(np.dot(self.cov_mean_, coef) + self.intercept_[col])
            values[valid_cov_mask] = values[valid_cov_mask] - (predicted - predicted_mean)
            adjusted[col] = values
        return adjusted

    def _encode_covariates(self, covariates: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        missing = [c for c in self.covariate_names if c not in covariates.columns]
        if missing:
            raise ValueError(f'Missing covariate columns: {missing}')
        cov_df = covariates[self.covariate_names].copy()
        categorical_cols = [c for c in self.covariate_names if c in {'sex', 'race'}]
        continuous_cols = [c for c in self.covariate_names if c not in {'sex', 'race'}]

        if categorical_cols:
            encoded = pd.get_dummies(cov_df[categorical_cols], drop_first=True, dtype=float)
        else:
            encoded = pd.DataFrame(index=cov_df.index)

        if fit:
            self.encoded_columns_ = list(encoded.columns)
        else:
            if self.encoded_columns_ is None:
                raise RuntimeError('CovariateAdjuster must be fit before transform')
            encoded = encoded.reindex(columns=self.encoded_columns_, fill_value=0.0)

        if continuous_cols:
            continuous = cov_df[continuous_cols].astype(float)
            return pd.concat([continuous, encoded], axis=1)
        return encoded

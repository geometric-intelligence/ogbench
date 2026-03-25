"""MotrPac dataset processor."""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface


def adjust_for_covariates(
    data: pd.DataFrame,
    covariates: pd.DataFrame,
    covariate_names: list[str],
) -> pd.DataFrame:
    """Adjust protein data for specified covariates using linear regression.

    For each protein, fits: protein ~ covariates, then adjusts to remove covariate
    effects while centering at mean covariate values. This avoids using target
    labels, preventing data leakage.
    Categorical variables (sex, race) are automatically one-hot encoded.

    Args:
        data: DataFrame with protein columns (already log-transformed)
        covariates: DataFrame with covariate columns
        covariate_names: List of covariate column names to adjust for

    Returns:
        Adjusted data with same shape as input
    """
    adjusted_data = data.copy()

    # Build covariate matrix with one-hot encoding for categoricals
    cov_df = covariates[covariate_names].copy()

    # Identify categorical columns (sex, race)
    categorical_cols = [col for col in covariate_names if col in {'sex', 'race'}]
    continuous_cols = [col for col in covariate_names if col not in categorical_cols]

    # One-hot encode categoricals and combine with continuous
    if categorical_cols:
        cov_encoded = pd.get_dummies(cov_df[categorical_cols], drop_first=True, dtype=float)
    else:
        cov_encoded = pd.DataFrame(index=cov_df.index)

    if continuous_cols:
        cov_continuous = cov_df[continuous_cols].astype(float)
        X_cov = pd.concat([cov_continuous, cov_encoded], axis=1)
    else:
        X_cov = cov_encoded

    # Find rows with complete covariate data
    valid_cov_mask = X_cov.notna().all(axis=1)

    if valid_cov_mask.sum() == 0:
        print('Warning: No samples with complete covariate data. Returning original data.')
        return adjusted_data

    X_cov_valid = X_cov.loc[valid_cov_mask].values
    X_cov_mean = X_cov_valid.mean(axis=0)

    print(f'Adjusting for covariates: {covariate_names}')
    print(f'  Encoded covariate columns: {list(X_cov.columns)}')
    print(f'  Samples with complete covariate data: {valid_cov_mask.sum()} / {len(data)}')

    # Track adjustment statistics
    adjustment_stats: dict[str, dict[str, float]] = {}

    # For each protein, fit protein ~ covariates and adjust
    for protein_col in data.columns:
        protein_values = data[protein_col].values

        # Find valid rows: complete covariates AND non-NaN protein
        valid_mask = valid_cov_mask.values & ~np.isnan(protein_values)

        if valid_mask.sum() < 10:
            # Not enough data to fit regression, skip adjustment
            continue

        # Get valid data for this protein
        X_valid = X_cov.loc[valid_mask].values
        y_valid = protein_values[valid_mask]

        # Fit: protein ~ covariates
        model = LinearRegression()
        model.fit(X_valid, y_valid)

        # Calculate expected value at mean covariates
        expected_at_mean = model.predict(X_cov_mean.reshape(1, -1))[0]

        # For all samples with valid covariates, calculate adjustment
        # adjusted = original - (predicted - expected_at_mean)
        predicted_all = model.predict(X_cov_valid)
        adjustment = predicted_all - expected_at_mean

        # Apply adjustment only to samples with valid covariates
        adjusted_values = protein_values.copy()
        adjusted_values[valid_cov_mask.values] = protein_values[valid_cov_mask.values] - adjustment
        adjusted_data[protein_col] = adjusted_values

        # Track mean absolute adjustment for this protein
        adjustment_stats[protein_col] = {
            'mean_abs_adjustment': float(np.abs(adjustment).mean()),
            'valid_samples': int(valid_mask.sum()),
        }

    # Print summary statistics
    mean_adjustments = [s['mean_abs_adjustment'] for s in adjustment_stats.values()]
    print(f'  Proteins adjusted: {len(adjustment_stats)} / {len(data.columns)}')
    print(f'  Mean absolute adjustment across proteins: {np.mean(mean_adjustments):.4f}')
    print(f'  Max absolute adjustment: {np.max(mean_adjustments):.4f}')

    return adjusted_data


def process_motrpac(output_dir: str = 'temp_data') -> None:
    """Download and process MotrPac dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Download URLs
    urls = {
        'proteomics': 'https://d1yw74buhe0ts0.cloudfront.net/static/motrpac-data-hub/publications/data/related-studies/heritage-proteomics/HERITAGE_proteomics_somalogic.xlsx',
        'analytes': 'https://d1yw74buhe0ts0.cloudfront.net/static/motrpac-data-hub/publications/data/related-studies/heritage-proteomics/HERITAGE_somalogic_analytes.xlsx',
    }

    # Download files
    for name, url in urls.items():
        file_path = os.path.join(output_dir, f'motrpac_{name}.xlsx')
        if not os.path.exists(file_path):
            print(f'Downloading {name}...')
            download_file(url, file_path)

    # Load data
    proteomics_df = pd.read_excel(os.path.join(output_dir, 'motrpac_proteomics.xlsx'), header=3)
    analytes_df = pd.read_excel(os.path.join(output_dir, 'motrpac_analytes.xlsx'))

    # 1) Use analytes.xlsx to select & name protein columns
    # The analytes file has Baseline/Post-training/Response columns with IDs
    if 'Baseline' in analytes_df.columns:
        analyte_ids = analytes_df['Baseline'].astype(str).tolist()
    else:
        # Fallback to original logic
        id_col = next(
            c
            for c in analytes_df.columns
            if c.lower() in {'somamer', 'seqid', 'target', 'analyte'}
        )
        analyte_ids = analytes_df[id_col].astype(str).tolist()

    # Keep only analyte columns that exist
    present = [c for c in proteomics_df.columns if str(c) in analyte_ids]
    raw_data = proteomics_df[present].copy()
    raw_data.columns = [str(c) for c in raw_data.columns]  # stable names

    # 2) Targets: compute relative ΔVO₂max and responder labels
    # Extract baselines and deltas robustly by column names
    col_base = next(c for c in proteomics_df.columns if 'baseline vo2' in c.lower())
    col_delta = next(c for c in proteomics_df.columns if 'delta vo2' in c.lower())

    vo2_base = pd.to_numeric(proteomics_df[col_base], errors='coerce')
    vo2_delta = pd.to_numeric(proteomics_df[col_delta], errors='coerce')
    vo2_post = vo2_base + vo2_delta

    mask = (~vo2_base.isna()) & (~vo2_post.isna())  # keep only valid pairs
    raw_data = raw_data.loc[mask]
    vo2_base = vo2_base.loc[mask]
    vo2_post = vo2_post.loc[mask]

    delta_rel = (vo2_post - vo2_base) / vo2_base  # relative improvement

    # Responder tasks
    responder15 = (delta_rel > 0.15).astype(int)  # binary

    # 3) Extract baseline covariates
    cov_cols = {
        'age': [c for c in proteomics_df.columns if c.lower() == 'age'],
        'sex': [c for c in proteomics_df.columns if c.lower() == 'sex'],
        'bmi': [c for c in proteomics_df.columns if 'bmi' in c.lower()],
        'race': [c for c in proteomics_df.columns if 'race' in c.lower()],
        'body_fat_pct': [c for c in proteomics_df.columns if 'body fat' in c.lower()],
        'ffm': [c for c in proteomics_df.columns if 'fat-free' in c.lower() or 'ffm' in c.lower()],
    }

    # Build covariates dataframe (use .loc[mask] for proteomics_df columns only)
    cov = pd.DataFrame(
        {
            'age': pd.to_numeric(proteomics_df.loc[mask, cov_cols['age'][0]], errors='coerce')
            if cov_cols['age']
            else np.nan,
            'sex': proteomics_df.loc[mask, cov_cols['sex'][0]].values
            if cov_cols['sex']
            else np.nan,
            'bmi': pd.to_numeric(proteomics_df.loc[mask, cov_cols['bmi'][0]], errors='coerce')
            if cov_cols['bmi']
            else np.nan,
            'race': proteomics_df.loc[mask, cov_cols['race'][0]].values
            if cov_cols['race']
            else np.nan,
            'body_fat_pct': pd.to_numeric(
                proteomics_df.loc[mask, cov_cols['body_fat_pct'][0]], errors='coerce'
            )
            if cov_cols['body_fat_pct']
            else np.nan,
            'vo2_baseline': vo2_base.values,  # Already filtered by mask
        }
    ).reset_index(drop=True)

    # 4) Light QC at ingest time (no learning from labels)
    col_na = raw_data.isna().mean()
    raw_data = raw_data.loc[:, col_na <= 0.1]  # drop analytes with >10% NA
    # Leave remaining NaNs as-is; downstream will impute on train only

    # 6) log transform (log2 is standard for SomaLogic RFU data)
    raw_data = np.log2(raw_data)
    raw_data = pd.DataFrame(raw_data, columns=raw_data.columns).reset_index(drop=True)

    # 6.5) Adjust for covariates (age, sex, bmi, race)
    print('\nCovariate selection:')
    print(f'  Available covariates in cov dataframe: {cov.columns.tolist()}')
    print(f'  Non-null counts: {cov.notna().sum().to_dict()}')

    covariates_to_adjust = ['age', 'sex', 'bmi', 'race']

    print(f'  Selected for adjustment: {covariates_to_adjust}')

    raw_data = adjust_for_covariates(raw_data, cov, covariates_to_adjust)

    # 7) Emit classification target only
    out = os.path.join(output_dir, 'motrpac')
    os.makedirs(out, exist_ok=True)

    raw_data.reset_index(drop=True).to_parquet(os.path.join(out, 'motrpac_data.parquet'))

    pd.DataFrame({'target': responder15.reset_index(drop=True).astype(np.int64)}).to_parquet(
        os.path.join(out, 'motrpac_targets.parquet')
    )

    cov.reset_index(drop=True).to_parquet(os.path.join(out, 'motrpac_covariates.parquet'))

    # 8) Create metadata
    target_stats = {
        'responder15': {
            'pos': int((responder15 == 1).sum()),
            'neg': int((responder15 == 0).sum()),
        },
    }

    metadata = create_dataset_metadata(
        dataset_name='motrpac',
        download_urls=urls,
        num_samples=len(raw_data),
        num_features=raw_data.shape[1],
        target_stats=target_stats,
        preprocessing_notes=(
            'Data is log2-transformed and adjusted for covariates (age, sex, bmi, race) '
            'using linear regression. For each protein, fits protein ~ covariates, then adjusts '
            'to remove covariate effects centered at mean covariate values. This approach avoids '
            'using target labels, preventing data leakage.'
        ),
    )

    # Upload to HuggingFace
    data_files = {
        'data': os.path.join(out, 'motrpac_data.parquet'),
        'targets': os.path.join(out, 'motrpac_targets.parquet'),
    }

    upload_to_huggingface('motrpac', data_files, metadata)

    print('Successfully processed and uploaded MotrPac dataset')
    print(f'  Samples: {len(raw_data)}')
    print(f'  Features: {raw_data.shape[1]}')
    print(f'  Analytes used: {len(analyte_ids)}')
    print(f'  Target stats: {target_stats}')

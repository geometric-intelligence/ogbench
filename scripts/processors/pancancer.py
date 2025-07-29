"""PanCancer dataset processor."""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface


def process_pancancer(output_dir: str = "temp_data") -> None:
    """Download and process PanCancer dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Download URLs
    urls = {
        "proteomics": "https://ars.els-cdn.com/content/image/1-s2.0-S1535610822002744-mmc3.xlsx",
        "drug_response": "https://figshare.com/ndownloader/files/34355645",
    }

    # Download files
    for name, url in urls.items():
        file_path = os.path.join(
            output_dir, f"pancancer_{name}.{'xlsx' if name == 'proteomics' else 'csv.gz'}"
        )
        if not os.path.exists(file_path):
            print(f"Downloading {name}...")
            download_file(url, file_path)

    # Load data
    print("Loading data...")
    proteomics_df = pd.read_excel(
        os.path.join(output_dir, "pancancer_proteomics.xlsx"),
        sheet_name="Full protein matrix",
        header=1,
    )
    drug_df = pd.read_csv(
        os.path.join(output_dir, "pancancer_drug_response.csv.gz"),
        compression="gzip",
        low_memory=False,
    )

    # Filter for Avagacestat
    drug_df = drug_df[drug_df["drug_name"] == "Avagacestat"]
    drug_df = drug_df.dropna(subset=["ln_IC50"])

    # Split Project_Identifier
    proteomics_df[["model_id", "cell_line_name"]] = proteomics_df["Project_Identifier"].str.split(
        ";", expand=True
    )

    # Merge datasets
    merged_df = pd.merge(proteomics_df, drug_df, on=["model_id", "cell_line_name"], how="inner")

    # Extract features and target
    raw_data = merged_df.iloc[
        :, 2:-4
    ]  # Protein expression values (skip Project_Identifier, model_id, cell_line_name, and drug-related columns)
    targets = merged_df["ln_IC50"].values

    # Remove rows with nan values
    mask = ~pd.isna(targets)
    raw_data = raw_data[mask]
    targets = targets[mask]

    # Convert to numeric and handle non-numeric columns
    numeric_data = pd.DataFrame()
    for col in raw_data.columns:
        try:
            numeric_data[col] = pd.to_numeric(raw_data[col], errors="coerce")
        except ValueError:
            print(f"Skipping non-numeric column: {col}")
            continue

    # Impute missing values in features
    imputer = SimpleImputer(strategy="mean")
    raw_data = pd.DataFrame(
        imputer.fit_transform(numeric_data), columns=numeric_data.columns, index=numeric_data.index
    )

    assert not raw_data.isna().any().any(), "Raw data has nan values"
    assert not np.isnan(targets).any(), "Targets have nan values"

    # Save as parquet
    data_file = os.path.join(output_dir, "pancancer_data.parquet")
    targets_file = os.path.join(output_dir, "pancancer_targets.parquet")

    # Reset index to make it a proper DataFrame
    raw_data = raw_data.reset_index(drop=True)
    raw_data.to_parquet(data_file)
    pd.DataFrame({"target": targets}).to_parquet(targets_file)

    # Create metadata
    target_stats = {
        "mean": float(np.mean(targets)),
        "std": float(np.std(targets)),
        "min": float(np.min(targets)),
        "max": float(np.max(targets)),
    }

    metadata = create_dataset_metadata(
        dataset_name="pancancer",
        download_urls=urls,
        num_samples=len(targets),
        num_features=raw_data.shape[1],
        target_stats=target_stats,
    )

    # Upload to HuggingFace
    data_files = {"data": data_file, "targets": targets_file}

    upload_to_huggingface("pancancer", data_files, metadata)

    print("Successfully processed and uploaded PanCancer dataset")
    print(f"  Samples: {len(targets)}")
    print(f"  Features: {raw_data.shape[1]}")
    print(f"  Target stats: {target_stats}")

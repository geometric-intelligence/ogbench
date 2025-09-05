"""MotrPac dataset processor."""

import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface


def process_motrpac(output_dir: str = "temp_data") -> None:
    """Download and process MotrPac dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Download URLs
    urls = {
        "proteomics": "https://d1yw74buhe0ts0.cloudfront.net/static/motrpac-data-hub/publications/data/related-studies/heritage-proteomics/HERITAGE_proteomics_somalogic.xlsx",
        "analytes": "https://d1yw74buhe0ts0.cloudfront.net/static/motrpac-data-hub/publications/data/related-studies/heritage-proteomics/HERITAGE_somalogic_analytes.xlsx",
    }

    # Download files
    for name, url in urls.items():
        file_path = os.path.join(output_dir, f"motrpac_{name}.xlsx")
        if not os.path.exists(file_path):
            print(f"Downloading {name}...")
            download_file(url, file_path)

    # Load data
    proteomics_df = pd.read_excel(os.path.join(output_dir, "motrpac_proteomics.xlsx"), header=3)
    _ = pd.read_excel(os.path.join(output_dir, "motrpac_analytes.xlsx"))

    # Extract features and target
    raw_data = proteomics_df.iloc[:, 9:]  # Protein expression values
    targets = proteomics_df["Delta VO2MX (ml/min)"].values

    # Remove rows with nan values
    mask = ~pd.isna(targets)
    raw_data = raw_data[mask]
    targets = targets[mask]

    # Impute missing values in features
    imputer = SimpleImputer(strategy="mean")
    raw_data = pd.DataFrame(
        imputer.fit_transform(raw_data),
        columns=raw_data.columns,
        index=raw_data.index,
    )

    assert not raw_data.isna().any().any(), "Raw data has nan values"
    assert not np.isnan(targets).any(), "Targets have nan values"

    # Save as parquet
    data_file = os.path.join(output_dir, "motrpac_data.parquet")
    targets_file = os.path.join(output_dir, "motrpac_targets.parquet")

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
        dataset_name="motrpac",
        download_urls=urls,
        num_samples=len(targets),
        num_features=raw_data.shape[1],
        target_stats=target_stats,
    )

    # Upload to HuggingFace
    data_files = {"data": data_file, "targets": targets_file}

    upload_to_huggingface("motrpac", data_files, metadata)

    print("Successfully processed and uploaded MotrPac dataset")
    print(f"  Samples: {len(targets)}")
    print(f"  Features: {raw_data.shape[1]}")
    print(f"  Target stats: {target_stats}")

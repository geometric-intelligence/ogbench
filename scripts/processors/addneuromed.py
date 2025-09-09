"""AddNeuroMed dataset processor."""

import gzip
import os

import numpy as np
import pandas as pd

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface


def process_addneuromed(output_dir: str = "temp_data") -> None:
    """Download and process AddNeuroMed dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Download URLs
    urls = {
        "GPL10558": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63063/matrix/GSE63063-GPL10558_series_matrix.txt.gz",
        "GPL6947": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63063/matrix/GSE63063-GPL6947_series_matrix.txt.gz",
    }

    raw_data: pd.DataFrame = pd.DataFrame()
    frames: list[pd.DataFrame] = []
    statuses: list[str] = []

    for dataset, url in urls.items():
        gz_path: str = os.path.join(output_dir, f"{dataset}.txt.gz")
        if not os.path.exists(gz_path):
            print(f"Downloading {dataset}...")
            try:
                download_file(url, gz_path)
                print(f"Successfully downloaded {dataset}")
            except Exception as e:
                print(f"Error downloading {dataset}: {str(e)}")
                raise

        # Read microarray data
        with gzip.open(gz_path, "rt") as f:
            data = pd.read_csv(f, sep="\t", comment="!", index_col="ID_REF").transpose()

        # Extract ages from the data
        with gzip.open(gz_path, "rt") as f:
            for line in f:
                if line.startswith("!Sample_characteristics_ch1") and "status:" in line:
                    statuses.extend(
                        [x.split(": ")[1].strip().strip('"') for x in line.split("\t")[1:]]
                    )
                    break

        frames.append(data)

    # Verify no common patients between datasets
    common_patients = set(frames[0].index).intersection(set(frames[1].index))
    assert len(common_patients) == 0, "Common patients found between the two datasets"

    # Find and use common genes
    common_genes = list(set(frames[0].columns).intersection(set(frames[1].columns)))
    frames[0] = frames[0][common_genes]
    frames[1] = frames[1][common_genes]

    # Combine datasets
    raw_data = pd.concat(frames, axis=0)
    targets = np.array(statuses)

    # Remove rows with nan values
    mask = (~pd.isna(raw_data)).any(axis=1).values & ~pd.isna(targets)
    raw_data = raw_data[mask]
    targets = targets[mask]

    # Raise if raw data or targets have nan values
    assert not raw_data.isna().any().any(), "Raw data has nan values"
    assert not (targets == "").any(), "Targets have empty strings"

    # Save as parquet
    data_file = os.path.join(output_dir, "addneuromed_data.parquet")
    targets_file = os.path.join(output_dir, "addneuromed_targets.parquet")

    # Reset index to make it a proper DataFrame
    raw_data = raw_data.reset_index(drop=True)
    raw_data.to_parquet(data_file)
    pd.DataFrame({"target": targets}).to_parquet(targets_file)

    # Create metadata
    target_stats = {}

    metadata = create_dataset_metadata(
        dataset_name="addneuromed",
        download_urls=urls,
        num_samples=len(targets),
        num_features=raw_data.shape[1],
        target_stats=target_stats,
    )

    # Upload to HuggingFace
    data_files = {"data": data_file, "targets": targets_file}

    upload_to_huggingface("addneuromed", data_files, metadata)

    print("Successfully processed and uploaded AddNeuroMed dataset")
    print(f"  Samples: {len(targets)}")
    print(f"  Features: {raw_data.shape[1]}")
    print(f"  Target stats: {target_stats}")

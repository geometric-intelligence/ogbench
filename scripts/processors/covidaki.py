"""CovidAKI dataset processor."""


import gzip
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface


def process_covidaki(output_dir: str = "temp_data") -> None:
    """Download and process CovidAKI dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Download URLs
    urls = {
        "GSE225349_metadata": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE225nnn/GSE225349/matrix/GSE225349_series_matrix.txt.gz",
        "GSE225349_covid_AKI": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE225nnn/GSE225349/suppl/GSE225349_non-normalized_counts_covid_AKI.txt.gz",
        "GSE225349_covid_only": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE225nnn/GSE225349/suppl/GSE225349_non-normalized_counts_covid_only.txt.gz",
    }

    raw_data: pd.DataFrame = pd.DataFrame()
    frames: list[pd.DataFrame] = []
    covid_status: list[int] = []
    sample_ids: list[str] = []

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

        if dataset == "GSE225349_metadata":
            with gzip.open(gz_path, "rt") as f:
                for line in f:
                    if (
                        line.startswith("!Sample_characteristics_ch1")
                        and "blood sample id" in line.lower()
                    ):
                        sample_ids.extend(
                            [x.split(": ")[1].strip().strip('"') for x in line.split("\t")[1:]]
                        )
                    if (
                        line.startswith("!Sample_characteristics_ch1")
                        and "covid-19 status:" in line
                    ):
                        covid_status.extend(
                            (x.split(": ")[1].strip().strip('"')) for x in line.split("\t")[1:]
                        )
                        break
            assert len(sample_ids) == len(covid_status), "Mismatch in metadata lengths"

            meta_df = pd.DataFrame(
                {
                    "blood_sample_id": sample_ids,
                    "covid_status": [1 if s == "Positive" else 0 for s in covid_status],
                }
            ).set_index("blood_sample_id")
        else:
            with gzip.open(gz_path, "rt") as f:
                data = pd.read_csv(
                    f, sep="\t", comment="!", index_col="ID REF"
                ).transpose()  # index_col=0 because one is ID_REF and the other is ID REF
            frames.append(data)

    # Remove overlapping patients
    common_patients = set(frames[0].index).intersection(set(frames[1].index))
    if common_patients:
        print(f"Common patients found: {common_patients}")
        # keep common patients data from first dataset
        frames[1] = frames[1].drop(index=common_patients)

    # Find and use common genes
    common_genes = list(set(frames[0].columns).intersection(set(frames[1].columns)))
    frames[0] = frames[0][common_genes]
    frames[1] = frames[1][common_genes]

    # Combine datasets
    raw_data = pd.concat(frames, axis=0)
    raw_data.columns = raw_data.columns.astype(str)
    shared_ids = raw_data.index.intersection(meta_df.index)
    raw_data = raw_data.loc[shared_ids]
    targets = meta_df.loc[shared_ids, "covid_status"].values

    # Remove rows with nan values
    mask = (~pd.isna(raw_data)).any(axis=1).values & ~pd.isna(targets)
    raw_data = raw_data[mask]
    targets = targets[mask]

    # Raise if raw data or targets have nan values
    assert not raw_data.isna().any().any(), "Raw data has nan values"
    assert not np.isnan(targets).any(), "Targets have nan values"

    # Save as parquet
    data_file = os.path.join(output_dir, "covidaki_data.parquet")
    targets_file = os.path.join(output_dir, "covidaki_targets.parquet")

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
        dataset_name="covidaki",
        download_urls=urls,
        num_samples=len(targets),
        num_features=raw_data.shape[1],
        target_stats=target_stats,
    )

    # Upload to HuggingFace
    data_files = {"data": data_file, "targets": targets_file}

    upload_to_huggingface("covidaki", data_files, metadata)

    print("Successfully processed and uploaded CovidAKI dataset")
    print(f"  Samples: {len(targets)}")
    print(f"  Features: {raw_data.shape[1]}")
    print(f"  Target stats: {target_stats}")

"""Parkinsons dataset processor."""

import os
import pandas as pd
import numpy as np
import gzip
import urllib.request
from typing import Dict, Tuple

from scripts.utils import download_file, create_dataset_metadata, upload_to_huggingface


def _map_probes_to_genes(df: pd.DataFrame, data_dir: str, collapse: bool = True) -> pd.DataFrame:
    """Map Affymetrix probe IDs to gene symbols using GPL570 annotation."""
    url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"
    annot_path = os.path.join(data_dir, "GPL570.annot.gz")

    if not os.path.exists(annot_path):
        print("Downloading GPL570 annotation...")
        urllib.request.urlretrieve(url, annot_path)

    # Skip to the data portion
    with gzip.open(annot_path, "rt") as f:
        lines = f.readlines()

    start = next(i for i, line in enumerate(lines) if line.startswith("ID\t"))
    from io import StringIO
    gpl = pd.read_csv(StringIO("".join(lines[start:])), sep="\t", dtype=str, low_memory=False)

    # Try to identify the gene symbol column flexibly
    gene_symbol_col = next((col for col in gpl.columns if "gene symbol" in col.lower()), None)
    if gene_symbol_col is None:
        raise ValueError("No column with gene symbols found in GPL annotation file.")

    print(f"Using gene symbol column: '{gene_symbol_col}'")

    probe_map = gpl[["ID", gene_symbol_col]].dropna()
    probe_map[gene_symbol_col] = probe_map[gene_symbol_col].str.split("///").str[0].str.strip()
    probe_map = probe_map[probe_map[gene_symbol_col] != ""].set_index("ID")

    df.columns = df.columns.astype(str)
    common_probes = df.columns.intersection(probe_map.index)
    df = df[common_probes]
    df.columns = probe_map.loc[common_probes, gene_symbol_col].values

    if collapse:
        df = df.groupby(df.columns, axis=1).mean()

    return df


def process_parkinsons(output_dir: str = "temp_data") -> None:
    """Download and process Parkinsons dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Download URL
    urls = {
        'GSE99039': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE99nnn/GSE99039/matrix/GSE99039_series_matrix.txt.gz'
    }
    
    dataset = "GSE99039_series_matrix"
    gz_path = os.path.join(output_dir, f"{dataset}.txt.gz")

    if not os.path.exists(gz_path):
        print(f"Downloading {dataset}...")
        try:
            download_file(urls['GSE99039'], gz_path)
            print(f"Successfully downloaded {dataset}")
        except Exception as e:
            print(f"Error downloading {dataset}: {str(e)}")
            raise

    # Read in metadata lines
    metadata_lines = []
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            if line.startswith("!Sample_characteristics_ch1"):
                metadata_lines.append(line.strip().split("\t")[1:])  # skip label

    if not metadata_lines:
        raise ValueError("No !Sample_characteristics_ch1 lines found.")

    # Transpose so each item corresponds to a sample
    sample_metadata = list(zip(*metadata_lines))

    # Extract 'moca score' for each sample
    moca_scores = []
    for fields in sample_metadata:
        moca = None
        for field in fields:
            if "moca score:" in field.lower():
                try:
                    moca = field.split(":")[1].strip().strip('"')
                except IndexError:
                    pass
        moca_scores.append(moca)

    moca_scores = pd.to_numeric(moca_scores, errors="coerce")

    # Load gene expression data (after metadata ends)
    with gzip.open(gz_path, "rt") as f:
        expression_df = pd.read_csv(f, sep="\t", comment="!", index_col="ID_REF").transpose()

    # Filter and map to gene symbols
    expression_df = _map_probes_to_genes(expression_df, output_dir, collapse=True)

    # Match metadata with expression samples
    assert len(moca_scores) == expression_df.shape[0], (
        f"Mismatched samples: {len(moca_scores)} scores vs {expression_df.shape[0]} samples"
    )

    valid_mask = ~np.isnan(moca_scores)
    raw_data = expression_df.loc[valid_mask]
    targets = moca_scores[valid_mask]

    assert not raw_data.isna().any().any(), "Raw data contains NaNs"
    assert not np.isnan(targets).any(), "Targets contain NaNs"
    
    # Save as parquet
    data_file = os.path.join(output_dir, "parkinsons_data.parquet")
    targets_file = os.path.join(output_dir, "parkinsons_targets.parquet")
    
    # Reset index to make it a proper DataFrame
    raw_data = raw_data.reset_index(drop=True)
    raw_data.to_parquet(data_file)
    pd.DataFrame({"target": targets}).to_parquet(targets_file)
    
    # Create metadata
    target_stats = {
        "mean": float(np.mean(targets)),
        "std": float(np.std(targets)),
        "min": float(np.min(targets)),
        "max": float(np.max(targets))
    }
    
    metadata = create_dataset_metadata(
        dataset_name="parkinsons",
        download_urls=urls,
        num_samples=len(targets),
        num_features=raw_data.shape[1],
        target_stats=target_stats
    )
    
    # Upload to HuggingFace
    data_files = {
        "data": data_file,
        "targets": targets_file
    }
    
    upload_to_huggingface("parkinsons", data_files, metadata)
    
    print(f"Successfully processed and uploaded Parkinsons dataset")
    print(f"  Samples: {len(targets)}")
    print(f"  Features: {raw_data.shape[1]}")
    print(f"  Target stats: {target_stats}") 
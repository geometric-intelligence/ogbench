"""AddNeuroMed dataset processor."""


import gzip
import os

import numpy as np
import pandas as pd
import requests
from combat.pycombat import pycombat
from tqdm import tqdm

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface


def download_platform_file(url: str) -> str:
    """Download platform file from URL and return content."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_gpl6947(content: str) -> pd.DataFrame:
    """Parse GPL6947 platform file content."""
    lines = content.strip().split('\n')

    # Find the header line (starts with ID)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('ID\t'):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError('Could not find header line in GPL6947 file')

    # Extract header and data
    header_line = lines[header_idx]
    data_lines = lines[header_idx + 1 :]

    # Create DataFrame
    data = []
    for line in data_lines:
        if line.strip():  # Skip empty lines
            data.append(line.split('\t'))

    df = pd.DataFrame(data, columns=header_line.split('\t'))

    # Select relevant columns
    relevant_cols = ['ID', 'ILMN_Gene', 'RefSeq_ID', 'Entrez_Gene_ID']
    available_cols = [col for col in relevant_cols if col in df.columns]

    return df[available_cols]


def parse_gpl10558(content: str) -> pd.DataFrame:
    """Parse GPL10558 platform file content."""
    lines = content.strip().split('\n')

    # Find the header line by looking for ILMN_Gene column
    header_idx = None
    for i, line in enumerate(lines):
        if 'ILMN_Gene' in line and '\t' in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError('Could not find header line with ILMN_Gene in GPL10558 file')

    # Extract header and data
    header_line = lines[header_idx]
    data_lines = lines[header_idx + 1 :]

    # Create DataFrame
    data = []
    for line in data_lines:
        if line.strip():  # Skip empty lines
            data.append(line.split('\t'))

    df = pd.DataFrame(data, columns=header_line.split('\t'))

    # Select relevant columns - GPL10558 has different column names
    relevant_cols = ['ID', 'ILMN_Gene', 'RefSeq_ID', 'Entrez_Gene_ID']
    available_cols = [col for col in relevant_cols if col in df.columns]

    return df[available_cols]


def create_gene_probe_mapping() -> dict[str, set[str]]:
    """Create mapping of genes to sets of probes from both platform files."""

    # URLs for the platform files
    gpl6947_url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?mode=raw&is_datatable=true&acc=GPL6947&id=13512&db=GeoDb_blob107'
    gpl10558_url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?mode=raw&is_datatable=true&acc=GPL10558&id=50081&db=GeoDb_blob135'

    print('Downloading GPL6947 platform file...')
    gpl6947_content = download_platform_file(gpl6947_url)
    gpl6947_df = parse_gpl6947(gpl6947_content)
    print(f'GPL6947: {len(gpl6947_df)} probes')

    print('Downloading GPL10558 platform file...')
    gpl10558_content = download_platform_file(gpl10558_url)
    gpl10558_df = parse_gpl10558(gpl10558_content)
    print(f'GPL10558: {len(gpl10558_df)} probes')

    gpl6947_gene_col = 'Entrez_Gene_ID' if 'Entrez_Gene_ID' in gpl6947_df.columns else 'ILMN_Gene'
    gpl10558_gene_col = (
        'Entrez_Gene_ID' if 'Entrez_Gene_ID' in gpl10558_df.columns else 'ILMN_Gene'
    )

    gpl6947_genes = set(gpl6947_df[gpl6947_gene_col].dropna().unique())
    gpl10558_genes = set(gpl10558_df[gpl10558_gene_col].dropna().unique())

    print(f'GPL6947 unique genes: {len(gpl6947_genes)}')
    print(f'GPL10558 unique genes: {len(gpl10558_genes)}')

    # Find intersection of genes
    common_genes = gpl6947_genes.intersection(gpl10558_genes)
    print(f'Common genes: {len(common_genes)}')

    # Create mapping: gene symbol -> set of probe_ids
    gene_probe_mapping: dict[str, set[str]] = {}

    # Add probes from GPL6947
    for _, row in gpl6947_df.iterrows():
        symbol = row[gpl6947_gene_col]
        probe_id = row['ID']
        if pd.notna(symbol) and symbol in common_genes:
            if symbol not in gene_probe_mapping:
                gene_probe_mapping[symbol] = set()
            gene_probe_mapping[symbol].add(probe_id)

    # Add probes from GPL10558
    for _, row in gpl10558_df.iterrows():
        symbol = row[gpl10558_gene_col]
        probe_id = row['ID']
        if pd.notna(symbol) and symbol in common_genes:
            if symbol not in gene_probe_mapping:
                gene_probe_mapping[symbol] = set()
            gene_probe_mapping[symbol].add(probe_id)

    return gene_probe_mapping


def process_addneuromed(output_dir: str = 'temp_data') -> None:
    """Download and process AddNeuroMed dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Download URLs
    urls = {
        'GPL10558': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63063/matrix/GSE63063-GPL10558_series_matrix.txt.gz',
        'GPL6947': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63063/matrix/GSE63063-GPL6947_series_matrix.txt.gz',
    }

    # Create gene-probe mapping
    print('Creating gene-probe mapping...')
    gene_probe_mapping = create_gene_probe_mapping()

    raw_data: pd.DataFrame = pd.DataFrame()
    frames: list[pd.DataFrame] = []
    statuses: list[str] = []
    batches: list[str] = []

    for dataset, url in urls.items():
        gz_path: str = os.path.join(output_dir, f'{dataset}.txt.gz')
        if not os.path.exists(gz_path):
            print(f'Downloading {dataset}...')
            try:
                download_file(url, gz_path)
                print(f'Successfully downloaded {dataset}')
            except Exception as e:
                print(f'Error downloading {dataset}: {str(e)}')
                raise

        # Read microarray data
        with gzip.open(gz_path, 'rt') as f:
            data = pd.read_csv(f, sep='\t', comment='!', index_col='ID_REF').transpose()

        # Extract statuses from the data
        dataset_statuses: list[str] = []
        with gzip.open(gz_path, 'rt') as f:
            for line in f:
                if line.startswith('!Sample_characteristics_ch1') and 'status:' in line:
                    dataset_statuses.extend(
                        [x.split(': ')[1].strip().strip('"') for x in line.split('\t')[1:]]
                    )
                    break

        # Track batch labels for this dataset
        dataset_batches = [dataset] * len(data.index)
        batches.extend(dataset_batches)
        statuses.extend(dataset_statuses)

        # Convert probe-level data to gene-level data
        print(f'Converting {dataset} from probe-level to gene-level data...')

        # Build gene expression data more efficiently
        gene_columns = {}
        for gene_symbol, probe_ids in tqdm(gene_probe_mapping.items(), desc='Processing genes'):
            # Find probes that exist in this dataset
            available_probes = [probe for probe in probe_ids if probe in data.columns]

            if available_probes:
                # Average expression across all probes for this gene
                gene_columns[gene_symbol] = data[available_probes].mean(axis=1)
        # Create DataFrame from all gene columns at once
        gene_data = pd.DataFrame(gene_columns, index=data.index)
        frames.append(gene_data)

    # Verify no common patients between datasets
    common_patients = set(frames[0].index).intersection(set(frames[1].index))
    assert len(common_patients) == 0, 'Common patients found between the two datasets'

    # Find and use common genes
    common_genes = list(set(frames[0].columns).intersection(set(frames[1].columns)))
    frames[0] = frames[0][common_genes]
    frames[1] = frames[1][common_genes]

    # Combine datasets
    raw_data = pd.concat(frames, axis=0)
    targets = np.array(statuses)
    batch_labels = np.array(batches)

    # Define classes to remove
    classes_to_remove = {'CTL to AD', 'MCI to CTL', 'OTHER', 'borderline MCI'}

    # Filter out unwanted classes
    mask = (
        (~pd.isna(raw_data)).any(axis=1).values
        & ~pd.isna(targets)
        & ~np.isin(targets, list(classes_to_remove))
    )
    raw_data = raw_data[mask]
    targets = targets[mask]
    batch_labels = batch_labels[mask]

    gene_map = pd.DataFrame(
        {
            'node_id': list(raw_data.columns),
            'string_id': list(raw_data.columns),  # Entrez IDs map directly to STRING
        }
    )
    gene_map['node_id'] = gene_map['node_id'].astype(str)
    gene_map['string_id'] = gene_map['string_id'].astype(str)
    print(f'Mapping: {len(gene_map)} genes with Entrez IDs for STRING')

    # Apply ComBat batch correction
    print('Applying ComBat batch correction...')

    # Create design matrix for status to preserve biological signal
    unique_statuses = np.unique(targets)
    design_matrix = pd.DataFrame(index=raw_data.index)
    for status in unique_statuses:
        design_matrix[f'status_{status}'] = (targets == status).astype(int)

    # Apply ComBat correction
    # pycombat expects genes x samples, so we transpose raw_data
    raw_data_transposed = raw_data.T  # Transpose to genes x samples

    # Convert batch labels to numeric indices for pycombat
    unique_batches = np.unique(batch_labels)
    batch_to_idx = {batch: i for i, batch in enumerate(unique_batches)}
    batch_indices = np.array([batch_to_idx[batch] for batch in batch_labels])

    # Apply ComBat correction without covariates for now
    # This will correct batch effects while preserving overall biological signal
    corrected_data = pycombat(
        raw_data_transposed,  # Pass DataFrame, not numpy array
        batch_indices,
    )

    # Convert back to DataFrame (transpose back to samples x genes)
    raw_data = corrected_data.T  # Transpose back to samples x features

    # Raise if raw data or targets have nan values
    assert not raw_data.isna().any().any(), 'Raw data has nan values'
    assert not (targets == '').any(), 'Targets have empty strings'

    # Convert string classes to integers
    unique_classes = np.unique(targets)
    class_to_int = {class_name: i for i, class_name in enumerate(unique_classes)}
    targets_int = np.array([class_to_int[class_name] for class_name in targets])

    # Save as parquet
    data_file = os.path.join(output_dir, 'addneuromed_data.parquet')
    targets_file = os.path.join(output_dir, 'addneuromed_targets.parquet')
    map_file = os.path.join(output_dir, 'addneuromed_map.parquet')

    # Reset index to make it a proper DataFrame
    raw_data = raw_data.reset_index(drop=True)
    raw_data.to_parquet(data_file)
    pd.DataFrame({'target': targets_int}).to_parquet(targets_file)
    gene_map.reset_index(drop=True).to_parquet(map_file, index=False)

    # Create metadata
    target_stats = {
        'class_mapping': class_to_int,
        'num_classes': len(unique_classes),
        'class_names': list(unique_classes),
    }

    metadata = create_dataset_metadata(
        dataset_name='addneuromed',
        download_urls=urls,
        num_samples=len(targets_int),
        num_features=raw_data.shape[1],
        target_stats=target_stats,
    )

    # Upload to HuggingFace
    data_files = {'data': data_file, 'targets': targets_file, 'map': map_file}

    upload_to_huggingface('addneuromed', data_files, metadata)

    print('Successfully processed and uploaded AddNeuroMed dataset')
    print(f'  Samples: {len(targets_int)}')
    print(f'  Features: {raw_data.shape[1]}')
    print(f'  Target stats: {target_stats}')

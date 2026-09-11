"""AddNeuroMed dataset processor."""


import gzip
import os
from io import StringIO

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface


def _load_illumina_probes(url: str, dest_path: str) -> pd.DataFrame:
    """Download an Illumina probe table (bgx/txt) and return probe / Entrez IDs."""
    if not os.path.exists(dest_path):
        print(f'Downloading {os.path.basename(dest_path)}...')
        download_file(url, dest_path)
    with gzip.open(dest_path, 'rt', encoding='utf-8', errors='replace') as handle:
        lines = handle.readlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith('[Probes]')), None)
    if header_idx is None:
        raise ValueError(f'Could not find [Probes] section in {dest_path}')
    df = pd.read_csv(
        StringIO(''.join(lines[header_idx + 1 :])),
        sep='\t',
        dtype=str,
        low_memory=False,
    )
    if 'Probe_Id' not in df.columns or 'Entrez_Gene_ID' not in df.columns:
        raise ValueError(f'Missing Probe_Id or Entrez_Gene_ID in {dest_path}')
    df = df.rename(columns={'Probe_Id': 'ID', 'Entrez_Gene_ID': 'gene_id'})
    df['gene_id'] = df['gene_id'].replace('', np.nan)
    return df[['ID', 'gene_id']].dropna(subset=['ID'])


def create_gene_probe_mapping(output_dir: str) -> dict[str, set[str]]:
    """Create mapping of Entrez genes to probes from both Illumina platforms."""
    gpl6947_df = _load_illumina_probes(
        'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6947/suppl/'
        'GPL6947_HumanHT-12_V3_0_R1_11283641_A.bgx.gz',
        os.path.join(output_dir, 'GPL6947_HumanHT-12_V3.bgx.gz'),
    )
    gpl10558_df = _load_illumina_probes(
        'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/GPL10558/suppl/'
        'GPL10558_HumanHT-12_V4_0_R2_15002873_B.txt.gz',
        os.path.join(output_dir, 'GPL10558_HumanHT-12_V4.txt.gz'),
    )
    print(f'GPL6947: {len(gpl6947_df)} probes')
    print(f'GPL10558: {len(gpl10558_df)} probes')

    gpl6947_genes = set(gpl6947_df['gene_id'].dropna().unique())
    gpl10558_genes = set(gpl10558_df['gene_id'].dropna().unique())

    print(f'GPL6947 unique genes: {len(gpl6947_genes)}')
    print(f'GPL10558 unique genes: {len(gpl10558_genes)}')

    common_genes = gpl6947_genes.intersection(gpl10558_genes)
    print(f'Common genes: {len(common_genes)}')

    gene_probe_mapping: dict[str, set[str]] = {}
    for frame in (gpl6947_df, gpl10558_df):
        for probe_id, gene_id in zip(frame['ID'], frame['gene_id'], strict=True):
            if pd.notna(gene_id) and gene_id in common_genes:
                gene_probe_mapping.setdefault(gene_id, set()).add(probe_id)
    return gene_probe_mapping


def process_addneuromed(output_dir: str = 'temp_data') -> None:
    """Download and process AddNeuroMed dataset."""
    os.makedirs(output_dir, exist_ok=True)

    urls = {
        'GPL10558': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63063/matrix/GSE63063-GPL10558_series_matrix.txt.gz',
        'GPL6947': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63063/matrix/GSE63063-GPL6947_series_matrix.txt.gz',
    }

    print('Creating gene-probe mapping...')
    gene_probe_mapping = create_gene_probe_mapping(output_dir)

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

        with gzip.open(gz_path, 'rt') as f:
            data = pd.read_csv(f, sep='\t', comment='!', index_col='ID_REF').transpose()

        dataset_statuses: list[str] = []
        with gzip.open(gz_path, 'rt') as f:
            for line in f:
                if line.startswith('!Sample_characteristics_ch1') and 'status:' in line:
                    dataset_statuses.extend(
                        [x.split(': ')[1].strip().strip('"') for x in line.split('\t')[1:]]
                    )
                    break

        dataset_batches = [dataset] * len(data.index)
        batches.extend(dataset_batches)
        statuses.extend(dataset_statuses)

        print(f'Converting {dataset} from probe-level to gene-level data...')
        values = data.to_numpy(dtype=np.float64, copy=False)
        col_index = {probe: i for i, probe in enumerate(data.columns)}
        gene_columns = {}
        for gene_symbol, probe_ids in tqdm(gene_probe_mapping.items(), desc='Processing genes'):
            idxs = [col_index[probe] for probe in probe_ids if probe in col_index]
            if idxs:
                gene_columns[gene_symbol] = values[:, idxs].mean(axis=1)
        gene_data = pd.DataFrame(gene_columns, index=data.index)
        frames.append(gene_data)

    common_patients = set(frames[0].index).intersection(set(frames[1].index))
    assert len(common_patients) == 0, 'Common patients found between the two datasets'

    common_genes = list(set(frames[0].columns).intersection(set(frames[1].columns)))
    frames[0] = frames[0][common_genes]
    frames[1] = frames[1][common_genes]

    raw_data = pd.concat(frames, axis=0)
    targets = np.array(statuses)
    batch_labels = np.array(batches)

    classes_to_remove = {'CTL to AD', 'MCI to CTL', 'OTHER', 'borderline MCI'}
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
            'string_id': list(raw_data.columns),
        }
    )
    gene_map['node_id'] = gene_map['node_id'].astype(str)
    gene_map['string_id'] = gene_map['string_id'].astype(str)
    print(f'Mapping: {len(gene_map)} genes with Entrez IDs for STRING')
    print(
        'Skipping full-data ComBat; platform batch labels are saved for train-only '
        'correction in HFOmics. Unique batches:',
        np.unique(batch_labels),
    )

    assert not raw_data.isna().any().any(), 'Raw data has nan values'
    assert not (targets == '').any(), 'Targets have empty strings'

    unique_classes = [str(name) for name in np.unique(targets)]
    class_to_int = {class_name: i for i, class_name in enumerate(unique_classes)}
    targets_int = np.array([class_to_int[str(class_name)] for class_name in targets])

    data_file = os.path.join(output_dir, 'addneuromed_data.parquet')
    targets_file = os.path.join(output_dir, 'addneuromed_targets.parquet')
    map_file = os.path.join(output_dir, 'addneuromed_map.parquet')
    batches_file = os.path.join(output_dir, 'addneuromed_batches.parquet')

    raw_data = raw_data.reset_index(drop=True)
    raw_data.to_parquet(data_file)
    pd.DataFrame({'target': targets_int}).to_parquet(targets_file)
    gene_map.reset_index(drop=True).to_parquet(map_file, index=False)
    pd.DataFrame({'batch': batch_labels}).to_parquet(batches_file, index=False)

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
        preprocessing_notes=(
            'Platforms GPL10558 and GPL6947 are concatenated without ComBat at ingest. '
            'ogbench fits ComBat on the training split only and applies frozen batch '
            'location/scale to val/test. Labels are not used as ComBat covariates.'
        ),
    )

    data_files = {
        'data': data_file,
        'targets': targets_file,
        'map': map_file,
        'batches': batches_file,
    }

    upload_to_huggingface('addneuromed', data_files, metadata)

    print('Successfully processed and uploaded AddNeuroMed dataset')
    print(f'  Samples: {len(targets_int)}')
    print(f'  Features: {raw_data.shape[1]}')
    print(f'  Target stats: {target_stats}')

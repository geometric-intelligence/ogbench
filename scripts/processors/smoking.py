"""Smoking dataset processor (GSE50660 Illumina 450k methylation)."""

import gzip
import os
from io import StringIO

import numpy as np
import pandas as pd

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface


PROMOTER_REGIONS = {'TSS1500', 'TSS200'}


def _build_probe_gene_mapping(manifest_path: str) -> pd.DataFrame:
    """Parse the Illumina HumanMethylation450 manifest and return TSS1500/TSS200 probe-gene rows.

    The manifest is a CSV with an Illumina header block, an `[Assay]` section that holds the
    probe table (its header line starts with `IlmnID,`), and a trailing `[Controls]` section
    that must be skipped because its column layout differs from the assay table.

    Returns:
        DataFrame with columns ['probe_id', 'gene', 'region'], one row per (probe, gene) pair
        whose annotation falls in TSS1500 or TSS200. Pairs are de-duplicated.
    """
    collected: list[str] = []
    started = False
    with open(manifest_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not started:
                if line.startswith('IlmnID,'):
                    started = True
                    collected.append(line)
                continue
            if line.startswith('[Controls]'):
                break
            collected.append(line)

    if not collected:
        raise ValueError(f'Could not find assay header (IlmnID,...) in manifest: {manifest_path}')

    manifest = pd.read_csv(
        StringIO(''.join(collected)),
        dtype=str,
        low_memory=False,
        usecols=['IlmnID', 'UCSC_RefGene_Name', 'UCSC_RefGene_Group'],
    )

    manifest = manifest.dropna(subset=['UCSC_RefGene_Name', 'UCSC_RefGene_Group'])

    genes = manifest['UCSC_RefGene_Name'].str.split(';')
    regions = manifest['UCSC_RefGene_Group'].str.split(';')

    probe_ids: list[str] = []
    gene_names: list[str] = []
    region_names: list[str] = []
    for probe_id, gene_list, region_list in zip(
        manifest['IlmnID'].values, genes.values, regions.values, strict=True
    ):
        if gene_list is None or region_list is None:
            continue
        # Pair gene and region by position; ignore mismatched lengths defensively
        for gene, region in zip(gene_list, region_list, strict=False):
            gene = gene.strip()
            region = region.strip()
            if gene and region in PROMOTER_REGIONS:
                probe_ids.append(probe_id)
                gene_names.append(gene)
                region_names.append(region)

    mapping = pd.DataFrame(
        {'probe_id': probe_ids, 'gene': gene_names, 'region': region_names}
    ).drop_duplicates(subset=['probe_id', 'gene'])

    print(
        f'Manifest: {len(mapping)} (probe, gene) promoter pairs '
        f'covering {mapping["gene"].nunique()} genes from {mapping["probe_id"].nunique()} probes'
    )
    return mapping


def _parse_series_matrix(gz_path: str) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Parse GSE50660 series matrix.

    Returns:
        sample_ids: list of GSM sample accessions (column order of the beta matrix).
        targets: int array of smoking status (0=never, 1=former, 2=current).
        beta: DataFrame of shape (n_samples, n_probes) indexed by GSM accession.
    """
    sample_ids: list[str] = []
    smoking_values: list[int] = []

    with gzip.open(gz_path, 'rt') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [x.strip().strip('"') for x in line.rstrip('\n').split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1') and 'smoking (0, 1 and 2' in line:
                for token in line.rstrip('\n').split('\t')[1:]:
                    cleaned = token.strip().strip('"')
                    # Format: "smoking (0, 1 and 2, which represent ...): <int>"
                    value = cleaned.split(':')[-1].strip()
                    smoking_values.append(int(value))
            elif line.startswith('!series_matrix_table_begin'):
                break

    if not sample_ids:
        raise ValueError('Could not find !Sample_geo_accession line in series matrix')
    if not smoking_values:
        raise ValueError('Could not find smoking !Sample_characteristics_ch1 line in series matrix')
    if len(sample_ids) != len(smoking_values):
        raise ValueError(
            f'Sample count mismatch: {len(sample_ids)} GSM ids vs {len(smoking_values)} smoking values'
        )

    with gzip.open(gz_path, 'rt') as f:
        beta = pd.read_csv(f, sep='\t', comment='!', index_col='ID_REF').transpose()

    # Align beta row order with sample_ids order from !Sample_geo_accession
    beta.index = beta.index.astype(str).str.strip().str.strip('"')
    missing = [sid for sid in sample_ids if sid not in beta.index]
    if missing:
        raise ValueError(f'{len(missing)} GSM ids missing from beta matrix (e.g. {missing[:3]})')
    beta = beta.loc[sample_ids]

    targets = np.array(smoking_values, dtype=np.int64)
    return sample_ids, targets, beta


def _select_min_promoter_per_gene(
    beta: pd.DataFrame, mapping: pd.DataFrame, never_mask: np.ndarray
) -> tuple[pd.DataFrame, pd.Series]:
    """For each gene pick the candidate probe with the minimum mean beta in never-smokers.

    Returns:
        gene_data: DataFrame (samples x genes) of selected probe betas, columns renamed to genes.
        gene_to_probe: Series mapping gene -> chosen probe_id.
    """
    available = mapping[mapping['probe_id'].isin(beta.columns)].copy()
    if available.empty:
        raise ValueError('No manifest probes overlap with the beta matrix columns')

    candidate_probes = available['probe_id'].unique().tolist()
    never_means = beta.loc[never_mask, candidate_probes].mean(axis=0)
    available['never_mean'] = available['probe_id'].map(never_means)

    available = available.dropna(subset=['never_mean'])
    if available.empty:
        raise ValueError('All candidate probes have NaN mean across never-smoker samples')

    idx_min = available.groupby('gene')['never_mean'].idxmin()
    chosen = available.loc[idx_min, ['gene', 'probe_id']]
    gene_to_probe = pd.Series(chosen['probe_id'].values, index=chosen['gene'].values, name='probe_id')

    gene_data = beta.loc[:, gene_to_probe.values].copy()
    gene_data.columns = gene_to_probe.index.astype(str)

    print(
        f'Selected {gene_data.shape[1]} gene-level features from '
        f'{len(candidate_probes)} candidate promoter probes'
    )
    return gene_data, gene_to_probe


def process_smoking(output_dir: str = 'temp_data') -> None:
    """Download and process the smoking (GSE50660) methylation dataset."""
    os.makedirs(output_dir, exist_ok=True)

    urls = {
        'GSE50660': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE50nnn/GSE50660/matrix/GSE50660_series_matrix.txt.gz',
        'HumanMethylation450_manifest': 'https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv',
    }

    gz_path = os.path.join(output_dir, 'GSE50660_series_matrix.txt.gz')
    if not os.path.exists(gz_path):
        print('Downloading GSE50660 series matrix...')
        download_file(urls['GSE50660'], gz_path)

    manifest_path = os.path.join(output_dir, 'humanmethylation450_15017482_v1-2.csv')
    if not os.path.exists(manifest_path):
        print('Downloading HumanMethylation450 manifest...')
        download_file(urls['HumanMethylation450_manifest'], manifest_path)

    print('Parsing series matrix...')
    sample_ids, targets, beta = _parse_series_matrix(gz_path)
    print(f'  Samples: {len(sample_ids)}, Probes: {beta.shape[1]}')

    print('Building probe-to-gene promoter mapping...')
    mapping = _build_probe_gene_mapping(manifest_path)

    print('Selecting per-gene minimum-beta promoter in never-smokers...')
    never_mask = targets == 0
    if never_mask.sum() == 0:
        raise ValueError('No never-smoker samples found (smoking == 0)')
    gene_data, gene_to_probe = _select_min_promoter_per_gene(beta, mapping, never_mask)

    all_nan_cols = gene_data.columns[gene_data.isna().all(axis=0)]
    if len(all_nan_cols) > 0:
        print(f'Dropping {len(all_nan_cols)} all-NaN gene columns')
        gene_data = gene_data.drop(columns=all_nan_cols)
        gene_to_probe = gene_to_probe.drop(index=all_nan_cols)

    if gene_data.isna().any().any():
        nan_cells = int(gene_data.isna().sum().sum())
        print(f'Imputing {nan_cells} remaining NaN cells with column means')
        gene_data = gene_data.fillna(gene_data.mean(axis=0))

    print('Median-centering per gene across samples...')
    gene_data = gene_data - gene_data.median(axis=0)

    assert not gene_data.isna().any().any(), 'Gene data has NaN values after processing'
    assert not np.isnan(targets).any(), 'Targets have NaN values'
    assert gene_data.shape[0] == len(targets), 'Sample count mismatch between data and targets'

    # Collapse the original 3-class GEO encoding (0=never, 1=former, 2=current) into a binary
    # never (0) vs ever-smoker (1 = former + current) target for downstream modeling.
    targets_binary = (targets > 0).astype(np.int64)
    class_names = ['never', 'ever']
    class_mapping = {'never': 0, 'ever': 1}
    original_class_counts = {
        'never': int((targets == 0).sum()),
        'former': int((targets == 1).sum()),
        'current': int((targets == 2).sum()),
    }

    data_file = os.path.join(output_dir, 'smoking_data.parquet')
    targets_file = os.path.join(output_dir, 'smoking_targets.parquet')

    gene_data.reset_index(drop=True).to_parquet(data_file)
    pd.DataFrame({'target': targets_binary}).to_parquet(targets_file)

    target_stats: dict = {
        'class_mapping': class_mapping,
        'num_classes': len(class_names),
        'class_names': class_names,
        'samples_per_class': {
            name: int((targets_binary == idx).sum()) for name, idx in class_mapping.items()
        },
        'original_geo_class_counts': original_class_counts,
    }

    metadata = create_dataset_metadata(
        dataset_name='smoking',
        download_urls=urls,
        num_samples=len(targets_binary),
        num_features=gene_data.shape[1],
        target_stats=target_stats,
        preprocessing_notes=(
            'GSE50660 Illumina 450k beta values mapped to genes using the HumanMethylation450 '
            'v1.2 manifest, restricted to probes annotated as TSS1500 or TSS200 promoter '
            'regions. For each gene, the candidate promoter probe with the minimum mean beta '
            'across never-smoker samples (original GEO smoking == 0) is kept as the gene-level '
            'feature. Values are then median-centered per gene across samples. The original '
            '3-class GEO smoking status (0=never, 1=former, 2=current) is collapsed into a '
            'binary target: 0=never, 1=ever (former or current).'
        ),
    )

    data_files = {'data': data_file, 'targets': targets_file}
    upload_to_huggingface('smoking', data_files, metadata)

    print('Successfully processed and uploaded smoking dataset')
    print(f'  Samples: {len(targets_binary)}')
    print(f'  Features (genes): {gene_data.shape[1]}')
    print(f'  Target stats: {target_stats}')

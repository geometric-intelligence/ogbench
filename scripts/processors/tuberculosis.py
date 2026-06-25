"""Tuberculosis dataset processor (GSE19433 M. tuberculosis full-proteome microarray).

GSE19433 profiles antibody responses against the entire Mycobacterium tuberculosis proteome (~4,000
proteins) in the sera of suspected TB patients, using a protein microarray (platform GPL9790, taxid
83332).

The classification target is the patient's chest x-ray result (NORMAL vs ABNORMAL); samples whose
chest x-ray is "not applicable" or "UNKNOWN" are discarded.

Feature identifiers in the series matrix are microarray spot coordinates (block_row_column, e.g.
"10_10_1"). These are mapped to M. tuberculosis ORFs (Rv locus tags) via the GPL9790 platform
table, and Rv locus tags are then mapped to UniProt accessions via the H37Rv reference proteome
(UP000001584). Spots whose Rv locus tag does not resolve to a UniProt accession are discarded.
"""

import gzip
import os
import re
from io import StringIO

import numpy as np
import pandas as pd

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface

# Array-specific suffixes on ORF names that denote multiple spots/segments printed
# for the same protein (e.g. "Rv0101-s1".."Rv0101-s4" segments, "Rv0018c-alt").
# These collapse to the base locus tag. Trailing letters such as "Rv0078a" are
# legitimate distinct loci and are preserved.
_ORF_SUFFIX_RE = re.compile(r'-(s\d+|alt)$', flags=re.IGNORECASE)

# M. tuberculosis H37Rv reference proteome on UniProt.
UNIPROT_PROTEOME = 'UP000001584'
UNIPROT_TAXID = 83332


def _normalize_orf(orf: str) -> str:
    """Reduce an array ORF name to its base Rv locus tag."""
    return _ORF_SUFFIX_RE.sub('', orf.strip())


def _parse_series_matrix(gz_path: str) -> tuple[list[str], list[str], pd.DataFrame]:
    """Parse the GSE19433 series matrix.

    Returns:
        sample_ids: GSM accessions in column order of the expression matrix.
        chest_xray: raw chest x-ray value per sample (e.g. "ABNORMAL", "NORMAL",
            "not applicable", "UNKNOWN"), aligned with sample_ids.
        expression: DataFrame of shape (n_samples, n_spots) indexed by GSM accession,
            columns are microarray spot ids.
    """
    sample_ids: list[str] = []
    chest_xray: list[str] = []

    with gzip.open(gz_path, 'rt') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [x.strip().strip('"') for x in line.rstrip('\n').split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1') and 'chest x-ray:' in line:
                # Skip the separate "chest x-ray description:" characteristics line.
                tokens = [x.strip().strip('"') for x in line.rstrip('\n').split('\t')[1:]]
                if tokens and tokens[0].lower().startswith('chest x-ray:'):
                    chest_xray = [t.split(':', 1)[-1].strip() for t in tokens]
            elif line.startswith('!series_matrix_table_begin'):
                break

    if not sample_ids:
        raise ValueError('Could not find !Sample_geo_accession line in series matrix')
    if not chest_xray:
        raise ValueError('Could not find the "chest x-ray:" characteristics line in series matrix')
    if len(sample_ids) != len(chest_xray):
        raise ValueError(
            f'Sample count mismatch: {len(sample_ids)} GSM ids vs {len(chest_xray)} chest x-ray values'
        )

    with gzip.open(gz_path, 'rt') as f:
        expression = pd.read_csv(f, sep='\t', comment='!', index_col='ID_REF').transpose()

    expression.index = expression.index.astype(str).str.strip().str.strip('"')
    missing = [sid for sid in sample_ids if sid not in expression.index]
    if missing:
        raise ValueError(
            f'{len(missing)} GSM ids missing from expression matrix (e.g. {missing[:3]})'
        )
    expression = expression.loc[sample_ids]

    return sample_ids, chest_xray, expression


def _build_spot_to_orf(soft_path: str) -> dict[str, str]:
    """Parse the GPL9790 platform table mapping spot id -> ORF (Rv locus tag)."""
    collected: list[str] = []
    started = False
    with gzip.open(soft_path, 'rt') as f:
        for line in f:
            if not started:
                if line.startswith('!platform_table_begin'):
                    started = True
                continue
            if line.startswith('!platform_table_end'):
                break
            collected.append(line)

    if not collected:
        raise ValueError(f'Could not find platform table in {soft_path}')

    platform = pd.read_csv(StringIO(''.join(collected)), sep='\t', dtype=str)
    if not {'ID', 'ORF'}.issubset(platform.columns):
        raise ValueError(f'Unexpected GPL9790 columns: {list(platform.columns)}')

    platform = platform.dropna(subset=['ID', 'ORF'])
    spot_to_orf = dict(zip(platform['ID'].astype(str), platform['ORF'].astype(str), strict=True))
    print(f'GPL9790: {len(spot_to_orf)} spot -> ORF mappings')
    return spot_to_orf


def _build_orf_to_uniprot(uniprot_tsv_path: str) -> dict[str, str]:
    """Build Rv locus tag -> UniProt accession mapping from the H37Rv proteome.

    Prefers reviewed (Swiss-Prot) entries when a locus tag maps to multiple accessions. A single
    UniProt entry may cover several loci (e.g. "Rv1180/Rv1181").
    """
    uni = pd.read_csv(uniprot_tsv_path, sep='\t', dtype=str).fillna('')
    uni.columns = ['accession', 'reviewed', 'ordered_locus']

    # Process reviewed entries first so they take precedence over unreviewed ones.
    uni = uni.sort_values('reviewed', key=lambda s: (s != 'reviewed'))

    orf_to_uniprot: dict[str, str] = {}
    for accession, _reviewed, ordered_locus in uni.itertuples(index=False):
        for locus in ordered_locus.split('/'):
            locus = locus.strip()
            if locus and locus not in orf_to_uniprot:
                orf_to_uniprot[locus] = accession

    print(f'UniProt {UNIPROT_PROTEOME}: {len(orf_to_uniprot)} Rv locus -> accession mappings')
    return orf_to_uniprot


def process_tuberculosis(output_dir: str = 'temp_data') -> None:
    """Download and process the tuberculosis (GSE19433) proteome microarray dataset."""
    os.makedirs(output_dir, exist_ok=True)

    urls = {
        'GSE19433': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19433/matrix/GSE19433_series_matrix.txt.gz',
        'GPL9790_platform': 'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL9nnn/GPL9790/soft/GPL9790_family.soft.gz',
        'UniProt_H37Rv_proteome': (
            f'https://rest.uniprot.org/uniprotkb/stream?query=proteome:{UNIPROT_PROTEOME}'
            '&fields=accession,reviewed,gene_oln&format=tsv'
        ),
    }

    gz_path = os.path.join(output_dir, 'GSE19433_series_matrix.txt.gz')
    if not os.path.exists(gz_path):
        print('Downloading GSE19433 series matrix...')
        download_file(urls['GSE19433'], gz_path)

    soft_path = os.path.join(output_dir, 'GPL9790_family.soft.gz')
    if not os.path.exists(soft_path):
        print('Downloading GPL9790 platform annotation...')
        download_file(urls['GPL9790_platform'], soft_path)

    uniprot_path = os.path.join(output_dir, 'uniprot_h37rv.tsv')
    if not os.path.exists(uniprot_path):
        print('Downloading UniProt H37Rv proteome mapping...')
        download_file(urls['UniProt_H37Rv_proteome'], uniprot_path)

    print('Parsing series matrix...')
    sample_ids, chest_xray, expression = _parse_series_matrix(gz_path)
    print(f'  Samples: {len(sample_ids)}, Spots: {expression.shape[1]}')

    # Build target: NORMAL=0, ABNORMAL=1; drop "not applicable" / "UNKNOWN" / anything else.
    label_map = {'NORMAL': 0, 'ABNORMAL': 1}
    targets_raw = np.array([label_map.get(v.strip().upper(), -1) for v in chest_xray])
    keep_samples = targets_raw != -1
    dropped = int((~keep_samples).sum())
    print(
        f'  Chest x-ray labels: keeping {int(keep_samples.sum())} samples '
        f'(NORMAL/ABNORMAL), dropping {dropped} (not applicable/unknown)'
    )

    expression = expression.loc[keep_samples]
    targets = targets_raw[keep_samples].astype(np.int64)

    print('Mapping spots -> Rv ORFs -> UniProt accessions...')
    spot_to_orf = _build_spot_to_orf(soft_path)
    orf_to_uniprot = _build_orf_to_uniprot(uniprot_path)

    # Map each spot column to (Rv locus, UniProt accession); keep only resolvable spots.
    spot_rv = {spot: _normalize_orf(orf) for spot, orf in spot_to_orf.items()}
    spot_uniprot = {
        spot: orf_to_uniprot[rv] for spot, rv in spot_rv.items() if rv in orf_to_uniprot
    }

    expression.columns = expression.columns.astype(str)
    mapped_spots = [s for s in expression.columns if s in spot_uniprot]
    n_total = expression.shape[1]
    print(
        f'  Mapped {len(mapped_spots)}/{n_total} spots to UniProt '
        f'({len(mapped_spots) / n_total * 100:.1f}%); discarding the rest'
    )
    expression = expression[mapped_spots]

    # Rename spots to their base Rv locus tag and collapse duplicate loci
    # (multi-segment / alt spots of the same protein) by averaging.
    expression.columns = [spot_rv[s] for s in mapped_spots]
    expression = expression.apply(pd.to_numeric, errors='coerce')
    expression = expression.T.groupby(level=0).mean().T
    print(f'  Collapsed to {expression.shape[1]} unique Rv loci')

    raw_data = expression.reset_index(drop=True)

    assert not raw_data.isna().any().any(), 'Raw data contains NaNs'
    assert raw_data.shape[0] == len(targets), 'Sample count mismatch between data and targets'

    # node_id = Rv locus tag, string_id = UniProt accession (STRING-compatible).
    rv_to_uniprot = {spot_rv[s]: spot_uniprot[s] for s in mapped_spots}
    gene_map = pd.DataFrame(
        {
            'node_id': list(raw_data.columns),
            'string_id': [rv_to_uniprot[rv] for rv in raw_data.columns],
        }
    )
    gene_map['node_id'] = gene_map['node_id'].astype(str)
    gene_map['string_id'] = gene_map['string_id'].astype(str)

    class_names = ['NORMAL', 'ABNORMAL']
    class_mapping = {name: i for i, name in enumerate(class_names)}

    print('Classification distribution:')
    unique_classes, counts = np.unique(targets, return_counts=True)
    for class_id, count in zip(unique_classes, counts, strict=True):
        print(f'  {class_names[class_id]}: {count} samples ({count / len(targets) * 100:.1f}%)')

    data_file = os.path.join(output_dir, 'tuberculosis_data.parquet')
    targets_file = os.path.join(output_dir, 'tuberculosis_targets.parquet')
    map_file = os.path.join(output_dir, 'tuberculosis_map.parquet')

    raw_data.to_parquet(data_file)
    pd.DataFrame({'target': targets}).to_parquet(targets_file)
    gene_map.reset_index(drop=True).to_parquet(map_file, index=False)

    target_stats = {
        'class_mapping': class_mapping,
        'num_classes': len(class_names),
        'class_names': class_names,
        'samples_per_class': {
            name: int((targets == idx).sum()) for name, idx in class_mapping.items()
        },
    }

    metadata = create_dataset_metadata(
        dataset_name='tuberculosis',
        download_urls=urls,
        num_samples=len(targets),
        num_features=raw_data.shape[1],
        target_stats=target_stats,
        preprocessing_notes=(
            'GSE19433 M. tuberculosis full-proteome antibody microarray (platform GPL9790, '
            'log-transformed median spot intensities). The classification target is the patient '
            'chest x-ray result (0=NORMAL, 1=ABNORMAL); samples with chest x-ray "not applicable" '
            'or "UNKNOWN" are dropped. Microarray spot ids (block_row_column) are mapped to '
            'M. tuberculosis ORFs (Rv locus tags) via the GPL9790 platform table; array-specific '
            'segment/alt spots ("-s1", "-alt") are collapsed to the base locus tag by averaging. '
            'Rv locus tags are mapped to UniProt accessions via the H37Rv reference proteome '
            f'({UNIPROT_PROTEOME}); spots that do not resolve to a UniProt accession are discarded.'
        ),
    )

    data_files = {'data': data_file, 'targets': targets_file, 'map': map_file}
    upload_to_huggingface('tuberculosis', data_files, metadata)

    print('Successfully processed and uploaded tuberculosis dataset')
    print(f'  Samples: {len(targets)}')
    print(f'  Features (Rv loci): {raw_data.shape[1]}')
    print(f'  Target stats: {target_stats}')

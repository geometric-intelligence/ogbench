"""OV (Ovarian Cancer) dataset processor."""

import logging
import os

import numpy as np
import pandas as pd

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface

logger = logging.getLogger(__name__)

CLASS_NAMES = ['Immunoreactive', 'Mesenchymal', 'Proliferative', 'Differentiated']


def _parquet_column_name(gene: str) -> str:
    """Return a parquet-safe feature name for HGNC symbols containing dots.

    Parquet treats '.' as a nested-field separator, so e.g. ``A1BG`` and ``A1BG.AS1``
    cannot coexist as flat columns.
    """
    return str(gene).replace('.', '_')


def _prepare_parquet_frame(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a parquet-safe expression matrix and aligned feature/gene symbol lists."""
    gene_symbols = [str(g) for g in raw_data.columns]
    feature_names = [_parquet_column_name(g) for g in gene_symbols]

    data = raw_data.copy()
    data.columns = feature_names

    dup_mask = data.columns.duplicated()
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        logger.info('Dropping %d duplicate feature columns (keeping first occurrence)', n_dup)
        data = data.loc[:, ~dup_mask]
        gene_symbols = [g for g, dup in zip(gene_symbols, dup_mask, strict=True) if not dup]
        feature_names = [f for f, dup in zip(feature_names, dup_mask, strict=True) if not dup]

    return data.reset_index(drop=True), feature_names, gene_symbols


def _drop_high_zero_genes(
    raw_data: pd.DataFrame,
    train_fraction: float = 0.70,
    zero_threshold: float = 0.50,
    seed: int = 0,
) -> pd.DataFrame:
    """Drop genes with >50%% zeros in a random training split (no label leakage)."""
    n_train = int(train_fraction * len(raw_data))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(raw_data))
    rng.shuffle(idx)
    train_df = raw_data.iloc[idx[:n_train]]

    zero_frac = (train_df == 0).mean(axis=0)
    before = raw_data.shape[1]
    filtered = raw_data.loc[:, zero_frac <= zero_threshold].copy()
    logger.info(
        'Gene zero-filter (>%.0f%% zeros in train): %d -> %d genes (dropped %d)',
        zero_threshold * 100,
        before,
        filtered.shape[1],
        before - filtered.shape[1],
    )
    return filtered


def process_ov(output_dir: str = 'temp_data') -> None:
    """Download and process OV transcriptomics dataset from MLOmics/TCGA."""
    os.makedirs(output_dir, exist_ok=True)

    urls = {
        'OV_mRNA': (
            'https://huggingface.co/datasets/AIBIC/MLOmics/resolve/main'
            '/Main_Dataset/Classification_datasets/GS-OV/Original/OV_mRNA.csv'
        ),
        'OV_label_num': (
            'https://huggingface.co/datasets/AIBIC/MLOmics/resolve/main'
            '/Main_Dataset/Classification_datasets/GS-OV/Original/OV_label_num.csv'
        ),
    }

    mrna_path = os.path.join(output_dir, 'OV_mRNA.csv')
    label_path = os.path.join(output_dir, 'OV_label_num.csv')

    downloads: list[tuple[str, str, str]] = [
        ('OV_mRNA', urls['OV_mRNA'], mrna_path),
        ('OV_label_num', urls['OV_label_num'], label_path),
    ]
    for name, url, path in downloads:
        if os.path.exists(path):
            continue
        logger.info('Downloading %s...', name)
        try:
            download_file(url, path)
            logger.info('Successfully downloaded %s to %s', name, path)
        except Exception as e:
            logger.error('Error downloading %s: %s', name, e)
            raise

    # MLOmics format: rows = genes, columns = samples -> transpose to samples x genes
    mrna = pd.read_csv(mrna_path, index_col=0, low_memory=False)
    raw_data = mrna.transpose().astype(float)

    label_df = pd.read_csv(label_path)
    label_col = 'Label' if 'Label' in label_df.columns else label_df.columns[0]
    targets = label_df[label_col].values.astype(int)

    assert len(targets) == len(raw_data), (
        f'Mismatched samples: {len(targets)} labels vs {len(raw_data)} expression rows'
    )

    flat = raw_data.values.flatten()
    assert flat.max() <= 50, f'Max value {flat.max():.1f} > 50 — may not be log-transformed'
    logger.info(
        'Loaded mRNA matrix: %d samples x %d genes (value range [%.3f, %.3f])',
        raw_data.shape[0],
        raw_data.shape[1],
        flat.min(),
        flat.max(),
    )

    raw_data = _drop_high_zero_genes(raw_data)

    assert not raw_data.isna().any().any(), 'Raw data contains NaNs after filtering'

    # Gene columns are HGNC symbols; STRING resolves them via alias lookup.
    raw_data, feature_names, gene_symbols = _prepare_parquet_frame(raw_data)
    n_dot = sum('.' in g for g in gene_symbols)
    logger.info(
        'Gene map: %d total, %d protein-coding (STRING-mappable), '
        '%d lncRNA/pseudogene (dot-notation)',
        len(gene_symbols),
        len(gene_symbols) - n_dot,
        n_dot,
    )

    gene_map = pd.DataFrame(
        {
            'node_id': feature_names,
            'string_id': gene_symbols,
        }
    )

    class_mapping = {name: i for i, name in enumerate(CLASS_NAMES)}

    logger.info('Classification distribution:')
    unique_classes, counts = np.unique(targets, return_counts=True)
    for class_id, count in zip(unique_classes, counts, strict=True):
        logger.info(
            '  %s: %d samples (%.1f%%)',
            CLASS_NAMES[class_id],
            count,
            count / len(targets) * 100,
        )

    data_file = os.path.join(output_dir, 'ov_data.parquet')
    targets_file = os.path.join(output_dir, 'ov_targets.parquet')
    map_file = os.path.join(output_dir, 'ov_map.parquet')

    raw_data.to_parquet(data_file)
    pd.DataFrame({'target': targets}).to_parquet(targets_file)
    gene_map.reset_index(drop=True).to_parquet(map_file, index=False)

    target_stats = {
        'class_mapping': class_mapping,
        'num_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES,
        'samples_per_class': {
            CLASS_NAMES[int(c)]: int(cnt) for c, cnt in zip(unique_classes, counts, strict=True)
        },
    }

    metadata = create_dataset_metadata(
        dataset_name='ov',
        download_urls=urls,
        num_samples=len(targets),
        num_features=raw_data.shape[1],
        target_stats=target_stats,
        preprocessing_notes=(
            'MLOmics log-transformed FPKM values (no additional log transform). '
            'Dropped genes with >50% zero values in a random 70% training split. '
            'Duplicate gene symbols are dropped keeping first occurrence (1-Mar, 2-Mar). '
            'Parquet feature names replace dots with underscores; gene map string_id '
            'retains original HGNC symbols.'
        ),
    )

    data_files = {'data': data_file, 'targets': targets_file, 'map': map_file}
    upload_to_huggingface('ov', data_files, metadata)

    logger.info('Successfully processed and uploaded OV dataset')
    logger.info('  Samples: %d', len(targets))
    logger.info('  Features: %d', raw_data.shape[1])
    logger.info('  Classes: %d (%s)', len(CLASS_NAMES), ', '.join(CLASS_NAMES))
    logger.info('  Target stats: %s', target_stats)

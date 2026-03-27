"""Breast Cancer dataset processor."""

import logging
import os

import numpy as np
import pandas as pd

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface

logger = logging.getLogger(__name__)


def process_brca(output_dir: str = 'temp_data') -> None:
    """Download and process BRCA methylation dataset from MLOmics/TCGA."""
    os.makedirs(output_dir, exist_ok=True)

    urls = {
        'BRCA_Methy': 'https://huggingface.co/datasets/AIBIC/MLOmics/resolve/main/Main_Dataset/Classification_datasets/GS-BRCA/Original/BRCA_Methy.csv',
        'BRCA_label_num': 'https://huggingface.co/datasets/AIBIC/MLOmics/resolve/main/Main_Dataset/Classification_datasets/GS-BRCA/Original/BRCA_label_num.csv',
    }

    csv_path = os.path.join(output_dir, 'BRCA_Methy.csv')
    label_path = os.path.join(output_dir, 'BRCA_label_num.csv')

    downloads: list[tuple[str, str, str]] = [
        ('BRCA_Methy', urls['BRCA_Methy'], csv_path),
        ('BRCA_label_num', urls['BRCA_label_num'], label_path),
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

    # Methylation matrix: rows = probes/genes, columns = samples (TCGA barcodes)
    methy = pd.read_csv(csv_path, index_col=0)
    raw_data = methy.transpose()

    label_df = pd.read_csv(label_path)
    targets = label_df['Label'].values.astype(int)

    # Match metadata with expression samples
    assert len(targets) == len(
        raw_data
    ), f'Mismatched samples: {len(targets)} scores vs {len(raw_data)} samples'

    assert not raw_data.isna().any().any(), 'Raw data contains NaNs'

    # PAM50 subtype mapping
    class_names = ['LumA', 'Her2', 'LumB', 'Normal', 'Basal']
    class_mapping = {name: i for i, name in enumerate(class_names)}

    logger.info('Classification distribution:')
    unique_classes, counts = np.unique(targets, return_counts=True)
    for class_id, count in zip(unique_classes, counts, strict=True):
        logger.info(
            '  %s: %d samples (%.1f%%)',
            class_names[class_id],
            count,
            count / len(targets) * 100,
        )

    # Build gene map — gene symbols passed directly to STRING alias lookup
    gene_map = pd.DataFrame(
        {
            'node_id': list(raw_data.columns),
            'string_id': list(raw_data.columns),
        }
    )
    gene_map['node_id'] = gene_map['node_id'].astype(str)
    gene_map['string_id'] = gene_map['string_id'].astype(str)

    data_file = os.path.join(output_dir, 'brca_data.parquet')
    targets_file = os.path.join(output_dir, 'brca_targets.parquet')
    map_file = os.path.join(output_dir, 'brca_map.parquet')

    raw_data.reset_index(drop=True).to_parquet(data_file)
    pd.DataFrame({'target': targets}).to_parquet(targets_file)
    gene_map.reset_index(drop=True).to_parquet(map_file, index=False)

    target_stats = {
        'class_mapping': class_mapping,
        'num_classes': len(class_names),
        'class_names': class_names,
    }

    metadata = create_dataset_metadata(
        dataset_name='brca',
        download_urls=urls,
        num_samples=len(targets),
        num_features=raw_data.shape[1],
        target_stats=target_stats,
    )

    data_files = {'data': data_file, 'targets': targets_file, 'map': map_file}
    upload_to_huggingface('brca', data_files, metadata)

    logger.info('  Successfully processed and uploaded BRCA dataset')
    logger.info('  Samples: %d', len(targets))
    logger.info('  Features: %d', raw_data.shape[1])
    logger.info('  Classes: %d (%s)', len(class_names), ', '.join(class_names))
    logger.info('  Target stats: %s', target_stats)

"""Check that the sklearn baselines see exactly the GNN node-selected features.

Compares ``load_and_prepare_data`` against the ``selected_data.parquet`` artifacts
written by ``HFOmicsDataset.download`` for every cached fold/ratio/method cell.

Usage:
    python scripts/verify_baseline_features.py --root-dir /scratch/lcornelis/ogbench
"""

import argparse
import glob
import os.path as osp
import re

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

import ogbench.baseline as baseline

CONFIG_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'configs')
CELL_PATTERN = re.compile(
    r'/omics/(?P<dataset>[^/]+)/[^/]+/adj_method_(?P<adj>[^/]+)/(?P<method>[^/]+)'
    r'/p_(?P<ratio>[^/]+)/[^/]+/split_k-fold_k_(?P<k>\d+)_fold_(?P<fold>\d+)'
)


def discover_cells(root_dir: str) -> list[dict]:
    pattern = osp.join(root_dir, 'data', 'omics', '**', 'selected_data.parquet')
    cells = {}
    for artifact in glob.glob(pattern, recursive=True):
        raw_dir = osp.dirname(artifact)
        match = CELL_PATTERN.search(raw_dir)
        if not match:
            continue
        cell = match.groupdict()
        key = (cell['dataset'], cell['method'], cell['ratio'], cell['fold'])
        cells.setdefault(key, {**cell, 'raw_dir': raw_dir})
    return sorted(cells.values(), key=lambda c: (c['dataset'], c['method'], c['ratio'], c['fold']))


def check_cell(cell: dict, root_dir: str) -> tuple[bool, str]:
    captured: dict = {}
    original = baseline._select_gnn_nodes

    def spy(train_data, train_targets, cfg):
        columns = original(train_data, train_targets, cfg)
        captured['columns'] = columns
        captured['train'] = train_data[columns]
        return columns

    baseline._select_gnn_nodes = spy
    try:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base='1.3'):
            cfg = compose(
                config_name='baseline.yaml',
                overrides=[
                    f'dataset={cell["dataset"]}',
                    'dataset.split_params.split_type=k-fold',
                    f'dataset.split_params.k={cell["k"]}',
                    f'dataset.split_params.data_seed={cell["fold"]}',
                    f'dataset.loader.parameters.node_sample_ratio={cell["ratio"]}',
                    f'dataset.loader.parameters.method={cell["method"]}',
                    f'paths.root_dir={root_dir}',
                    'seed=42',
                ],
            )
            baseline.load_and_prepare_data(cfg)
    finally:
        baseline._select_gnn_nodes = original

    cached = pd.read_parquet(osp.join(cell['raw_dir'], 'selected_data.parquet'))
    train = captured['train']
    if list(cached.columns) != list(captured['columns']):
        return False, f'column mismatch ({len(cached.columns)} cached vs {len(train.columns)})'
    diff = np.abs(cached.iloc[: len(train)].values - train.values)
    max_diff = float(np.nanmax(diff)) if diff.size else 0.0
    if max_diff != 0.0:
        return False, f'value mismatch, max abs diff {max_diff:.3e}'
    return True, f'{len(train.columns)} features, {len(train)} train rows'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--fold', default=None)
    parser.add_argument('--ratio', default=None)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    cells = discover_cells(args.root_dir)
    if args.dataset:
        cells = [c for c in cells if c['dataset'] == args.dataset]
    if args.fold is not None:
        cells = [c for c in cells if c['fold'] == args.fold]
    if args.ratio is not None:
        cells = [c for c in cells if c['ratio'] == args.ratio]
    if args.limit:
        cells = cells[: args.limit]

    failures = 0
    for cell in cells:
        try:
            ok, detail = check_cell(cell, args.root_dir)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f'{type(exc).__name__}: {exc}'
        failures += not ok
        status = 'OK  ' if ok else 'FAIL'
        print(
            f'{status} {cell["dataset"]:<14} {cell["method"]:<20} '
            f'p={cell["ratio"]:<4} fold={cell["fold"]}  {detail}',
            flush=True,
        )
    print(f'\n{len(cells) - failures}/{len(cells)} cells match the GNN feature set.')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()

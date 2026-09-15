#!/usr/bin/env python3
"""Offline: calibrate WGCNA adjacency thresholds to ~10% connectivity per k-fold.

STRING configs are left at 0.4 (not calibrated here).

For each unique WGCNA (dataset, node method, node_sample_ratio) used in
``scripts/rerun_best_kfold.sh``, and for each fold in 0..k-1, rebuild the
fold-specific train graph (same split / impute / node selection as HFOmics)
and set the threshold to the (1 - target_connectivity) quantile of the
upper-triangle continuous adjacency — equivalent to binarizing at that cut
for the target edge density.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from sklearn.impute import SimpleImputer
from tqdm import tqdm

from ogbench.data.adjacency import get_adjacency_builder
from ogbench.data.selectors import get_selector
from ogbench.data.utils.split_utils import compute_omics_split_indices

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / 'configs'
KFOLD_SCRIPT = ROOT / 'scripts' / 'rerun_best_kfold.sh'
HF_REPO_ID = 'geometric-intelligence/ogbench'
TARGET_CONNECTIVITY = 0.10
K = 5
RANDOM_STATE = 42


def load_hf_revision() -> str:
    with open(CONFIGS_DIR / 'hf' / 'default.yaml') as f:
        return yaml.safe_load(f).get('revision', 'main')


def parse_wgcna_recipes(script_path: Path) -> list[tuple[str, str, float]]:
    """Unique (dataset, node_method, ratio) for WGCNA launches in the kfold script."""
    text = script_path.read_text()
    recipes: set[tuple[str, str, float]] = set()
    for m in re.finditer(
        r'dataset=(\w+).*?'
        r'adjacency_method=(\w+).*?'
        r'node_sample_ratio=([\d.]+).*?'
        r'method=(\w+).*?'
        r'adjacency_threshold=([\d.]+)',
        text,
        re.S,
    ):
        dataset, adj_method, ratio, method, _thr = m.groups()
        if adj_method != 'wgcna':
            continue
        recipes.add((dataset, method, float(ratio)))
    return sorted(recipes)


def threshold_for_connectivity(continuous_adj: np.ndarray, target: float) -> float:
    """Threshold such that fraction of off-diagonal pairs above it ≈ target."""
    n = continuous_adj.shape[0]
    if n < 2:
        return 0.0
    upper = continuous_adj[np.triu_indices(n, k=1)]
    upper = np.asarray(upper, dtype=np.float64)
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return 0.0
    # Keep top `target` fraction of pairs → quantile at 1 - target
    thr = float(np.quantile(upper, 1.0 - target))
    return thr


def measure_connectivity(continuous_adj: np.ndarray, threshold: float) -> float:
    n = continuous_adj.shape[0]
    adj = (continuous_adj > threshold).astype(np.float64)
    np.fill_diagonal(adj, 0)
    n_edges = adj.sum() / 2.0
    max_edges = n * (n - 1) / 2.0
    return float(n_edges / max_edges) if max_edges > 0 else 0.0


def prepare_fold_features(
    raw_data: pd.DataFrame,
    targets: np.ndarray,
    *,
    fold: int,
    method: str,
    node_sample_ratio: float,
) -> pd.DataFrame:
    split = compute_omics_split_indices(
        targets,
        split_type='k-fold',
        k=K,
        fold=fold,
        random_state=RANDOM_STATE,
    )
    train_data = raw_data.iloc[split['train']].copy()
    train_targets = targets[split['train']]

    if train_data.isna().to_numpy().any() or raw_data.isna().to_numpy().any():
        imputer = SimpleImputer(strategy='mean')
        train_data = pd.DataFrame(
            imputer.fit_transform(train_data),
            columns=train_data.columns,
            index=train_data.index,
        )

    n_train = len(train_targets)
    n_nodes = int(n_train / node_sample_ratio)
    n_nodes = min(n_nodes, train_data.shape[1])
    selected = get_selector(method).select(train_data.values, train_targets, n_nodes)
    return train_data.iloc[:, selected]


def calibrate() -> pd.DataFrame:
    revision = load_hf_revision()
    recipes = parse_wgcna_recipes(KFOLD_SCRIPT)
    print(f'WGCNA recipes: {len(recipes)} × {K} folds = {len(recipes) * K} calibrations')
    for r in recipes:
        print(f'  {r}')

    cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    rows: list[dict] = []

    for dataset, method, ratio in tqdm(recipes, desc='recipes'):
        if dataset not in cache:
            data_file = hf_hub_download(  # nosec
                repo_id=HF_REPO_ID,
                repo_type='dataset',
                revision=revision,
                filename=f'{dataset}_data.parquet',
            )
            targets_file = hf_hub_download(  # nosec
                repo_id=HF_REPO_ID,
                repo_type='dataset',
                revision=revision,
                filename=f'{dataset}_targets.parquet',
            )
            raw = pd.read_parquet(data_file)
            if 'target' in raw.columns:
                raw = raw.drop(columns=['target'])
            targets = pd.read_parquet(targets_file)['target'].to_numpy()
            cache[dataset] = (raw, targets)

        raw_data, targets = cache[dataset]
        builder = get_adjacency_builder('wgcna')

        for fold in range(K):
            train_selected = prepare_fold_features(
                raw_data,
                targets,
                fold=fold,
                method=method,
                node_sample_ratio=ratio,
            )
            continuous = builder.build(train_selected)
            continuous = np.nan_to_num(np.asarray(continuous, dtype=np.float64), nan=0.0)
            thr = threshold_for_connectivity(continuous, TARGET_CONNECTIVITY)
            conn = measure_connectivity(continuous, thr)
            rows.append(
                {
                    'dataset': dataset,
                    'adjacency_method': 'wgcna',
                    'method': method,
                    'node_sample_ratio': ratio,
                    'fold': fold,
                    'n_nodes': train_selected.shape[1],
                    'n_train': train_selected.shape[0],
                    'threshold': thr,
                    'connectivity': conn,
                    'target_connectivity': TARGET_CONNECTIVITY,
                }
            )
            print(
                f'{dataset} {method} p={ratio} fold={fold}: '
                f'n_nodes={train_selected.shape[1]} thr={thr:.6g} conn={conn:.4f}'
            )

    return pd.DataFrame(rows)


def patch_kfold_script(df: pd.DataFrame, script_path: Path, dry_run: bool = False) -> int:
    """Replace WGCNA adjacency_threshold=...

    in each launch block with fold-specific value.
    """
    text = script_path.read_text()
    lookup = {
        (r.dataset, r.method, float(r.node_sample_ratio), int(r.fold)): float(r.threshold)
        for r in df.itertuples()
    }

    # Walk line-by-line, accumulate blocks starting with "launch "
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n_patched = 0
    while i < len(lines):
        line = lines[i]
        if not re.match(r'^launch\s+\d+\s+python\s+-m\s+ogbench', line):
            out.append(line)
            i += 1
            continue

        block_lines = [line]
        i += 1
        while i < len(lines):
            block_lines.append(lines[i])
            # block ends when we hit a line that does not continue the command
            # commands use trailing \ ; last line has no \
            if not lines[i].rstrip().endswith('\\'):
                i += 1
                break
            i += 1

        block = ''.join(block_lines)
        ds_m = re.search(r'dataset=(\w+)', block)
        adj_m = re.search(r'adjacency_method=(\w+)', block)
        ratio_m = re.search(r'node_sample_ratio=([\d.]+)', block)
        method_m = re.search(r'\n\s*dataset\.loader\.parameters\.method=(\w+)', block)
        # method= also appears as model=; prefer loader method
        if method_m is None:
            method_m = re.search(r'dataset\.loader\.parameters\.method=(\w+)', block)
        fold_m = re.search(r'dataset\.split_params\.data_seed=(\d+)', block)
        thr_m = re.search(r'adjacency_threshold=([\d.eE+-]+)', block)

        if not all([ds_m, adj_m, ratio_m, method_m, fold_m, thr_m]):
            out.extend(block_lines)
            continue

        if adj_m.group(1) != 'wgcna':
            out.extend(block_lines)
            continue

        key = (
            ds_m.group(1),
            method_m.group(1),
            float(ratio_m.group(1)),
            int(fold_m.group(1)),
        )
        if key not in lookup:
            raise KeyError(f'No calibrated threshold for {key}')

        new_thr = lookup[key]
        # keep reasonable precision for shell / cache path stability
        thr_str = f'{new_thr:.6g}'
        new_block = re.sub(
            r'adjacency_threshold=[\d.eE+-]+',
            f'adjacency_threshold={thr_str}',
            block,
            count=1,
        )
        if new_block != block:
            n_patched += 1
        out.append(new_block)

    if not dry_run:
        script_path.write_text(''.join(out))
    return n_patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'stats' / 'kfold_wgcna_thresholds_10pct.csv',
    )
    parser.add_argument('--patch-script', action='store_true', help='Update rerun_best_kfold.sh')
    parser.add_argument('--dry-run-patch', action='store_true')
    parser.add_argument(
        '--from-csv',
        type=Path,
        default=None,
        help='Skip calibration; patch from an existing CSV',
    )
    args = parser.parse_args()

    if args.from_csv is not None:
        df = pd.read_csv(args.from_csv)
    else:
        df = calibrate()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f'\nWrote {args.output}')

    print('\nPer-recipe threshold ranges:')
    for (ds, method, ratio), g in df.groupby(['dataset', 'method', 'node_sample_ratio']):
        print(
            f'  {ds} {method} p={ratio}: '
            f'{g["threshold"].min():.6g} .. {g["threshold"].max():.6g} '
            f'(conn {g["connectivity"].min():.4f}-{g["connectivity"].max():.4f})'
        )

    if args.patch_script or args.dry_run_patch:
        n = patch_kfold_script(df, KFOLD_SCRIPT, dry_run=args.dry_run_patch)
        print(
            f'Patched {n} WGCNA launch blocks in {KFOLD_SCRIPT}'
            + (' (dry-run)' if args.dry_run_patch else '')
        )


if __name__ == '__main__':
    main()

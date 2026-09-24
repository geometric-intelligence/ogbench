#!/usr/bin/env python3
"""Split ratio-0.3 cells that Parka will not reach between the A30 servers.

The first tier is every cell of --first-models, cheapest projected cell first.
The second tier is --second-models in reverse Parka priority, so it meets
Parka's forward queue only once both sides have covered the whole model.
Within each tier, cells projected to exceed the timeout on an A30 go last and
consecutive cells alternate between servers. Cells Parka has already started
are skipped.

Projected A30 fold time = source estimate x the median slowdown Parka measured
at ratio 0.3 (by adjacency method and source ratio) x --a30-factor.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

ADJACENCY = 'dataset.loader.parameters.adjacency_method'
REFERENCE_MODELS = ('gcn', 'chebnet')


def parka_attempts(ledger_path: Path) -> pd.DataFrame:
    with sqlite3.connect(f'file:{ledger_path}?mode=ro', uri=True) as connection:
        return pd.read_sql_query(
            'SELECT study_name, status, elapsed_time FROM fold_attempts', connection
        )


def source_ratio(candidates: pd.DataFrame) -> pd.Series:
    return candidates['source_rule'].str.extract(r'r([\d.]+)$')[0].astype(float)


def slowdown_factors(
    candidates: pd.DataFrame,
    attempts: pd.DataFrame,
    reference_models: Sequence[str] = REFERENCE_MODELS,
) -> pd.Series:
    """Median measured / estimated fold time, indexed by (adjacency, source ratio)."""
    frame = candidates.assign(src_ratio=source_ratio(candidates))
    folds = attempts[attempts['status'] == 'success'].merge(
        frame[['study_name', 'model', ADJACENCY, 'src_ratio', 'estimated_fold_seconds']],
        on='study_name',
    )
    folds = folds[folds['model'].isin(reference_models)]
    ratio = folds['elapsed_time'] / folds['estimated_fold_seconds']
    return ratio.groupby([folds[ADJACENCY], folds['src_ratio']]).median()


def build_queue(
    candidates: pd.DataFrame,
    started: set[str],
    factors: pd.Series,
    *,
    servers: Sequence[str],
    first_models: Sequence[str],
    second_models: Sequence[str],
    a30_factor: float,
    timeout: float,
) -> pd.DataFrame:
    frame = candidates[~candidates['study_name'].isin(started)].copy()
    frame['src_ratio'] = source_ratio(frame)
    fallback = factors.groupby(level=0).max()
    frame['slowdown'] = [
        factors.get((adjacency, ratio), fallback[adjacency])
        for adjacency, ratio in zip(frame[ADJACENCY], frame['src_ratio'], strict=True)
    ]
    frame['projected_a30_fold_seconds'] = (
        frame['estimated_fold_seconds'] * frame['slowdown'] * a30_factor
    ).round()
    frame['likely_timeout'] = frame['projected_a30_fold_seconds'] > timeout

    first = frame[frame['model'].isin(first_models)].sort_values(
        ['likely_timeout', 'projected_a30_fold_seconds', 'priority']
    )
    second = frame[frame['model'].isin(second_models)].sort_values(
        ['likely_timeout', 'priority'], ascending=[True, False]
    )
    tiers = []
    for tier, cells in enumerate((first, second), start=1):
        cells = cells.reset_index(drop=True)
        tiers.append(
            cells.assign(
                tier=tier,
                server=[servers[index % len(servers)] for index in range(len(cells))],
            )
        )
    queue = pd.concat(tiers, ignore_index=True)
    queue['server_rank'] = queue.groupby('server').cumcount() + 1
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('/scratch/lcornelis/ogbench/search_results/sep24_ratio03_transfer'),
        help='Parka campaign root with run_ledger.sqlite3',
    )
    parser.add_argument('--prepare-dir', type=Path, help='Default: ROOT/candidates_r05')
    parser.add_argument('--output-dir', type=Path, help='Default: ROOT/a30_handoff')
    parser.add_argument('--servers', nargs='+', default=['frank', 'hall'])
    parser.add_argument('--first-models', nargs='+', default=['gatv2'])
    parser.add_argument('--second-models', nargs='+', default=['graph_sage'])
    parser.add_argument('--a30-factor', type=float, default=1.6)
    parser.add_argument('--timeout', type=float, default=3600.0)
    args = parser.parse_args()

    prepare_dir = args.prepare_dir or args.root / 'candidates_r05'
    output_dir = args.output_dir or args.root / 'a30_handoff'
    candidates = pd.read_csv(prepare_dir / 'candidates.csv')
    attempts = parka_attempts(args.root / 'run_ledger.sqlite3')
    queue = build_queue(
        candidates,
        set(attempts['study_name']),
        slowdown_factors(candidates, attempts),
        servers=args.servers,
        first_models=args.first_models,
        second_models=args.second_models,
        a30_factor=args.a30_factor,
        timeout=args.timeout,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for server in args.servers:
        names = queue.loc[queue['server'] == server, 'study_name']
        (output_dir / f'{server}.txt').write_text('\n'.join(names) + '\n')
    columns = [
        'server',
        'server_rank',
        'tier',
        'study_name',
        'model',
        'dataset',
        ADJACENCY,
        'priority',
        'source_rule',
        'projected_a30_fold_seconds',
        'likely_timeout',
    ]
    queue[columns].to_csv(output_dir / 'a30_queue.csv', index=False)
    summary = queue.groupby(['server', 'tier', 'model', 'likely_timeout']).size()
    print(summary.to_string())


if __name__ == '__main__':
    main()

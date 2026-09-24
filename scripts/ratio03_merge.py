#!/usr/bin/env python3
"""Merge the ratio-0.3 results of Parka and the A30 servers into one table.

Each server's results_ratio03.csv covers all 816 cells with that server's
status. For every cell the merge keeps the best status (complete, then
incomplete, failed, pending), preferring more successful folds and then the
server listed first (Parka). Complete runs of the same cell on other servers
are listed in duplicate_servers. Writes results_ratio03.csv, coverage.md,
status_latest.json, and the concatenated trial and fold tables to --output-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ratio03_collect import coverage_markdown  # noqa: E402

STATUS_RANK = {'complete': 0, 'incomplete': 1, 'in_progress': 1, 'failed': 2, 'pending': 3}
COUNTERS = ('folds_last_hour', 'failed_attempts', 'oom_attempts', 'timeouts')
TABLES = ('trials.csv', 'fold_attempts.csv', 'failures.csv')


def merge_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pick one row per cell from per-server results, in server preference order."""
    frames = []
    for order, (server, frame) in enumerate(results.items()):
        frames.append(frame.assign(server=server, server_order=order))
    rows = pd.concat(frames, ignore_index=True)

    params = rows.groupby('study_name')['sampled_params'].nunique()
    if (params > 1).any():
        mismatched = list(params[params > 1].index[:3])
        raise ValueError(f'Servers ran different configurations, e.g. {mismatched}')

    rows['status_rank'] = rows['status'].map(STATUS_RANK).fillna(len(STATUS_RANK))
    rows = rows.sort_values(
        ['study_name', 'status_rank', 'folds_done', 'server_order'],
        ascending=[True, True, False, True],
    )
    chosen = rows.drop_duplicates('study_name', keep='first').set_index('study_name')
    complete = rows[rows['status'] == 'complete']
    duplicates = {
        study: ','.join(group['server'][group['server'] != chosen.at[study, 'server']])
        for study, group in complete.groupby('study_name')
    }
    chosen['duplicate_servers'] = pd.Series(duplicates).reindex(chosen.index).fillna('')
    merged = chosen.reset_index().sort_values('priority')
    return merged.drop(columns=['status_rank', 'server_order']).reset_index(drop=True)


def read_server(directory: Path) -> tuple[pd.DataFrame, dict]:
    results = pd.read_csv(directory / 'results_ratio03.csv')
    status_path = directory / 'status_latest.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    return results, status


def download(project: str, server: str, destination: Path) -> Path:
    import wandb

    artifact = wandb.Api().artifact(f'{project}/ratio03-a30-results-{server}:latest')
    return Path(artifact.download(root=str(destination / server)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('/scratch/lcornelis/ogbench/search_results/sep24_ratio03_transfer'),
        help='Parka campaign root with the final results_ratio03.csv',
    )
    parser.add_argument(
        '--server',
        action='append',
        default=[],
        metavar='NAME=DIR',
        help='Results directory of another server; repeat per server',
    )
    parser.add_argument(
        '--download',
        nargs='*',
        default=[],
        metavar='NAME',
        help='Download ratio03-a30-results-NAME:latest into ROOT/a30_results/NAME first',
    )
    parser.add_argument('--project', default='bioshape-lab/ogbench_sep24_ratio03_transfer')
    parser.add_argument('--output-dir', type=Path, help='Default: ROOT/merged')
    args = parser.parse_args()

    directories = {'parka': args.root}
    for server in args.download:
        directories[server] = download(args.project, server, args.root / 'a30_results')
    for entry in args.server:
        name, _, directory = entry.partition('=')
        directories[name] = Path(directory)

    loaded = {server: read_server(directory) for server, directory in directories.items()}
    merged = merge_results({server: results for server, (results, _) in loaded.items()})

    output_dir = args.output_dir or args.root / 'merged'
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / 'results_ratio03.csv', index=False)
    for table in TABLES:
        frames = [
            pd.read_csv(directory / table).assign(server=server)
            for server, directory in directories.items()
            if (directory / table).is_file() and (directory / table).stat().st_size > 1
        ]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output_dir / table, index=False)

    counts = merged['status'].value_counts().to_dict()
    statuses = [status for _, status in loaded.values()]
    summary = {
        'time': datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z'),
        'cells': len(merged),
        'status': counts,
        'folds_succeeded': int(merged['folds_done'].sum()),
        'folds_target': len(merged) * 5,
        **{key: int(sum(status.get(key, 0) for status in statuses)) for key in COUNTERS},
        'servers': merged.loc[merged['status'] == 'complete', 'server'].value_counts().to_dict(),
        'duplicate_complete_cells': int((merged['duplicate_servers'] != '').sum()),
    }
    notes = [
        '- Complete cells by server: '
        + ', '.join(f'{server} {count}' for server, count in summary['servers'].items()),
        f'- Complete cells also completed on another server: '
        f'{summary["duplicate_complete_cells"]}',
    ]
    (output_dir / 'coverage.md').write_text(
        coverage_markdown(merged, summary, final=True, notes=notes)
    )
    (output_dir / 'status_latest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()

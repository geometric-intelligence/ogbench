#!/usr/bin/env python3
"""Export W&B runs to CSV with progress bar.

Usage:
    python scripts/export_wandb_runs.py --entity bioshape-lab --project bgbench_multi_dataset_grid_search --output runs.csv

For multiple projects:
    python scripts/export_wandb_runs.py --entity bioshape-lab --project proj1 proj2 --output runs.csv
"""

import argparse
import os
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    import wandb
except ImportError:
    print('Please install wandb: pip install wandb')
    exit(1)


MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds, will be multiplied by attempt number


def flatten_dict(d: dict[str, Any], parent_key: str = '', sep: str = '.') -> dict[str, Any]:
    """Flatten nested dictionary with dot-separated keys."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten config and remove private keys."""
    flat = flatten_dict(config)
    return {k: v for k, v in flat.items() if not k.startswith('_')}


def process_run_with_retry(
    run, project: str, entity: str, max_retries: int = MAX_RETRIES
) -> dict[str, Any] | None:
    """Process a single run with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            config = normalize_config(run.config or {})
            summary = normalize_config(run.summary or {})

            row = {
                'run_id': run.id,
                'project': project,
                'entity': entity,
                'name': run.name,
                'state': run.state,
                'tags': ';'.join(run.tags or []),
                'created_at': run.created_at,
                'updated_at': run.updated_at,
                **config,
                **{f'summary.{k}': v for k, v in summary.items()},
            }
            return row
        except Exception as e:
            if attempt < max_retries:
                wait_time = RETRY_DELAY * attempt
                tqdm.write(
                    f'Error processing run {run.id}: {e}. Retrying in {wait_time}s... (attempt {attempt}/{max_retries})'
                )
                time.sleep(wait_time)
            else:
                tqdm.write(f'Failed to process run {run.id} after {max_retries} attempts: {e}')
                return None
    return None


def fetch_runs_with_retry(api, path: str, max_retries: int = MAX_RETRIES) -> list:
    """Fetch runs list with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            runs = api.runs(path)
            return list(runs)
        except Exception as e:
            if attempt < max_retries:
                wait_time = RETRY_DELAY * attempt
                print(
                    f'Error fetching runs from {path}: {e}. Retrying in {wait_time}s... (attempt {attempt}/{max_retries})'
                )
                time.sleep(wait_time)
            else:
                print(f'Failed to fetch runs from {path} after {max_retries} attempts: {e}')
                raise


def load_wandb_runs(
    entity: str,
    projects: list[str],
    output_path: str,
) -> pd.DataFrame:
    """Load all runs from W&B projects and save to CSV."""

    if not os.getenv('WANDB_API_KEY'):
        print('Warning: WANDB_API_KEY not set. You may need to run `wandb login` first.')

    api = wandb.Api()
    all_rows = []

    for project in projects:
        print(f'\nFetching runs from {entity}/{project}...')

        # Get all runs with retry logic
        runs_list = fetch_runs_with_retry(api, f'{entity}/{project}')
        print(f'Found {len(runs_list)} runs')

        failed_runs = 0
        for run in tqdm(runs_list, desc=f'Processing {project}', unit='run'):
            row = process_run_with_retry(run, project, entity)
            if row is not None:
                all_rows.append(row)
            else:
                failed_runs += 1

        if failed_runs > 0:
            print(f'Warning: {failed_runs} runs failed to process in {project}')

    df = pd.DataFrame(all_rows)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f'\nSaved {len(df)} runs to {output_path}')

    return df


def main():
    parser = argparse.ArgumentParser(description='Export W&B runs to CSV')
    parser.add_argument('--entity', '-e', required=True, help='W&B entity (username or team)')
    parser.add_argument('--project', '-p', nargs='+', required=True, help='W&B project name(s)')
    parser.add_argument('--output', '-o', default='wandb_runs.csv', help='Output CSV path')

    args = parser.parse_args()

    load_wandb_runs(
        entity=args.entity,
        projects=args.project,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()

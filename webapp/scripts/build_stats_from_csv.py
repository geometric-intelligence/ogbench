#!/usr/bin/env python3
"""Build webapp/public/data/stats.json from stats/<dataset>/graph_stats_comprehensive.csv.

Key format matches getStatsKey in the webapp: dataset|ratio|method|threshold|adjacency_method.

Node sample ratio is the sample-to-node ratio used by HFOmicsDataset
(``n_nodes = n_train / ratio``), so "full" (all features) is a distinct graph from
"1.0" and is kept as its own ratio value. Numeric ratios are normalized to the JS
string form (1.0 -> "1", 0.5 -> "0.5").

Usage:
  From repo root: python webapp/scripts/build_stats_from_csv.py
  From webapp:    python scripts/build_stats_from_csv.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = SCRIPT_DIR.parent
REPO_ROOT = WEBAPP_DIR.parent
STATS_DIR = REPO_ROOT / 'stats'
OUTPUT_PATH = WEBAPP_DIR / 'public' / 'data' / 'stats.json'

CSV_FILES = sorted(STATS_DIR.glob('*/graph_stats_comprehensive.csv'))

INT_METRICS = ('num_nodes', 'num_edges', 'num_connected_components')
FLOAT_METRICS = (
    'avg_degree',
    'density_pct',
    'largest_cc_ratio_pct',
    'degree_std',
    'clustering_coefficient',
    'diameter',
    'modularity',
    'homophily',
)


def normalize_ratio(node_sample_ratio: str) -> str:
    """Keep 'full' as-is; format numeric ratios the way JS stringifies numbers."""
    s = node_sample_ratio.strip().lower()
    if s == 'full':
        return 'full'
    return format_number(float(s))


def format_number(value: float) -> str:
    """Format a float so it matches JavaScript's String(number) (1.0 -> '1', 0.11 -> '0.11')."""
    if value == int(value):
        return str(int(value))
    return str(value)


def _float(val: str | None) -> float | None:
    if val is None or val.strip() == '':
        return None
    try:
        parsed = float(val)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def row_to_key_and_stats(row: dict[str, str]) -> tuple[str, dict[str, object]] | None:
    """Convert a CSV row to (key, stats) for the webapp JSON.

    Returns None to skip the row.
    """
    dataset = row.get('dataset', '').strip()
    method = row.get('method', '').strip()
    adjacency_method = row.get('adjacency_method', '').strip()
    thresh_val = _float(row.get('adj_thresh'))
    if not dataset or not method or not adjacency_method or thresh_val is None:
        return None
    if row.get('error', '').strip() or _float(row.get('num_nodes')) is None:
        return None

    ratio_key = normalize_ratio(row.get('node_sample_ratio', ''))
    key = f'{dataset}|{ratio_key}|{method}|{format_number(thresh_val)}|{adjacency_method}'

    stats: dict[str, object] = {}
    for metric in INT_METRICS:
        value = _float(row.get(metric))
        stats[metric] = int(value) if value is not None else 0
    for metric in FLOAT_METRICS:
        # null (not NaN) so the JSON stays parseable by fetch().json()
        stats[metric] = _float(row.get(metric))
    stats['dataset'] = dataset
    stats['adjacency_method'] = adjacency_method
    return key, stats


def main() -> None:
    result: dict[str, dict[str, object]] = {}
    for csv_path in CSV_FILES:
        count = 0
        with csv_path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                pair = row_to_key_and_stats(row)
                if pair is None:
                    continue
                key, stats = pair
                result[key] = stats
                count += 1
        print(f'Read {csv_path.relative_to(REPO_ROOT)} ({count} entries)')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(f'Wrote {len(result)} entries to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()

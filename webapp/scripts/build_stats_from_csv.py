#!/usr/bin/env python3
"""Build webapp/public/data/stats.json from the graph_stats_comprehensive_*.csv files.

Key format matches getStatsKey in webapp: dataset|ratio|method|threshold[|adjacency_method].
Node sample ratio "full" is normalized to 1.0 so the Explorer can look up by 1.0.
If the CSV contains an adjacency_method column, it is appended to the key.

Usage:
  From repo root: python webapp/scripts/build_stats_from_csv.py
  From webapp:    python scripts/build_stats_from_csv.py
"""

from __future__ import annotations

import json
from pathlib import Path

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = SCRIPT_DIR.parent
REPO_ROOT = WEBAPP_DIR.parent
STATS_DIR = REPO_ROOT / "tutorials" / "stats"
OUTPUT_PATH = WEBAPP_DIR / "public" / "data" / "stats.json"

WGCNA_CSV_FILES = [
    STATS_DIR / "addneuromed" / "graph_stats_comprehensive_addneuro.csv",
    STATS_DIR / "motrpac" / "graph_stats_comprehensive_motrpac.csv",
    STATS_DIR / "parkinsons" / "graph_stats_comprehensive_parkinsons.csv",
]

STRING_STATS_DIR = REPO_ROOT / "stats"
STRING_CSV_FILES = [
    STRING_STATS_DIR / "addneuromed" / "graph_stats_comprehensive.csv",
    STRING_STATS_DIR / "motrpac" / "graph_stats_comprehensive.csv",
    STRING_STATS_DIR / "parkinsons" / "graph_stats_comprehensive.csv",
]


def normalize_ratio(node_sample_ratio: str) -> str:
    """Map 'full' to '1'; keep numeric strings so they match JS (1.0 -> '1', 0.5 -> '0.5')."""
    s = node_sample_ratio.strip().lower()
    if s == "full":
        return "1"
    if s in ("1.0", "1"):
        return "1"
    return node_sample_ratio.strip()


def format_threshold(adj_thresh: float) -> str:
    """Format threshold so it matches JavaScript string representation (e.g. 0.11 not 0.1)."""
    t = float(adj_thresh)
    if t == int(t):
        return str(int(t))
    return str(t)


def row_to_key_and_stats(
    row: dict, *, default_adjacency_method: str = ""
) -> tuple[str, dict] | None:
    """Convert a CSV row to (key, stats) for the webapp JSON. Returns None to skip row."""
    dataset = row.get("dataset", "").strip()
    if not dataset:
        return None
    adj_thresh = row.get("adj_thresh", "")
    try:
        thresh_val = float(adj_thresh)
    except (TypeError, ValueError):
        return None
    ratio_key = normalize_ratio(row.get("node_sample_ratio", ""))
    method = row.get("method", "").strip()
    if not method:
        return None
    adjacency_method = row.get("adjacency_method", "").strip() or default_adjacency_method

    key = f"{dataset}|{ratio_key}|{method}|{format_threshold(thresh_val)}"
    if adjacency_method:
        key += f"|{adjacency_method}"

    def num(val, default=0):
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    stats = {
        "num_nodes": int(num(row.get("num_nodes"), 0)),
        "num_edges": int(num(row.get("num_edges"), 0)),
        "avg_degree": num(row.get("avg_degree")),
        "density_pct": num(row.get("density_pct")),
        "largest_cc_ratio_pct": num(row.get("largest_cc_ratio_pct")),
        "num_connected_components": int(num(row.get("num_connected_components"), 0)),
        "degree_std": num(row.get("degree_std")),
        "avg_clustering_coeff": num(row.get("avg_clustering_coeff")),
        "avg_shortest_path_length": num(row.get("avg_shortest_path_length")),
        "dataset": dataset,
    }
    if adjacency_method:
        stats["adjacency_method"] = adjacency_method
    return (key, stats)


def _read_csvs(
    csv_files: list[Path],
    result: dict[str, dict],
    *,
    default_adjacency_method: str = "",
) -> None:
    import csv as csv_module

    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"Skip (not found): {csv_path}")
            continue
        count = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                pair = row_to_key_and_stats(
                    row, default_adjacency_method=default_adjacency_method
                )
                if pair:
                    key, stats = pair
                    result[key] = stats
                    count += 1
        print(f"Read {csv_path} ({count} entries)")


def main() -> None:
    result: dict[str, dict] = {}
    _read_csvs(WGCNA_CSV_FILES, result, default_adjacency_method="wgcna")
    _read_csvs(STRING_CSV_FILES, result, default_adjacency_method="string")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {len(result)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

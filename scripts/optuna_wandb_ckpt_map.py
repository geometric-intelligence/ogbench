#!/usr/bin/env python3
"""Build a checkpoint manifest by querying W&B as the single source of truth.

For every study across ALL servers and ALL campaigns (original + rebalanced),
this script:

  1. Fetches all finished training runs from one or more W&B projects.
  2. Identifies the best validation trial per study (highest mean val/f1_macro
     across folds).
  3. Records the exact checkpoint directory from
     ``callbacks.model_checkpoint.dirpath`` — the absolute path on whichever
     server ran that trial.
  4. Derives ``server_root`` from the ``ckpt_dir`` prefix so the per-server
     filter in ``optuna_ckpt_test_eval.py --server-root`` is always reliable
     (``paths.root_dir`` in the W&B config can retain the yaml default if
     ``--root-dir`` was not passed at launch).

The output is a small CSV that you copy to every server.  Each server runs:

    python scripts/optuna_ckpt_test_eval.py \\
        --config configs/hparams_search/sep24_factorial_optuna.yaml \\
        --ckpt-map /path/to/wandb_ckpt_manifest.csv \\
        --server-root /scratch/<user>/ogbench \\
        --output-dir /scratch/<user>/ogbench/search_results/sep24_test_rebal \\
        --root-dir /scratch/<user>/ogbench \\
        --gpus 0 1 2 3 4 5 6 7 --jobs-per-gpu 2

Usage:

    python scripts/optuna_wandb_ckpt_map.py \\
        --project bioshape-lab/ogbench_sep24_factorial_optuna \\
        --project bioshape-lab/ogbench_sep24_factorial_tc10 \\
        --output /tmp/wandb_ckpt_manifest.csv

Columns in the output CSV
--------------------------
study_name, model, dataset, experiment,
dataset.loader.parameters.adjacency_method,
dataset.loader.parameters.node_sample_ratio,
dataset.loader.parameters.method,
dataset.loader.parameters.adjacency_threshold,
dataset.loader.parameters.adjacency_target_connectivity,
trial_number, fold,
ckpt_dir        -- path to checkpoints/ directory ON the server that ran it
server_root     -- derived from ckpt_dir prefix (e.g. /scratch/lcornelis/ogbench)
host            -- hostname logged by W&B
sampled_params  -- JSON of Optuna-sampled hyperparameters (from W&B args list)
val_f1_mean     -- mean val F1 of the best trial across folds (for reference)
wandb_run_id    -- W&B run ID for the best-trial fold run (for debugging)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Args that are NOT sampled hyperparameters: skip them when building
# sampled_params so the dict contains only Optuna-tuned keys.
# (used for runs that DO log _wandb.value.args — legacy / baseline format)
# ---------------------------------------------------------------------------
_SKIP_ARG_STARTS = (
    'model=',
    'dataset=',
    'logger.',
    'hydra.',
    'paths.',
    'seed=',
    'test=',
    'train=',
    '+optimized_metric=',
    'experiment=',
    'dataset.split_params.',
    'dataset.dataloader_params.',
    'dataset.loader.parameters.',  # outer cell: adjacency_method etc.
)

# Metadata keys that come from the args list (outer cell / structural fields)
_METADATA_ARG_KEYS = [
    'model',
    'dataset',
    'experiment',
    'dataset.loader.parameters.adjacency_method',
    'dataset.loader.parameters.node_sample_ratio',
    'dataset.loader.parameters.method',
    'dataset.loader.parameters.adjacency_threshold',
    'dataset.loader.parameters.adjacency_target_connectivity',
]

# ---------------------------------------------------------------------------
# Nested Hydra config extraction (Optuna runs — no _wandb.value.args)
# ---------------------------------------------------------------------------

# Sampled paths common to all models (nested → dot-notation key)
_COMMON_SAMPLED_NESTED = [
    (('optimizer', 'parameters', 'lr'), 'optimizer.parameters.lr'),
    (('optimizer', 'parameters', 'weight_decay'), 'optimizer.parameters.weight_decay'),
    (('model', 'backbone', 'dropout'), 'model.backbone.dropout'),
]
# Per-model extra sampled paths (superset — only present paths are emitted)
_EXTRA_SAMPLED_NESTED = [
    (('model', 'feature_encoder', 'out_channels'), 'model.feature_encoder.out_channels'),
    (('model', 'backbone', 'num_layers'), 'model.backbone.num_layers'),
    (('model', 'backbone', 'heads'), 'model.backbone.heads'),
    (('model', 'backbone', 'num_heads'), 'model.backbone.num_heads'),
    (('model', 'backbone', 'hidden_channels'), 'model.backbone.hidden_channels'),
    (('model', 'encodings'), 'model.encodings'),
]

# Known model config-group names (lowercase) used in Hydra overrides
_MODEL_NAME_MAP = {
    'gps': 'gps',
    'gin': 'gin',
    'gcn': 'gcn',
    'gatv2': 'gatv2',
    'sage': 'sage',
    'graphsage': 'sage',
    'graph_sage': 'sage',
    'sagn': 'sagn',
    'chebnet': 'chebnet',
    'mlagnn': 'gatv4',
    'gatv4': 'gatv4',
    'mlp': 'mlp',
}


def _nested_get(d: dict, *keys):
    """Walk a nested dict; return None if any key is missing."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _extract_from_nested_config(config: dict) -> dict:
    """Extract metadata + sampled_params from a nested Hydra W&B config dict.

    Used for Optuna search runs which do not log ``_wandb.value.args``.
    Returns a dict with the same keys as ``_METADATA_ARG_KEYS`` plus
    ``_sampled_params`` (dict, not JSON string yet).
    """
    # --- model name (Hydra config-group key) ---------------------------------
    model_raw = _nested_get(config, 'model', 'model_name') or ''
    model = _MODEL_NAME_MAP.get(str(model_raw).strip().lower(), str(model_raw).strip().lower())

    # --- dataset / loader parameters -----------------------------------------
    lp = _nested_get(config, 'dataset', 'loader', 'parameters') or {}
    dataset = lp.get('data_name')
    adj_meth = lp.get('adjacency_method')
    ratio = lp.get('node_sample_ratio')
    method = lp.get('method')
    threshold = lp.get('adjacency_threshold')
    conn = lp.get('adjacency_target_connectivity')

    # --- experiment (readout type only — adjacency is a separate override) ---
    # Tags: ['omics_readout', 'GPS', 'smoking', 'correlation']
    # Hydra config group is 'omics_readout' or 'no_readout'; adjacency_method
    # is passed separately via dataset.loader.parameters.adjacency_method.
    tags = config.get('tags') or []
    readout_tag = next((str(t) for t in tags if 'readout' in str(t).lower()), None)
    experiment = readout_tag  # e.g. 'omics_readout' or 'no_readout'

    # --- sampled hyper-parameters --------------------------------------------
    sampled: dict = {}
    for nested_keys, dot_key in _COMMON_SAMPLED_NESTED:
        v = _nested_get(config, *nested_keys)
        if v is not None:
            sampled[dot_key] = v
    for nested_keys, dot_key in _EXTRA_SAMPLED_NESTED:
        v = _nested_get(config, *nested_keys)
        if v is not None:
            sampled[dot_key] = v

    return {
        'model': model or None,
        'dataset': dataset,
        'experiment': experiment,
        'dataset.loader.parameters.adjacency_method': adj_meth,
        'dataset.loader.parameters.node_sample_ratio': ratio,
        'dataset.loader.parameters.method': method,
        'dataset.loader.parameters.adjacency_threshold': threshold,
        'dataset.loader.parameters.adjacency_target_connectivity': conn,
        '_sampled_params': sampled,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_run_name(run_name: str, group: str) -> tuple[int, int, int] | None:
    """Return (trial_number, fold, attempt) from a search run name, or None."""
    suffix = run_name[len(group) + 1 :] if run_name.startswith(group + '_') else run_name
    m = re.match(r'^trial(\d+)_fold(\d+)_attempt(\d+)$', suffix)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _val_f1(summary: dict) -> float | None:
    for key in ('best_val/f1_macro', 'val/f1_macro'):
        v = summary.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _ckpt_dir(config: dict) -> str | None:
    try:
        return config['callbacks']['model_checkpoint']['dirpath']
    except (KeyError, TypeError):
        return None


def _host(config: dict) -> str | None:
    try:
        return config['_wandb']['value']['host']
    except (KeyError, TypeError):
        return None


def _server_root_from_ckpt_dir(ckpt_dir: str | None) -> str | None:
    """Derive the server root from the ckpt_dir prefix.

    ``ckpt_dir`` is set from ``--output-dir`` at launch time and always
    reflects the actual server filesystem, unlike ``paths.root_dir`` which
    may retain the yaml default if ``--root-dir`` was not passed.

    Returns the first three meaningful path components, e.g.
    ``/scratch/lcornelis/ogbench`` from a path like
    ``/scratch/lcornelis/ogbench/search_results/…/checkpoints``.
    """
    if not ckpt_dir:
        return None
    # Use forward-slash string ops so the result is always a POSIX path,
    # regardless of the platform where this script is run.
    normalised = ckpt_dir.replace('\\', '/')
    parts = [p for p in normalised.split('/') if p]  # strip empty segments
    # Expect at least /scratch/<user>/<project>/…
    if len(parts) >= 3:
        return '/' + '/'.join(parts[:3])  # e.g. /scratch/lcornelis/ogbench
    return None


def _parse_all_args(config: dict) -> dict:
    """Parse the W&B args list into a flat {key: parsed_value} dict."""
    try:
        args: list = config['_wandb']['value']['args']
    except (KeyError, TypeError):
        args = []
    result: dict = {}
    for arg in args:
        if not isinstance(arg, str) or '=' not in arg:
            continue
        key, _, raw_val = arg.partition('=')
        try:
            result[key] = json.loads(raw_val)
        except (json.JSONDecodeError, ValueError):
            result[key] = raw_val
    return result


def _extract_metadata(all_args: dict) -> dict:
    """Pull out the structural / outer-cell fields from the parsed args."""
    return {key: all_args.get(key) for key in _METADATA_ARG_KEYS}


def _extract_sampled_params(all_args: dict) -> str:
    """Return a JSON string of Optuna-sampled keys only (no fixed/structural keys).

    ``_SKIP_ARG_STARTS`` entries ending in ``=`` are treated as exact key
    matches (e.g. ``model=`` skips only the key ``model``, not
    ``model.backbone.dropout``).  Entries ending in ``.`` are treated as
    prefix matches (e.g. ``logger.`` skips any key starting with ``logger.``).
    """
    sampled: dict = {}
    for k, v in all_args.items():
        skip = False
        for s in _SKIP_ARG_STARTS:
            if s.endswith('='):
                # Exact key match only
                if k == s[:-1]:
                    skip = True
                    break
            else:
                # Prefix match
                if k.startswith(s):
                    skip = True
                    break
        if not skip:
            sampled[k] = v
    return json.dumps(sampled, sort_keys=True)


# ---------------------------------------------------------------------------
# W&B fetch
# ---------------------------------------------------------------------------


def fetch_runs(projects: list[str]) -> pd.DataFrame:
    """Fetch all finished training runs from W&B and return a flat DataFrame."""
    api = wandb.Api(timeout=120)
    records: list[dict] = []

    for project in projects:
        print(f'Querying {project} …', flush=True)
        filters = {
            '$and': [
                {'tags': {'$in': ['optuna']}},
                {'tags': {'$nin': ['test_eval']}},
                {'state': 'finished'},
                # Baseline runs (e.g. baseline_svm_smoking_…) share the
                # project but do not follow the Optuna naming convention and
                # have no checkpoint to evaluate.  Filter them server-side so
                # they are never even fetched.
                {'displayName': {'$regex': '_trial\\d{4}_fold\\d+_attempt\\d+'}},
            ]
        }
        skipped = fetched = 0
        for run in api.runs(project, filters=filters, per_page=500):
            parsed = _parse_run_name(run.name, run.group or '')
            if parsed is None:
                skipped += 1
                continue
            trial_number, fold, attempt = parsed
            study_name = run.group or ''

            v = _val_f1(run.summary._json_dict)
            cd = _ckpt_dir(run.config)
            if cd is None:
                skipped += 1
                continue

            all_args = _parse_all_args(run.config)

            # For Optuna runs, _wandb.value.args is absent → extract from
            # the nested Hydra config dict instead.
            if all_args:
                # args-based format (legacy / baseline Hydra runs)
                meta = _extract_metadata(all_args)
                sampled_params = _extract_sampled_params(all_args)
                meta['_sampled_params'] = json.loads(sampled_params)
            else:
                meta = _extract_from_nested_config(run.config)
                sampled_params = json.dumps(meta.pop('_sampled_params', {}), sort_keys=True)
                meta['_sampled_params'] = json.loads(sampled_params)

            records.append(
                {
                    'study_name': study_name,
                    'trial_number': trial_number,
                    'fold': fold,
                    'attempt': attempt,
                    'val_f1': v,
                    'ckpt_dir': cd,
                    'server_root': _server_root_from_ckpt_dir(cd),
                    'host': _host(run.config),
                    'wandb_run_id': run.id,
                    'wandb_project': project,
                    '_all_args': meta,  # carries metadata + _sampled_params
                }
            )
            fetched += 1
            if fetched % 500 == 0:
                print(f'  … {fetched} runs collected', flush=True)

        print(f'  {project}: {fetched} training runs, {skipped} skipped')

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Best-trial selection
# ---------------------------------------------------------------------------


def select_best_trials(runs: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (study, fold) for the best trial per study.

    Best trial = highest mean val_f1 across folds, ties broken by highest trial_number.
    """
    if runs.empty:
        return runs

    # Latest attempt per (study, trial, fold)
    latest = runs.sort_values('attempt').drop_duplicates(
        ['study_name', 'trial_number', 'fold'], keep='last'
    )

    trial_means = (
        latest.groupby(['study_name', 'trial_number'])['val_f1']
        .mean()
        .reset_index()
        .rename(columns={'val_f1': 'val_f1_mean'})
    )
    best = (
        trial_means.sort_values(['val_f1_mean', 'trial_number'], ascending=[False, False])
        .drop_duplicates('study_name', keep='first')
        .rename(columns={'trial_number': 'best_trial_number'})
    )

    merged = latest.merge(
        best[['study_name', 'best_trial_number', 'val_f1_mean']],
        left_on=['study_name', 'trial_number'],
        right_on=['study_name', 'best_trial_number'],
    )
    return merged


# ---------------------------------------------------------------------------
# Build manifest
# ---------------------------------------------------------------------------


def build_manifest(best_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in best_runs.iterrows():
        meta: dict = row.get('_all_args') or {}
        # Extract metadata: prefer pre-computed keys from _extract_from_nested_config
        # (they sit directly in meta), else fall back to _extract_metadata(meta).
        if meta.get('model') is not None or meta.get('dataset') is not None:
            # Nested-config path: keys are already top-level in meta
            metadata = {k: meta.get(k) for k in _METADATA_ARG_KEYS}
            sampled_params = json.dumps(meta.get('_sampled_params', {}), sort_keys=True)
        else:
            # args-based path (should rarely happen after the fetch_runs fix)
            metadata = _extract_metadata(meta)
            sampled_params = _extract_sampled_params(meta)
        rows.append(
            {
                'study_name': row['study_name'],
                **metadata,
                'trial_number': int(row['trial_number']),
                'fold': int(row['fold']),
                'ckpt_dir': row['ckpt_dir'],
                'server_root': row.get('server_root'),
                'host': row.get('host'),
                'sampled_params': sampled_params,
                'val_f1_mean': row.get('val_f1_mean'),
                'wandb_run_id': row.get('wandb_run_id'),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project',
        action='append',
        required=True,
        metavar='ENTITY/PROJECT',
        help=(
            'W&B project to query. Repeat to cover multiple campaigns:\n'
            '  --project bioshape-lab/ogbench_sep24_factorial_optuna\n'
            '  --project bioshape-lab/ogbench_sep24_factorial_tc10'
        ),
    )
    parser.add_argument(
        '--output',
        required=True,
        type=Path,
        help='Destination CSV for the manifest.',
    )
    args = parser.parse_args()

    runs_df = fetch_runs(args.project)
    if runs_df.empty:
        print('ERROR: no runs fetched — check project names and W&B credentials')
        sys.exit(1)

    print(f'\nTotal runs fetched: {len(runs_df)} ' f'({runs_df["study_name"].nunique()} studies)')

    best_runs = select_best_trials(runs_df)
    print(
        f'Best-trial fold rows: {len(best_runs)} '
        f'({best_runs["study_name"].nunique()} studies × folds)'
    )

    manifest = build_manifest(best_runs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print(f'\nManifest: {len(manifest)} rows ' f'({manifest["study_name"].nunique()} studies)')
    print('By server_root:')
    for root, n in manifest['server_root'].value_counts().items():
        print(f'  {root}: {n} fold rows')
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()

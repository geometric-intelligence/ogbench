#!/usr/bin/env python3
"""Select one transferred configuration for every node_sample_ratio 0.3 cell.

The configuration of a ratio-0.3 cell is the best complete trial of the study
with the same model, dataset, experiment, adjacency method, and selection
method at ratio 0.5, falling back to the best partial 0.5 trial and then to the
best complete 0.8 trial. Trials are reconstructed from the finished fold runs
logged to W&B, because the tc10 campaign's studies are split across servers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.optuna_search import (  # noqa: E402
    ADJACENCY_METHOD,
    EXPERIMENT,
    NODE_SAMPLE_RATIO,
    SELECTION_METHOD,
    OptunaSearchConfig,
    OuterCell,
    _stable_hash,
    build_outer_cells,
    validate_sampled_parameters,
)

RUN_NAME = re.compile(
    r'^(?P<study>.+)_trial(?P<trial>\d{4})_fold(?P<fold>\d+)_attempt(?P<attempt>\d+)$'
)
# Mean ratio-0.5 fold time per model on Parka, cheapest first.
MODEL_COST_ORDER = (
    'mlp',
    'sagn',
    'gcn',
    'chebnet',
    'graph_sage',
    'gatv2',
    'gin',
    'gatv4',
    'gps',
)
SOURCE_RATIOS = (0.5, 0.8)


def cell_key(cell: OuterCell) -> tuple[str, str, str, str, str]:
    """Identify a cell independently of its node sample ratio."""
    return (
        cell.model,
        cell.dataset,
        str(cell.values[EXPERIMENT]),
        str(cell.values[ADJACENCY_METHOD]),
        str(cell.values[SELECTION_METHOD]),
    )


def ratio_token(ratio: float) -> str:
    return f'_r{str(ratio).replace(".", "p")}_'


def lookup(config: dict[str, Any], dotted: str) -> Any:
    """Read a dotted Hydra key from a nested W&B config."""
    value: Any = config
    for part in dotted.split('.'):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def search_keys(config: OptunaSearchConfig) -> list[str]:
    keys = set(config.search_space)
    for space in config.per_model_search_space.values():
        keys.update(space)
    return sorted(keys)


def fetch_runs(
    project: str,
    ratio: float,
    keys: list[str],
    objective_metric: str,
    cache_path: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Download finished fold runs for one source ratio, caching them as JSONL."""
    if cache_path.exists() and not refresh:
        return pd.read_json(cache_path, lines=True)
    import wandb

    api = wandb.Api(timeout=120)
    # lazy=False embeds config and summary in each page instead of one request per run.
    runs = api.runs(
        project,
        filters={'group': {'$regex': ratio_token(ratio)}, 'state': 'finished'},
        per_page=500,
        lazy=False,
    )
    records = []
    for index, run in enumerate(runs, start=1):
        match = RUN_NAME.match(run.name or '')
        if match is None:
            continue
        config = run.config
        values = {}
        for key in keys:
            try:
                values[key] = lookup(config, key)
            except KeyError:
                continue
        summary = run.summary
        records.append(
            {
                'run_id': run.id,
                'run_name': run.name,
                'study_name': match['study'],
                'trial_number': int(match['trial']),
                'fold': int(match['fold']),
                'attempt': int(match['attempt']),
                'objective': summary.get(objective_metric),
                'runtime': summary.get('_runtime'),
                'created_at': run.created_at,
                'config_values': json.dumps(values, sort_keys=True),
            }
        )
        if index % 2000 == 0:
            print(f'  ratio {ratio}: fetched {index} runs', flush=True)
    frame = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(cache_path, orient='records', lines=True)
    print(f'Fetched {len(frame)} finished ratio-{ratio} runs from {project}', flush=True)
    return frame


def sampled_from_values(
    config: OptunaSearchConfig, model: str, values: dict[str, Any]
) -> dict[str, Any] | None:
    """Rebuild a trial's sampled parameters from the values logged to W&B."""
    space = {**config.search_space, **config.per_model_search_space.get(model, {})}
    if any(name not in values for name in space):
        return None
    try:
        return validate_sampled_parameters(config, model, {name: values[name] for name in space})
    except ValueError:
        return None


@dataclass(frozen=True)
class TrialSummary:
    study_name: str
    param_hash: str
    sampled: dict[str, Any]
    fold_scores: dict[int, float]
    fold_runtimes: dict[int, float]
    trial_numbers: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return len(self.fold_scores) == 5

    @property
    def mean(self) -> float:
        return statistics.fmean(self.fold_scores.values())


def summarize_trials(
    config: OptunaSearchConfig,
    runs: pd.DataFrame,
    source_cells: dict[str, OuterCell],
) -> dict[str, list[TrialSummary]]:
    """Group finished fold runs by study and parameter set.

    Retried trials reuse durable folds under a new trial number, so folds are
    grouped by the parameter hash rather than by the trial number.
    """
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in runs.itertuples(index=False):
        cell = source_cells.get(row.study_name)
        if cell is None:
            continue
        objective = row.objective
        if objective is None or not math.isfinite(float(objective)):
            continue
        sampled = sampled_from_values(config, cell.model, json.loads(row.config_values))
        if sampled is None:
            continue
        param_hash = _stable_hash(sampled)
        entry = grouped.setdefault(
            (row.study_name, param_hash),
            {'sampled': sampled, 'scores': {}, 'runtimes': {}, 'created': {}, 'trials': set()},
        )
        fold = int(row.fold)
        created = str(row.created_at)
        if fold not in entry['created'] or created > entry['created'][fold]:
            entry['created'][fold] = created
            entry['scores'][fold] = float(objective)
            entry['runtimes'][fold] = float(row.runtime or 0.0)
        entry['trials'].add(int(row.trial_number))

    by_study: dict[str, list[TrialSummary]] = defaultdict(list)
    for (study_name, param_hash), entry in grouped.items():
        by_study[study_name].append(
            TrialSummary(
                study_name=study_name,
                param_hash=param_hash,
                sampled=entry['sampled'],
                fold_scores=dict(sorted(entry['scores'].items())),
                fold_runtimes=dict(sorted(entry['runtimes'].items())),
                trial_numbers=tuple(sorted(entry['trials'])),
            )
        )
    return by_study


def ledger_trials(
    config: OptunaSearchConfig, ledger_path: Path, source_cells: dict[str, OuterCell]
) -> dict[str, list[TrialSummary]]:
    """Rebuild trials from a campaign ledger, which also covers runs missing from W&B."""
    if not ledger_path.exists():
        return {}
    with sqlite3.connect(f'file:{ledger_path}?mode=ro', uri=True) as connection:
        rows = connection.execute(
            """
            SELECT study_name, param_hash, params_json, fold, metric, elapsed_time,
                   trial_number
            FROM fold_attempts WHERE status='success'
            ORDER BY attempt
            """
        ).fetchall()
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for study, param_hash, params_json, fold, metric, elapsed, trial_number in rows:
        cell = source_cells.get(study)
        if cell is None:
            continue
        sampled = sampled_from_values(config, cell.model, json.loads(params_json))
        if sampled is None or _stable_hash(sampled) != param_hash:
            continue
        entry = grouped.setdefault(
            (study, param_hash),
            {'sampled': sampled, 'scores': {}, 'runtimes': {}, 'trials': set()},
        )
        entry['scores'][int(fold)] = float(metric)
        entry['runtimes'][int(fold)] = float(elapsed)
        entry['trials'].add(int(trial_number))
    by_study: dict[str, list[TrialSummary]] = defaultdict(list)
    for (study, param_hash), entry in grouped.items():
        by_study[study].append(
            TrialSummary(
                study_name=study,
                param_hash=param_hash,
                sampled=entry['sampled'],
                fold_scores=dict(sorted(entry['scores'].items())),
                fold_runtimes=dict(sorted(entry['runtimes'].items())),
                trial_numbers=tuple(sorted(entry['trials'])),
            )
        )
    return by_study


def merge_trials(
    *sources: dict[str, list[TrialSummary]],
) -> dict[str, list[TrialSummary]]:
    """Union trials by study and parameter hash, keeping the version with most folds."""
    merged: dict[tuple[str, str], TrialSummary] = {}
    for source in sources:
        for trials in source.values():
            for trial in trials:
                key = (trial.study_name, trial.param_hash)
                if key not in merged or len(trial.fold_scores) > len(merged[key].fold_scores):
                    merged[key] = trial
    by_study: dict[str, list[TrialSummary]] = defaultdict(list)
    for (study, _param_hash), trial in merged.items():
        by_study[study].append(trial)
    return by_study


def best_trial(trials: list[TrialSummary], *, complete_only: bool) -> TrialSummary | None:
    pool = [trial for trial in trials if trial.complete or not complete_only]
    if not pool:
        return None
    # Optuna keeps the first maximum; the lowest trial number approximates it.
    return max(pool, key=lambda trial: (trial.mean, -min(trial.trial_numbers)))


def local_best_hashes(storage_path: Path, studies: set[str]) -> dict[str, str]:
    """Return the best complete trial's parameter hash for local Parka studies."""
    if not storage_path.exists():
        return {}
    storage = optuna.storages.RDBStorage(url=f'sqlite:///{storage_path}')
    best: dict[str, str] = {}
    for summary in optuna.study.get_all_study_summaries(storage=storage):
        if summary.study_name not in studies or summary.best_trial is None:
            continue
        param_hash = summary.best_trial.user_attrs.get('param_hash')
        if param_hash:
            best[summary.study_name] = str(param_hash)
    return best


def local_complete_means(ledger_path: Path) -> dict[tuple[str, str], float]:
    """Return the 5-fold mean of every complete parameter set in the Parka ledger."""
    if not ledger_path.exists():
        return {}
    with sqlite3.connect(f'file:{ledger_path}?mode=ro', uri=True) as connection:
        rows = connection.execute(
            """
            SELECT study_name, param_hash, fold, metric
            FROM fold_attempts WHERE status='success'
            ORDER BY attempt
            """
        ).fetchall()
    folds: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for study, param_hash, fold, metric in rows:
        folds[(study, param_hash)][int(fold)] = float(metric)
    return {
        key: statistics.fmean(scores.values())
        for key, scores in folds.items()
        if len(scores) == 5
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-config', default='configs/hparams_search/sep24_factorial_optuna.yaml'
    )
    parser.add_argument(
        '--target-config', default='configs/hparams_search/sep24_ratio03_transfer.yaml'
    )
    parser.add_argument('--project', default='bioshape-lab/ogbench_sep24_factorial_tc10')
    parser.add_argument(
        '--local-root',
        default='/scratch/lcornelis/ogbench/search_results/sep24_factorial_tc10',
        help='Parka campaign root used to cross-check the W&B reconstruction',
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--refresh', action='store_true', help='Re-download W&B runs')
    parser.add_argument(
        '--source-ratios',
        nargs='+',
        type=float,
        default=list(SOURCE_RATIOS),
        help='Ratios to read; 0.8 is only a fallback for cells without any 0.5 trial',
    )
    parser.add_argument(
        '--max-mismatch-fraction',
        type=float,
        default=0.02,
        help='Abort if more local best trials disagree with W&B than this fraction',
    )
    args = parser.parse_args()

    source = OptunaSearchConfig.from_yaml(args.source_config)
    target = OptunaSearchConfig.from_yaml(args.target_config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_cells = build_outer_cells(source)
    target_cells = build_outer_cells(target)
    source_by_ratio: dict[float, dict[tuple[str, ...], OuterCell]] = defaultdict(dict)
    for cell in source_cells:
        source_by_ratio[float(cell.values[NODE_SAMPLE_RATIO])][cell_key(cell)] = cell
    source_lookup = {cell.study_name: cell for cell in source_cells}

    keys = search_keys(source)
    source_ratios = tuple(args.source_ratios)
    if 0.5 not in source_ratios:
        parser.error('--source-ratios must include 0.5')
    trials_by_study: dict[str, list[TrialSummary]] = {}
    for ratio in source_ratios:
        runs = fetch_runs(
            args.project,
            ratio,
            keys,
            source.objective_metric,
            output_dir / 'wandb_cache' / f'ratio_{ratio}.jsonl',
            args.refresh,
        )
        trials_by_study.update(summarize_trials(source, runs, source_lookup))

    # Cross-check the W&B reconstruction against Parka's own ledger and studies:
    # the same complete parameter sets must have the same 5-fold means, and,
    # restricted to those sets, W&B must pick the same best trial as Optuna.
    local_root = Path(args.local_root)
    source_studies = {
        cell.study_name for ratio in source_ratios for cell in source_by_ratio[ratio].values()
    }
    local_means = {
        key: mean
        for key, mean in local_complete_means(local_root / 'run_ledger.sqlite3').items()
        if key[0] in source_studies
    }
    wandb_trials = {
        (trial.study_name, trial.param_hash): trial
        for trials in trials_by_study.values()
        for trial in trials
        if trial.complete
    }
    matched = [key for key in local_means if key in wandb_trials]
    missing_from_wandb = len(local_means) - len(matched)
    max_mean_gap = max(
        (abs(local_means[key] - wandb_trials[key].mean) for key in matched), default=0.0
    )
    print(
        f'Parka complete parameter sets: {len(local_means)}; on W&B: {len(matched)} '
        f'(missing {missing_from_wandb}); max 5-fold mean gap {max_mean_gap:.2e}'
    )

    local_best = local_best_hashes(local_root / 'studies.db', source_studies)
    compared = 0
    mismatches = []
    for study_name, local_hash in sorted(local_best.items()):
        local_hashes = {param_hash for study, param_hash in local_means if study == study_name}
        pool = [
            trial
            for trial in trials_by_study.get(study_name, [])
            if trial.param_hash in local_hashes
        ]
        if local_hash not in {trial.param_hash for trial in pool}:
            continue
        wandb_best = best_trial(pool, complete_only=True)
        compared += 1
        if wandb_best is not None and wandb_best.param_hash != local_hash:
            mismatches.append(
                {
                    'study_name': study_name,
                    'local_best_hash': local_hash,
                    'wandb_best_hash': wandb_best.param_hash,
                    'wandb_best_mean': wandb_best.mean,
                    'local_best_mean': local_means[(study_name, local_hash)],
                }
            )
    pd.DataFrame(mismatches).to_csv(output_dir / 'crosscheck_mismatches.csv', index=False)
    mismatch_fraction = len(mismatches) / compared if compared else 1.0
    print(
        f'Cross-check against Parka studies.db: {compared} studies compared, '
        f'{len(mismatches)} best-trial mismatches ({mismatch_fraction:.1%})'
    )
    if compared == 0 or mismatch_fraction > args.max_mismatch_fraction or max_mean_gap > 1e-6:
        raise SystemExit('W&B reconstruction disagrees with the local Optuna studies')

    # W&B only holds runs from Parka-origin studies, so Parka's ledger adds the
    # other source ratios for cells whose ratio-0.5 study ran on Frank or Hall.
    trials_by_study = merge_trials(
        trials_by_study,
        ledger_trials(source, local_root / 'run_ledger.sqlite3', source_lookup),
    )

    def best_for(keys: list[tuple[str, ...]], ratio: float) -> TrialSummary | None:
        pool = [
            trial
            for key in keys
            if key in source_by_ratio[ratio]
            for trial in trials_by_study.get(source_by_ratio[ratio][key].study_name, [])
        ]
        return best_trial(pool, complete_only=True)

    all_keys = list(source_by_ratio[0.5])
    candidates: dict[str, dict[str, Any]] = {}
    rows = []
    for cell in target_cells:
        key = cell_key(cell)
        siblings = [
            other for other in all_keys if other[:4] == key[:4] and other != key
        ]
        cousins = [
            other for other in all_keys if other[:3] == key[:3] and other[:4] != key[:4]
        ]
        relatives = [
            other for other in all_keys if other[:2] == key[:2] and other[:3] != key[:3]
        ]
        chosen = None
        rule = None
        for label, keys in (
            ('same_cell', [key]),
            ('other_selection_method', siblings),
            ('other_adjacency', cousins),
            ('other_experiment', relatives),
        ):
            for ratio in (0.5, 0.8, 1.0):
                chosen = best_for(keys, ratio)
                if chosen is not None:
                    rule = f'{label}_r{ratio}'
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise SystemExit(f'No transferable trial for {cell.study_name}')
        validate_sampled_parameters(target, cell.model, chosen.sampled)
        runtimes = list(chosen.fold_runtimes.values())
        candidates[cell.study_name] = {
            'sampled_params': chosen.sampled,
            'param_hash': chosen.param_hash,
            'source_study': chosen.study_name,
            'source_rule': rule,
            'source_trial_numbers': list(chosen.trial_numbers),
            'source_mean': chosen.mean,
            'source_folds': len(chosen.fold_scores),
        }
        rows.append(
            {
                **cell.result_fields(),
                'study_name': cell.study_name,
                'source_study': chosen.study_name,
                'source_rule': rule,
                'source_mean': chosen.mean,
                'source_folds': len(chosen.fold_scores),
                'source_mean_fold_seconds': statistics.fmean(runtimes) if runtimes else None,
                'param_hash': chosen.param_hash,
                'sampled_params': json.dumps(chosen.sampled, sort_keys=True),
            }
        )

    table = pd.DataFrame(rows)
    model_rank = {model: index for index, model in enumerate(MODEL_COST_ORDER)}
    fallback_seconds = table.groupby('model')['source_mean_fold_seconds'].transform('median')
    table['estimated_fold_seconds'] = table['source_mean_fold_seconds'].fillna(fallback_seconds)
    table['model_rank'] = table['model'].map(model_rank)
    table = table.sort_values(['model_rank', 'estimated_fold_seconds', 'study_name'])
    table['priority'] = range(1, len(table) + 1)

    candidates_path = output_dir / 'candidates.json'
    candidates_path.write_text(json.dumps(candidates, indent=2, sort_keys=True) + '\n')
    priority_path = output_dir / 'priority.txt'
    priority_path.write_text('\n'.join(table['study_name']) + '\n')
    table_path = output_dir / 'candidates.csv'
    table.to_csv(table_path, index=False)

    checksum_lines = [
        f'{sha256(path)}  {path.name}' for path in (candidates_path, priority_path, table_path)
    ]
    (output_dir / 'SHA256SUMS.candidates').write_text('\n'.join(checksum_lines) + '\n')
    print(f'Wrote {len(candidates)} candidates to {candidates_path}')
    print(table['source_rule'].value_counts().to_string())
    hours = table['estimated_fold_seconds'].sum() * len(target.folds) / 3600
    print(f'Estimated slot-hours at source-ratio speed: {hours:.0f}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run resumable Optuna studies across fixed ablation cells.

Each Optuna trial evaluates one hyperparameter configuration over every configured k-fold split.
Dataset/model/experiment/data-construction choices remain outer ablation axes and are never
optimized against one another.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import random
import re
import shlex
import sqlite3
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from joblib import Parallel, delayed, parallel_backend
from omegaconf import OmegaConf
from optuna.trial import TrialState

from ogbench.utils.config_resolvers import register_all_resolvers
from ogbench.utils.hparam_search import (
    OOM_ERROR_PREFIX,
    GpuDevice,
    acquire_gpu,
    enforce_single_thread_process,
    is_oom_failure,
    populate_gpu_queue,
    run_training,
    to_override,
    visible_gpu_devices,
)

register_all_resolvers()

ADJACENCY_METHOD = 'dataset.loader.parameters.adjacency_method'
NODE_SAMPLE_RATIO = 'dataset.loader.parameters.node_sample_ratio'
SELECTION_METHOD = 'dataset.loader.parameters.method'
ADJACENCY_THRESHOLD = 'dataset.loader.parameters.adjacency_threshold'
ADJACENCY_TARGET_CONNECTIVITY = 'dataset.loader.parameters.adjacency_target_connectivity'
EXPERIMENT = 'experiment'
REQUIRED_ABLATIONS = (
    EXPERIMENT,
    ADJACENCY_METHOD,
    NODE_SAMPLE_RATIO,
    SELECTION_METHOD,
)
SCALAR_TYPES = (str, int, float, bool, type(None))


class FoldExecutionError(RuntimeError):
    """A fold could not produce the configured objective metric."""


@dataclass(frozen=True)
class ExecutionPolicy:
    """Runtime-only scheduling choices that never affect study comparability."""

    min_free_gpu_mib: int | None = None
    oom_retries: int = 0
    oom_min_free_gpu_mib: int | None = None
    oom_backoff_seconds: float = 60.0
    oom_gpu_wait_seconds: float = 900.0
    gpu_poll_seconds: float = 30.0


DEFAULT_POLICY = ExecutionPolicy()


@dataclass(frozen=True)
class SearchSpaceSpec:
    """One typed Optuna search-space entry."""

    kind: str
    choices: tuple[Any, ...] = ()
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    log: bool = False

    @classmethod
    def from_raw(cls, name: str, raw: Any) -> SearchSpaceSpec:
        """Parse a search-space entry from YAML."""
        if isinstance(raw, list):
            if not raw:
                raise ValueError(f"Search space '{name}' has no choices")
            return cls(kind='categorical', choices=tuple(raw))
        if not isinstance(raw, dict):
            raise TypeError(f"Search space '{name}' must be a list or mapping")

        kind = str(raw.get('type', '')).lower()
        if kind in {'categorical', 'choice'}:
            choices = raw.get('choices')
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Categorical search space '{name}' requires non-empty choices")
            return cls(kind='categorical', choices=tuple(choices))
        if kind not in {'int', 'float'}:
            raise ValueError(
                f"Search space '{name}' has unsupported type {kind!r}; "
                'expected categorical, int, or float'
            )
        if 'low' not in raw or 'high' not in raw:
            raise ValueError(f"Search space '{name}' requires low and high")
        low = raw['low']
        high = raw['high']
        if low >= high:
            raise ValueError(f"Search space '{name}' requires low < high")
        step = raw.get('step')
        log = bool(raw.get('log', False))
        if log and step is not None:
            raise ValueError(f"Search space '{name}' cannot specify both log and step")
        return cls(kind=kind, low=low, high=high, step=step, log=log)

    @property
    def structured(self) -> bool:
        """Whether Optuna stores these categorical choices as JSON strings."""
        return self.kind == 'categorical' and any(
            not isinstance(choice, SCALAR_TYPES) for choice in self.choices
        )

    def suggest(self, trial: optuna.Trial, name: str) -> Any:
        """Sample this entry, decoding structured categorical values."""
        if self.kind == 'categorical':
            if self.structured:
                encoded = tuple(json.dumps(choice, sort_keys=True) for choice in self.choices)
                return json.loads(trial.suggest_categorical(name, encoded))
            return trial.suggest_categorical(name, self.choices)
        if self.kind == 'int':
            kwargs: dict[str, Any] = {'log': self.log}
            if self.step is not None:
                kwargs['step'] = int(self.step)
            return trial.suggest_int(name, int(self.low), int(self.high), **kwargs)
        kwargs = {'log': self.log}
        if self.step is not None:
            kwargs['step'] = float(self.step)
        return trial.suggest_float(name, float(self.low), float(self.high), **kwargs)

    def canonical(self, value: Any) -> Any:
        """Return the in-space value equal to a sampled value, or raise ValueError."""
        if self.kind == 'categorical':
            for choice in self.choices:
                if isinstance(choice, bool) == isinstance(value, bool) and choice == value:
                    return choice
            raise ValueError(f'{value!r} is not one of {list(self.choices)}')
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f'{value!r} is not numeric')
        tolerance = 1e-9 * max(1.0, abs(float(self.high)))
        if not float(self.low) - tolerance <= value <= float(self.high) + tolerance:
            raise ValueError(f'{value!r} is outside [{self.low}, {self.high}]')
        if self.step is not None:
            steps = (value - self.low) / self.step
            if abs(steps - round(steps)) > 1e-6:
                raise ValueError(f'{value!r} is not on the step grid of {self.step}')
        return int(value) if self.kind == 'int' else float(value)

    def raw(self, value: Any) -> Any:
        """Return the value Optuna stores in trial.params for a sampled value."""
        return json.dumps(value, sort_keys=True) if self.structured else value


@dataclass(frozen=True)
class OuterCell:
    """A fixed ablation cell with its resolved adjacency threshold."""

    model: str
    dataset: str
    values: dict[str, Any]
    study_name: str

    def result_fields(self) -> dict[str, Any]:
        return {'model': self.model, 'dataset': self.dataset, **self.values}


@dataclass
class OptunaSearchConfig:
    """Configuration for the multi-study Optuna launcher."""

    source_path: Path
    datasets: list[str]
    models: list[str]
    folds: list[int]
    k: int
    training_seed: int
    fixed: dict[str, Any]
    ablations: dict[str, list[Any]]
    ablation_mode: str
    ablation_baseline: dict[str, Any]
    per_model_ablation_baseline: dict[str, dict[str, Any]]
    exclude_cells: list[dict[str, Any]]
    search_space: dict[str, SearchSpaceSpec]
    per_model_search_space: dict[str, dict[str, SearchSpaceSpec]]
    thresholds: dict[tuple[str, float | str, str], float]
    string_adjacency_threshold: float
    wgcna_target_connectivity: float | None
    objective_metric: str
    direction: str
    n_trials: int
    sampler_seed: int
    n_startup_trials: int
    storage: str
    study_name_prefix: str
    heartbeat_interval: int
    grace_period: int
    max_retries: int
    timeout: int
    output_dir: Path
    tags: list[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> OptunaSearchConfig:
        source_path = Path(path).resolve()
        with source_path.open() as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise TypeError('Optuna search config must contain a YAML mapping')

        folds = [int(fold) for fold in raw.get('folds', range(5))]
        k = int(raw.get('k', 5))
        if sorted(set(folds)) != list(range(k)):
            raise ValueError(f'folds must contain every fold exactly once from 0 to k-1: k={k}')

        ablations = raw.get('ablations', {})
        if not isinstance(ablations, dict):
            raise TypeError('ablations must be a mapping')
        missing = [key for key in REQUIRED_ABLATIONS if key not in ablations]
        if missing:
            raise ValueError(f'Missing required outer ablation axes: {missing}')
        for name, values in ablations.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"Ablation axis '{name}' must be a non-empty list")
        ablation_design = raw.get('ablation_design', {})
        if not isinstance(ablation_design, dict):
            raise TypeError('ablation_design must be a mapping')
        ablation_mode = str(ablation_design.get('mode', 'full_factorial'))
        if ablation_mode not in {'full_factorial', 'one_factor_at_a_time'}:
            raise ValueError(
                "ablation_design.mode must be 'full_factorial' or 'one_factor_at_a_time'"
            )
        ablation_baseline = dict(ablation_design.get('baseline', {}))
        per_model_ablation_baseline = {
            str(model): dict(values)
            for model, values in ablation_design.get('per_model_baseline', {}).items()
        }
        if ablation_mode == 'one_factor_at_a_time':
            missing_baselines = set(ablations) - set(ablation_baseline)
            if missing_baselines:
                raise ValueError(
                    'one_factor_at_a_time requires a baseline for every ablation axis: '
                    f'{sorted(missing_baselines)}'
                )
            _validate_ablation_baseline('baseline', ablation_baseline, ablations)
            for model, baseline in per_model_ablation_baseline.items():
                if model not in raw['models']:
                    raise ValueError(f'Unknown per-model ablation baseline: {model}')
                merged = {**ablation_baseline, **baseline}
                _validate_ablation_baseline(f'per_model_baseline.{model}', merged, ablations)
        exclude_cells = raw.get('exclude_cells', [])
        if not isinstance(exclude_cells, list) or any(
            not isinstance(item, dict) for item in exclude_cells
        ):
            raise TypeError('exclude_cells must be a list of mappings')

        search_space = _parse_search_space(raw.get('search_space', {}))
        per_model_search_space = {
            model: _parse_search_space(space)
            for model, space in raw.get('per_model_search_space', {}).items()
        }

        has_legacy_thresholds = raw.get('per_dataset_ratio_method_grid') is not None or bool(
            raw.get('thresholds_from')
        )
        if 'wgcna_target_connectivity' in raw:
            target_connectivity_raw = raw.get('wgcna_target_connectivity')
            wgcna_target_connectivity = (
                None if target_connectivity_raw is None else float(target_connectivity_raw)
            )
        else:
            wgcna_target_connectivity = None if has_legacy_thresholds else 0.10
        if wgcna_target_connectivity is not None and not 0 <= wgcna_target_connectivity <= 1:
            raise ValueError('wgcna_target_connectivity must be between 0 and 1')

        threshold_raw = raw.get('per_dataset_ratio_method_grid')
        if threshold_raw is None:
            threshold_source = raw.get('thresholds_from')
            if not threshold_source and wgcna_target_connectivity is None:
                raise ValueError(
                    'Set per_dataset_ratio_method_grid, thresholds_from, or '
                    'wgcna_target_connectivity in the Optuna config'
                )
            if threshold_source:
                threshold_path = Path(threshold_source)
                if not threshold_path.is_absolute():
                    threshold_path = source_path.parent / threshold_path
                with threshold_path.open() as handle:
                    threshold_config = yaml.safe_load(handle)
                threshold_raw = threshold_config.get('per_dataset_ratio_method_grid', {})
            else:
                threshold_raw = {}
        thresholds = _parse_thresholds(threshold_raw)

        objective = raw.get('objective', {})
        direction = str(objective.get('direction', 'maximize'))
        if direction not in {'maximize', 'minimize'}:
            raise ValueError("objective.direction must be 'maximize' or 'minimize'")

        optuna_config = raw.get('optuna', {})
        training = raw.get('training', {})
        path_base = _find_project_root(source_path)
        output_dir = Path(training.get('output_dir', './search_results/optuna'))
        if not output_dir.is_absolute():
            output_dir = path_base / output_dir
        output_dir = output_dir.resolve()
        storage = str(optuna_config.get('storage') or f'sqlite:///{output_dir / "studies.db"}')
        storage = _normalize_sqlite_url(storage, path_base)

        config = cls(
            source_path=source_path,
            datasets=list(raw['datasets']),
            models=list(raw['models']),
            folds=folds,
            k=k,
            training_seed=int(raw.get('training_seed', 42)),
            fixed=dict(raw.get('fixed', {})),
            ablations={key: list(values) for key, values in ablations.items()},
            ablation_mode=ablation_mode,
            ablation_baseline=ablation_baseline,
            per_model_ablation_baseline=per_model_ablation_baseline,
            exclude_cells=[dict(item) for item in exclude_cells],
            search_space=search_space,
            per_model_search_space=per_model_search_space,
            thresholds=thresholds,
            string_adjacency_threshold=float(raw.get('string_adjacency_threshold', 0.4)),
            wgcna_target_connectivity=wgcna_target_connectivity,
            objective_metric=str(objective.get('metric', 'best_val/f1_macro')),
            direction=direction,
            n_trials=int(optuna_config.get('n_trials', 20)),
            sampler_seed=int(optuna_config.get('sampler_seed', 1234)),
            n_startup_trials=int(optuna_config.get('n_startup_trials', 10)),
            storage=storage,
            study_name_prefix=str(optuna_config.get('study_name_prefix', 'ogbench')),
            heartbeat_interval=int(optuna_config.get('heartbeat_interval', 60)),
            grace_period=int(optuna_config.get('grace_period', 300)),
            max_retries=int(optuna_config.get('max_retries', 1)),
            timeout=int(training.get('timeout', 3600)),
            output_dir=output_dir,
            tags=list(raw.get('tags', [])),
        )
        if config.n_trials < 1:
            raise ValueError('optuna.n_trials must be positive')
        if config.heartbeat_interval < 1 or config.grace_period < 1:
            raise ValueError('Optuna heartbeat_interval and grace_period must be positive')
        if config.max_retries < 0:
            raise ValueError('optuna.max_retries cannot be negative')
        unknown_models = set(config.per_model_search_space) - set(config.models)
        if unknown_models:
            raise ValueError(f'Per-model spaces reference unknown models: {unknown_models}')
        return config

    @property
    def fingerprint(self) -> str:
        """Hash settings that affect objective comparability."""
        comparable_fixed = {
            key: value for key, value in self.fixed.items() if key != 'paths.root_dir'
        }
        payload = {
            'datasets': self.datasets,
            'models': self.models,
            'folds': self.folds,
            'k': self.k,
            'training_seed': self.training_seed,
            'fixed': comparable_fixed,
            'ablations': self.ablations,
            'ablation_mode': self.ablation_mode,
            'ablation_baseline': self.ablation_baseline,
            'per_model_ablation_baseline': self.per_model_ablation_baseline,
            'exclude_cells': self.exclude_cells,
            'search_space': {
                key: asdict(value) for key, value in sorted(self.search_space.items())
            },
            'per_model_search_space': {
                model: {key: asdict(value) for key, value in sorted(space.items())}
                for model, space in sorted(self.per_model_search_space.items())
            },
            'thresholds': sorted((str(key), value) for key, value in self.thresholds.items()),
            'string_adjacency_threshold': self.string_adjacency_threshold,
            'wgcna_target_connectivity': self.wgcna_target_connectivity,
            'objective_metric': self.objective_metric,
            'direction': self.direction,
            'sampler_seed': self.sampler_seed,
            'n_startup_trials': self.n_startup_trials,
        }
        return _stable_hash(payload)


class RunLedger:
    """Durable append-only records for every fold attempt."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fold_attempts (
                    study_name TEXT NOT NULL,
                    param_hash TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    fold INTEGER NOT NULL,
                    training_seed INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    metric REAL,
                    elapsed_time REAL NOT NULL,
                    error TEXT,
                    log_path TEXT,
                    trial_number INTEGER NOT NULL,
                    gpu_logical_id INTEGER,
                    gpu_visibility_token TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (
                        study_name, param_hash, fold, training_seed, attempt
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS fold_attempt_lookup
                ON fold_attempts (
                    study_name, param_hash, fold, training_seed, status
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        return connection

    def next_attempt(self, study_name: str, param_hash: str, fold: int, training_seed: int) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(attempt), 0) AS attempt
                FROM fold_attempts
                WHERE study_name=? AND param_hash=? AND fold=? AND training_seed=?
                """,
                (study_name, param_hash, fold, training_seed),
            ).fetchone()
        return int(row['attempt']) + 1

    def successful_metric(
        self, study_name: str, param_hash: str, fold: int, training_seed: int
    ) -> float | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT metric FROM fold_attempts
                WHERE study_name=? AND param_hash=? AND fold=? AND training_seed=?
                  AND status='success'
                ORDER BY attempt DESC LIMIT 1
                """,
                (study_name, param_hash, fold, training_seed),
            ).fetchone()
        return None if row is None else float(row['metric'])

    def record(
        self,
        *,
        study_name: str,
        param_hash: str,
        params: dict[str, Any],
        fold: int,
        training_seed: int,
        attempt: int,
        status: str,
        metric: float | None,
        elapsed_time: float,
        error: str | None,
        log_path: Path,
        trial_number: int,
        gpu: GpuDevice | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO fold_attempts (
                    study_name, param_hash, params_json, fold, training_seed,
                    attempt, status, metric, elapsed_time, error, log_path,
                    trial_number, gpu_logical_id, gpu_visibility_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study_name,
                    param_hash,
                    json.dumps(params, sort_keys=True),
                    fold,
                    training_seed,
                    attempt,
                    status,
                    metric,
                    elapsed_time,
                    error,
                    str(log_path),
                    trial_number,
                    None if gpu is None else gpu.logical_id,
                    None if gpu is None else gpu.visibility_token,
                ),
            )

    def attempt_budget_used(
        self,
        study_name: str,
        param_hash: str,
        fold: int,
        training_seed: int,
        oom_retries: int = 0,
    ) -> int:
        """Count failed attempts, excluding the first ``oom_retries`` out-of-memory failures."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN error LIKE ? THEN 0 ELSE 1 END), 0) AS normal,
                    COALESCE(SUM(CASE WHEN error LIKE ? THEN 1 ELSE 0 END), 0) AS oom
                FROM fold_attempts
                WHERE study_name=? AND param_hash=? AND fold=? AND training_seed=?
                  AND status='failed'
                """,
                (
                    f'{OOM_ERROR_PREFIX}%',
                    f'{OOM_ERROR_PREFIX}%',
                    study_name,
                    param_hash,
                    fold,
                    training_seed,
                ),
            ).fetchone()
        return int(row['normal']) + max(0, int(row['oom']) - oom_retries)

    def retryable(
        self,
        study_name: str,
        param_hash: str,
        max_attempts: int,
        expected_folds: Sequence[int] | None = None,
        oom_retries: int = 0,
    ) -> bool:
        """Return whether any unresolved fold can still be attempted."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT fold,
                       SUM(CASE WHEN status='failed' AND error LIKE ? THEN 1 ELSE 0 END)
                           AS oom,
                       SUM(CASE WHEN status='failed' AND (error IS NULL OR error NOT LIKE ?)
                           THEN 1 ELSE 0 END) AS normal,
                       MAX(CASE WHEN status='success' THEN 1 ELSE 0 END) AS succeeded
                FROM fold_attempts
                WHERE study_name=? AND param_hash=?
                GROUP BY fold
                """,
                (f'{OOM_ERROR_PREFIX}%', f'{OOM_ERROR_PREFIX}%', study_name, param_hash),
            ).fetchall()
        by_fold = {int(row['fold']): row for row in rows}

        def budget_used(row: sqlite3.Row) -> int:
            return int(row['normal']) + max(0, int(row['oom']) - oom_retries)

        if expected_folds is None:
            return any(
                not row['succeeded'] and budget_used(row) < max_attempts for row in rows
            )
        # All folds may already be durable when a worker dies before Optuna
        # commits the aggregate objective. Re-enqueueing finalizes it without
        # rerunning those folds.
        if all(by_fold.get(fold) and by_fold[fold]['succeeded'] for fold in expected_folds):
            return True
        # Folds run in order. A permanently failed earlier fold blocks later
        # missing folds, so only the first incomplete fold determines whether
        # this parameter set is retryable.
        for fold in expected_folds:
            row = by_fold.get(fold)
            if row and row['succeeded']:
                continue
            return row is None or budget_used(row) < max_attempts
        return False

    def unresolved_failures(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT failed.*
                FROM fold_attempts AS failed
                WHERE failed.status='failed'
                  AND failed.attempt = (
                      SELECT MAX(latest.attempt)
                      FROM fold_attempts AS latest
                      WHERE latest.study_name=failed.study_name
                        AND latest.param_hash=failed.param_hash
                        AND latest.fold=failed.fold
                        AND latest.training_seed=failed.training_seed
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM fold_attempts AS succeeded
                      WHERE succeeded.study_name=failed.study_name
                        AND succeeded.param_hash=failed.param_hash
                        AND succeeded.fold=failed.fold
                        AND succeeded.training_seed=failed.training_seed
                        AND succeeded.status='success'
                  )
                ORDER BY failed.study_name, failed.param_hash, failed.fold
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def all_attempts(self) -> list[dict[str, Any]]:
        """Return the durable fold-attempt history for CSV export."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM fold_attempts
                ORDER BY study_name, param_hash, fold, attempt
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def interruption_recorded(self, study_name: str, param_hash: str, trial_number: int) -> bool:
        """Return whether this stale Optuna trial already has a ledger marker."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM fold_attempts
                WHERE study_name=? AND param_hash=? AND trial_number=?
                  AND status='failed'
                  AND error='Worker interrupted before this fold produced a durable result'
                LIMIT 1
                """,
                (study_name, param_hash, trial_number),
            ).fetchone()
        return row is not None


def _parse_search_space(raw: Any) -> dict[str, SearchSpaceSpec]:
    if not isinstance(raw, dict):
        raise TypeError('search_space entries must be mappings')
    return {name: SearchSpaceSpec.from_raw(name, value) for name, value in raw.items()}


def _validate_ablation_baseline(
    label: str,
    baseline: dict[str, Any],
    ablations: dict[str, list[Any]],
) -> None:
    unknown = set(baseline) - set(ablations)
    if unknown:
        raise ValueError(f'{label} references unknown ablation axes: {sorted(unknown)}')
    invalid = {key: value for key, value in baseline.items() if value not in ablations[key]}
    if invalid:
        raise ValueError(f'{label} contains values outside their ablation axes: {invalid}')


def _parse_thresholds(raw: Any) -> dict[tuple[str, float | str, str], float]:
    if not isinstance(raw, dict):
        raise TypeError('per_dataset_ratio_method_grid must be a mapping')
    thresholds: dict[tuple[str, float | str, str], float] = {}
    for raw_key, values in raw.items():
        parts = [part.strip() for part in str(raw_key).split(',')]
        if len(parts) != 3:
            raise ValueError(f'Invalid threshold key {raw_key!r}; expected dataset,ratio,method')
        dataset, ratio_raw, method = parts
        try:
            ratio: float | str = float(ratio_raw)
        except ValueError:
            ratio = ratio_raw
        threshold_values = values.get(ADJACENCY_THRESHOLD, [])
        if len(threshold_values) != 1:
            raise ValueError(
                f"Threshold entry '{raw_key}' must contain exactly one {ADJACENCY_THRESHOLD}"
            )
        thresholds[(dataset, ratio, method)] = float(threshold_values[0])
    return thresholds


def _normalize_sqlite_url(url: str, base_dir: Path) -> str:
    prefix = 'sqlite:///'
    if not url.startswith(prefix):
        return url
    raw_path = url.removeprefix(prefix)
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f'sqlite:///{path}'


def _find_project_root(source_path: Path) -> Path:
    """Find a stable base for runtime paths independent of the launch CWD."""
    for candidate in (source_path.parent, *source_path.parents):
        if (candidate / '.project-root').exists() or (candidate / 'pyproject.toml').exists():
            return candidate
    return source_path.parent


def _enable_sqlite_wal(url: str) -> None:
    """Improve safe concurrency when several study workers share SQLite."""
    prefix = 'sqlite:///'
    if not url.startswith(prefix):
        return
    path = Path(url.removeprefix(prefix))
    with sqlite3.connect(path, timeout=60) as connection:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA busy_timeout=60000')


def _stable_hash(value: Any, length: int = 16) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def _slug(value: Any) -> str:
    text = str(value).lower().replace('.', 'p')
    return re.sub(r'[^a-z0-9_-]+', '-', text).strip('-')


def _adjacency_parameters_for(
    config: OptunaSearchConfig, dataset: str, values: dict[str, Any]
) -> dict[str, float]:
    if values[ADJACENCY_METHOD] == 'string':
        return {ADJACENCY_THRESHOLD: config.string_adjacency_threshold}
    if config.wgcna_target_connectivity is not None:
        return {ADJACENCY_TARGET_CONNECTIVITY: config.wgcna_target_connectivity}
    ratio = values[NODE_SAMPLE_RATIO]
    method = str(values[SELECTION_METHOD])
    keys = [(dataset, ratio, method)]
    try:
        keys.append((dataset, float(ratio), method))
    except (TypeError, ValueError):
        pass
    for key in keys:
        if key in config.thresholds:
            return {ADJACENCY_THRESHOLD: config.thresholds[key]}
    raise ValueError(
        f'No adjacency threshold for dataset={dataset}, ratio={ratio}, method={method}'
    )


def build_outer_cells(
    config: OptunaSearchConfig,
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
) -> list[OuterCell]:
    """Build every fixed ablation cell."""
    selected_models = list(models or config.models)
    selected_datasets = list(datasets or config.datasets)
    unknown_models = set(selected_models) - set(config.models)
    unknown_datasets = set(selected_datasets) - set(config.datasets)
    if unknown_models:
        raise ValueError(f'Unknown model filters: {sorted(unknown_models)}')
    if unknown_datasets:
        raise ValueError(f'Unknown dataset filters: {sorted(unknown_datasets)}')

    cells: list[OuterCell] = []
    for model, dataset in itertools.product(selected_models, selected_datasets):
        if config.ablation_mode == 'one_factor_at_a_time':
            baseline = {
                **config.ablation_baseline,
                **config.per_model_ablation_baseline.get(model, {}),
            }
            combinations = [baseline]
            combinations.extend(
                {**baseline, axis: value}
                for axis, choices in config.ablations.items()
                for value in choices
                if value != baseline[axis]
            )
        else:
            axis_names = list(config.ablations)
            combinations = [
                dict(zip(axis_names, combination, strict=True))
                for combination in itertools.product(
                    *(config.ablations[name] for name in axis_names)
                )
            ]
        for combination in combinations:
            values = dict(combination)
            candidate = {'model': model, 'dataset': dataset, **values}
            if any(
                all(candidate.get(key) == value for key, value in exclusion.items())
                for exclusion in config.exclude_cells
            ):
                continue
            values.update(_adjacency_parameters_for(config, dataset, values))
            name_parts = [
                config.study_name_prefix,
                model,
                dataset,
                values[EXPERIMENT],
                values[ADJACENCY_METHOD],
                values[SELECTION_METHOD],
                f'r{values[NODE_SAMPLE_RATIO]}',
            ]
            readable_name = '_'.join(_slug(part) for part in name_parts)
            cell_hash = _stable_hash({'model': model, 'dataset': dataset, **values}, length=8)
            cells.append(
                OuterCell(
                    model=model,
                    dataset=dataset,
                    values=values,
                    study_name=f'{readable_name}_{cell_hash}',
                )
            )
    return cells


def study_shard(study_name: str, num_shards: int) -> int:
    """Return the deterministic virtual shard for a study."""
    if num_shards < 1:
        raise ValueError('num_shards must be positive')
    return int(_stable_hash(study_name), 16) % num_shards


def select_study_shards(
    cells: Sequence[OuterCell],
    num_shards: int,
    shard_indices: Sequence[int] | None,
) -> list[OuterCell]:
    """Select a disjoint union of deterministic virtual study shards."""
    if num_shards < 1:
        raise ValueError('num_shards must be positive')
    selected = list(range(num_shards)) if shard_indices is None else list(shard_indices)
    if len(set(selected)) != len(selected):
        raise ValueError('shard_indices must not contain duplicates')
    invalid = [index for index in selected if index < 0 or index >= num_shards]
    if invalid:
        raise ValueError(f'shard_indices must be between 0 and {num_shards - 1}: {invalid}')
    selected_set = set(selected)
    return [cell for cell in cells if study_shard(cell.study_name, num_shards) in selected_set]


def read_study_manifest(path: str | Path) -> list[str]:
    """Read unique study names from a newline-delimited manifest."""
    manifest_path = Path(path).resolve()
    studies = [
        line.strip()
        for line in manifest_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]
    if not studies:
        raise ValueError(f'Study manifest is empty: {manifest_path}')
    duplicates = sorted({study for study in studies if studies.count(study) > 1})
    if duplicates:
        raise ValueError(f'Study manifest contains duplicate names: {duplicates}')
    return studies


def _model_space(config: OptunaSearchConfig, model: str) -> dict[str, SearchSpaceSpec]:
    return {**config.search_space, **config.per_model_search_space.get(model, {})}


def _sample_parameters(
    trial: optuna.Trial, config: OptunaSearchConfig, model: str
) -> dict[str, Any]:
    space = _model_space(config, model)
    return {name: spec.suggest(trial, name) for name, spec in space.items()}


def validate_sampled_parameters(
    config: OptunaSearchConfig, model: str, sampled: dict[str, Any]
) -> dict[str, Any]:
    """Return canonical in-space values for a complete parameter set, or raise ValueError."""
    space = _model_space(config, model)
    if set(sampled) != set(space):
        raise ValueError(
            f'{model} parameters {sorted(sampled)} do not match its search space {sorted(space)}'
        )
    canonical = {}
    for name, spec in space.items():
        try:
            canonical[name] = spec.canonical(sampled[name])
        except ValueError as error:
            raise ValueError(f'{model} parameter {name}: {error}') from error
    return canonical


def _trial_hyperparameters(cell: OuterCell, sampled: dict[str, Any]) -> dict[str, Any]:
    hyperparameters = {**cell.values, **sampled}
    is_no_readout = hyperparameters.get(EXPERIMENT) == 'no_readout'
    if (
        not is_no_readout
        and 'model.backbone.dropout' in hyperparameters
        and 'model.readout.fc_dropout' not in hyperparameters
    ):
        hyperparameters['model.readout.fc_dropout'] = hyperparameters['model.backbone.dropout']
    if is_no_readout:
        hyperparameters.pop('model.readout.fc_dim', None)
        hyperparameters.pop('model.readout.fc_dropout', None)
    return hyperparameters


def _fold_overrides(
    config: OptunaSearchConfig,
    cell: OuterCell,
    hyperparameters: dict[str, Any],
    trial_number: int,
    fold: int,
    attempt: int,
) -> list[str]:
    tags = [cell.model, cell.dataset, 'optuna', f'fold{fold}', *config.tags]
    run_name = f'{cell.study_name}_trial{trial_number:04d}_fold{fold}_attempt{attempt}'
    output_dir = config.output_dir / 'runs' / cell.study_name / run_name
    parameters = {**config.fixed, **hyperparameters}
    parameters.update(
        {
            'seed': config.training_seed,
            'dataset.split_params.split_type': 'k-fold',
            'dataset.split_params.k': config.k,
            'dataset.split_params.data_seed': fold,
            'test': False,
            'logger.wandb.tags': tags,
            'logger.wandb.group': cell.study_name,
            'logger.wandb.name': run_name,
            'hydra.run.dir': str(output_dir),
        }
    )
    overrides = [
        f'model={cell.model}',
        f'dataset={cell.dataset}',
        f'+optimized_metric={config.objective_metric}',
    ]
    overrides.extend(to_override(key, value) for key, value in parameters.items())
    return overrides


def _storage(config: OptunaSearchConfig) -> optuna.storages.RDBStorage:
    engine_kwargs = (
        {'connect_args': {'timeout': 60}} if config.storage.startswith('sqlite:///') else {}
    )
    return optuna.storages.RDBStorage(
        url=config.storage,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=config.heartbeat_interval,
        grace_period=config.grace_period,
    )


def _load_or_create_study(config: OptunaSearchConfig, cell: OuterCell) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(
        seed=config.sampler_seed,
        n_startup_trials=config.n_startup_trials,
    )
    study = optuna.create_study(
        storage=_storage(config),
        study_name=cell.study_name,
        sampler=sampler,
        direction=config.direction,
        load_if_exists=True,
    )
    existing_fingerprint = study.user_attrs.get('config_fingerprint')
    if existing_fingerprint and existing_fingerprint != config.fingerprint:
        raise ValueError(
            f"Study '{cell.study_name}' was created with a different search configuration "
            f'({existing_fingerprint} != {config.fingerprint})'
        )
    study.set_user_attr('config_fingerprint', config.fingerprint)
    study.set_user_attr('outer_cell', cell.result_fields())
    study.set_user_attr('folds', config.folds)
    study.set_user_attr('training_seed', config.training_seed)
    return study


def _fail_stale_trials(study: optuna.Study) -> None:
    fail_stale = getattr(optuna.storages, 'fail_stale_trials', None)
    if fail_stale is not None:
        fail_stale(study)


def _record_interrupted_trials(
    study: optuna.Study,
    config: OptunaSearchConfig,
    cell: OuterCell,
    ledger: RunLedger,
) -> None:
    """Represent stale worker failures in the fold ledger and failure manifest."""
    for trial in study.get_trials(deepcopy=False, states=(TrialState.FAIL,)):
        if trial.user_attrs.get('failure'):
            continue
        sampled = trial.user_attrs.get('sampled_params')
        if not isinstance(sampled, dict):
            continue
        param_hash = _stable_hash(sampled)
        if ledger.interruption_recorded(cell.study_name, param_hash, trial.number):
            continue
        for fold in config.folds:
            if (
                ledger.successful_metric(cell.study_name, param_hash, fold, config.training_seed)
                is not None
            ):
                continue
            attempt = ledger.next_attempt(cell.study_name, param_hash, fold, config.training_seed)
            ledger.record(
                study_name=cell.study_name,
                param_hash=param_hash,
                params=sampled,
                fold=fold,
                training_seed=config.training_seed,
                attempt=attempt,
                status='failed',
                metric=None,
                elapsed_time=0.0,
                error='Worker interrupted before this fold produced a durable result',
                log_path=(
                    config.output_dir
                    / 'logs'
                    / cell.study_name
                    / param_hash
                    / f'interrupted_trial_{trial.number}_fold_{fold}.log'
                ),
                trial_number=trial.number,
                gpu=None,
            )
            break


def _enqueue_failed_trials(
    study: optuna.Study,
    ledger: RunLedger,
    max_attempts: int,
    expected_folds: Sequence[int] | None = None,
    oom_retries: int = 0,
) -> dict[str, int]:
    waiting_by_fixed_params = {
        json.dumps(trial.system_attrs.get('fixed_params', trial.params), sort_keys=True): trial
        for trial in study.trials
        if trial.state == TrialState.WAITING
    }
    running_fixed_params = {
        json.dumps(trial.system_attrs.get('fixed_params', trial.params), sort_keys=True)
        for trial in study.trials
        if trial.state == TrialState.RUNNING
    }
    completed_hashes = {
        trial.user_attrs.get('param_hash')
        for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    }
    queued_hashes: set[str] = set()
    enqueued: dict[str, int] = {}
    for failed in study.get_trials(deepcopy=False, states=(TrialState.FAIL,)):
        if not failed.params:
            continue
        sampled = failed.user_attrs.get('sampled_params')
        if not isinstance(sampled, dict):
            continue
        param_hash = _stable_hash(sampled)
        fixed_key = json.dumps(failed.params, sort_keys=True)
        if param_hash in completed_hashes or param_hash in queued_hashes:
            continue
        if fixed_key in waiting_by_fixed_params:
            queued_hashes.add(param_hash)
            enqueued[param_hash] = failed.number
            continue
        if fixed_key in running_fixed_params:
            continue
        if not ledger.retryable(
            study.study_name,
            param_hash,
            max_attempts,
            expected_folds=expected_folds,
            oom_retries=oom_retries,
        ):
            continue
        try:
            study.enqueue_trial(failed.params, user_attrs={'retry_of': failed.number})
        except TypeError:
            study.enqueue_trial(failed.params)
        queued_hashes.add(param_hash)
        enqueued[param_hash] = failed.number
    return enqueued


def _enqueue_candidate(
    study: optuna.Study,
    config: OptunaSearchConfig,
    cell: OuterCell,
    candidate: dict[str, Any],
) -> int:
    """Queue a fixed candidate configuration once and return how many new trials to run.

    Failed evaluations of the candidate are left to the ``--retry-failed`` path.
    """
    param_hash = candidate['param_hash']
    space = _model_space(config, cell.model)
    fixed = {name: spec.raw(candidate['sampled_params'][name]) for name, spec in space.items()}
    fixed_key = json.dumps(fixed, sort_keys=True)
    for trial in study.trials:
        if trial.user_attrs.get('param_hash') == param_hash:
            return 0
        if trial.state == TrialState.WAITING and (
            json.dumps(trial.system_attrs.get('fixed_params', trial.params), sort_keys=True)
            == fixed_key
        ):
            return 1
    study.enqueue_trial(fixed)
    return 1


def _acquire_gpu(gpu_queue: Any, policy: ExecutionPolicy, *, after_oom: bool) -> GpuDevice:
    if after_oom and policy.oom_min_free_gpu_mib:
        return acquire_gpu(
            gpu_queue,
            policy.oom_min_free_gpu_mib,
            poll_seconds=policy.gpu_poll_seconds,
            fallback_min_free_mib=policy.min_free_gpu_mib,
            fallback_after_seconds=policy.oom_gpu_wait_seconds,
        )
    return acquire_gpu(gpu_queue, policy.min_free_gpu_mib, poll_seconds=policy.gpu_poll_seconds)


def _objective(
    trial: optuna.Trial,
    *,
    config: OptunaSearchConfig,
    cell: OuterCell,
    ledger_path: Path,
    gpu_queue: Any | None,
    retry_sources: dict[str, int] | None = None,
    candidate: dict[str, Any] | None = None,
    policy: ExecutionPolicy = DEFAULT_POLICY,
) -> float:
    ledger = RunLedger(ledger_path)
    sampled = _sample_parameters(trial, config, cell.model)
    trial.set_user_attr('sampled_params', sampled)
    hyperparameters = _trial_hyperparameters(cell, sampled)
    param_hash = _stable_hash(sampled)
    trial.set_user_attr('param_hash', param_hash)
    if retry_sources and param_hash in retry_sources:
        trial.set_user_attr('retry_of', retry_sources.pop(param_hash))
    if candidate is not None:
        if param_hash != candidate['param_hash']:
            error = (
                f'Sampled parameters {param_hash} differ from the fixed candidate '
                f"{candidate['param_hash']}"
            )
            trial.set_user_attr('failure', error)
            raise FoldExecutionError(error)
        trial.set_user_attr(
            'transfer_source',
            {
                key: candidate.get(key)
                for key in ('source_study', 'source_rule', 'source_trial_numbers', 'source_mean')
            },
        )

    fold_scores: dict[str, float] = {}
    reused_folds: list[int] = []
    max_attempts = config.max_retries + 1
    for fold in config.folds:
        cached_metric = ledger.successful_metric(
            cell.study_name, param_hash, fold, config.training_seed
        )
        if cached_metric is not None:
            fold_scores[str(fold)] = cached_metric
            reused_folds.append(fold)
            continue

        after_oom = False
        while True:
            used = ledger.attempt_budget_used(
                cell.study_name, param_hash, fold, config.training_seed, policy.oom_retries
            )
            if used >= max_attempts:
                error = f'Fold {fold} exceeded the maximum of {max_attempts} attempts'
                trial.set_user_attr('failed_fold', fold)
                trial.set_user_attr('failure', error)
                raise FoldExecutionError(error)
            attempt = ledger.next_attempt(
                cell.study_name, param_hash, fold, config.training_seed
            )

            gpu: GpuDevice | None = None
            if gpu_queue is not None:
                gpu = _acquire_gpu(gpu_queue, policy, after_oom=after_oom)
            log_path = (
                config.output_dir
                / 'logs'
                / cell.study_name
                / param_hash
                / f'fold_{fold}_attempt_{attempt}.log'
            )
            started = time.time()
            try:
                overrides = _fold_overrides(
                    config, cell, hyperparameters, trial.number, fold, attempt
                )
                success, error, metrics = run_training(
                    overrides,
                    timeout=config.timeout,
                    gpu_id=None if gpu is None else gpu.visibility_token,
                    n_threads=1,
                    log_path=log_path,
                )
            finally:
                if gpu_queue is not None:
                    gpu_queue.put(gpu)
            elapsed = time.time() - started

            metric = None if not metrics else metrics.get('objective')
            if success and metric is None:
                success = False
                error = (
                    f"Training succeeded but did not emit objective '{config.objective_metric}'"
                )
            elif success and not math.isfinite(float(metric)):
                success = False
                error = f'Training emitted a non-finite objective value: {metric}'
            oom = not success and is_oom_failure(error, log_path)
            if oom:
                error = f'{OOM_ERROR_PREFIX}{error}'
            ledger.record(
                study_name=cell.study_name,
                param_hash=param_hash,
                params=sampled,
                fold=fold,
                training_seed=config.training_seed,
                attempt=attempt,
                status='success' if success else 'failed',
                metric=None if metric is None else float(metric),
                elapsed_time=elapsed,
                error=error,
                log_path=log_path,
                trial_number=trial.number,
                gpu=gpu,
            )
            if success:
                break
            if oom and policy.oom_retries > 0:
                after_oom = True
                time.sleep(policy.oom_backoff_seconds)
                continue
            trial.set_user_attr('failed_fold', fold)
            trial.set_user_attr('failure', error or 'unknown training failure')
            raise FoldExecutionError(f'Fold {fold} failed: {error}')
        fold_scores[str(fold)] = float(metric)

    values = list(fold_scores.values())
    mean_score = statistics.fmean(values)
    std_score = statistics.pstdev(values)
    trial.set_user_attr('fold_scores', fold_scores)
    trial.set_user_attr('reused_folds', reused_folds)
    trial.set_user_attr('fold_mean', mean_score)
    trial.set_user_attr('fold_std', std_score)
    return mean_score


def _trial_rows(study: optuna.Study, cell: OuterCell) -> list[dict[str, Any]]:
    rows = []
    for trial in study.trials:
        row = {
            **cell.result_fields(),
            'study_name': study.study_name,
            'trial_number': trial.number,
            'state': trial.state.name,
            'objective': trial.value,
            'sampled_params': json.dumps(
                trial.user_attrs.get('sampled_params', {}), sort_keys=True
            ),
            'fold_scores': json.dumps(trial.user_attrs.get('fold_scores', {}), sort_keys=True),
            'fold_mean': trial.user_attrs.get('fold_mean'),
            'fold_std': trial.user_attrs.get('fold_std'),
            'failed_fold': trial.user_attrs.get('failed_fold'),
            'failure': trial.user_attrs.get('failure'),
            'retry_of': trial.user_attrs.get('retry_of'),
        }
        if 'transfer_source' in trial.user_attrs:
            row['transfer_source'] = json.dumps(trial.user_attrs['transfer_source'], sort_keys=True)
        rows.append(row)
    return rows


def _run_study(
    config: OptunaSearchConfig,
    cell: OuterCell,
    ledger_path: Path,
    gpu_queue: Any | None,
    retry_failed: bool,
    candidate: dict[str, Any] | None = None,
    policy: ExecutionPolicy = DEFAULT_POLICY,
) -> list[dict[str, Any]]:
    study = _load_or_create_study(config, cell)
    _fail_stale_trials(study)
    ledger = RunLedger(ledger_path)
    _record_interrupted_trials(study, config, cell, ledger)
    if candidate is not None:
        remaining = _enqueue_candidate(study, config, cell, candidate)
    else:
        # Compute the new-suggestion budget before adding retries. The objective
        # marks fallback Optuna 2.10 retries with retry_of as soon as they start.
        original_trials = [
            trial
            for trial in study.trials
            if trial.user_attrs.get('retry_of') is None
            and trial.state not in {TrialState.WAITING, TrialState.RUNNING}
        ]
        remaining = max(0, config.n_trials - len(original_trials))
    retry_sources = (
        _enqueue_failed_trials(
            study,
            ledger,
            config.max_retries + 1,
            expected_folds=config.folds,
            oom_retries=policy.oom_retries,
        )
        if retry_failed
        else {}
    )
    trials_to_run = remaining + len(retry_sources)
    if trials_to_run:
        study.optimize(
            lambda trial: _objective(
                trial,
                config=config,
                cell=cell,
                ledger_path=ledger_path,
                gpu_queue=gpu_queue,
                retry_sources=retry_sources,
                candidate=candidate,
                policy=policy,
            ),
            n_trials=trials_to_run,
            catch=(FoldExecutionError,),
        )
    return _trial_rows(study, cell)


def _cache_configs(config: OptunaSearchConfig, cells: Sequence[OuterCell]) -> list[Any]:
    """Compose and deduplicate complete loader configs for all folds."""
    config_dir = Path(__file__).resolve().parent.parent / 'configs'
    composed: dict[str, Any] = {}
    requests: dict[str, tuple[OuterCell, int, dict[str, Any]]] = {}
    for cell in cells:
        for fold in config.folds:
            parameters = {**config.fixed, **cell.values}
            parameters.update(
                {
                    'dataset.split_params.split_type': 'k-fold',
                    'dataset.split_params.k': config.k,
                    'dataset.split_params.data_seed': fold,
                }
            )
            # Models do not affect dataset caches. Collapse equivalent requests
            # before the comparatively expensive Hydra composition step.
            request_key = _stable_hash(
                {'dataset': cell.dataset, 'parameters': parameters}, length=32
            )
            requests[request_key] = (cell, fold, parameters)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            config_dir=str(config_dir), job_name='optuna_cache_warmup', version_base='1.3'
        ):
            for cell, _fold, parameters in requests.values():
                overrides = [
                    f'dataset={cell.dataset}',
                ]
                overrides.extend(to_override(key, value) for key, value in parameters.items())
                cfg = compose(config_name='train.yaml', overrides=overrides)
                loader = OmegaConf.to_container(cfg.dataset.loader, resolve=True)
                signature = _stable_hash(loader, length=32)
                composed[signature] = OmegaConf.create(loader)
    finally:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
    return list(composed.values())


def _seed_cache_randomness(seed: int) -> None:
    """Mirror Lightning's RNG reset before each cache is generated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _warmup_cache(
    loader_config: Any,
    *,
    index: int,
    total: int,
    training_seed: int,
) -> None:
    """Build one cache in an isolated one-thread worker."""
    enforce_single_thread_process()
    _seed_cache_randomness(training_seed)
    params = loader_config.parameters
    print(
        f'  [{index}/{total}] {params.data_name} '
        f'fold={params.fold}/{params.k} method={params.method} '
        f'adjacency={params.adjacency_method}',
        flush=True,
    )
    loader = instantiate(loader_config)
    loader.load_dataset()


def warmup_caches(
    config: OptunaSearchConfig,
    cells: Sequence[OuterCell],
    n_jobs: int = 1,
) -> None:
    """Build unique fold-aware dataset caches in parallel."""
    enforce_single_thread_process()
    if n_jobs < 1:
        raise ValueError('warmup_jobs must be positive')
    loaders = _cache_configs(config, cells)
    workers = min(n_jobs, len(loaders))
    print(
        f'Warming {len(loaders)} unique fold-aware dataset caches ' f'with {workers} workers...',
        flush=True,
    )
    if workers == 1:
        for index, loader_config in enumerate(loaders, start=1):
            _warmup_cache(
                loader_config,
                index=index,
                total=len(loaders),
                training_seed=config.training_seed,
            )
    else:
        with parallel_backend('loky', inner_max_num_threads=1):
            Parallel(n_jobs=workers, verbose=10)(
                delayed(_warmup_cache)(
                    loader_config,
                    index=index,
                    total=len(loaders),
                    training_seed=config.training_seed,
                )
                for index, loader_config in enumerate(loaders, start=1)
            )


def _dry_run(
    config: OptunaSearchConfig,
    cells: Sequence[OuterCell],
    candidates: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Sample once per cell (or use its fixed candidate) and validate model construction."""
    from scripts.hyperparam_search import dry_run_config

    rows = []
    for index, cell in enumerate(cells, start=1):
        study = optuna.create_study(
            direction=config.direction,
            sampler=optuna.samplers.RandomSampler(seed=config.sampler_seed),
        )
        if candidates is not None:
            _enqueue_candidate(study, config, cell, candidates[cell.study_name])
        trial = study.ask()
        sampled = _sample_parameters(trial, config, cell.model)
        hyperparameters = _trial_hyperparameters(cell, sampled)
        overrides = _fold_overrides(config, cell, hyperparameters, trial.number, 0, 1)
        params, error = dry_run_config(overrides)
        rows.append(
            {
                **cell.result_fields(),
                'study_name': cell.study_name,
                'success': params is not None,
                'params': params,
                'error': error,
                'sampled_params': json.dumps(sampled, sort_keys=True),
            }
        )
        print(f'[{index}/{len(cells)}] {cell.study_name}: {"OK" if params else "FAIL"}')
    return pd.DataFrame(rows)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp-{os.getpid()}')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _best_rows(trials: pd.DataFrame, direction: str) -> pd.DataFrame:
    complete = trials[(trials['state'] == 'COMPLETE') & trials['objective'].notna()]
    if complete.empty:
        return trials.iloc[0:0].copy()
    index = (
        complete.groupby('study_name')['objective'].idxmax()
        if direction == 'maximize'
        else complete.groupby('study_name')['objective'].idxmin()
    )
    return complete.loc[index].reset_index(drop=True)


def _failure_frame(
    ledger: RunLedger,
    config_path: Path,
    jobs_per_gpu: int,
    study_names: set[str] | None = None,
) -> pd.DataFrame:
    failures = [
        failure
        for failure in ledger.unresolved_failures()
        if study_names is None or failure['study_name'] in study_names
    ]
    for failure in failures:
        failure['retry_command'] = (
            f'python scripts/optuna_search.py --config {shlex.quote(str(config_path))} '
            f'--studies {shlex.quote(failure["study_name"])} --retry-failed '
            f'--jobs-per-gpu {jobs_per_gpu}'
        )
    columns = [
        'study_name',
        'param_hash',
        'params_json',
        'fold',
        'training_seed',
        'attempt',
        'status',
        'metric',
        'elapsed_time',
        'error',
        'log_path',
        'trial_number',
        'gpu_logical_id',
        'gpu_visibility_token',
        'created_at',
        'retry_command',
    ]
    return pd.DataFrame(failures, columns=columns)


def run_search(
    config: OptunaSearchConfig,
    *,
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    studies: Sequence[str] | None = None,
    num_shards: int = 1,
    shard_indices: Sequence[int] | None = None,
    requested_gpus: Sequence[int] | None = None,
    jobs_per_gpu: int = 1,
    n_jobs: int | None = None,
    warmup_jobs: int | None = None,
    retry_failed: bool = False,
    skip_warmup: bool = False,
    warmup_only: bool = False,
    dry_run: bool = False,
    export_only: bool = False,
    candidates: dict[str, dict[str, Any]] | None = None,
    policy: ExecutionPolicy = DEFAULT_POLICY,
) -> pd.DataFrame:
    """Run or resume all selected Optuna studies."""
    enforce_single_thread_process()
    if warmup_only and skip_warmup:
        raise ValueError('warmup_only and skip_warmup cannot be used together')
    cells = build_outer_cells(config, models=models, datasets=datasets)
    if studies:
        # Requested studies run in the order given, so manifests set priority.
        by_name = {cell.study_name: cell for cell in cells}
        missing = set(studies) - set(by_name)
        if missing:
            raise ValueError(f'Unknown study filters: {sorted(missing)}')
        cells = [by_name[name] for name in dict.fromkeys(studies)]
    cells = select_study_shards(cells, num_shards, shard_indices)
    if not cells:
        raise ValueError('No ablation cells selected after applying filters and shards')
    if candidates is not None:
        _validate_candidates(config, cells, candidates)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        frame = _dry_run(config, cells, candidates)
        _atomic_write_csv(frame, config.output_dir / 'dry_run.csv')
        return frame
    if export_only:
        return _export_existing(config, cells, jobs_per_gpu)

    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = visible_gpu_devices(requested_gpus, device_count=available)
    if requested_gpus and not devices:
        raise ValueError('GPU IDs were requested but CUDA is unavailable')
    slots = len(devices) * jobs_per_gpu if devices else 1
    workers = slots if n_jobs is None else min(n_jobs, slots)
    if workers < 1:
        raise ValueError('n_jobs must be positive')
    cache_workers = workers if warmup_jobs is None else warmup_jobs
    if cache_workers < 1:
        raise ValueError('warmup_jobs must be positive')

    print('=' * 72)
    print('OPTUNA ABLATION SWEEP')
    print(
        f'Cells: {len(cells)} | folds/trial: {config.folds} | target trials/cell: {config.n_trials}'
    )
    print(f'GPU devices: {[device.logical_id for device in devices] or ["CPU"]}')
    print(f'Virtual shards: {list(shard_indices or range(num_shards))}/{num_shards}')
    print(f'Jobs/GPU: {jobs_per_gpu} | parallel workers: {workers} | CPU threads/job: 1')
    print(f'Cache warmup workers: {cache_workers} | CPU threads/worker: 1')
    if policy != DEFAULT_POLICY:
        print(f'Execution policy: {policy}')
    if candidates is not None:
        print(f'Fixed candidates: {len(cells)} studies, one configuration each')
    print(f'Storage: {config.storage}')
    print(f'Output: {config.output_dir}')
    print('=' * 72)

    if not skip_warmup:
        warmup_caches(config, cells, n_jobs=cache_workers)
    if warmup_only:
        print(f'Cache warmup complete for {len(cells)} selected studies')
        return pd.DataFrame()

    ledger_path = config.output_dir / 'run_ledger.sqlite3'
    RunLedger(ledger_path)
    # Initialize/migrate the Optuna schema once before worker processes open
    # the shared SQLite database concurrently.
    _storage(config)
    _enable_sqlite_wal(config.storage)

    def study_task(cell: OuterCell, gpu_queue: Any | None) -> Any:
        candidate = None if candidates is None else candidates[cell.study_name]
        return delayed(_run_study)(
            config, cell, ledger_path, gpu_queue, retry_failed, candidate, policy
        )

    if workers > 1:
        manager = multiprocessing.Manager()
        gpu_queue = manager.Queue() if devices else None
        if gpu_queue is not None:
            populate_gpu_queue(gpu_queue, devices, jobs_per_gpu)
        with parallel_backend('loky', inner_max_num_threads=1):
            nested_rows = Parallel(n_jobs=workers, verbose=10)(
                study_task(cell, gpu_queue) for cell in cells
            )
    else:
        gpu_queue = None
        if devices:
            manager = multiprocessing.Manager()
            gpu_queue = manager.Queue()
            populate_gpu_queue(gpu_queue, devices, jobs_per_gpu)
        nested_rows = []
        for cell in cells:
            function, args, kwargs = study_task(cell, gpu_queue)
            nested_rows.append(function(*args, **kwargs))

    trials = pd.DataFrame(list(itertools.chain.from_iterable(nested_rows)))
    _write_exports(config, cells, trials, jobs_per_gpu)
    return trials


def _write_exports(
    config: OptunaSearchConfig,
    cells: Sequence[OuterCell],
    trials: pd.DataFrame,
    jobs_per_gpu: int,
) -> None:
    ledger = RunLedger(config.output_dir / 'run_ledger.sqlite3')
    best = _best_rows(trials, config.direction) if not trials.empty else trials
    selected_studies = {cell.study_name for cell in cells}
    failures = _failure_frame(
        ledger,
        config.source_path,
        jobs_per_gpu,
        study_names=selected_studies,
    )
    fold_attempts = pd.DataFrame(ledger.all_attempts())
    if not fold_attempts.empty:
        fold_attempts = fold_attempts.loc[
            fold_attempts['study_name'].isin(selected_studies)
        ].copy()
    _atomic_write_csv(trials, config.output_dir / 'trials.csv')
    _atomic_write_csv(best, config.output_dir / 'best_trials.csv')
    _atomic_write_csv(fold_attempts, config.output_dir / 'fold_attempts.csv')
    _atomic_write_csv(failures, config.output_dir / 'failures.csv')
    print(
        f'Completed export: {len(trials)} trials, {len(best)} best configurations, '
        f'{len(failures)} unresolved fold failures'
    )


def _export_existing(
    config: OptunaSearchConfig, cells: Sequence[OuterCell], jobs_per_gpu: int
) -> pd.DataFrame:
    """Export CSVs from existing studies without creating studies or training."""
    storage_path = config.storage.removeprefix('sqlite:///')
    if config.storage.startswith('sqlite:///') and not Path(storage_path).exists():
        raise FileNotFoundError(f'No Optuna storage to export: {storage_path}')
    storage = _storage(config)
    existing = {summary.study_name for summary in optuna.get_all_study_summaries(storage)}
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if cell.study_name not in existing:
            continue
        study = optuna.load_study(study_name=cell.study_name, storage=storage)
        rows.extend(_trial_rows(study, cell))
    trials = pd.DataFrame(rows)
    _write_exports(config, cells, trials, jobs_per_gpu)
    return trials


def load_candidates(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load fixed per-study configurations written by scripts/ratio03_prepare.py."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f'Candidate file must map study names to candidates: {path}')
    return raw


def _validate_candidates(
    config: OptunaSearchConfig,
    cells: Sequence[OuterCell],
    candidates: dict[str, dict[str, Any]],
) -> None:
    missing = [cell.study_name for cell in cells if cell.study_name not in candidates]
    if missing:
        raise ValueError(f'{len(missing)} selected studies have no candidate, e.g. {missing[:3]}')
    for cell in cells:
        candidate = candidates[cell.study_name]
        canonical = validate_sampled_parameters(config, cell.model, candidate['sampled_params'])
        if _stable_hash(canonical) != candidate['param_hash']:
            raise ValueError(f'Candidate hash mismatch for {cell.study_name}')


def _apply_runtime_overrides(
    config: OptunaSearchConfig,
    *,
    output_dir: str | None = None,
    storage: str | None = None,
    root_dir: str | None = None,
) -> None:
    """Apply server-local paths supplied by the launcher CLI."""
    if output_dir:
        config.output_dir = Path(output_dir).resolve()
    if storage:
        config.storage = _normalize_sqlite_url(storage, Path.cwd())
    if root_dir:
        config.fixed['paths.root_dir'] = str(Path(root_dir).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Optuna search YAML')
    parser.add_argument('--models', nargs='+', help='Only run these configured models')
    parser.add_argument('--datasets', nargs='+', help='Only run these configured datasets')
    parser.add_argument('--studies', nargs='+', help='Only run exact deterministic study names')
    parser.add_argument(
        '--studies-file',
        help='Only run study names in this newline-delimited manifest',
    )
    parser.add_argument(
        '--num-shards',
        type=int,
        default=1,
        help='Deterministically partition studies into this many virtual shards',
    )
    parser.add_argument(
        '--shard-indices',
        nargs='+',
        type=int,
        help='Run this union of zero-based virtual shard indices (default: all)',
    )
    parser.add_argument(
        '--gpus',
        nargs='+',
        type=int,
        help='Launcher-visible logical GPU IDs (default: all visible devices)',
    )
    parser.add_argument(
        '--jobs-per-gpu',
        type=int,
        default=1,
        help='Concurrent training jobs allowed on each selected GPU',
    )
    parser.add_argument('--n-jobs', type=int, help='Optional cap on total parallel workers')
    parser.add_argument(
        '--warmup-jobs',
        type=int,
        help='Parallel cache builders (default: total training workers)',
    )
    parser.add_argument(
        '--retry-failed',
        action='store_true',
        help='Re-enqueue failed trials and rerun only missing/failed folds',
    )
    parser.add_argument('--skip-warmup', action='store_true')
    parser.add_argument(
        '--warmup-only',
        action='store_true',
        help='Build selected caches and exit without opening studies or training',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Sample and instantiate one configuration per cell without training or persistence',
    )
    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Write trial/failure CSVs from existing studies without training',
    )
    parser.add_argument(
        '--candidates-file',
        help='JSON of one fixed configuration per study; replaces sampling for every study',
    )
    parser.add_argument(
        '--min-free-gpu-mib',
        type=int,
        help='Only start a fold on a GPU with at least this much free memory',
    )
    parser.add_argument(
        '--oom-retries',
        type=int,
        default=0,
        help='Out-of-memory failures per fold retried immediately without using max_retries',
    )
    parser.add_argument(
        '--oom-min-free-gpu-mib',
        type=int,
        help='Free memory required when retrying a fold after an out-of-memory failure',
    )
    parser.add_argument('--output-dir', help='Override training.output_dir from YAML')
    parser.add_argument('--storage', help='Override optuna.storage from YAML')
    parser.add_argument('--root-dir', help='Override fixed paths.root_dir from YAML')
    args = parser.parse_args()
    if args.oom_retries < 0:
        parser.error('--oom-retries cannot be negative')

    config = OptunaSearchConfig.from_yaml(args.config)
    _apply_runtime_overrides(
        config,
        output_dir=args.output_dir,
        storage=args.storage,
        root_dir=args.root_dir,
    )
    requested_studies = list(args.studies or [])
    if args.studies_file:
        requested_studies.extend(read_study_manifest(args.studies_file))
    if len(requested_studies) != len(set(requested_studies)):
        parser.error('Study filters contain duplicate names')
    run_search(
        config,
        models=args.models,
        datasets=args.datasets,
        studies=requested_studies or None,
        num_shards=args.num_shards,
        shard_indices=args.shard_indices,
        requested_gpus=args.gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        n_jobs=args.n_jobs,
        warmup_jobs=args.warmup_jobs,
        retry_failed=args.retry_failed,
        skip_warmup=args.skip_warmup,
        warmup_only=args.warmup_only,
        dry_run=args.dry_run,
        export_only=args.export_only,
        candidates=load_candidates(args.candidates_file) if args.candidates_file else None,
        policy=ExecutionPolicy(
            min_free_gpu_mib=args.min_free_gpu_mib,
            oom_retries=args.oom_retries,
            oom_min_free_gpu_mib=args.oom_min_free_gpu_mib,
        ),
    )


if __name__ == '__main__':
    main()

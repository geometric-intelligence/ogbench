#!/usr/bin/env python3
"""Hyperparameter grid search script.

Cleaner implementation of hyperparameter search that loads search space from YAML config.
Supports parallel execution via joblib across multiple GPUs.

Usage:
    python scripts/hyperparam_search.py --config configs/hparams_search/parkinsons_grid.yaml
    python scripts/hyperparam_search.py --config configs/hparams_search/parkinsons_grid.yaml --dry-run
    python scripts/hyperparam_search.py --config configs/hparams_search/parkinsons_grid.yaml --models gcn gin
"""

import argparse
import itertools
import json
import multiprocessing
import os
import subprocess  # nosec B404
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from joblib import Parallel, delayed

from ogbench.utils.config_resolvers import (
    register_all_resolvers,
)

register_all_resolvers()


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""

    datasets: list[str]
    models: list[str]
    seeds: list[int]
    fixed: dict[str, Any]
    shared_grid: dict[str, list[Any]]
    per_model_grid: dict[str, dict[str, list[Any]]]
    per_model_dataset_grid: dict[tuple[str, str], dict[str, list[Any]]]
    per_dataset_ratio_method_grid: dict[tuple[str, float | str, str], dict[str, list[Any]]]
    string_adjacency_threshold: float
    timeout: int
    output_dir: str
    tags: list[str]

    @classmethod
    def from_yaml(cls, path: str) -> 'SearchConfig':
        """Load search configuration from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)

        # Handle both single dataset (backward compatible) and multiple datasets
        if 'datasets' in config:
            datasets = config['datasets']
        elif 'dataset' in config:
            datasets = [config['dataset']]
        else:
            raise ValueError("Config must specify either 'dataset' or 'datasets'")

        # Parse per_model_dataset_grid with string keys "model,dataset" -> tuple keys
        per_model_dataset_raw = config.get('per_model_dataset_grid', {})
        per_model_dataset_grid = {}
        for key, value in per_model_dataset_raw.items():
            if isinstance(key, str) and ',' in key:
                # Parse "model,dataset" -> (model, dataset)
                parts = key.split(',')
                if len(parts) == 2:
                    model, dataset = parts[0].strip(), parts[1].strip()
                    per_model_dataset_grid[(model, dataset)] = value
            elif isinstance(key, tuple) and len(key) == 2:
                # Already a tuple (shouldn't happen with YAML but handle it)
                per_model_dataset_grid[key] = value

        # Parse per_dataset_ratio_method_grid with string keys "dataset,ratio,method" -> tuple keys
        per_dataset_ratio_raw = config.get('per_dataset_ratio_method_grid', {})
        per_dataset_ratio_method_grid = {}
        for key, value in per_dataset_ratio_raw.items():
            if isinstance(key, str) and ',' in key:
                # Parse "dataset,ratio,method" -> (dataset, ratio, method)
                parts = key.split(',')
                if len(parts) == 3:
                    dataset = parts[0].strip()
                    ratio_str = parts[1].strip()
                    method = parts[2].strip()
                    # Try to convert ratio to float, otherwise keep as string
                    try:
                        ratio = float(ratio_str)
                    except ValueError:
                        ratio = ratio_str
                    per_dataset_ratio_method_grid[(dataset, ratio, method)] = value
            elif isinstance(key, tuple) and len(key) == 3:
                # Already a tuple (shouldn't happen with YAML but handle it)
                per_dataset_ratio_method_grid[key] = value

        return cls(
            datasets=datasets,
            models=config['models'],
            seeds=config['seeds'],
            fixed=config.get('fixed', {}),
            shared_grid=config.get('shared_grid', {}),
            per_model_grid=config.get('per_model_grid', {}),
            per_model_dataset_grid=per_model_dataset_grid,
            per_dataset_ratio_method_grid=per_dataset_ratio_method_grid,
            string_adjacency_threshold=config.get('string_adjacency_threshold', 0.4),
            timeout=config.get('training', {}).get('timeout', 3600),
            output_dir=config.get('training', {}).get('output_dir', './search_results'),
            tags=config.get('tags', []),
        )


@dataclass
class RunConfig:
    """Configuration for a single training run."""

    run_id: int
    model: str
    dataset: str
    seed: int
    overrides: list[str]
    hyperparams: dict[str, Any]
    timeout: int | None = None
    gpu_id: int | None = None
    n_threads: int | None = None


def to_override(key: str, value: Any) -> str:
    """Convert a key-value pair to Hydra override string."""
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    if isinstance(value, str):
        return f'{key}={value}'
    if isinstance(value, list | tuple):
        inner = ','.join(str(x) for x in value)
        return f'{key}=[{inner}]'
    if value is None:
        return f'{key}=null'
    return f'{key}={value}'


def product_dict(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate all combinations from a parameter grid."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    vals = [list(v) for v in grid.values()]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*vals)]


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def dry_run_config(overrides: list[str]) -> tuple[int | None, str | None]:
    """Evaluate a configuration without training (count parameters only)."""
    try:
        if GlobalHydra().is_initialized():
            GlobalHydra().clear()

        register_all_resolvers()

        # Get absolute path to configs directory
        script_dir = Path(__file__).resolve().parent
        config_dir = script_dir.parent / 'configs'

        initialize_config_dir(config_dir=str(config_dir), job_name='dry_run', version_base=None)

        cfg = compose(
            config_name='train.yaml',
            overrides=overrides,
            return_hydra_config=True,
        )

        model = instantiate(
            cfg.model, evaluator=cfg.evaluator, optimizer=cfg.optimizer, loss=cfg.loss
        ).cpu()

        n_params = count_trainable_params(model)
        return n_params, None

    except Exception as e:
        return None, str(e)
    finally:
        # Always clean up GlobalHydra to prevent state leakage in parallel execution
        if GlobalHydra().is_initialized():
            GlobalHydra().clear()


def run_training(
    overrides: list[str],
    timeout: int | None = None,
    gpu_id: int | None = None,
    n_threads: int | None = None,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Run training via subprocess."""
    cmd = ['ogbench-train'] + overrides

    env = os.environ.copy()
    if gpu_id is not None and torch.cuda.is_available():
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if n_threads is not None:
        env['OMP_NUM_THREADS'] = str(n_threads)
        env['MKL_NUM_THREADS'] = str(n_threads)

    try:
        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )

        if result.returncode == 0:
            metrics = None
            try:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('{') and line.endswith('}'):
                        metrics = json.loads(line)
                        break
            except (json.JSONDecodeError, ValueError):
                pass
            return True, None, metrics
        else:
            # Truncate error message to prevent memory issues with large outputs
            # Show first 500 chars (where error typically is) and last 500 chars
            stderr_truncated = (
                result.stderr[:500] + '...' + result.stderr[-500:]
                if len(result.stderr) > 1000
                else result.stderr
            )
            error_msg = f'Return code {result.returncode}\nSTDERR: {stderr_truncated}'
            return False, error_msg, None

    except subprocess.TimeoutExpired:
        return False, f'Timeout after {timeout}s', None
    except Exception as e:
        return False, str(e), None


def execute_run(config: RunConfig, dry_run: bool = False) -> dict[str, Any]:
    """Execute a single run configuration."""
    print(f'[{config.run_id:04d}] {config.model} | seed={config.seed} | GPU={config.gpu_id}')

    start_time = time.time()

    if dry_run:
        n_params, error = dry_run_config(config.overrides)
        success = n_params is not None
        metrics = {'params': n_params} if success else None
        error_msg = error
    else:
        success, error_msg, metrics = run_training(
            config.overrides, config.timeout, config.gpu_id, config.n_threads
        )

    elapsed = time.time() - start_time

    result = {
        'run_id': config.run_id,
        'model': config.model,
        'dataset': config.dataset,
        'seed': config.seed,
        'success': success,
        'elapsed_time': elapsed,
        'overrides': ' '.join(config.overrides),
        'error': error_msg,
        'gpu_id': config.gpu_id,
        **config.hyperparams,
    }

    if dry_run and metrics:
        result['params'] = metrics.get('params')
    elif metrics:
        result.update(metrics)

    status = 'OK' if success else 'FAIL'
    print(f'[{config.run_id:04d}] {status} ({elapsed:.1f}s)')
    if not success and error_msg:
        print(f'         Error: {error_msg[:100]}...')

    return result


def _execute_with_gpu_pool(config: RunConfig, gpu_queue, dry_run: bool = False) -> dict[str, Any]:
    """Execute a run with dynamic GPU assignment from a shared pool.

    Acquires a GPU slot from the queue before running and releases it after,
    guaranteeing that no more than `jobs_per_gpu` jobs share a single GPU.
    """
    gpu_id = gpu_queue.get()
    try:
        config.gpu_id = gpu_id
        return execute_run(config, dry_run)
    finally:
        gpu_queue.put(gpu_id)


def build_run_configs(
    search_config: SearchConfig,
    models_filter: list[str] | None = None,
    n_gpus: int = 1,
    parallel: bool = True,
    dry_run: bool = False,
    n_threads: int | None = None,
) -> list[RunConfig]:
    """Build all run configurations from search config."""
    configs = []
    run_id = 0

    models = models_filter if models_filter else search_config.models

    for model in models:
        for dataset in search_config.datasets:
            # Multi-level grid merge (priority: shared < per_model < per_model_dataset < per_dataset_ratio)
            # Get model-specific grid
            model_grid = search_config.per_model_grid.get(model, {})

            # Get model+dataset-specific grid
            model_dataset_grid = search_config.per_model_dataset_grid.get((model, dataset), {})

            # Combine base grids (per_dataset_ratio applied conditionally after generating combinations)
            base_grid = {
                **search_config.shared_grid,
                **model_grid,
                **model_dataset_grid,
            }

            # Generate all hyperparameter combinations
            for hp_combo in product_dict(base_grid):
                # Apply per_dataset_ratio_method_grid if dataset, node_sample_ratio, and method match
                node_sample_ratio_key = 'dataset.loader.parameters.node_sample_ratio'
                method_key = 'dataset.loader.parameters.method'
                dataset_ratio_grid = {}

                if node_sample_ratio_key in hp_combo and method_key in hp_combo:
                    node_sample_ratio_value = hp_combo[node_sample_ratio_key]
                    method_value = hp_combo[method_key]
                    # Try to match as float or string
                    ratio_key = None
                    # Try exact match first
                    if (
                        dataset,
                        node_sample_ratio_value,
                        method_value,
                    ) in search_config.per_dataset_ratio_method_grid:
                        ratio_key = (dataset, node_sample_ratio_value, method_value)
                    else:
                        # Try float conversion for matching
                        try:
                            float_value = float(node_sample_ratio_value)
                            if (
                                dataset,
                                float_value,
                                method_value,
                            ) in search_config.per_dataset_ratio_method_grid:
                                ratio_key = (dataset, float_value, method_value)
                        except (ValueError, TypeError):
                            pass

                    if ratio_key:
                        dataset_ratio_grid = search_config.per_dataset_ratio_method_grid[ratio_key]
                        # Generate combinations from dataset_ratio_grid and merge each into hp_combo
                        for ratio_hp_combo in product_dict(dataset_ratio_grid):
                            final_hp_combo = {**hp_combo, **ratio_hp_combo}

                            if (
                                final_hp_combo.get('dataset.loader.parameters.adjacency_method')
                                == 'string'
                            ):
                                final_hp_combo[
                                    'dataset.loader.parameters.adjacency_threshold'
                                ] = search_config.string_adjacency_threshold

                            for seed in search_config.seeds:
                                run_id += 1

                                # Build overrides list
                                all_tags = [model, dataset, 'hpsearch'] + search_config.tags
                                tags_str = ','.join(all_tags)
                                overrides = [
                                    f'model={model}',
                                    f'dataset={dataset}',
                                    f'seed={seed}',
                                    f'logger.wandb.tags=[{tags_str}]',
                                    f'hydra.run.dir=${{paths.log_dir}}/${{task_name}}/runs/hpsearch_{run_id:04d}',
                                ]

                                # Add fixed parameters
                                for key, value in search_config.fixed.items():
                                    overrides.append(to_override(key, value))

                                # Add hyperparameters (skip OmicsReadOut params if NoReadOut)
                                readout_name = final_hp_combo.get('model.readout.readout_name')
                                experiment = final_hp_combo.get(
                                    'experiment', search_config.fixed.get('experiment')
                                )
                                is_no_readout = (
                                    readout_name == 'NoReadOut' or experiment == 'no_readout'
                                )
                                # Auto-set fc_dropout to match backbone.dropout if not explicitly set
                                if (
                                    not is_no_readout
                                    and 'model.readout.fc_dropout' not in final_hp_combo
                                    and 'model.backbone.dropout' in final_hp_combo
                                ):
                                    final_hp_combo['model.readout.fc_dropout'] = final_hp_combo[
                                        'model.backbone.dropout'
                                    ]

                                for key, value in final_hp_combo.items():
                                    if is_no_readout and key in (
                                        'model.readout.fc_dim',
                                        'model.readout.fc_dropout',
                                    ):
                                        continue
                                    overrides.append(to_override(key, value))

                                # Assign GPU round-robin
                                gpu_id = None
                                if parallel and not dry_run and n_gpus > 0:
                                    gpu_id = (run_id - 1) % n_gpus

                                configs.append(
                                    RunConfig(
                                        run_id=run_id,
                                        model=model,
                                        dataset=dataset,
                                        seed=seed,
                                        overrides=overrides,
                                        hyperparams=final_hp_combo,
                                        timeout=search_config.timeout,
                                        gpu_id=gpu_id,
                                        n_threads=n_threads,
                                    )
                                )
                        continue  # Skip the else block below

                # No per_dataset_ratio_method_grid match, use hp_combo as-is
                if hp_combo.get('dataset.loader.parameters.adjacency_method') == 'string':
                    hp_combo[
                        'dataset.loader.parameters.adjacency_threshold'
                    ] = search_config.string_adjacency_threshold
                elif 'dataset.loader.parameters.adjacency_threshold' not in hp_combo:
                    adj_method = hp_combo.get(
                        'dataset.loader.parameters.adjacency_method', 'unknown'
                    )
                    ratio = hp_combo.get('dataset.loader.parameters.node_sample_ratio', '?')
                    method = hp_combo.get('dataset.loader.parameters.method', '?')
                    raise ValueError(
                        f'No adjacency_threshold found in per_dataset_ratio_method_grid for '
                        f'({dataset}, {ratio}, {method}) with adjacency_method={adj_method}. '
                        f"Add an entry to per_dataset_ratio_method_grid or set adjacency_method to 'string'."
                    )

                for seed in search_config.seeds:
                    run_id += 1

                    # Build overrides list
                    all_tags = [model, dataset, 'hpsearch'] + search_config.tags
                    tags_str = ','.join(all_tags)
                    overrides = [
                        f'model={model}',
                        f'dataset={dataset}',
                        f'seed={seed}',
                        f'logger.wandb.tags=[{tags_str}]',
                        f'hydra.run.dir=${{paths.log_dir}}/${{task_name}}/runs/hpsearch_{run_id:04d}',
                    ]

                    # Add fixed parameters
                    for key, value in search_config.fixed.items():
                        overrides.append(to_override(key, value))

                    # Add hyperparameters (skip OmicsReadOut params if NoReadOut)
                    readout_name = hp_combo.get('model.readout.readout_name')
                    experiment = hp_combo.get('experiment', search_config.fixed.get('experiment'))
                    is_no_readout = readout_name == 'NoReadOut' or experiment == 'no_readout'
                    if (
                        not is_no_readout
                        and 'model.readout.fc_dropout' not in hp_combo
                        and 'model.backbone.dropout' in hp_combo
                    ):
                        hp_combo['model.readout.fc_dropout'] = hp_combo['model.backbone.dropout']

                    for key, value in hp_combo.items():
                        if is_no_readout and key in (
                            'model.readout.fc_dim',
                            'model.readout.fc_dropout',
                        ):
                            continue
                        overrides.append(to_override(key, value))

                    # Assign GPU round-robin
                    gpu_id = None
                    if parallel and not dry_run and n_gpus > 0:
                        gpu_id = (run_id - 1) % n_gpus

                    configs.append(
                        RunConfig(
                            run_id=run_id,
                            model=model,
                            dataset=dataset,
                            seed=seed,
                            overrides=overrides,
                            hyperparams=hp_combo,
                            timeout=search_config.timeout,
                            gpu_id=gpu_id,
                            n_threads=n_threads,
                        )
                    )

    return configs


def warmup_caches(dataset_configs: list[dict[str, Any]], data_dir: str) -> None:
    """Pre-generate dataset caches to avoid race conditions in parallel training.

    Sequentially instantiates HFOmicsDataset for each unique config, triggering the download() and
    process() methods if the cache doesn't exist.
    """
    from ogbench.data.datasets import HFOmicsDataset

    print(f'Warming up {len(dataset_configs)} dataset caches...')
    for i, config in enumerate(dataset_configs, 1):
        print(
            f'  [{i}/{len(dataset_configs)}] {config["data_name"]} | '
            f'ratio={config["node_sample_ratio"]} | method={config["method"]} | '
            f'threshold={config["adjacency_threshold"]} | '
            f'adj_method={config["adjacency_method"]}'
        )

        HFOmicsDataset(
            root=data_dir,
            data_name=config['data_name'],
            node_sample_ratio=config['node_sample_ratio'],
            method=config['method'],
            adjacency_threshold=config['adjacency_threshold'],
            adjacency_method=config['adjacency_method'],
        )

    print('Cache warmup complete.')


def extract_unique_dataset_configs(search_config: SearchConfig) -> list[dict[str, Any]]:
    """Extract unique dataset configurations that need cache warmup.

    Identifies all unique (dataset, node_sample_ratio, method, adjacency_threshold) combinations
    from the search config to pre-generate caches before parallel training.
    """
    unique_configs: set[tuple[str, float, str, float, str]] = set()
    adj_method_key = 'dataset.loader.parameters.adjacency_method'
    if adj_method_key in search_config.fixed:
        adjacency_methods = [search_config.fixed[adj_method_key]]
    elif adj_method_key in search_config.shared_grid:
        adjacency_methods = search_config.shared_grid[adj_method_key]
    else:
        raise ValueError(
            f"'{adj_method_key}' must be set in the search config's "
            f"'fixed' or 'shared_grid' section for cache warmup to work correctly."
        )

    for dataset in search_config.datasets:
        ratios = search_config.shared_grid.get(
            'dataset.loader.parameters.node_sample_ratio', [0.5]
        )
        methods = search_config.shared_grid.get('dataset.loader.parameters.method', ['variance'])

        for ratio in ratios:
            for method in methods:
                ratio_key = (dataset, float(ratio), method)
                threshold_grid = search_config.per_dataset_ratio_method_grid.get(ratio_key, {})
                thresholds = threshold_grid.get(
                    'dataset.loader.parameters.adjacency_threshold', [0.05]
                )

                for threshold in thresholds:
                    for adj_method in adjacency_methods:
                        effective_threshold = (
                            search_config.string_adjacency_threshold
                            if adj_method == 'string'
                            else float(threshold)
                        )
                        unique_configs.add(
                            (dataset, float(ratio), method, effective_threshold, adj_method)
                        )

    return [
        {
            'data_name': d,
            'node_sample_ratio': r,
            'method': m,
            'adjacency_threshold': t,
            'adjacency_method': am,
        }
        for d, r, m, t, am in sorted(unique_configs)
    ]


def run_search(
    search_config: SearchConfig,
    models_filter: list[str] | None = None,
    dry_run: bool = False,
    parallel: bool = True,
    n_jobs: int | None = None,
    jobs_per_gpu: int = 1,
    skip_warmup: bool = False,
) -> pd.DataFrame:
    """Run the full hyperparameter search."""
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_jobs is None:
        n_jobs = max(n_gpus * jobs_per_gpu, 1)

    # Cap CPU threads per job to avoid thrashing when running in parallel.
    # Each job gets an equal share of available cores.
    n_cpu = os.cpu_count() or 1
    n_threads = max(1, n_cpu // n_jobs)

    # Build all run configurations
    configs = build_run_configs(
        search_config,
        models_filter=models_filter,
        n_gpus=n_gpus,
        parallel=parallel,
        dry_run=dry_run,
        n_threads=n_threads,
    )

    # Print summary
    print('=' * 60)
    print('HYPERPARAMETER SEARCH')
    print('=' * 60)
    print(f'Datasets: {search_config.datasets}')
    print(f'Models: {models_filter or search_config.models}')
    print(f'Seeds: {search_config.seeds}')
    print(f'Total runs: {len(configs)}')
    print(f"Mode: {'DRY RUN' if dry_run else 'TRAINING'}")
    print(f'Parallel: {parallel} (n_jobs={n_jobs}, n_gpus={n_gpus}, jobs_per_gpu={jobs_per_gpu})')
    print(f'CPU threads per job: {n_threads} (of {n_cpu} total cores)')
    print(f'Output: {search_config.output_dir}')
    print('=' * 60)

    # Warmup dataset caches to avoid race conditions in parallel training
    if not dry_run and not skip_warmup:
        root_dir = search_config.fixed.get('paths.root_dir', os.environ.get('PROJECT_ROOT', '.'))
        data_dir = os.path.join(root_dir, 'data', 'omics')
        dataset_configs = extract_unique_dataset_configs(search_config)
        print(f'\nUnique dataset configs to cache: {len(dataset_configs)}')
        warmup_caches(dataset_configs, data_dir)

    # Create output directory
    os.makedirs(search_config.output_dir, exist_ok=True)

    # Execute runs
    if parallel and not dry_run and n_jobs > 1:
        print(f'\nRunning {len(configs)} configs in parallel...')
        if n_gpus > 0:
            manager = multiprocessing.Manager()
            gpu_queue = manager.Queue()
            for gpu_id in range(n_gpus):
                for _ in range(jobs_per_gpu):
                    gpu_queue.put(gpu_id)
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_execute_with_gpu_pool)(config, gpu_queue) for config in configs
            )
        else:
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(execute_run)(config, dry_run) for config in configs
            )
    else:
        print(f'\nRunning {len(configs)} configs sequentially...')
        results = [execute_run(config, dry_run) for config in configs]

    # Save results
    df = pd.DataFrame(results)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    mode_suffix = 'dryrun' if dry_run else 'train'
    detailed_path = os.path.join(
        search_config.output_dir, f'results_{mode_suffix}_{timestamp}.csv'
    )
    df.to_csv(detailed_path, index=False)

    # Create summary
    summary_rows = []
    for (model, dataset), group in df.groupby(['model', 'dataset']):
        total = len(group)
        success = group['success'].sum()
        summary_row = {
            'model': model,
            'dataset': dataset,
            'total_runs': total,
            'successful': success,
            'failed': total - success,
            'success_rate': success / total if total > 0 else 0,
            'avg_time': group['elapsed_time'].mean(),
        }
        if dry_run and 'params' in group.columns:
            valid = group.dropna(subset=['params'])
            if not valid.empty:
                summary_row['min_params'] = int(valid['params'].min())
                summary_row['max_params'] = int(valid['params'].max())
                summary_row['avg_params'] = int(valid['params'].mean())
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(search_config.output_dir, f'summary_{mode_suffix}_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)

    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(summary_df.to_string(index=False))
    print(f'\nDetailed results: {detailed_path}')
    print(f'Summary: {summary_path}')

    return df


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter grid search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full search
    python scripts/hyperparam_search.py --config configs/hparams_search/parkinsons_grid.yaml

    # Dry run (count parameters only)
    python scripts/hyperparam_search.py --config configs/hparams_search/parkinsons_grid.yaml --dry-run

    # Search specific models only
    python scripts/hyperparam_search.py --config configs/hparams_search/parkinsons_grid.yaml --models gcn gin

    # Sequential execution (no parallelism)
    python scripts/hyperparam_search.py --config configs/hparams_search/parkinsons_grid.yaml --no-parallel
        """,
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to search config YAML',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Count parameters only, no training',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Filter to specific models (default: all from config)',
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel execution',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        help='Number of parallel jobs (default: n_gpus * jobs_per_gpu)',
    )
    parser.add_argument(
        '--jobs-per-gpu',
        type=int,
        default=1,
        help='Max concurrent jobs per GPU (default: 1). Total jobs = n_gpus * jobs_per_gpu.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config',
    )
    parser.add_argument(
        '--skip-warmup',
        action='store_true',
        help='Skip dataset cache warmup (use when caches already exist)',
    )

    args = parser.parse_args()

    # Load search configuration
    search_config = SearchConfig.from_yaml(args.config)

    # Override output directory if specified
    if args.output_dir:
        search_config.output_dir = args.output_dir

    # Disable gradient computation for dry runs
    if args.dry_run:
        torch.set_grad_enabled(False)
        if not torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Run search
    run_search(
        search_config,
        models_filter=args.models,
        dry_run=args.dry_run,
        parallel=not args.no_parallel,
        n_jobs=args.n_jobs,
        jobs_per_gpu=args.jobs_per_gpu,
        skip_warmup=args.skip_warmup,
    )


if __name__ == '__main__':
    main()

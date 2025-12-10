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
from omegaconf import OmegaConf

from ogbench.utils.config_resolvers import (
    calculate_num_nodes,
    get_default_metrics,
    get_default_trainer,
    get_default_transform,
    get_flattened_channels,
    get_gatv4_output_dim,
    get_monitor_metric,
    get_monitor_mode,
    get_non_relational_out_channels,
    get_required_lifting,
    infer_in_channels,
    infer_num_cell_dimensions,
)


def register_resolvers() -> None:
    """Register all custom OmegaConf resolvers."""
    OmegaConf.register_new_resolver('calculate_num_nodes', calculate_num_nodes, replace=True)
    OmegaConf.register_new_resolver('get_default_metrics', get_default_metrics, replace=True)
    OmegaConf.register_new_resolver('get_default_trainer', get_default_trainer, replace=True)
    OmegaConf.register_new_resolver('get_default_transform', get_default_transform, replace=True)
    OmegaConf.register_new_resolver('get_flattened_channels', get_flattened_channels, replace=True)
    OmegaConf.register_new_resolver('get_required_lifting', get_required_lifting, replace=True)
    OmegaConf.register_new_resolver('get_monitor_metric', get_monitor_metric, replace=True)
    OmegaConf.register_new_resolver('get_monitor_mode', get_monitor_mode, replace=True)
    OmegaConf.register_new_resolver('get_gatv4_output_dim', get_gatv4_output_dim, replace=True)
    OmegaConf.register_new_resolver(
        'get_non_relational_out_channels', get_non_relational_out_channels, replace=True
    )
    OmegaConf.register_new_resolver('infer_in_channels', infer_in_channels, replace=True)
    OmegaConf.register_new_resolver(
        'infer_num_cell_dimensions', infer_num_cell_dimensions, replace=True
    )


register_resolvers()


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""

    dataset: str
    models: list[str]
    seeds: list[int]
    fixed: dict[str, Any]
    shared_grid: dict[str, list[Any]]
    per_model_grid: dict[str, dict[str, list[Any]]]
    timeout: int
    output_dir: str

    @classmethod
    def from_yaml(cls, path: str) -> 'SearchConfig':
        """Load search configuration from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)

        return cls(
            dataset=config['dataset'],
            models=config['models'],
            seeds=config['seeds'],
            fixed=config.get('fixed', {}),
            shared_grid=config.get('shared_grid', {}),
            per_model_grid=config.get('per_model_grid', {}),
            timeout=config.get('training', {}).get('timeout', 3600),
            output_dir=config.get('training', {}).get('output_dir', './search_results'),
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

        register_resolvers()

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


def run_training(
    overrides: list[str], timeout: int | None = None, gpu_id: int | None = None
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Run training via subprocess."""
    cmd = ['ogbench-train'] + overrides

    env = os.environ.copy()
    if gpu_id is not None and torch.cuda.is_available():
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

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
            error_msg = f'Return code {result.returncode}\nSTDERR: {result.stderr[-1000:]}'
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
        success, error_msg, metrics = run_training(config.overrides, config.timeout, config.gpu_id)

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


def build_run_configs(
    search_config: SearchConfig,
    models_filter: list[str] | None = None,
    n_gpus: int = 1,
    parallel: bool = True,
    dry_run: bool = False,
) -> list[RunConfig]:
    """Build all run configurations from search config."""
    configs = []
    run_id = 0

    models = models_filter if models_filter else search_config.models

    for model in models:
        # Get model-specific grid
        model_grid = search_config.per_model_grid.get(model, {})

        # Combine shared and model-specific grids
        full_grid = {**search_config.shared_grid, **model_grid}

        # Generate all hyperparameter combinations
        for hp_combo in product_dict(full_grid):
            for seed in search_config.seeds:
                run_id += 1

                # Build overrides list
                overrides = [
                    f'model={model}',
                    f'dataset={search_config.dataset}',
                    f'seed={seed}',
                    f'logger.wandb.tags=[{model},{search_config.dataset},hpsearch]',
                ]

                # Add fixed parameters
                for key, value in search_config.fixed.items():
                    overrides.append(to_override(key, value))

                # Add hyperparameters
                for key, value in hp_combo.items():
                    overrides.append(to_override(key, value))

                # Assign GPU round-robin
                gpu_id = None
                if parallel and not dry_run and n_gpus > 0:
                    gpu_id = (run_id - 1) % n_gpus

                configs.append(
                    RunConfig(
                        run_id=run_id,
                        model=model,
                        dataset=search_config.dataset,
                        seed=seed,
                        overrides=overrides,
                        hyperparams=hp_combo,
                        timeout=search_config.timeout,
                        gpu_id=gpu_id,
                    )
                )

    return configs


def run_search(
    search_config: SearchConfig,
    models_filter: list[str] | None = None,
    dry_run: bool = False,
    parallel: bool = True,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Run the full hyperparameter search."""
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_jobs = n_jobs or max(n_gpus, 1)

    # Build all run configurations
    configs = build_run_configs(
        search_config,
        models_filter=models_filter,
        n_gpus=n_gpus,
        parallel=parallel,
        dry_run=dry_run,
    )

    # Print summary
    print('=' * 60)
    print('HYPERPARAMETER SEARCH')
    print('=' * 60)
    print(f'Dataset: {search_config.dataset}')
    print(f'Models: {models_filter or search_config.models}')
    print(f'Seeds: {search_config.seeds}')
    print(f'Total runs: {len(configs)}')
    print(f"Mode: {'DRY RUN' if dry_run else 'TRAINING'}")
    print(f'Parallel: {parallel} (n_jobs={n_jobs}, n_gpus={n_gpus})')
    print(f'Output: {search_config.output_dir}')
    print('=' * 60)

    # Create output directory
    os.makedirs(search_config.output_dir, exist_ok=True)

    # Execute runs
    if parallel and not dry_run and n_jobs > 1:
        print(f'\nRunning {len(configs)} configs in parallel...')
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
    for model, group in df.groupby('model'):
        total = len(group)
        success = group['success'].sum()
        summary_row = {
            'model': model,
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
        help='Number of parallel jobs (default: number of GPUs)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config',
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
    )


if __name__ == '__main__':
    main()

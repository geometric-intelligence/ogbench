#!/usr/bin/env python3
"""Hyperparameter search driver script.

This script performs grid search over hyperparameters by calling run.py via subprocess. Supports
both normal execution and dry-run mode for parameter counting.
"""

import argparse
import itertools
import json
import os
import subprocess  # nosec B404
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf

# Import resolvers
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


def register_resolvers():
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


# Register resolvers immediately when module is imported
register_resolvers()


def run_single_config_worker(config_data: dict[str, Any]) -> dict[str, Any]:
    """Worker function for parallel execution using joblib."""
    # Create a temporary search instance for this worker
    search = HyperparameterSearch()

    run_id = config_data['run_id']
    model_key = config_data['model']
    dataset_key = config_data['dataset']
    seed = config_data['seed']
    hp = config_data['hp']
    overrides = config_data['overrides']
    timeout = config_data.get('timeout')
    gpu_id = config_data.get('gpu_id')
    dry_run = config_data.get('dry_run', False)

    print(f'[{run_id}] Model: {model_key}, Dataset: {dataset_key}, Seed: {seed}, GPU: {gpu_id}')
    print(f"[{run_id}] Overrides: {' '.join(overrides)}")

    start_time = time.time()

    if dry_run:
        n_params, error = search.dry_run_config(overrides)
        success = n_params is not None
        metrics = {'params': n_params} if success else None
        error_msg = error
    else:
        success, error_msg, metrics = search.run_config(overrides, timeout, gpu_id)
        n_params = None

    elapsed = time.time() - start_time

    result = {
        'run_id': run_id,
        'model': model_key,
        'dataset': dataset_key,
        'seed': seed,
        'success': success,
        'elapsed_time': elapsed,
        'overrides': ' '.join(overrides),
        'error': error_msg,
        'gpu_id': gpu_id,
        **hp,
    }

    if dry_run:
        result['params'] = n_params
    elif metrics:
        result.update(metrics)

    status = '✅ SUCCESS' if success else '❌ FAILED'
    print(f'[{run_id}] {status} ({elapsed:.1f}s)')
    if not success:
        print(f'[{run_id}] Error: {error_msg}')
    print()

    return result


class HyperparameterSearch:
    """Driver for hyperparameter search via subprocess calls to run.py."""

    def __init__(
        self,
        config_path: str = 'configs',
        project_cfg: str = 'train.yaml',
        n_jobs: int | None = None,
    ):
        self.config_path = config_path
        self.project_cfg = project_cfg
        self.results: list[dict[str, Any]] = []
        self.n_jobs = n_jobs or min(
            torch.cuda.device_count() if torch.cuda.is_available() else -1, 8
        )

        # Stay off GPU for dry runs
        if not torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.set_grad_enabled(False)

    def product_dict(self, grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
        """Generate all combinations from a parameter grid."""
        keys = list(grid.keys())
        vals = [list(v) for v in grid.values()]
        return [dict(zip(keys, tup, strict=True)) for tup in itertools.product(*vals)]

    def to_override(self, k: str, v: Any) -> str:
        """Convert parameter to Hydra override string."""
        if isinstance(v, bool):
            return f"{k}={'true' if v else 'false'}"
        if isinstance(v, str):
            return f'{k}={v}'
        if isinstance(v, list | tuple):
            inner = ','.join(str(x) for x in v)
            return f'{k}=[{inner}]'
        return f'{k}={v}'

    def build_overrides(self, base_overrides: list[str], hp_dict: dict[str, Any]) -> list[str]:
        """Build complete override list from base + hyperparameters."""
        return base_overrides + [self.to_override(k, v) for k, v in hp_dict.items()]

    def count_trainable_params(self, model: torch.nn.Module) -> int:
        """Count trainable parameters in a model."""
        return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)

    def dry_run_config(self, overrides: list[str]) -> tuple[int | None, str | None]:
        """Evaluate a configuration without running training (dry run)"""
        try:
            # Clear GlobalHydra instance if already initialized
            if GlobalHydra().is_initialized():
                GlobalHydra().clear()

            # Register resolvers before initializing Hydra
            register_resolvers()

            initialize(config_path=self.config_path, job_name='dry_run')

            # Use overrides as-is since dataset is now explicitly included in grid
            final_overrides = overrides.copy()

            cfg = compose(
                config_name=self.project_cfg,
                overrides=final_overrides,
                return_hydra_config=True,
            )

            # Instantiate model only (no training)
            model = instantiate(
                cfg.model, evaluator=cfg.evaluator, optimizer=cfg.optimizer, loss=cfg.loss
            ).cpu()

            n_params = self.count_trainable_params(model)
            return n_params, None

        except Exception as e:
            return None, str(e)

    def run_config(
        self, overrides: list[str], timeout: int | None = None, gpu_id: int | None = None
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Run a configuration via subprocess call to run.py."""
        # Use overrides as-is since dataset is now explicitly included in grid
        final_overrides = overrides.copy()

        cmd = [sys.executable, 'ogbench/run.py'] + final_overrides

        # Set GPU environment if specified
        env = os.environ.copy()
        if gpu_id is not None and torch.cuda.is_available():
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(__file__).parent,
                env=env,
                check=False,  # nosec B603
            )

            if result.returncode == 0:
                # Try to parse metrics from stdout if available
                metrics = None
                try:
                    # Look for JSON-like output in stdout
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.startswith('{') and line.endswith('}'):
                            metrics = json.loads(line)
                            break
                except (json.JSONDecodeError, ValueError, KeyError):
                    # Ignore JSON parsing errors
                    pass

                return True, None, metrics
            else:
                error_msg = f'Return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}'
                return False, error_msg, None

        except subprocess.TimeoutExpired:
            return False, f'Timeout after {timeout} seconds', None
        except Exception as e:
            return False, str(e), None

    def run_single_config(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Run a single configuration (for sequential execution)."""
        run_id = config_data['run_id']
        model_key = config_data['model']
        dataset_key = config_data['dataset']
        seed = config_data['seed']
        hp = config_data['hp']
        overrides = config_data['overrides']
        timeout = config_data.get('timeout')
        gpu_id = config_data.get('gpu_id')
        dry_run = config_data.get('dry_run', False)

        print(
            f'[{run_id}] Model: {model_key}, Dataset: {dataset_key}, Seed: {seed}, GPU: {gpu_id}'
        )
        print(f"[{run_id}] Overrides: {' '.join(overrides)}")

        start_time = time.time()

        if dry_run:
            n_params, error = self.dry_run_config(overrides)
            success = n_params is not None
            metrics = {'params': n_params} if success else None
            error_msg = error
        else:
            success, error_msg, metrics = self.run_config(overrides, timeout, gpu_id)
            n_params = None

        elapsed = time.time() - start_time

        result = {
            'run_id': run_id,
            'model': model_key,
            'dataset': dataset_key,
            'seed': seed,
            'success': success,
            'elapsed_time': elapsed,
            'overrides': ' '.join(overrides),
            'error': error_msg,
            'gpu_id': gpu_id,
            **hp,
        }

        if dry_run:
            result['params'] = n_params
        elif metrics:
            result.update(metrics)

        status = '✅ SUCCESS' if success else '❌ FAILED'
        print(f'[{run_id}] {status} ({elapsed:.1f}s)')
        if not success:
            print(f'[{run_id}] Error: {error_msg}')
        print()

        return result

    def search(
        self,
        models: list[str],
        datasets: list[str],
        shared_grid: dict[str, Iterable[Any]],
        per_model_dataset_grid: dict[tuple[str, str], dict[str, Iterable[Any]]],
        seeds: list[int],
        dry_run: bool = False,
        timeout: int | None = None,
        output_dir: str = './search_results',
        parallel: bool = True,
    ) -> pd.DataFrame:
        """Perform hyperparameter search.

        Parameters
        ----------
        models : List[str]
            List of model names to search over
        datasets : List[str]
            List of dataset names to search over
        shared_grid : Dict[str, Iterable[Any]]
            Shared hyperparameters for all models and datasets
        per_model_dataset_grid : Dict[Tuple[str, str], Dict[str, Iterable[Any]]]
            Model and dataset-specific hyperparameters indexed by (model, dataset)
        seeds : List[int]
            List of random seeds to iterate over for each configuration
        dry_run : bool
            If True, only count parameters without training
        timeout : Optional[int]
            Timeout in seconds for each run (None for no timeout)
        output_dir : str
            Directory to save results
        parallel : bool
            If True, run configurations in parallel across GPUs

        Returns
        -------
        pd.DataFrame
            Results dataframe
        """
        os.makedirs(output_dir, exist_ok=True)

        total_combinations = 0
        for model_key in models:
            for dataset_key in datasets:
                model_dataset_specific = per_model_dataset_grid.get((model_key, dataset_key), {})
                full_grid = dict(shared_grid)
                full_grid.update(model_dataset_specific)
                total_combinations += len(self.product_dict(full_grid)) * len(seeds)

        print('Starting hyperparameter search...')
        print(f'Models: {models}')
        print(f'Datasets: {datasets}')
        print(f'Seeds: {seeds}')
        print(f'Total combinations: {total_combinations}')
        print(f"Mode: {'DRY RUN' if dry_run else 'TRAINING'}")
        print(f'Parallel: {parallel} ({self.n_jobs} jobs)')
        print(f'Output directory: {output_dir}')
        print('-' * 50)

        # Prepare all configurations
        configs = []
        current_run = 0
        available_gpus = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
        )

        for seed in seeds:
            for model_key in models:
                for dataset_key in datasets:
                    model_dataset_specific = per_model_dataset_grid.get(
                        (model_key, dataset_key), {}
                    )
                    full_grid = dict(shared_grid)
                    full_grid.update(model_dataset_specific)

                    for hp in self.product_dict(full_grid):
                        current_run += 1
                        overrides = [
                            f'model={model_key}',
                            f'dataset={dataset_key}',
                            f'seed={seed}',
                        ]
                        overrides = self.build_overrides(overrides, hp)

                        gpu_id = (
                            available_gpus[(current_run - 1) % len(available_gpus)]
                            if parallel and not dry_run
                            else None
                        )

                        config_data = {
                            'run_id': current_run,
                            'model': model_key,
                            'dataset': dataset_key,
                            'seed': seed,
                            'hp': hp,
                            'overrides': overrides,
                            'timeout': timeout,
                            'gpu_id': gpu_id,
                            'dry_run': dry_run,
                        }
                        configs.append(config_data)

        # Run configurations
        if parallel and not dry_run:
            # Parallel execution for training runs using joblib
            print(f'Running {len(configs)} configurations in parallel with {self.n_jobs} jobs...')
            results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(run_single_config_worker)(config) for config in configs
            )
            self.results = results
        else:
            # Sequential execution for dry runs or when parallel is disabled
            print(f'Running {len(configs)} configurations sequentially...')
            for config in configs:
                result = self.run_single_config(config)
                self.results.append(result)

        # Save results
        df = pd.DataFrame(self.results)

        # Save detailed results
        detailed_csv = os.path.join(output_dir, 'search_results_detailed.csv')
        df.to_csv(detailed_csv, index=False)

        # Create summary
        summary_rows = []
        for (model, dataset), group in df.groupby(['model', 'dataset']):
            total_runs = len(group)
            successful_runs = group['success'].sum()
            failed_runs = total_runs - successful_runs

            summary_row = {
                'model': model,
                'dataset': dataset,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
                'avg_time': group['elapsed_time'].mean(),
            }

            if dry_run:
                valid_params = group.dropna(subset=['params'])
                if not valid_params.empty:
                    summary_row.update(
                        {
                            'min_params': valid_params['params'].min(),
                            'max_params': valid_params['params'].max(),
                            'avg_params': valid_params['params'].mean(),
                        }
                    )

            summary_rows.append(summary_row)

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(output_dir, 'search_summary.csv')
        summary_df.to_csv(summary_csv, index=False)

        print('Search completed!')
        print('Results saved to:')
        print(f'  - {detailed_csv}')
        print(f'  - {summary_csv}')

        return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Hyperparameter search driver')
    parser.add_argument(
        '--dry-run', action='store_true', help='Dry run mode (count parameters only)'
    )
    parser.add_argument('--timeout', type=int, help='Timeout in seconds for each run')
    parser.add_argument('--output-dir', default='./search_results', help='Output directory')
    parser.add_argument('--config-path', default='configs', help='Hydra config path')
    parser.add_argument('--models', nargs='+', help='Models to search (default: all)')
    parser.add_argument('--n-jobs', type=int, help='Number of parallel jobs (default: auto)')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel execution')

    args = parser.parse_args()

    # Define hyperparameter grids (from your notebook)
    DATASETS = ['motrpac', 'addneuromed', 'parkinsons']
    NODE_SAMPLE_RATIOS = [1.0, 0.8, 0.5]  # , 0.2]  # , 0.125]
    SAMPLE_METHODS = ['variance', 'random']

    OPT_LRS = [0.001]
    OPT_WD = [0.0004]

    READOUT_POOL = ['mean']  # "sum"

    # Models
    MODEL_KEYS = ['sagn', 'chebnet', 'mlp', 'gcn', 'gin', 'gatv4', 'gatv2', 'graph_sage']

    # Seeds for reproducibility
    SEEDS = [42, 123, 456]

    # Dataset-specific adjacency thresholds
    DATASET_ADJ_THRESHOLDS = {
        'addneuromed': [0.3],
        'motrpac': [0.03],
        'parkinsons': [0.03],
    }

    # Model and dataset-specific grids
    PER_MODEL_DATASET_GRID = {
        ('gcn', 'motrpac'): {
            'model.backbone.num_layers': [4],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['motrpac'],
        },
        ('gcn', 'addneuromed'): {
            'model.backbone.num_layers': [4],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['addneuromed'],
        },
        ('gcn', 'parkinsons'): {
            'model.backbone.num_layers': [4],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['parkinsons'],
        },
        ('gin', 'motrpac'): {
            'model.backbone.num_layers': [8],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['motrpac'],
        },
        ('gin', 'addneuromed'): {
            'model.backbone.num_layers': [8],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['addneuromed'],
        },
        ('gin', 'parkinsons'): {
            'model.backbone.num_layers': [8],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['parkinsons'],
        },
        ('gatv2', 'motrpac'): {
            'model.backbone.v2': [True],
            'model.backbone.heads': [8],
            'model.backbone.num_layers': [8],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['motrpac'],
        },
        ('gatv2', 'addneuromed'): {
            'model.backbone.v2': [True],
            'model.backbone.heads': [8],
            'model.backbone.num_layers': [8],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['addneuromed'],
        },
        ('gatv2', 'parkinsons'): {
            'model.backbone.v2': [True],
            'model.backbone.heads': [8],
            'model.backbone.num_layers': [8],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['parkinsons'],
        },
        ('gatv4', 'motrpac'): {
            'model.backbone.hidden_channels': [[16, 32]],
            'model.backbone.heads': [[4, 4]],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['motrpac'],
        },
        ('gatv4', 'addneuromed'): {
            'model.backbone.hidden_channels': [[16, 32]],
            'model.backbone.heads': [[4, 4]],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['addneuromed'],
        },
        ('gatv4', 'parkinsons'): {
            'model.backbone.hidden_channels': [[16, 32]],
            'model.backbone.heads': [[4, 4]],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['parkinsons'],
        },
        ('graph_sage', 'motrpac'): {
            'model.backbone.num_layers': [8],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['motrpac'],
        },
        ('graph_sage', 'addneuromed'): {
            'model.backbone.num_layers': [8],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['addneuromed'],
        },
        ('graph_sage', 'parkinsons'): {
            'model.backbone.num_layers': [8],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['parkinsons'],
        },
        ('chebnet', 'motrpac'): {
            'model.backbone.K': [2],
            'model.backbone.num_layers': [2],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['motrpac'],
        },
        ('chebnet', 'addneuromed'): {
            'model.backbone.K': [2],
            'model.backbone.num_layers': [2],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['addneuromed'],
        },
        ('chebnet', 'parkinsons'): {
            'model.backbone.K': [2],
            'model.backbone.num_layers': [2],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['parkinsons'],
        },
        ('mlp', 'motrpac'): {
            'model.backbone.hidden_channels': [[8, 16, 4]],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['motrpac'],
        },
        ('mlp', 'addneuromed'): {
            'model.backbone.hidden_channels': [[32, 16, 4]],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['addneuromed'],
        },
        ('mlp', 'parkinsons'): {
            'model.backbone.hidden_channels': [[24, 16, 4]],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['parkinsons'],
        },
        ('sagn', 'motrpac'): {
            'model.backbone.hidden_channels': [32],
            'model.backbone.dropout': [0.2],
            'model.backbone.num_layers': [4],
            'model.backbone.alpha': [0.5],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['motrpac'],
        },
        ('sagn', 'addneuromed'): {
            'model.backbone.hidden_channels': [32],
            'model.backbone.dropout': [0.2],
            'model.backbone.num_layers': [4],
            'model.backbone.alpha': [0.5],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['addneuromed'],
        },
        ('sagn', 'parkinsons'): {
            'model.backbone.hidden_channels': [32],
            'model.backbone.dropout': [0.2],
            'model.backbone.num_layers': [4],
            'model.backbone.alpha': [0.5],
            'dataset.loader.parameters.adjacency_threshold': DATASET_ADJ_THRESHOLDS['parkinsons'],
        },
    }

    # Shared grid
    SHARED_GRID = {
        'optimizer.parameters.lr': OPT_LRS,
        'optimizer.parameters.weight_decay': OPT_WD,
        'model.readout.pooling_type': READOUT_POOL,
        'dataset.loader.parameters.node_sample_ratio': NODE_SAMPLE_RATIOS,
        'dataset.loader.parameters.method': SAMPLE_METHODS,
    }

    # Filter models if specified
    models_to_search = args.models if args.models else MODEL_KEYS

    # Initialize search
    search = HyperparameterSearch(config_path=args.config_path, n_jobs=args.n_jobs)

    # Run search
    results_df = search.search(
        models=models_to_search,
        datasets=DATASETS,
        shared_grid=SHARED_GRID,
        per_model_dataset_grid=PER_MODEL_DATASET_GRID,
        seeds=SEEDS,
        dry_run=args.dry_run,
        timeout=args.timeout,
        output_dir=args.output_dir,
        parallel=not args.no_parallel,
    )

    # Print summary
    print('\n' + '=' * 50)
    print('SEARCH SUMMARY')
    print('=' * 50)
    print(results_df.groupby(['model', 'dataset'])['success'].agg(['count', 'sum']).to_string())


if __name__ == '__main__':
    main()

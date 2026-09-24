"""Shared helpers for grid and Optuna hyperparameter-search launchers."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import sys
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

OBJECTIVE_PAYLOAD_PREFIX = 'OGBENCH_OBJECTIVE='
METRICS_PAYLOAD_PREFIX = 'OGBENCH_METRICS='
THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
)


@dataclass(frozen=True)
class GpuDevice:
    """A CUDA device visible to the launcher and its child visibility token."""

    logical_id: int
    visibility_token: str


def to_override(key: str, value: Any) -> str:
    """Convert a key-value pair to a Hydra override string."""
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


def objective_payload(metric_name: str, metric_value: float) -> str:
    """Build the machine-readable objective line emitted by the training process."""
    payload = {'metric_name': metric_name, 'metric_value': float(metric_value)}
    return OBJECTIVE_PAYLOAD_PREFIX + json.dumps(payload, sort_keys=True)


def metrics_payload(metric_dict: dict[str, Any]) -> str:
    """Serialize scalar Lightning metrics for the parent launcher to parse."""
    scalars: dict[str, float] = {}
    for key, value in metric_dict.items():
        try:
            if hasattr(value, 'item') and callable(value.item):
                value = value.item()
            scalars[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return METRICS_PAYLOAD_PREFIX + json.dumps(scalars, sort_keys=True)


def parse_metrics_payload(stdout: str) -> dict[str, float]:
    """Parse the last OGBENCH_METRICS line from training stdout."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith(METRICS_PAYLOAD_PREFIX):
            continue
        try:
            payload = json.loads(line.removeprefix(METRICS_PAYLOAD_PREFIX))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        parsed: dict[str, float] = {}
        for key, value in payload.items():
            try:
                parsed[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return parsed
    return {}


def parse_objective_payload(stdout: str) -> dict[str, Any] | None:
    """Parse the last valid objective payload from subprocess output."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith(OBJECTIVE_PAYLOAD_PREFIX):
            continue
        try:
            payload = json.loads(line.removeprefix(OBJECTIVE_PAYLOAD_PREFIX))
            metric_name = str(payload['metric_name'])
            metric_value = float(payload['metric_value'])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        return {
            'optimized_metric': metric_name,
            'objective': metric_value,
            metric_name: metric_value,
        }
    return None


def single_thread_environment(
    base: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that restricts native numerical libraries to one thread."""
    env = dict(os.environ if base is None else base)
    for name in THREAD_ENV_VARS:
        env[name] = '1'
    env['OGBENCH_NUM_THREADS'] = '1'
    return env


def configure_torch_threads_from_env() -> None:
    """Apply the explicit launcher thread limit inside the training process."""
    raw_limit = os.environ.get('OGBENCH_NUM_THREADS')
    if raw_limit is None:
        return
    limit = int(raw_limit)
    if limit < 1:
        raise ValueError(f'OGBENCH_NUM_THREADS must be positive, got {limit}')
    torch.set_num_threads(limit)
    try:
        torch.set_num_interop_threads(limit)
    except RuntimeError:
        # PyTorch only allows setting inter-op threads before parallel work starts.
        pass


def enforce_single_thread_process() -> None:
    """Apply the search thread limit to the current process and its children."""
    os.environ.update(single_thread_environment())
    configure_torch_threads_from_env()


def _force_zero_dataloader_workers(overrides: Sequence[str]) -> list[str]:
    """Ensure a training subprocess cannot spawn additional dataloader workers."""
    worker_key = 'dataset.dataloader_params.num_workers'
    persistent_key = 'dataset.dataloader_params.persistent_workers'
    filtered = [
        override
        for override in overrides
        if not override.startswith(f'{worker_key}=')
        and not override.startswith(f'{persistent_key}=')
    ]
    return [*filtered, f'{worker_key}=0', f'{persistent_key}=false']


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp-{os.getpid()}')
    temporary.write_text(content)
    os.replace(temporary, path)


def run_training(
    overrides: Sequence[str],
    timeout: int | None = None,
    gpu_id: int | str | None = None,
    n_threads: int | None = 1,
    log_path: str | Path | None = None,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Run one training process with strict resource isolation."""
    effective_overrides = _force_zero_dataloader_workers(overrides)
    # Use the launcher's interpreter instead of relying on an activated shell
    # or an environment-specific console-script path.
    cmd = [sys.executable, '-m', 'ogbench.run', *effective_overrides]

    env = single_thread_environment()
    if gpu_id is not None and torch.cuda.is_available():
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # One thread is an invariant for search jobs. Retain the argument for API
    # compatibility with the existing grid launcher.
    if n_threads not in (None, 1):
        raise ValueError(f'Hyperparameter-search jobs require n_threads=1, got {n_threads}')

    try:
        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )

        if log_path is not None:
            log_content = (
                f'COMMAND: {" ".join(cmd)}\n'
                f'RETURN_CODE: {result.returncode}\n\n'
                f'STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n'
            )
            _atomic_write_text(Path(log_path), log_content)

        if result.returncode == 0:
            metrics = parse_objective_payload(result.stdout) or {}
            metrics.update(parse_metrics_payload(result.stdout))
            return True, None, metrics or None

        stderr_truncated = (
            result.stderr[:500] + '...' + result.stderr[-500:]
            if len(result.stderr) > 1000
            else result.stderr
        )
        error_msg = f'Return code {result.returncode}\nSTDERR: {stderr_truncated}'
        return False, error_msg, None

    except subprocess.TimeoutExpired as error:
        if log_path is not None:
            stdout = error.stdout or ''
            stderr = error.stderr or ''
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors='replace')
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors='replace')
            _atomic_write_text(
                Path(log_path),
                f'COMMAND: {" ".join(cmd)}\nTIMEOUT: {timeout}s\n\n'
                f'STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n',
            )
        return False, f'Timeout after {timeout}s', None
    except Exception as error:
        if log_path is not None:
            _atomic_write_text(
                Path(log_path),
                f'COMMAND: {" ".join(cmd)}\nEXCEPTION: {error!r}\n',
            )
        return False, str(error), None


def visible_gpu_devices(
    requested: Sequence[int] | None,
    device_count: int | None = None,
    cuda_visible_devices: str | None = None,
) -> list[GpuDevice]:
    """Resolve selected logical GPU IDs to child-process visibility tokens."""
    count = torch.cuda.device_count() if device_count is None else device_count
    visible = (
        os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_visible_devices is None
        else cuda_visible_devices
    )
    tokens = [token.strip() for token in visible.split(',')] if visible else []
    if tokens and len(tokens) < count:
        count = len(tokens)
    if not tokens:
        tokens = [str(index) for index in range(count)]

    logical_ids = list(range(count)) if requested is None else list(requested)
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError(f'Duplicate GPU IDs are not allowed: {logical_ids}')
    invalid = [gpu_id for gpu_id in logical_ids if gpu_id < 0 or gpu_id >= count]
    if invalid:
        raise ValueError(
            f'GPU IDs {invalid} are not visible; valid launcher-visible IDs are '
            f'{list(range(count))}'
        )
    return [
        GpuDevice(logical_id=gpu_id, visibility_token=tokens[gpu_id]) for gpu_id in logical_ids
    ]


def populate_gpu_queue(queue: Any, devices: Sequence[GpuDevice], jobs_per_gpu: int) -> None:
    """Populate a shared queue with the requested number of slots per GPU."""
    if jobs_per_gpu < 1:
        raise ValueError(f'jobs_per_gpu must be positive, got {jobs_per_gpu}')
    for device in devices:
        for _ in range(jobs_per_gpu):
            queue.put(device)

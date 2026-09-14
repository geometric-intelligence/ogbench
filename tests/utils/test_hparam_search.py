"""Tests for shared hyperparameter-search execution helpers."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
from unittest.mock import Mock

import pytest

from ogbench.utils.hparam_search import (
    OBJECTIVE_PAYLOAD_PREFIX,
    THREAD_ENV_VARS,
    configure_torch_threads_from_env,
    objective_payload,
    parse_objective_payload,
    populate_gpu_queue,
    run_training,
    single_thread_environment,
    visible_gpu_devices,
)


def test_objective_payload_round_trip_ignores_unrelated_json() -> None:
    line = objective_payload('best_val/f1_macro', 0.75)
    parsed = parse_objective_payload(f'{{"noise": true}}\n{line}\n')

    assert line.startswith(OBJECTIVE_PAYLOAD_PREFIX)
    assert parsed == {
        'optimized_metric': 'best_val/f1_macro',
        'objective': 0.75,
        'best_val/f1_macro': 0.75,
    }


def test_single_thread_environment_overrides_existing_values() -> None:
    env = single_thread_environment({'OMP_NUM_THREADS': '12', 'OTHER': 'kept'})

    assert env['OTHER'] == 'kept'
    assert env['OGBENCH_NUM_THREADS'] == '1'
    assert all(env[name] == '1' for name in THREAD_ENV_VARS)


def test_run_training_enforces_thread_and_dataloader_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=objective_payload('best_val/f1_macro', 0.5),
        stderr='',
    )
    run = Mock(return_value=completed)
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)

    success, error, metrics = run_training(
        [
            'dataset.dataloader_params.num_workers=8',
            'dataset.dataloader_params.persistent_workers=true',
        ],
        gpu_id='GPU-token',
    )

    assert success is True
    assert error is None
    assert metrics is not None and metrics['objective'] == 0.5
    command = run.call_args.args[0]
    assert command[:3] == [sys.executable, '-m', 'ogbench.run']
    assert 'dataset.dataloader_params.num_workers=0' in command
    assert 'dataset.dataloader_params.persistent_workers=false' in command
    env = run.call_args.kwargs['env']
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-token'
    assert all(env[name] == '1' for name in THREAD_ENV_VARS)


def test_run_training_rejects_more_than_one_thread() -> None:
    with pytest.raises(ValueError, match='require n_threads=1'):
        run_training([], n_threads=2)


def test_configure_torch_threads_uses_launcher_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_threads = Mock()
    set_interop_threads = Mock()
    monkeypatch.setenv('OGBENCH_NUM_THREADS', '1')
    monkeypatch.setattr('torch.set_num_threads', set_threads)
    monkeypatch.setattr('torch.set_num_interop_threads', set_interop_threads)

    configure_torch_threads_from_env()

    set_threads.assert_called_once_with(1)
    set_interop_threads.assert_called_once_with(1)


def test_visible_gpu_devices_maps_launcher_logical_ids_to_tokens() -> None:
    devices = visible_gpu_devices(
        requested=[1, 0],
        device_count=2,
        cuda_visible_devices='GPU-a,GPU-b',
    )

    assert [(device.logical_id, device.visibility_token) for device in devices] == [
        (1, 'GPU-b'),
        (0, 'GPU-a'),
    ]


def test_gpu_queue_contains_requested_slots_only() -> None:
    devices = visible_gpu_devices(
        requested=[1],
        device_count=2,
        cuda_visible_devices='GPU-a,GPU-b',
    )
    slots: queue.Queue = queue.Queue()

    populate_gpu_queue(slots, devices, jobs_per_gpu=2)

    queued = [slots.get_nowait(), slots.get_nowait()]
    assert [(device.logical_id, device.visibility_token) for device in queued] == [
        (1, 'GPU-b'),
        (1, 'GPU-b'),
    ]
    assert slots.empty()


@pytest.mark.parametrize('requested', [[0, 0], [2]])
def test_visible_gpu_devices_rejects_invalid_selection(requested: list[int]) -> None:
    with pytest.raises(ValueError):
        visible_gpu_devices(
            requested=requested,
            device_count=2,
            cuda_visible_devices='0,1',
        )


def test_parser_skips_malformed_prefixed_payload() -> None:
    malformed = OBJECTIVE_PAYLOAD_PREFIX + json.dumps({'metric_name': 'metric'})
    assert parse_objective_payload(malformed) is None

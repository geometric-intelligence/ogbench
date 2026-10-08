"""Tests for the launcher's follow mode (append-only manifest fed by the campaign coordinator)."""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import BrokenExecutor, Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts import optuna_search
from scripts.optuna_search import (
    EXPERIMENT,
    SELECTION_METHOD,
    FollowLauncher,
    OptunaSearchConfig,
    build_outer_cells,
    read_follow_manifest,
)

CONFIG_PATH = Path('configs/hparams_search/optuna_smoke_test.yaml')


def _config(tmp_path: Path, prefix: str = 'smoke') -> OptunaSearchConfig:
    config = OptunaSearchConfig.from_yaml(CONFIG_PATH)
    config.output_dir = tmp_path / prefix
    config.storage = f'sqlite:///{tmp_path / prefix / "studies.db"}'
    config.study_name_prefix = prefix
    config.ablations[EXPERIMENT] = ['no_readout', 'omics_readout']
    config.ablations[SELECTION_METHOD] = ['variance', 'random']
    return config


def _thread_pool(workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=workers)


class _Recorder:
    """Fake study runner and cache builder that record what the launcher asked for."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.studies: list[str] = []
        self.candidates: dict[str, dict | None] = {}
        self.warmed: list[tuple] = []
        self.exports: list[list[str]] = []
        self.on_study = None

    def run_study(self, config, cell, ledger_path, gpu_queue, retry_failed, candidate, policy):
        with self.lock:
            self.studies.append(cell.study_name)
            self.candidates[cell.study_name] = candidate
        if self.on_study is not None:
            self.on_study(cell.study_name)
        return [{'state': 'COMPLETE'}, {'state': 'COMPLETE'}, {'state': 'FAIL'}]

    @staticmethod
    def cache_configs(config, cells):
        (cell,) = cells
        return [
            OmegaConf.create({'method': cell.values[SELECTION_METHOD], 'fold': fold})
            for fold in range(2)
        ]

    def warmup(self, loader_configs, training_seed):
        with self.lock:
            self.warmed.extend((loader.method, loader.fold) for loader in loader_configs)
        return len(loader_configs)

    def export(self, config, cells, jobs_per_gpu):
        self.exports.append([cell.study_name for cell in cells])


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(optuna_search, '_run_study', recorder.run_study)
    monkeypatch.setattr(optuna_search, '_cache_configs', recorder.cache_configs)
    monkeypatch.setattr(optuna_search, '_follow_warmup', recorder.warmup)
    monkeypatch.setattr(optuna_search, '_export_existing', recorder.export)
    return recorder


def _launcher(configs, manifest: Path, **kwargs) -> FollowLauncher:
    kwargs.setdefault('executor_factory', _thread_pool)
    kwargs.setdefault('poll_seconds', 0.01)
    kwargs.setdefault('requeue_delay', 0.0)
    return FollowLauncher(configs, manifest, **kwargs)


def _events(manifest: Path) -> list[dict]:
    events_path = manifest.with_name(manifest.name + '.events.jsonl')
    return [json.loads(line) for line in events_path.read_text().splitlines()]


def test_manifest_reader_ignores_partial_lines_duplicates_and_text_after_end(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / 'manifest.txt'
    assert read_follow_manifest(manifest) == ([], False)

    manifest.write_text('\n'.join(['# header', 'first', '', 'second', 'first', 'partial']))
    assert read_follow_manifest(manifest) == (['first', 'second'], False)

    manifest.write_text('\n'.join(['first', 'second', '#END', 'late', '']))
    assert read_follow_manifest(manifest) == (['first', 'second'], True)


def test_follow_runs_appended_studies_and_warms_each_cache_once(
    tmp_path: Path, recorder: _Recorder
) -> None:
    config = _config(tmp_path)
    names = [cell.study_name for cell in build_outer_cells(config)]
    assert len(names) == 4
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text(f'{names[0]}\n{names[1]}\n')

    def append_rest(study_name: str) -> None:
        if study_name == names[0]:
            with manifest.open('a') as handle:
                handle.write(f'{names[2]}\n{names[3]}\n#END\n')

    recorder.on_study = append_rest
    result = _launcher([config], manifest, workers=2).run()

    assert sorted(result['completed']) == sorted(names)
    assert result['failed'] == {}
    assert sorted(recorder.studies) == sorted(names)
    assert sorted(recorder.warmed) == [
        ('random', 0),
        ('random', 1),
        ('variance', 0),
        ('variance', 1),
    ]
    assert recorder.exports == [sorted(names)]

    state = json.loads(manifest.with_name('manifest.txt.state.json').read_text())
    assert state['finished'] is True
    assert state['end_seen'] is True
    assert state['completed'] == 4
    assert state['pending'] == state['ready'] == state['warming'] == []
    events = _events(manifest)
    completed = [event for event in events if event['event'] == 'completed']
    assert {event['study'] for event in completed} == set(names)
    assert all(event['complete'] == 2 and event['failed'] == 1 for event in completed)


def test_single_worker_starts_studies_in_manifest_order(
    tmp_path: Path, recorder: _Recorder
) -> None:
    config = _config(tmp_path)
    names = [cell.study_name for cell in build_outer_cells(config)]
    order = [names[3], names[0], names[2], names[1]]
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text('\n'.join([*order, '#END']) + '\n')

    _launcher([config], manifest, workers=1, skip_warmup=True).run()

    assert recorder.studies == order
    assert recorder.warmed == []


def test_restarted_launcher_skips_studies_completed_by_an_earlier_launcher(
    tmp_path: Path, recorder: _Recorder
) -> None:
    config = _config(tmp_path)
    names = [cell.study_name for cell in build_outer_cells(config)]
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text('\n'.join([*names[:2], '#END']) + '\n')
    _launcher([config], manifest, workers=1, skip_warmup=True).run()
    assert recorder.studies == names[:2]

    manifest.write_text('\n'.join([*names, '#END']) + '\n')
    result = _launcher([config], manifest, workers=1, skip_warmup=True).run()

    assert recorder.studies == names
    assert sorted(result['completed']) == sorted(names)


def test_unknown_studies_and_missing_candidates_fail_without_stopping(
    tmp_path: Path, recorder: _Recorder
) -> None:
    main = _config(tmp_path, 'main')
    followup = replace(_config(tmp_path, 'followup'), candidates_required=True)
    main_name = build_outer_cells(main)[0].study_name
    with_candidate, without_candidate = (cell for cell in build_outer_cells(followup)[:2])
    candidate = {
        'sampled_params': {
            'optimizer.parameters.lr': 0.0005,
            'model.backbone.dropout': 0.1,
            'model.feature_encoder.out_channels': 32,
            'model.backbone.num_layers': 2,
        },
    }
    canonical = optuna_search.validate_sampled_parameters(
        followup, with_candidate.model, candidate['sampled_params']
    )
    candidate['param_hash'] = optuna_search._stable_hash(canonical)
    candidates = tmp_path / 'candidates.json'
    candidates.write_text(json.dumps({with_candidate.study_name: candidate}))
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text(
        '\n'.join(
            [
                'not-a-study',
                main_name,
                with_candidate.study_name,
                without_candidate.study_name,
                '#END',
            ]
        )
        + '\n'
    )

    result = _launcher(
        [main, followup], manifest, candidates_path=candidates, workers=2, skip_warmup=True
    ).run()

    assert sorted(result['completed']) == sorted([main_name, with_candidate.study_name])
    assert set(result['failed']) == {'not-a-study', without_candidate.study_name}
    assert 'No fixed candidate' in result['failed'][without_candidate.study_name]
    assert recorder.candidates[main_name] is None
    assert recorder.candidates[with_candidate.study_name] == candidate
    assert {tuple(sorted(names)) for names in recorder.exports} == {
        (main_name,),
        tuple(sorted([with_candidate.study_name, without_candidate.study_name])),
    }


def _worker_threads() -> tuple[str | None, int]:
    import os

    import torch

    return os.environ.get('OMP_NUM_THREADS'), torch.get_num_threads()


def test_process_executor_runs_launcher_tasks_single_threaded() -> None:
    executor = optuna_search._process_executor(1)
    try:
        assert executor.submit(optuna_search._follow_warmup, [], 42).result(timeout=120) == 0
        assert executor.submit(_worker_threads).result(timeout=120) == ('1', 1)
    finally:
        executor.shutdown(wait=True)


def test_configs_with_overlapping_study_names_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='more than one config'):
        FollowLauncher._index([_config(tmp_path), _config(tmp_path)])


class _CrashingPool:
    """Executor whose first ``crashes`` submissions fail as if the worker process died."""

    def __init__(self, inner: ThreadPoolExecutor, budget: list[int]) -> None:
        self.inner = inner
        self.budget = budget

    def submit(self, function, *args, **kwargs):
        if self.budget[0] > 0:
            self.budget[0] -= 1
            future: Future = Future()
            future.set_exception(BrokenExecutor('worker died'))
            return future
        return self.inner.submit(function, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        self.inner.shutdown(wait=wait)


@pytest.mark.parametrize(('crashes', 'completes'), [(1, True), (3, False)])
def test_crashed_worker_requeues_the_study_until_the_crash_limit(
    tmp_path: Path, recorder: _Recorder, crashes: int, completes: bool
) -> None:
    config = _config(tmp_path)
    name = build_outer_cells(config)[0].study_name
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text(f'{name}\n#END\n')
    budget = [crashes]
    pools: list[_CrashingPool] = []

    def factory(workers: int) -> _CrashingPool:
        pools.append(_CrashingPool(ThreadPoolExecutor(max_workers=workers), budget))
        return pools[-1]

    result = _launcher(
        [config], manifest, workers=1, skip_warmup=True, executor_factory=factory
    ).run()

    requeues = [event for event in _events(manifest) if event['event'] == 'requeued']
    if completes:
        assert result['completed'] == [name]
        assert len(requeues) == 1
        assert len(pools) == 2
    else:
        assert result['completed'] == []
        assert 'crashed 3 times' in result['failed'][name]
        assert len(requeues) == 2


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, 'argv', ['optuna_search.py', *argv])
    optuna_search.main()


def test_cli_requires_follow_mode_for_several_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _main(monkeypatch, '--config', str(CONFIG_PATH), str(CONFIG_PATH), '--dry-run')


def test_cli_rejects_static_filters_in_follow_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        _main(
            monkeypatch,
            '--config',
            str(CONFIG_PATH),
            '--follow-manifest',
            str(tmp_path / 'manifest.txt'),
            '--models',
            'gcn',
        )


def test_cli_gives_each_config_its_own_output_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    second = tmp_path / 'second.yaml'
    second.write_text(CONFIG_PATH.read_text().replace('prefix: smoke', 'prefix: other'))
    captured = {}

    def fake_follow(configs, manifest_path, **kwargs):
        captured['configs'] = configs

    monkeypatch.setattr(optuna_search, 'run_follow', fake_follow)
    _main(
        monkeypatch,
        '--config',
        str(CONFIG_PATH),
        str(second),
        '--follow-manifest',
        str(tmp_path / 'manifest.txt'),
        '--output-dir',
        str(tmp_path / 'out'),
    )

    first_config, second_config = captured['configs']
    assert first_config.output_dir == (tmp_path / 'out' / 'optuna_smoke_test').resolve()
    assert second_config.output_dir == (tmp_path / 'out' / 'second').resolve()
    assert (
        second_config.storage == f'sqlite:///{(tmp_path / "out" / "second").resolve()}/studies.db'
    )

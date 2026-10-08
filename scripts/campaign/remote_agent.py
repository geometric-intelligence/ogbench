"""Server-side agent of the k-fold campaign coordinator.

Each subcommand reads one JSON object on stdin and prints one JSON object as the last line of
stdout, so the coordinator drives Parka (locally) and Frank and Hall (over ssh) the same way.
Commands that change state are idempotent: the coordinator may repeat any call whose reply it
lost.

Run from the pinned checkout: ``python -m scripts.campaign.remote_agent <command>``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import inspect
import json
import os
import shutil
import signal
import socket
import subprocess  # nosec B404
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MANIFEST = 'manifest.txt'
CANDIDATES = 'candidates.json'
SUPERVISOR_PID = 'supervisor.pid'
SUPERVISOR_SCRIPT = REPO_ROOT / 'scripts' / 'campaign' / 'server_supervisor.sh'
FOLD_CACHE_MARKER = 'split_k-fold'
EXPORTED_TABLES = ('trials.csv', 'best_trials.csv', 'fold_attempts.csv', 'failures.csv')
EVENTS_CHUNK_BYTES = 8 << 20


def manifest_path(run_root: Path) -> Path:
    return run_root / MANIFEST


def state_path(run_root: Path) -> Path:
    return run_root / f'{MANIFEST}.state.json'


def events_path(run_root: Path) -> Path:
    return run_root / f'{MANIFEST}.events.jsonl'


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _process_matches(pid: Any, marker: str) -> bool:
    """Whether ``pid`` is alive and its command line contains ``marker`` (guards PID reuse)."""
    try:
        cmdline = Path(f'/proc/{int(pid)}/cmdline').read_bytes()
    except (OSError, TypeError, ValueError):
        return False
    return marker.encode() in cmdline


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def read_events(path: Path, offset: int) -> tuple[list[dict[str, Any]], int, bool]:
    """Return complete event lines after ``offset``, the next offset, and whether it was reset."""
    if not path.exists():
        return [], 0, offset > 0
    reset = path.stat().st_size < offset
    start = 0 if reset else offset
    with path.open('rb') as handle:
        handle.seek(start)
        chunk = handle.read(EVENTS_CHUNK_BYTES)
    end = chunk.rfind(b'\n') + 1
    events = []
    for line in chunk[:end].splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events, start + end, reset


def gpu_status() -> list[dict[str, int]]:
    query = 'index,memory.used,memory.total,utilization.gpu'
    try:
        output = subprocess.run(  # nosec B603 B607
            ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 4 and all(field.isdigit() for field in fields):
            index, used, total, utilization = map(int, fields)
            rows.append(
                {'index': index, 'used_mib': used, 'total_mib': total, 'util': utilization}
            )
    return rows


def _free_gib(path: Path) -> float | None:
    try:
        return round(shutil.disk_usage(path).free / 2**30, 1)
    except OSError:
        return None


def cmd_status(payload: dict[str, Any]) -> dict[str, Any]:
    from scripts.optuna_search import read_follow_manifest

    run_root = Path(payload['run_root'])
    names, end = read_follow_manifest(manifest_path(run_root))
    state = _read_json(state_path(run_root))
    events, offset, reset = read_events(
        events_path(run_root), int(payload.get('events_offset', 0))
    )
    launcher_alive = (
        isinstance(state, dict)
        and not state.get('finished')
        and state.get('host') == socket.gethostname()
        and _process_matches(state.get('pid'), 'optuna_search')
    )
    supervisor_pid = _read_pid(run_root / SUPERVISOR_PID)
    return {
        'host': socket.gethostname(),
        'time': time.time(),
        'manifest': {'studies': len(names), 'end': end},
        'state': state,
        'launcher_alive': launcher_alive,
        'supervisor_alive': _process_matches(supervisor_pid, 'server_supervisor'),
        'events': events,
        'events_offset': offset,
        'events_reset': reset,
        'disk_free_gib': {
            'data_root': _free_gib(Path(payload['data_root'])),
            'run_root': _free_gib(run_root),
        },
        'gpus': gpu_status(),
        'load': list(os.getloadavg()),
    }


def append_manifest(path: Path, studies: Sequence[str], *, end: bool = False) -> dict[str, Any]:
    """Append new study names (and optionally ``#END``) in one write; repeats are skipped."""
    from scripts.optuna_search import END_SENTINEL, read_follow_manifest

    with _locked(path.with_name(path.name + '.lock')):
        if path.exists():
            content = path.read_bytes()
            if content and not content.endswith(b'\n'):
                # A torn last line was never read by the launcher; drop it.
                with path.open('r+b') as handle:
                    handle.truncate(content.rfind(b'\n') + 1)
        names, ended = read_follow_manifest(path)
        known = set(names)
        new = [name for name in dict.fromkeys(studies) if name not in known]
        if ended and new:
            raise ValueError(f'{path} already ends with {END_SENTINEL}; cannot add {new[:3]}')
        lines = [*new, *([END_SENTINEL] if end and not ended else [])]
        if lines:
            with path.open('a') as handle:
                handle.write('\n'.join(lines) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
    return {'appended': new, 'studies': len(names) + len(new), 'end': ended or end}


def cmd_append(payload: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(payload['run_root'])
    run_root.mkdir(parents=True, exist_ok=True)
    return append_manifest(
        manifest_path(run_root), payload.get('studies', []), end=bool(payload.get('end'))
    )


def cmd_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    from scripts.optuna_search import read_follow_manifest

    names, end = read_follow_manifest(manifest_path(Path(payload['run_root'])))
    return {'studies': names, 'end': end}


def _load_campaign_configs(payload: dict[str, Any]) -> list[Any]:
    """Load main and follow-up configs with the same paths the server's launcher uses."""
    from scripts.optuna_search import load_follow_configs

    configs = [REPO_ROOT / path for path in payload['configs']]
    return load_follow_configs(
        configs, output_dir=payload['run_root'], root_dir=payload['data_root']
    )


def cell_key(cell: Any) -> str:
    return json.dumps([cell.model, cell.dataset, cell.values], sort_keys=True, default=str)


def best_candidate(study: Any, direction: str) -> dict[str, Any] | None:
    """The best COMPLETE trial of a main study as a fixed follow-up candidate."""
    from optuna.trial import TrialState

    complete = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
        and 'sampled_params' in trial.user_attrs
    ]
    if not complete:
        return None
    pick = max if direction == 'maximize' else min
    best = pick(complete, key=lambda trial: (trial.value, -trial.number))
    return {
        'sampled_params': best.user_attrs['sampled_params'],
        'param_hash': best.user_attrs['param_hash'],
        'source_study': study.study_name,
        'source_rule': f'best_complete_{direction}',
        'source_trial_numbers': [best.number],
        'source_mean': best.value,
        'source_folds': len(best.user_attrs.get('fold_scores', {})),
    }


def merge_candidates(path: Path, new: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Add candidates atomically; return studies whose existing candidate differs (kept)."""
    conflicts = {}
    with _locked(path.with_name(path.name + '.lock')):
        current = _read_json(path) if path.exists() else {}
        if not isinstance(current, dict):
            raise ValueError(f'Candidate file is not a JSON object: {path}')
        for name, candidate in new.items():
            existing = current.get(name)
            if existing is None:
                current[name] = candidate
            elif existing['param_hash'] != candidate['param_hash']:
                conflicts[name] = existing['param_hash']
        _atomic_write_json(path, current)
    return conflicts


def cmd_followup(payload: dict[str, Any]) -> dict[str, Any]:
    """Queue the follow-up of finished main studies with their best main configuration."""
    import optuna

    from scripts.optuna_search import (
        _validate_candidates,
        build_outer_cells,
        read_follow_manifest,
    )

    main, followup = _load_campaign_configs(payload)
    run_root = Path(payload['run_root'])
    main_cells = {cell.study_name: cell for cell in build_outer_cells(main)}
    follow_cells = {cell_key(cell): cell for cell in build_outer_cells(followup)}
    listed, _ = read_follow_manifest(manifest_path(run_root))
    listed_set = set(listed)

    candidates: dict[str, dict[str, Any]] = {}
    targets: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for name in payload['studies']:
        cell = main_cells.get(name)
        if cell is None:
            skipped[name] = 'not a main study'
            continue
        target = follow_cells.get(cell_key(cell))
        if target is None:
            skipped[name] = 'no follow-up cell'
            continue
        if target.study_name in listed_set:
            skipped[name] = 'already queued'
            continue
        try:
            study = optuna.load_study(study_name=name, storage=main.storage)
        except KeyError:
            skipped[name] = 'main study not in storage'
            continue
        candidate = best_candidate(study, main.direction)
        if candidate is None:
            skipped[name] = 'no complete trial'
            continue
        try:
            _validate_candidates(followup, [target], {target.study_name: candidate})
        except ValueError as error:
            skipped[name] = f'invalid candidate: {error}'
            continue
        candidates[target.study_name] = candidate
        targets[name] = target.study_name

    conflicts = merge_candidates(run_root / CANDIDATES, candidates)
    for name, target in list(targets.items()):
        if target in conflicts:
            skipped[name] = f'follow-up candidate already set to {conflicts[target]}'
            del targets[name]
    appended = append_manifest(manifest_path(run_root), list(targets.values()))['appended']
    return {'followups': targets, 'appended': appended, 'skipped': skipped}


def fold_cache_dir(parameters: dict[str, Any]) -> Path:
    """The directory ``HFOmicsDataset`` caches one fold in, from resolved loader parameters."""
    from ogbench.data.datasets import HFOmicsDataset
    from ogbench.data.utils.split_utils import build_omics_cache_relative_name

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(HFOmicsDataset.__init__).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    params = {**defaults, **parameters}
    grouping = params['grouping']
    relative = build_omics_cache_relative_name(
        data_name=params['data_name'],
        adjacency_threshold=params['adjacency_threshold'],
        adjacency_method=params['adjacency_method'],
        method=params['method'],
        node_sample_ratio=params['node_sample_ratio'],
        train_split=(params['train_val_test_split'] or [0.7, 0.15, 0.15])[0],
        split_type=params['split_type'],
        k=int(params['k']),
        fold=int(params['fold']),
        corrections=list(params['corrections'] or []),
        grouping=grouping if grouping not in (None, 'null', '') else None,
        adjacency_target_connectivity=params['adjacency_target_connectivity'],
        revision=params['revision'],
        imputation_method=params['imputation_method'],
        wgcna_binarization=params['wgcna_binarization'],
    )
    return Path(params['data_dir']) / relative


def _checked_cache_dir(path: Path, data_root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(data_root.resolve()):
        raise ValueError(f'Refusing to remove {resolved}: outside {data_root}')
    if 'string_cache' in resolved.parts:
        raise ValueError(f'Refusing to remove the shared STRING cache: {resolved}')
    if not any(part.startswith(FOLD_CACHE_MARKER) for part in resolved.parts):
        raise ValueError(f'Refusing to remove {resolved}: not a k-fold cache')
    return resolved


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())


def cmd_gc(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove the fold caches of the given studies (all folds) on this server."""
    from omegaconf import OmegaConf

    from scripts.optuna_search import _cache_configs, build_outer_cells

    main = _load_campaign_configs(payload)[0]
    data_root = Path(payload['data_root'])
    cells = {cell.study_name: cell for cell in build_outer_cells(main)}
    unknown = [name for name in payload['studies'] if name not in cells]
    if unknown:
        raise ValueError(f'Unknown main studies: {unknown[:3]}')
    selected = [cells[name] for name in payload['studies']]
    removed, missing, freed = [], [], 0
    for loader in _cache_configs(main, selected):
        parameters = OmegaConf.to_container(loader.parameters, resolve=True)
        path = _checked_cache_dir(fold_cache_dir(parameters), data_root)
        if not path.exists():
            missing.append(str(path))
            continue
        freed += _tree_bytes(path)
        if not payload.get('dry_run'):
            shutil.rmtree(path)
        removed.append(str(path))
    return {'removed': removed, 'missing': missing, 'freed_gib': round(freed / 2**30, 2)}


def cmd_export(payload: dict[str, Any]) -> dict[str, Any]:
    """Export CSVs for every listed study and snapshot both SQLite databases."""
    from scripts.optuna_rebalance import backup_sqlite
    from scripts.optuna_search import _export_existing, build_outer_cells, read_follow_manifest

    run_root = Path(payload['run_root'])
    snapshot = run_root / 'export' / payload['stamp']
    listed, _ = read_follow_manifest(manifest_path(run_root))
    listed_set = set(listed)
    files = []
    for config in _load_campaign_configs(payload):
        cells = [cell for cell in build_outer_cells(config) if cell.study_name in listed_set]
        target = snapshot / config.output_dir.name
        target.mkdir(parents=True, exist_ok=True)
        if cells and (config.output_dir / 'studies.db').exists():
            _export_existing(config, cells, int(payload.get('jobs_per_gpu', 1)))
            for table in EXPORTED_TABLES:
                if (config.output_dir / table).exists():
                    shutil.copy2(config.output_dir / table, target / table)
                    files.append(str(target / table))
        for database in ('studies.db', 'run_ledger.sqlite3'):
            if (config.output_dir / database).exists():
                backup_sqlite(config.output_dir / database, target / database)
                files.append(str(target / database))
    for source in (
        manifest_path(run_root),
        state_path(run_root),
        events_path(run_root),
        run_root / CANDIDATES,
    ):
        if source.exists():
            shutil.copy2(source, snapshot / source.name)
            files.append(str(snapshot / source.name))
    return {'snapshot': str(snapshot), 'files': files}


def cmd_fold_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """Per-study fold durations and peak memory from this server's run ledgers."""
    import sqlite3

    since = payload.get('since', '')
    stats: dict[str, dict[str, Any]] = {}
    for config in _load_campaign_configs(payload):
        ledger = config.output_dir / 'run_ledger.sqlite3'
        if not ledger.exists():
            continue
        with sqlite3.connect(f'file:{ledger}?mode=ro', uri=True, timeout=60) as connection:
            rows = connection.execute(
                'SELECT study_name, status, elapsed_time, time_budget_hit, peak_memory_mib '
                'FROM fold_attempts WHERE created_at >= ?',
                (since,),
            ).fetchall()
        for study, status, elapsed, budget_hit, peak in rows:
            entry = stats.setdefault(
                study, {'seconds': [], 'failed': 0, 'budget_hits': 0, 'peak_mib': 0.0}
            )
            if status == 'success':
                entry['seconds'].append(elapsed)
            else:
                entry['failed'] += 1
            entry['budget_hits'] += int(budget_hit or 0)
            entry['peak_mib'] = max(entry['peak_mib'], float(peak or 0.0))
    return {'studies': stats}


def cmd_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Start the server supervisor detached in its own session, unless it already runs."""
    run_root = Path(payload['run_root'])
    run_root.mkdir(parents=True, exist_ok=True)
    pid_file = run_root / SUPERVISOR_PID
    current = _read_pid(pid_file)
    if _process_matches(current, 'server_supervisor'):
        return {'started': False, 'pid': current}
    env = {
        **os.environ,
        **{key: str(value) for key, value in payload.get('env', {}).items()},
        'REPO': str(REPO_ROOT),
        'PYTHON': payload['python'],
        'RUN_ROOT': str(run_root),
        'DATA_ROOT': payload['data_root'],
        'CONFIGS': ' '.join(payload['configs']),
        'GPUS': ' '.join(str(gpu) for gpu in payload['gpus']),
        'JOBS_PER_GPU': str(payload['jobs_per_gpu']),
        'WARMUP_JOBS': str(payload['warmup_jobs']),
        'MIN_FREE_GPU_MIB': str(payload['min_free_gpu_mib']),
        'OOM_RETRIES': str(payload['oom_retries']),
        'OOM_MIN_FREE_GPU_MIB': str(payload['oom_min_free_gpu_mib']),
        'PYTHONUNBUFFERED': '1',
    }
    for key in ('WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_DATA_DIR', 'HF_HOME', 'TMPDIR'):
        if key in env:
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    with (run_root / 'supervisor.log').open('a') as log:
        process = subprocess.Popen(  # nosec B603 B607
            ['bash', str(SUPERVISOR_SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(f'{process.pid}\n')
    return {'started': True, 'pid': process.pid}


def cmd_stop(payload: dict[str, Any]) -> dict[str, Any]:
    """Stop the supervisor's whole session: launcher, workers and training jobs."""
    pid = _read_pid(Path(payload['run_root']) / SUPERVISOR_PID)
    if not _process_matches(pid, 'server_supervisor'):
        return {'stopped': False, 'pid': pid}
    os.killpg(pid, signal.SIGTERM)
    return {'stopped': True, 'pid': pid}


COMMANDS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    'status': cmd_status,
    'append': cmd_append,
    'manifest': cmd_manifest,
    'followup': cmd_followup,
    'gc': cmd_gc,
    'export': cmd_export,
    'fold-stats': cmd_fold_stats,
    'start': cmd_start,
    'stop': cmd_stop,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    payload = json.loads(sys.stdin.read() or '{}')
    reply_stream = sys.stdout
    try:
        # Libraries print progress; keep stdout for the single JSON reply.
        with contextlib.redirect_stdout(sys.stderr):
            reply = COMMANDS[args.command](payload)
    except Exception as error:
        reply_stream.write(json.dumps({'error': f'{type(error).__name__}: {error}'}) + '\n')
        return 1
    reply_stream.write(json.dumps(reply, default=str) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

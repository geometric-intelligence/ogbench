"""Calibration of the October k-fold campaign before the full launch.

The smoke run starts every server's supervisor on the same reduced studies
(``oct_kfold_smoke*.yaml``, under ``<campaign>_smoke/<server>``), so results can be compared
across servers; the throughput run replays one fold at several jobs per GPU::

    python -m scripts.campaign.calibrate smoke start
    python -m scripts.campaign.calibrate smoke status
    python -m scripts.campaign.calibrate smoke followup   # gene_identity candidates
    python -m scripts.campaign.calibrate smoke end        # supervisors exit when done
    python -m scripts.campaign.calibrate smoke report
    python -m scripts.campaign.calibrate throughput --server parka --study S --gpus 0 1 3 \
        --copies 1 2 3 --budget 600
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.campaign.coordinator import (  # noqa: E402
    Server,
    Settings,
    ShellAgent,
    supervisor_settings,
    utc_now,
)
from scripts.optuna_search import OptunaSearchConfig, build_outer_cells  # noqa: E402

SMOKE_CONFIGS = (
    'configs/hparams_search/oct_kfold_smoke.yaml',
    'configs/hparams_search/oct_kfold_smoke_gene_identity.yaml',
)
TEST_METRIC = 'test/f1_macro'


def smoke_server(server: Server) -> Server:
    """The server with its run root moved from ``<campaign>/<name>`` to ``<campaign>_smoke``."""
    campaign = Path(server.run_root).parent
    return dataclasses.replace(
        server, run_root=str(campaign.with_name(f'{campaign.name}_smoke') / server.name)
    )


def smoke_studies() -> list[str]:
    main = OptunaSearchConfig.from_yaml(REPO_ROOT / SMOKE_CONFIGS[0])
    return [cell.study_name for cell in build_outer_cells(main)]


def call(server: Server, command: str, timeout: float = 1800.0, **extra: Any) -> dict[str, Any]:
    payload = {
        'run_root': server.run_root,
        'data_root': server.data_root,
        'configs': list(SMOKE_CONFIGS),
        **extra,
    }
    return ShellAgent(server, timeout=timeout).call(command, payload)


def on_servers(servers: Sequence[Server], function: Any) -> dict[str, Any]:
    """Run ``function(server)`` on every server at once; errors become ``{'error': ...}``."""

    def guarded(server: Server) -> Any:
        try:
            return function(server)
        except Exception as error:
            return {'error': f'{type(error).__name__}: {error}'}

    with ThreadPoolExecutor(len(servers)) as pool:
        replies = list(pool.map(guarded, servers))
    return dict(zip([server.name for server in servers], replies, strict=True))


def smoke_status(server: Server) -> dict[str, Any]:
    reply = call(server, 'status')
    events = collections.Counter(event.get('event') for event in reply['events'])
    failed = {
        event['study']: str(event.get('error', ''))[-300:]
        for event in reply['events']
        if event.get('event') == 'failed'
    }
    state = reply['state'] or {}
    return {
        'supervisor_alive': reply['supervisor_alive'],
        'launcher_alive': reply['launcher_alive'],
        'manifest': reply['manifest'],
        'events': dict(events),
        'running': sorted(state.get('running', {})),
        'failed': failed,
        'gpu_used_mib': [gpu['used_mib'] for gpu in reply['gpus']],
        'disk_free_gib': reply['disk_free_gib'],
    }


def _short(study: str) -> str:
    return study.split('_', 1)[1].rsplit('_', 1)[0]


def _fmt(value: Any) -> str:
    return '-' if value is None else f'{value:.3f}'


def smoke_report(servers: Sequence[Server], studies: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """Per-fold results, required ledger fields, cross-server agreement and cache fingerprints."""
    rows = on_servers(servers, lambda server: call(server, 'ledger-rows')['rows'])
    prints = on_servers(
        servers, lambda server: call(server, 'fingerprint', studies=list(studies))['fingerprints']
    )
    lines = [f'# Smoke report ({utc_now()})', '']
    problems: list[str] = []
    lines += [
        '| server | study | status | s | epochs | budget hit | peak MiB | val obj | test F1 |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    by_study: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for name, server_rows in rows.items():
        if isinstance(server_rows, dict):
            problems.append(f'{name}: ledger unreadable: {server_rows["error"]}')
            continue
        for row in sorted(server_rows, key=lambda row: (row['study_name'], row['attempt'])):
            test = row['metrics'].get(TEST_METRIC)
            lines.append(
                f'| {name} | {_short(row["study_name"])} | {row["status"]} '
                f'| {row["elapsed_time"]:.0f} | {row["epochs_completed"]} '
                f'| {row["time_budget_hit"]} | {row["peak_memory_mib"] or 0:.0f} '
                f'| {_fmt(row["metric"])} | {_fmt(test)} |'
            )
            if row['status'] != 'success':
                problems.append(f'{name} {row["study_name"]}: {row["status"]} {row["error"]}')
                continue
            missing = [
                field
                for field, value in (
                    ('objective', row['metric']),
                    (TEST_METRIC, test),
                    ('epochs_completed', row['epochs_completed']),
                    ('time_budget_hit', row['time_budget_hit']),
                    ('peak_memory_mib', row['peak_memory_mib']),
                )
                if value is None
            ]
            if missing:
                problems.append(f'{name} {row["study_name"]}: missing {missing}')
            by_study[row['study_name']][name] = row

    lines += ['', '## Same study across servers', '']
    lines += [
        '| study | params equal | budget hits | val obj | test F1 |',
        '|---|---|---|---|---|',
    ]
    for study, per_server in sorted(by_study.items()):
        if len(per_server) < 2:
            continue
        params = {json.dumps(row['params'], sort_keys=True) for row in per_server.values()}
        values = {
            name: (row['metric'], row['metrics'].get(TEST_METRIC), row['time_budget_hit'])
            for name, row in per_server.items()
        }
        lines.append(
            f'| {_short(study)} | {len(params) == 1} '
            f'| {" ".join(str(value[2]) for value in values.values())} '
            f'| {" ".join(_fmt(value[0]) for value in values.values())} '
            f'| {" ".join(_fmt(value[1]) for value in values.values())} |'
        )
        if len(params) != 1:
            problems.append(f'{study}: sampled parameters differ across servers')

    lines += ['', '## Fold caches across servers', '']
    caches: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    for name, server_prints in prints.items():
        if 'error' in server_prints:
            problems.append(f'{name}: fingerprints failed: {server_prints["error"]}')
            continue
        for key, fingerprint in server_prints.items():
            caches[key][name] = fingerprint
    tests_by_dataset: dict[str, set[str]] = collections.defaultdict(set)
    for key, per_server in sorted(caches.items()):
        present = {name: value for name, value in per_server.items() if value is not None}
        distinct = {json.dumps(value, sort_keys=True) for value in present.values()}
        sample = next(iter(present.values()), None)
        if sample is not None:
            tests_by_dataset[key.split('/')[2]].update(value['test'] for value in present.values())
        lines.append(
            f'- `{key}`: {len(present)} servers, '
            + (
                'identical'
                if len(distinct) == 1
                else ('missing' if not distinct else f'**{len(distinct)} variants**')
            )
            + (
                f' (genes {sample["n_genes"]}, nonzero {sample["nonzero"]}, '
                f'test {sample["n_test"]})'
                if sample
                else ''
            )
        )
        if len(distinct) > 1:
            problems.append(f'{key}: fold cache differs across servers {present}')
    for dataset, tests in sorted(tests_by_dataset.items()):
        lines.append(f'- test IDs of {dataset}: {len(tests)} distinct hash(es)')
        if len(tests) != 1:
            problems.append(f'{dataset}: test sample IDs differ across caches or servers')

    lines += ['', '## Problems', ''] + ([f'- {problem}' for problem in problems] or ['- none'])
    return '\n'.join(lines) + '\n', {'problems': problems, 'rows': rows}


def throughput(
    server: Server,
    study: str,
    fold: int,
    gpus: Sequence[int],
    copies: Sequence[int],
    budget: float,
) -> dict[str, Any]:
    """Replay one smoke fold with ``copies[i]`` concurrent jobs on ``gpus[i]``, all at once."""
    rows = call(server, 'ledger-rows')['rows']
    source = next(
        row
        for row in rows
        if row['study_name'] == study and row['fold'] == fold and row['status'] == 'success'
    )
    tag = f'{utc_now().replace(":", "")}_{_short(study)}'

    def replay(pair: tuple[int, int]) -> dict[str, Any]:
        gpu, count = pair
        return call(
            server,
            'replay',
            timeout=budget + 3600,
            study=study,
            params=source['params'],
            fold=fold,
            trial_number=source['trial_number'],
            gpu=gpu,
            copies=count,
            budget=budget,
            tag=f'{tag}/gpu{gpu}_x{count}',
            env=server.env,
        )

    with ThreadPoolExecutor(len(gpus)) as pool:
        replies = list(pool.map(replay, zip(gpus, copies, strict=True)))
    summary = []
    for count, reply in zip(copies, replies, strict=True):
        epochs = [copy['metrics'].get('train/epochs_completed') for copy in reply['copies']]
        done = [value for value in epochs if value is not None]
        summary.append(
            {
                'gpu': reply['gpu'],
                'jobs': count,
                'succeeded': sum(copy['success'] for copy in reply['copies']),
                'epochs_per_job': done,
                'epochs_per_gpu_hour': round(sum(done) * 3600 / budget, 1),
                'budget_hits': [
                    copy['metrics'].get('train/time_budget_hit') for copy in reply['copies']
                ],
                'peak_mib': [
                    round(copy['metrics'].get('gpu/peak_memory_allocated_mib', 0))
                    for copy in reply['copies']
                ],
                'errors': [copy['error'] for copy in reply['copies'] if not copy['success']],
            }
        )
    return {'server': server.name, 'study': study, 'budget': budget, 'results': summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default=str(REPO_ROOT / 'scripts/campaign/servers.yaml'))
    commands = parser.add_subparsers(dest='command', required=True)
    smoke = commands.add_parser('smoke', help='Smoke run on every server')
    smoke.add_argument('action', choices=['start', 'status', 'followup', 'end', 'report', 'stop'])
    smoke.add_argument('--servers', nargs='+')
    rate = commands.add_parser('throughput', help='Replay one fold at several jobs per GPU')
    rate.add_argument('--server', required=True)
    rate.add_argument('--study', required=True)
    rate.add_argument('--fold', type=int, default=0)
    rate.add_argument('--gpus', type=int, nargs='+', required=True)
    rate.add_argument('--copies', type=int, nargs='+', required=True)
    rate.add_argument('--budget', type=float, default=600.0)
    args = parser.parse_args(argv)

    settings = Settings.from_yaml(args.settings)
    if args.command == 'throughput':
        if len(args.gpus) != len(args.copies):
            parser.error('--gpus and --copies need the same length')
        server = smoke_server(settings.servers[args.server])
        result = throughput(server, args.study, args.fold, args.gpus, args.copies, args.budget)
        print(json.dumps(result, indent=1))
        return 0

    servers = [smoke_server(settings.servers[name]) for name in args.servers or settings.servers]
    studies = smoke_studies()
    if args.action == 'start':

        def start(server: Server) -> Any:
            appended = call(server, 'append', studies=studies)
            return appended, call(server, 'start', **supervisor_settings(server))

        result = on_servers(servers, start)
    elif args.action == 'status':
        result = on_servers(servers, smoke_status)
    elif args.action == 'followup':
        result = on_servers(servers, lambda server: call(server, 'followup', studies=studies))
    elif args.action == 'end':
        result = on_servers(servers, lambda server: call(server, 'append', studies=[], end=True))
    elif args.action == 'stop':
        result = on_servers(servers, lambda server: call(server, 'stop'))
    else:
        markdown, result = smoke_report(servers, studies)
        campaign = Path(settings.coordinator_root).parent
        destination = campaign.with_name(f'{campaign.name}_smoke') / 'smoke_report.md'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown)
        print(markdown)
        print(f'written to {destination}')
        return 1 if result['problems'] else 0
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())

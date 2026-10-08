"""Tests for the k-fold campaign coordinator, its server agent and the server supervisor."""

from __future__ import annotations

import collections
import io
import json
import os
import re
import socket
import sqlite3
import stat
import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from omegaconf import OmegaConf
from optuna.trial import TrialState

from ogbench.data.loaders.graph import omics_datasets
from ogbench.data.loaders.graph.omics_datasets import OmicsDatasetLoader
from scripts.campaign import remote_agent
from scripts.campaign.coordinator import (
    Cell,
    Coordinator,
    CostModel,
    Rule,
    Server,
    Settings,
    ShellAgent,
    build_cost_table,
    plan_homes,
    preflight,
)
from scripts.campaign.simulate import SimulatedCluster
from scripts.optuna_search import (
    OptunaSearchConfig,
    RunLedger,
    _model_space,
    _stable_hash,
    build_outer_cells,
    validate_sampled_parameters,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = 'configs/hparams_search/oct_kfold_factorial.yaml'
FOLLOWUP_CONFIG = 'configs/hparams_search/oct_kfold_gene_identity.yaml'


# --------------------------------------------------------------------------- fixtures


def _server(
    name: str,
    *,
    gpus: int,
    factor: float,
    budget: float,
    home: list[dict],
    allowed: list[dict] = (),
) -> Server:
    return Server(
        name=name,
        host='local',
        repo=str(REPO_ROOT),
        python=sys.executable,
        data_root=f'/data/{name}',
        run_root=f'/runs/{name}',
        gpus=tuple(range(gpus)),
        jobs_per_gpu=1,
        warmup_jobs=1,
        min_free_gpu_mib=0,
        oom_retries=0,
        oom_min_free_gpu_mib=0,
        fold_time_factor=factor,
        cache_budget_gib=budget,
        home=tuple(Rule.from_raw(rule) for rule in home),
        allowed=tuple(Rule.from_raw(rule) for rule in allowed),
    )


def _servers() -> dict[str, Server]:
    a30_ratio03 = [
        {'ratios': [0.3], 'adjacency': ['string']},
        {'ratios': [0.3], 'adjacency': ['wgcna'], 'models': ['gcn', 'sagn']},
    ]
    return {
        'big': _server(
            'big',
            gpus=4,
            factor=1.0,
            budget=100.0,
            home=[{'ratios': [0.3]}],
            allowed=[{'ratios': [1.0], 'models': ['gps']}],
        ),
        'small1': _server(
            'small1',
            gpus=2,
            factor=0.5,
            budget=30.0,
            home=[{'ratios': [1.0]}],
            allowed=a30_ratio03,
        ),
        'small2': _server(
            'small2',
            gpus=2,
            factor=0.5,
            budget=30.0,
            home=[{'ratios': [1.0]}],
            allowed=a30_ratio03,
        ),
    }


def _cells() -> tuple[list[Cell], dict[str, Cell]]:
    seconds = {'gcn': 100.0, 'gps': 400.0, 'sagn': 50.0}
    cells, followups = [], {}
    for dataset in ('a', 'b', 'c'):
        for ratio in (1.0, 0.3):
            for adjacency in ('string', 'wgcna'):
                for model in ('gcn', 'gps', 'sagn'):
                    name = f'{model}_{dataset}_{adjacency}_{ratio}'
                    common = {
                        'model': model,
                        'dataset': dataset,
                        'experiment': 'no_readout',
                        'adjacency': adjacency,
                        'method': 'variance',
                        'ratio': ratio,
                        'fold_seconds': seconds[model] * (3 if ratio == 0.3 else 1),
                        'folds': 2,
                        'cache_gib': 20.0 if (adjacency, ratio) == ('wgcna', 0.3) else 1.0,
                    }
                    follow = None if model == 'sagn' else f'id_{name}'
                    cells.append(Cell(study=f'main_{name}', trials=2, followup=follow, **common))
                    if follow:
                        followups[follow] = Cell(
                            study=follow, trials=1, main=f'main_{name}', **common
                        )
    return cells, followups


def _settings(root: Path, servers: dict[str, Server] | None = None) -> Settings:
    return Settings(
        campaign='test',
        main_config=MAIN_CONFIG,
        followup_config=FOLLOWUP_CONFIG,
        coordinator_root=root,
        cost_table=root / 'unused.csv',
        cache_table=root / 'unused.csv',
        servers=servers or _servers(),
        poll_seconds=60.0,
        report_seconds=600.0,
    )


def _simulation(
    root: Path, *, failing: frozenset[str] = frozenset()
) -> tuple[Coordinator, SimulatedCluster]:
    cells, followups = _cells()
    settings = _settings(root)
    cluster = SimulatedCluster(
        settings.servers, {cell.study: cell for cell in cells}, followups, failing=failing
    )
    coordinator = Coordinator(settings, cells, followups, cluster.agents(), clock=cluster.clock)
    return coordinator, cluster


def _completions(cluster: SimulatedCluster) -> collections.Counter[tuple[str, str]]:
    return collections.Counter(
        (name, event['study'])
        for name, sim in cluster.servers.items()
        for event in sim.events
        if event['event'] == 'completed'
    )


# --------------------------------------------------------------------------- cost model


def test_cost_model_prefers_exact_medians_then_scales_unseen_ratios() -> None:
    table = pd.DataFrame(
        [
            ('gcn', 'a', 'wgcna', 1.0, 100.0, 5),
            ('gcn', 'a', 'wgcna', 0.3, 300.0, 5),
            ('gin', 'a', 'wgcna', 1.0, 200.0, 5),
            ('gin', 'a', 'wgcna', 0.3, 800.0, 5),
            ('gps', 'a', 'string', 1.0, 50.0, 5),
            ('sagn', 'b', 'string', 0.3, 9000.0, 5),
        ],
        columns=['model', 'dataset', 'adjacency', 'ratio', 'fold_seconds', 'count'],
    )
    costs = CostModel(table, cap=3720.0, default=900.0)

    assert costs.fold_seconds('gcn', 'a', 'wgcna', 0.3) == 300.0
    assert costs.fold_seconds('gcn', 'b', 'wgcna', 0.3) == 300.0
    # gps was never timed at 0.3: its 1.0 time times the median 0.3 slowdown (3x and 4x).
    assert costs.fold_seconds('gps', 'a', 'string', 0.3) == pytest.approx(50.0 * 3.5)
    assert costs.fold_seconds('sagn', 'b', 'string', 0.3) == 3720.0
    assert costs.fold_seconds('mlp', 'a', 'string', 0.8) == 900.0


def test_cost_table_counts_successes_and_timeouts_but_not_quick_failures(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / 'run_ledger.sqlite3'
    RunLedger(ledger_path)
    name = 'octkfoldtc10_gcn_brca_no_readout_wgcna_variance_r0p3_0123abcd'
    rows = [
        ('success', 100.0, None),
        ('success', 300.0, None),
        ('failed', 3600.0, 'Timeout after 3600s'),
        ('failed', 5.0, 'OOM: CUDA out of memory'),
    ]
    with sqlite3.connect(ledger_path) as connection:
        for attempt, (status, elapsed, error) in enumerate(rows, start=1):
            connection.execute(
                'INSERT INTO fold_attempts (study_name, param_hash, params_json, fold, '
                'training_seed, attempt, status, metric, elapsed_time, error, trial_number) '
                "VALUES (?, 'h', '{}', 0, 0, ?, ?, NULL, ?, ?, 0)",
                (name, attempt, status, elapsed, error),
            )

    table = build_cost_table([ledger_path])

    assert table.to_dict('records') == [
        {
            'model': 'gcn',
            'dataset': 'brca',
            'adjacency': 'wgcna',
            'ratio': 0.3,
            'fold_seconds': 300.0,
            'count': 3,
        }
    ]


# --------------------------------------------------------------------------- placement


def test_home_plan_splits_shared_homes_by_dataset_and_puts_exclusive_work_first() -> None:
    cells, _ = _cells()
    servers = _servers()
    plan = plan_homes(cells, servers)
    by_study = {cell.study: cell for cell in cells}

    assert {by_study[s].ratio for s in plan['big']} == {0.3}
    small = [{by_study[s].dataset for s in plan[name]} for name in ('small1', 'small2')]
    assert small[0].isdisjoint(small[1]) and small[0] | small[1] == {'a', 'b', 'c'}
    assert sorted(s for queue in plan.values() for s in queue) == sorted(by_study)

    def movable(study: str) -> bool:
        return servers['small1'].may_run(by_study[study])

    flags = [movable(study) for study in plan['big']]
    assert flags == sorted(flags), 'studies only big may run must come first'
    exclusive = [by_study[s] for s in plan['big'] if not movable(s)]
    assert all(cell.adjacency == 'wgcna' and cell.model == 'gps' for cell in exclusive)


def test_rules_limit_where_cells_may_run() -> None:
    servers = _servers()
    cells = {cell.study: cell for cell in _cells()[0]}

    assert servers['small1'].may_run(cells['main_gps_a_string_0.3'])
    assert servers['small1'].may_run(cells['main_gcn_a_wgcna_0.3'])
    assert not servers['small1'].may_run(cells['main_gps_a_wgcna_0.3'])
    assert servers['big'].may_run(cells['main_gps_a_wgcna_1.0'])
    assert not servers['big'].may_run(cells['main_gcn_a_wgcna_1.0'])
    with pytest.raises(ValueError, match='Unknown placement rule'):
        Rule.from_raw({'ratio': [0.3]})


# --------------------------------------------------------------------------- coordinator


def test_simulated_campaign_runs_every_study_once_where_it_may_run(tmp_path: Path) -> None:
    coordinator, cluster = _simulation(tmp_path)
    coordinator.run(sleep=cluster.advance)

    completions = _completions(cluster)
    assert set(completions.values()) == {1}
    ran_on = {study: name for name, study in completions}
    assert sorted(ran_on) == sorted([*coordinator.cells, *coordinator.followups])
    for study, name in ran_on.items():
        cell = coordinator.cells.get(study) or coordinator.followups[study]
        assert coordinator.servers[name].may_run(cell), (study, name)
    for follow, cell in coordinator.followups.items():
        assert ran_on[follow] == ran_on[cell.main]

    records = [json.loads(line) for line in coordinator.log_path.read_text().splitlines()]
    assert any(record['type'] == 'move' for record in records)
    assert {record['server'] for record in records if record['type'] == 'end'} == set(
        coordinator.servers
    )
    assert all(not groups for groups in coordinator.open_groups.values())
    for name in ('small1', 'small2'):
        assert coordinator.peak_cache_gib[name] <= coordinator.servers[name].cache_budget_gib
    assert all(sim.finished for sim in cluster.servers.values())

    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['main']['complete'] == len(coordinator.cells)
    assert status['followup']['complete'] == len(coordinator.followups)
    assert '| 0.3 | gps |' in (tmp_path / 'coverage.md').read_text()


def test_restarted_coordinator_resumes_from_its_log(tmp_path: Path) -> None:
    coordinator, cluster = _simulation(tmp_path)
    coordinator.reconcile()
    for _ in range(40):
        coordinator.cycle()
        cluster.advance(coordinator.settings.poll_seconds)

    cells, followups = _cells()
    restarted = Coordinator(
        coordinator.settings, cells, followups, cluster.agents(), clock=cluster.clock
    )
    assert restarted.assigned == coordinator.assigned
    assert restarted.open_groups == coordinator.open_groups
    remaining = {
        name: [study for study in queue if study not in coordinator.assigned]
        for name, queue in coordinator.queues.items()
    }
    assert {
        name: [study for study in queue if study not in restarted.assigned]
        for name, queue in restarted.queues.items()
    } == remaining

    restarted.run(sleep=cluster.advance)
    assert set(_completions(cluster).values()) == {1}
    assert len(_completions(cluster)) == len(cells) + len(followups)


def test_reconcile_adopts_studies_whose_append_reply_was_lost(tmp_path: Path) -> None:
    coordinator, cluster = _simulation(tmp_path)
    study = coordinator.queues['small1'][0]
    cluster.handle('small1', 'append', {'studies': [study]})

    coordinator.reconcile()

    assert coordinator.assigned[study] == 'small1'
    record = json.loads(coordinator.log_path.read_text().splitlines()[-1])
    assert record['reason'] == 'reconciled'
    coordinator.run(sleep=cluster.advance)
    assert set(_completions(cluster).values()) == {1}


def test_unreachable_server_gets_nothing_until_it_returns(tmp_path: Path) -> None:
    coordinator, cluster = _simulation(tmp_path)
    coordinator.reconcile()
    cluster.servers['small2'].down = True
    for _ in range(5):
        coordinator.cycle()
        cluster.advance(coordinator.settings.poll_seconds)
    assert 'small2' not in coordinator.assigned.values()
    assert coordinator.views['small2'].error

    cluster.servers['small2'].down = False
    coordinator.run(sleep=cluster.advance)
    assert 'small2' in coordinator.assigned.values()
    assert set(_completions(cluster).values()) == {1}


def test_main_study_without_complete_trials_gets_no_follow_up(tmp_path: Path) -> None:
    failing = 'main_gcn_a_string_1.0'
    coordinator, cluster = _simulation(tmp_path, failing=frozenset({failing}))
    coordinator.run(sleep=cluster.advance)

    follow = coordinator.cells[failing].followup
    assert follow not in coordinator.assigned
    assert coordinator.followup_handled[failing] == 'skipped: no complete trial'
    status = coordinator.status()
    assert status['main']['partial'] == 1
    assert status['followup']['not_queued'] == {'no complete trial': 1}


def test_shell_agent_runs_locally_or_over_ssh() -> None:
    servers = _servers()
    local, env = ShellAgent(servers['big']).command('status')
    assert local[-3:] == ['-m', 'scripts.campaign.remote_agent', 'status']
    assert env['PYTHONPATH'] == str(REPO_ROOT)

    remote_server = Server(**{**servers['big'].__dict__, 'host': 'frank', 'repo': '/r e/po'})
    remote, env = ShellAgent(remote_server).command('gc')
    assert env is None
    assert remote[:2] == ['ssh', '-o'] and 'frank' in remote
    assert remote[-1].startswith("cd '/r e/po' && PYTHONPATH='/r e/po' ")
    assert remote[-1].endswith('-m scripts.campaign.remote_agent gc')


def test_preflight_reports_packages_as_differences_from_the_first_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages = {
        'big': ['optuna==2.10.1', 'torch==2.8.0'],
        'small1': ['optuna==2.10.1', 'torch==2.8.0'],
        'small2': ['extra==1.0', 'optuna==2.10.1', 'torch==2.7.1'],
    }

    def call(self: ShellAgent, command: str, payload: dict) -> dict:
        assert command == 'preflight' and payload['env'] == self.server.env
        return {'packages': packages[self.server.name]}

    monkeypatch.setattr(ShellAgent, 'call', call)
    replies = preflight(_settings(tmp_path), ['big', 'small1', 'small2'])
    assert replies['small1'] == {'packages_missing': [], 'packages_extra': []}
    assert replies['small2'] == {
        'packages_missing': ['torch==2.8.0'],
        'packages_extra': ['extra==1.0', 'torch==2.7.1'],
    }
    installed = remote_agent.installed_packages()
    assert installed == sorted(installed)
    assert any(package.startswith('optuna==') for package in installed)
    assert not any(package.startswith('ogbench==') for package in installed)


# --------------------------------------------------------------------------- agent


def test_append_manifest_is_idempotent_ends_once_and_drops_torn_lines(tmp_path: Path) -> None:
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text('one\ntwo\npartial-wri')

    reply = remote_agent.append_manifest(manifest, ['two', 'three', 'three'])
    assert reply == {'appended': ['three'], 'studies': 3, 'end': False}
    assert manifest.read_text() == 'one\ntwo\nthree\n'

    assert remote_agent.append_manifest(manifest, ['three'], end=True)['end']
    assert remote_agent.append_manifest(manifest, [], end=True)['appended'] == []
    assert manifest.read_text().count('#END') == 1
    with pytest.raises(ValueError, match='already ends'):
        remote_agent.append_manifest(manifest, ['four'])


def test_read_events_returns_complete_lines_and_detects_a_new_file(tmp_path: Path) -> None:
    events = tmp_path / 'events.jsonl'
    events.write_text('{"event": "a"}\n{"event": "b"}\n{"event": "c"')

    first, offset, reset = remote_agent.read_events(events, 0)
    assert [event['event'] for event in first] == ['a', 'b'] and not reset
    with events.open('a') as handle:
        handle.write('}\n')
    second, offset, _ = remote_agent.read_events(events, offset)
    assert [event['event'] for event in second] == ['c']

    events.write_text('{"event": "d"}\n')
    third, _, reset = remote_agent.read_events(events, offset)
    assert reset and [event['event'] for event in third] == ['d']


def test_merge_candidates_keeps_existing_ones_and_reports_conflicts(tmp_path: Path) -> None:
    path = tmp_path / 'candidates.json'
    assert remote_agent.merge_candidates(path, {'s1': {'param_hash': 'a'}}) == {}
    conflicts = remote_agent.merge_candidates(
        path, {'s1': {'param_hash': 'b'}, 's2': {'param_hash': 'c'}}
    )

    assert conflicts == {'s1': 'a'}
    assert json.loads(path.read_text()) == {'s1': {'param_hash': 'a'}, 's2': {'param_hash': 'c'}}


def test_status_reports_manifest_state_and_new_events(tmp_path: Path) -> None:
    run_root = tmp_path / 'run'
    run_root.mkdir()
    (run_root / 'manifest.txt').write_text('s1\ns2\n')
    state = {'pid': os.getpid(), 'host': socket.gethostname(), 'finished': False, 'known': 2}
    (run_root / 'manifest.txt.state.json').write_text(json.dumps(state))
    (run_root / 'manifest.txt.events.jsonl').write_text('{"event": "completed", "study": "s1"}\n')

    reply = remote_agent.cmd_status({'run_root': str(run_root), 'data_root': str(tmp_path)})

    assert reply['manifest'] == {'studies': 2, 'end': False}
    assert reply['state']['known'] == 2
    assert reply['events'] == [{'event': 'completed', 'study': 's1'}]
    # The pid is alive but is not a launcher.
    assert reply['launcher_alive'] is False
    assert reply['disk_free_gib']['data_root'] > 0


def test_agent_main_prints_one_json_reply_even_when_libraries_print(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def noisy(payload: dict) -> dict:
        print('progress output')
        return {'echo': payload['value']}

    monkeypatch.setitem(remote_agent.COMMANDS, 'status', noisy)
    monkeypatch.setattr(sys, 'stdin', io.StringIO('{"value": 3}'))
    assert remote_agent.main(['status']) == 0
    assert capsys.readouterr().out == '{"echo": 3}\n'

    monkeypatch.setitem(remote_agent.COMMANDS, 'status', lambda payload: 1 / 0)
    monkeypatch.setattr(sys, 'stdin', io.StringIO('{}'))
    assert remote_agent.main(['status']) == 1
    assert 'ZeroDivisionError' in json.loads(capsys.readouterr().out)['error']


def _complete_trial(config: OptunaSearchConfig, model: str, value: float, pick: int):
    sampled = {
        name: spec.choices[pick % len(spec.choices)] if spec.kind == 'categorical' else spec.low
        for name, spec in _model_space(config, model).items()
    }
    param_hash = _stable_hash(validate_sampled_parameters(config, model, sampled))
    return optuna.trial.create_trial(
        state=TrialState.COMPLETE,
        value=value,
        params={},
        distributions={},
        user_attrs={
            'sampled_params': sampled,
            'param_hash': param_hash,
            'fold_scores': {'0': value},
        },
    )


def test_followup_queues_the_best_main_configuration_as_a_candidate(tmp_path: Path) -> None:
    run_root = tmp_path / 'run'
    payload = {
        'run_root': str(run_root),
        'data_root': str(tmp_path / 'data'),
        'configs': [MAIN_CONFIG, FOLLOWUP_CONFIG],
    }
    main, followup = remote_agent._load_campaign_configs(payload)
    cells = build_outer_cells(main)
    gcn = next(cell for cell in cells if cell.model == 'gcn')
    sagn = next(cell for cell in cells if cell.model == 'sagn')
    unfinished = next(cell for cell in cells if cell.model == 'gin')
    main.output_dir.mkdir(parents=True)
    for cell in (gcn, sagn, unfinished):
        optuna.create_study(study_name=cell.study_name, storage=main.storage, direction='maximize')
    study = optuna.load_study(study_name=gcn.study_name, storage=main.storage)
    study.add_trial(_complete_trial(main, 'gcn', 0.6, pick=0))
    study.add_trial(_complete_trial(main, 'gcn', 0.8, pick=1))

    reply = remote_agent.cmd_followup(
        {**payload, 'studies': [gcn.study_name, sagn.study_name, unfinished.study_name]}
    )

    target = reply['followups'][gcn.study_name]
    assert target.startswith(followup.study_name_prefix)
    assert reply['skipped'] == {
        sagn.study_name: 'no follow-up cell',
        unfinished.study_name: 'no complete trial',
    }
    candidate = json.loads((run_root / 'candidates.json').read_text())[target]
    assert candidate['source_mean'] == 0.8 and candidate['source_trial_numbers'] == [1]
    assert (run_root / 'manifest.txt').read_text() == f'{target}\n'

    again = remote_agent.cmd_followup({**payload, 'studies': [gcn.study_name]})
    assert again['skipped'] == {gcn.study_name: 'already queued'}


def test_fold_cache_dir_matches_the_directory_the_dataset_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = tmp_path / 'hub'
    hub.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame(rng.normal(size=(40, 6)), columns=[f'g{j}' for j in range(6)]).to_parquet(
        hub / 'toy_data.parquet'
    )
    pd.DataFrame({'target': np.arange(40) % 2}).to_parquet(hub / 'toy_targets.parquet')

    def local_download(self, filename: str) -> str:
        if not (hub / filename).exists():
            raise FileNotFoundError(filename)
        return str(hub / filename)

    # ogbench.data.datasets registers its modules dynamically, so patch the class the loader uses.
    monkeypatch.setattr(omics_datasets.HFOmicsDataset, '_hf_download', local_download)
    data_root = tmp_path / 'data'
    parameters = {
        'data_type': 'omics',
        'data_domain': 'graph',
        'data_name': 'toy',
        'data_dir': str(data_root / 'omics'),
        'adjacency_method': 'wgcna',
        'adjacency_target_connectivity': 0.1,
        'wgcna_binarization': 'target_connectivity',
        'node_sample_ratio': 1.0,
        'method': 'variance',
        'imputation_method': 'mean',
        'split_type': 'k-fold',
        'k': 5,
        'fold': 2,
        'grouping': None,
    }
    dataset, _ = OmicsDatasetLoader(OmegaConf.create(parameters)).load()

    path = remote_agent.fold_cache_dir(parameters)

    assert path == Path(dataset.get_data_dir())
    assert remote_agent._checked_cache_dir(path, data_root) == path.resolve()
    with pytest.raises(ValueError, match='outside'):
        remote_agent._checked_cache_dir(path, tmp_path / 'elsewhere')
    with pytest.raises(ValueError, match='STRING'):
        remote_agent._checked_cache_dir(data_root / 'omics/string_cache/split_k-fold', data_root)
    with pytest.raises(ValueError, match='not a k-fold cache'):
        remote_agent._checked_cache_dir(data_root / 'omics/toy', data_root)


def test_gc_targets_the_five_fold_caches_of_a_group(tmp_path: Path) -> None:
    data_root = tmp_path / 'data'
    payload = {
        'run_root': str(tmp_path / 'run'),
        'data_root': str(data_root),
        'configs': [MAIN_CONFIG, FOLLOWUP_CONFIG],
    }
    main = remote_agent._load_campaign_configs(payload)[0]
    cells = build_outer_cells(main)
    first = cells[0]
    group = [
        cell
        for cell in cells
        if cell.dataset == first.dataset
        and all(
            cell.values[key] == first.values[key]
            for key in (
                'dataset.loader.parameters.adjacency_method',
                'dataset.loader.parameters.method',
                'dataset.loader.parameters.node_sample_ratio',
            )
        )
    ]
    assert len(group) == 17
    existing = None

    reply = remote_agent.cmd_gc(
        {**payload, 'studies': [cell.study_name for cell in group], 'dry_run': True}
    )

    paths = [Path(path) for path in reply['missing']]
    assert reply['removed'] == [] and len(paths) == 5
    folds = sorted(re.search(r'_fold_(\d+)', path.name).group(1) for path in paths)
    assert folds == ['0', '1', '2', '3', '4']
    for path in paths:
        assert path.is_relative_to(data_root) and first.dataset in path.parts
        existing = path
    existing.mkdir(parents=True)
    (existing / 'data.pt').write_bytes(b'x' * 2048)
    reply = remote_agent.cmd_gc({**payload, 'studies': [first.study_name]})
    assert reply['removed'] == [str(existing)] and not existing.exists()


# --------------------------------------------------------------------------- supervisor


def test_supervisor_restarts_the_launcher_until_the_manifest_is_finished(tmp_path: Path) -> None:
    run_root = tmp_path / 'run'
    calls = tmp_path / 'calls.txt'
    fake_python = tmp_path / 'python'
    fake_python.write_text(
        '#!/usr/bin/env bash\n'
        f'if [[ "$1" == -c ]]; then exec {sys.executable} "$@"; fi\n'
        f'echo "$*" >> {calls}\n'
        f'if [[ $(wc -l < {calls}) -ge 2 ]]; then\n'
        f'  echo \'{{"finished": true}}\' > {run_root}/manifest.txt.state.json\n'
        'fi\n'
        'exit 1\n'
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)
    env = {
        **os.environ,
        'REPO': str(REPO_ROOT),
        'PYTHON': str(fake_python),
        'RUN_ROOT': str(run_root),
        'DATA_ROOT': str(tmp_path / 'data'),
        'CONFIGS': f'{MAIN_CONFIG} {FOLLOWUP_CONFIG}',
        'GPUS': '0 1',
        'RELAUNCH_WAIT': '0',
        'GPU_LOG_EVERY': '3600',
    }

    result = subprocess.run(  # nosec B603 B607
        ['bash', str(REPO_ROOT / 'scripts/campaign/server_supervisor.sh')],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    launches = calls.read_text().splitlines()
    assert len(launches) == 2
    assert f'--config {MAIN_CONFIG} {FOLLOWUP_CONFIG}' in launches[0]
    assert f'--follow-manifest {run_root}/manifest.txt' in launches[0]
    assert '--gpus 0 1' in launches[0] and '--retry-failed' in launches[0]
    assert 'manifest finished' in result.stdout

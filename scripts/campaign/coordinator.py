"""Coordinate the October k-fold campaign on Parka, Frank and Hall from Parka.

Each server runs one follow-mode launcher (``server_supervisor.sh``) that trains the studies
appended to its manifest. The coordinator decides what goes where:

- placement: ratio 0.3 lives on Parka (A100), the larger ratios on the A30s, whose datasets are
  split to balance projected time; each server's queue is ordered by cache group (dataset,
  adjacency, selection method, ratio), most expensive group first;
- feeding: a server's manifest only grows when its backlog drops below ``low_water`` x slots,
  so unstarted work can still move;
- stealing: a server whose queue is empty takes the first unstarted cache group it is allowed to
  run from the server with the most remaining work;
- follow-ups: when a main study finishes, its best configuration is queued as the
  gene-identity follow-up on the same server, while its caches are still there;
- cleanup: a cache group's fold caches are removed from a server once every study of the group
  is assigned and the server's share of it (follow-ups included) has finished;
- reports: ``status.json`` and ``coverage.md`` with the projected finish of every server.

Every decision is appended to ``assignments.jsonl`` and the per-server event logs are re-read
from the start, so a restarted coordinator resumes exactly where it stopped.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import statistics
import subprocess  # nosec B404
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.campaign.remote_agent import cell_key  # noqa: E402
from scripts.optuna_search import FOLLOW_WARM_TTL_SECONDS  # noqa: E402

STUDY_PATTERN = re.compile(
    r'^[a-z0-9]+_(?P<model>gcn|gin|gatv2|gatv4|graph_sage|sagn|chebnet|mlp|gps)_'
    r'(?P<dataset>[a-z]+)_(?P<experiment>omics_readout|no_readout)_'
    r'(?P<adjacency>wgcna|string)_(?P<method>.+)_r(?P<ratio>[0-9p]+)_[0-9a-f]+$'
)
COST_LEVELS = (
    ('model', 'dataset', 'adjacency', 'ratio'),
    ('model', 'adjacency', 'ratio'),
    ('model', 'ratio'),
    ('ratio',),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(text)
    os.replace(temporary, path)


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class Rule:
    """Cells matching every given field (an omitted field matches anything)."""

    ratios: tuple[float, ...] | None = None
    models: tuple[str, ...] | None = None
    adjacency: tuple[str, ...] | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> Rule:
        unknown = set(raw) - {'ratios', 'models', 'adjacency'}
        if unknown:
            raise ValueError(f'Unknown placement rule fields: {sorted(unknown)}')
        return cls(
            ratios=None if 'ratios' not in raw else tuple(float(v) for v in raw['ratios']),
            models=None if 'models' not in raw else tuple(raw['models']),
            adjacency=None if 'adjacency' not in raw else tuple(raw['adjacency']),
        )

    def matches(self, cell: Cell) -> bool:
        return (
            (self.ratios is None or any(math.isclose(cell.ratio, r) for r in self.ratios))
            and (self.models is None or cell.model in self.models)
            and (self.adjacency is None or cell.adjacency in self.adjacency)
        )


@dataclass(frozen=True)
class Server:
    name: str
    host: str
    repo: str
    python: str
    data_root: str
    run_root: str
    gpus: tuple[int, ...]
    jobs_per_gpu: int
    warmup_jobs: int
    min_free_gpu_mib: int
    oom_retries: int
    oom_min_free_gpu_mib: int
    fold_time_factor: float
    cache_budget_gib: float
    home: tuple[Rule, ...]
    allowed: tuple[Rule, ...]
    env: dict[str, str] = field(default_factory=dict)

    @property
    def slots(self) -> int:
        return len(self.gpus) * self.jobs_per_gpu

    @property
    def is_local(self) -> bool:
        return self.host == 'local'

    def is_home(self, cell: Cell) -> bool:
        return any(rule.matches(cell) for rule in self.home)

    def may_run(self, cell: Cell) -> bool:
        return self.is_home(cell) or any(rule.matches(cell) for rule in self.allowed)


@dataclass(frozen=True)
class Settings:
    campaign: str
    main_config: str
    followup_config: str
    coordinator_root: Path
    cost_table: Path
    cache_table: Path
    servers: dict[str, Server]
    low_water: float = 1.5
    high_water: float = 2.0
    poll_seconds: float = 300.0
    report_seconds: float = 1800.0
    cleanup: bool = True
    fold_seconds_cap: float = 3720.0
    default_fold_seconds: float = 900.0
    followup_cost_factor: float = 1.0
    cache_safety: float = 1.25
    default_fold_cache_gib: float = 1.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        raw = yaml.safe_load(Path(path).read_text())
        feed = raw.get('feed', {})
        cost = raw.get('cost', {})
        servers = {}
        for name, entry in raw['servers'].items():
            entry = dict(entry)
            home = tuple(Rule.from_raw(rule) for rule in entry.pop('home'))
            allowed = tuple(Rule.from_raw(rule) for rule in entry.pop('allowed', []))
            env = {key: str(value) for key, value in entry.pop('env', {}).items()}
            servers[name] = Server(
                name=name,
                home=home,
                allowed=allowed,
                env=env,
                gpus=tuple(int(gpu) for gpu in entry.pop('gpus')),
                **entry,
            )
        settings = cls(
            campaign=raw['campaign'],
            main_config=raw['main_config'],
            followup_config=raw['followup_config'],
            coordinator_root=Path(raw['coordinator_root']),
            cost_table=REPO_ROOT / raw['cost_table'],
            cache_table=REPO_ROOT / raw['cache_table'],
            servers=servers,
            low_water=float(feed.get('low_water', 1.5)),
            high_water=float(feed.get('high_water', 2.0)),
            poll_seconds=float(feed.get('poll_seconds', 300)),
            report_seconds=float(feed.get('report_seconds', 1800)),
            cleanup=bool(feed.get('cleanup', True)),
            fold_seconds_cap=float(cost.get('fold_seconds_cap', 3720)),
            default_fold_seconds=float(cost.get('default_fold_seconds', 900)),
            followup_cost_factor=float(cost.get('followup_cost_factor', 1.0)),
            cache_safety=float(cost.get('cache_safety', 1.25)),
            default_fold_cache_gib=float(cost.get('default_fold_cache_gib', 1.0)),
        )
        if not 0 < settings.low_water <= settings.high_water:
            raise ValueError('feed needs 0 < low_water <= high_water')
        return settings


# --------------------------------------------------------------------------- cost model


def _ratio_from_slug(slug: str) -> float:
    return float(slug.replace('p', '.'))


def parse_study_name(name: str) -> dict[str, Any] | None:
    match = STUDY_PATTERN.match(name)
    if match is None:
        return None
    fields = match.groupdict()
    fields['ratio'] = _ratio_from_slug(fields['ratio'])
    return fields


def build_cost_table(ledgers: Sequence[Path], since: str = '') -> pd.DataFrame:
    """Median fold seconds by (model, dataset, adjacency, ratio) from run ledgers.

    Timed-out folds count at their elapsed time; other failures (mostly quick OOMs) do not.
    """
    samples: dict[tuple[str, str, str, float], list[float]] = collections.defaultdict(list)
    for ledger in ledgers:
        with sqlite3.connect(f'file:{ledger}?mode=ro', uri=True, timeout=60) as connection:
            rows = connection.execute(
                'SELECT study_name, elapsed_time FROM fold_attempts WHERE created_at >= ? '
                "AND elapsed_time > 0 AND (status = 'success' OR error LIKE 'Timeout after%')",
                (since,),
            ).fetchall()
        for study, elapsed in rows:
            fields = parse_study_name(study)
            if fields is not None:
                key = (fields['model'], fields['dataset'], fields['adjacency'], fields['ratio'])
                samples[key].append(float(elapsed))
    return pd.DataFrame(
        [
            {
                'model': model,
                'dataset': dataset,
                'adjacency': adjacency,
                'ratio': ratio,
                'fold_seconds': round(statistics.median(values), 1),
                'count': len(values),
            }
            for (model, dataset, adjacency, ratio), values in sorted(samples.items())
        ],
        columns=['model', 'dataset', 'adjacency', 'ratio', 'fold_seconds', 'count'],
    )


class CostModel:
    """Reference (Parka) seconds per fold, falling back to coarser medians when unseen.

    A model never timed at some ratio gets its time at the largest ratio, scaled by the median
    slowdown of the models timed at both ratios.
    """

    def __init__(self, table: pd.DataFrame, *, cap: float, default: float) -> None:
        self.cap = cap
        self.default = default
        self.levels: list[tuple[tuple[str, ...], dict[tuple[Any, ...], float]]] = []
        for columns in COST_LEVELS:
            medians = {}
            if not table.empty:
                for key, group in table.groupby(list(columns)):
                    key = key if isinstance(key, tuple) else (key,)
                    medians[tuple(_normalize(v) for v in key)] = float(
                        group['fold_seconds'].median()
                    )
            self.levels.append((columns, medians))
        by_model_ratio = self.levels[2][1]
        ratios = {ratio for _, ratio in by_model_ratio}
        self.reference_ratio = max(ratios) if ratios else None
        self.ratio_scale = {}
        for ratio in ratios:
            slowdowns = [
                seconds / by_model_ratio[(model, self.reference_ratio)]
                for (model, other), seconds in by_model_ratio.items()
                if other == ratio and (model, self.reference_ratio) in by_model_ratio
            ]
            if slowdowns:
                self.ratio_scale[ratio] = statistics.median(slowdowns)

    @classmethod
    def from_csv(cls, path: Path, *, cap: float, default: float) -> CostModel:
        table = pd.read_csv(path) if path.exists() else pd.DataFrame()
        return cls(table, cap=cap, default=default)

    def fold_seconds(self, model: str, dataset: str, adjacency: str, ratio: float) -> float:
        values = {'model': model, 'dataset': dataset, 'adjacency': adjacency, 'ratio': ratio}
        *per_model, (_, by_ratio) = self.levels
        for columns, medians in per_model:
            seconds = medians.get(self._key(columns, values))
            if seconds is not None:
                return min(seconds, self.cap)
        scale = self.ratio_scale.get(_normalize(ratio))
        if scale is not None:
            reference = {**values, 'ratio': self.reference_ratio}
            for columns, medians in per_model[1:]:
                seconds = medians.get(self._key(columns, reference))
                if seconds is not None:
                    return min(seconds * scale, self.cap)
        return min(by_ratio.get((_normalize(ratio),), self.default), self.cap)

    @staticmethod
    def _key(columns: Sequence[str], values: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(_normalize(values[column]) for column in columns)


def _normalize(value: Any) -> Any:
    return round(float(value), 6) if isinstance(value, int | float) else value


class CacheSizes:
    """GiB of one fold cache by (dataset, adjacency, ratio), measured on September caches."""

    def __init__(self, table: pd.DataFrame, *, default: float) -> None:
        self.default = default
        self.sizes = {
            (row.dataset, row.adjacency, _normalize(row.ratio)): float(row.fold_gib)
            for row in table.itertuples()
        }
        self.by_ratio: dict[tuple[str, float], float] = {}
        for (_, adjacency, ratio), size in self.sizes.items():
            key = (adjacency, ratio)
            self.by_ratio[key] = max(self.by_ratio.get(key, 0.0), size)

    @classmethod
    def from_csv(cls, path: Path, *, default: float) -> CacheSizes:
        table = pd.read_csv(path) if path.exists() else pd.DataFrame()
        return cls(table, default=default)

    def fold_gib(self, dataset: str, adjacency: str, ratio: float) -> float:
        ratio = _normalize(ratio)
        size = self.sizes.get((dataset, adjacency, ratio))
        if size is None:
            size = self.by_ratio.get((adjacency, ratio), self.default)
        return size


# --------------------------------------------------------------------------- cells and plan


@dataclass(frozen=True)
class Cell:
    study: str
    model: str
    dataset: str
    experiment: str
    adjacency: str
    method: str
    ratio: float
    fold_seconds: float
    trials: int
    folds: int
    followup: str | None = None
    main: str | None = None
    cache_gib: float = 0.0

    @property
    def group(self) -> tuple[str, str, str, float]:
        return (self.dataset, self.adjacency, self.method, self.ratio)

    @property
    def cost(self) -> float:
        """Reference slot-seconds for the whole study."""
        return self.fold_seconds * self.trials * self.folds


def campaign_cells(
    settings: Settings, costs: CostModel, caches: CacheSizes | None = None
) -> tuple[list[Cell], dict[str, Cell]]:
    """Main cells in config order, and follow-up cells by study name.

    ``cache_gib`` is the disk taken by the cell's cache group (all folds) on a server.
    """
    from scripts.optuna_search import (
        ADJACENCY_METHOD,
        EXPERIMENT,
        NODE_SAMPLE_RATIO,
        SELECTION_METHOD,
        OptunaSearchConfig,
        build_outer_cells,
    )

    main = OptunaSearchConfig.from_yaml(REPO_ROOT / settings.main_config)
    follow = OptunaSearchConfig.from_yaml(REPO_ROOT / settings.followup_config)
    follow_by_key = {cell_key(cell): cell for cell in build_outer_cells(follow)}
    cells: list[Cell] = []
    followups: dict[str, Cell] = {}
    for outer in build_outer_cells(main):
        values = outer.values
        ratio = float(values[NODE_SAMPLE_RATIO])
        fold_seconds = costs.fold_seconds(
            outer.model, outer.dataset, values[ADJACENCY_METHOD], ratio
        )
        cache_gib = 0.0
        if caches is not None:
            fold_gib = caches.fold_gib(outer.dataset, values[ADJACENCY_METHOD], ratio)
            cache_gib = fold_gib * len(main.folds) * settings.cache_safety
        common = {
            'model': outer.model,
            'dataset': outer.dataset,
            'experiment': values[EXPERIMENT],
            'adjacency': values[ADJACENCY_METHOD],
            'method': values[SELECTION_METHOD],
            'ratio': ratio,
            'cache_gib': cache_gib,
        }
        target = follow_by_key.get(cell_key(outer))
        cells.append(
            Cell(
                study=outer.study_name,
                fold_seconds=fold_seconds,
                trials=main.n_trials,
                folds=len(main.folds),
                followup=None if target is None else target.study_name,
                **common,
            )
        )
        if target is not None:
            followups[target.study_name] = Cell(
                study=target.study_name,
                fold_seconds=fold_seconds * settings.followup_cost_factor,
                trials=follow.n_trials,
                folds=len(follow.folds),
                main=outer.study_name,
                **common,
            )
    return cells, followups


def _order_queue(cells: Iterable[Cell], movable: Callable[[Cell], bool]) -> list[str]:
    """Studies only this server may run first, then the ones other servers may take.

    Idle servers steal from the back, so no server is left with a tail nobody else can share. Each
    part runs cache group by cache group, least movable and most expensive group first.
    """
    groups: dict[tuple[Any, ...], list[Cell]] = collections.defaultdict(list)
    for cell in cells:
        groups[cell.group].append(cell)
    ranked = sorted(
        groups.values(),
        key=lambda members: (
            sum(movable(c) for c in members) / len(members),
            -sum(c.cost for c in members),
            members[0].group,
        ),
    )
    rank = {members[0].group: index for index, members in enumerate(ranked)}
    ordered = sorted(
        (cell for members in ranked for cell in members),
        key=lambda c: (movable(c), rank[c.group], -c.cost, c.study),
    )
    return [cell.study for cell in ordered]


def plan_homes(cells: Sequence[Cell], servers: dict[str, Server]) -> dict[str, list[str]]:
    """Home queues: each cell goes to a server whose home rules match it.

    When several servers share a home, whole datasets are dealt to them (largest first, to the
    server with the least projected time), so each one builds caches for fewer datasets.
    """
    load = dict.fromkeys(servers, 0.0)
    assigned: dict[str, list[Cell]] = {name: [] for name in servers}
    by_homes: dict[tuple[str, ...], list[Cell]] = collections.defaultdict(list)
    for cell in cells:
        homes = tuple(name for name, server in servers.items() if server.is_home(cell))
        if not homes:
            raise ValueError(f'No server has {cell.study} in its home rules')
        by_homes[homes].append(cell)
    for homes, members in sorted(by_homes.items(), key=lambda item: (len(item[0]), item[0])):
        by_dataset: dict[str, list[Cell]] = collections.defaultdict(list)
        for cell in members:
            by_dataset[cell.dataset].append(cell)
        datasets = sorted(by_dataset, key=lambda d: (-sum(c.cost for c in by_dataset[d]), d))
        for dataset in datasets:
            target = min(
                homes,
                key=lambda name: (
                    load[name] * servers[name].fold_time_factor / servers[name].slots,
                    name,
                ),
            )
            assigned[target].extend(by_dataset[dataset])
            load[target] += sum(c.cost for c in by_dataset[dataset])

    def movable_from(home: str) -> Callable[[Cell], bool]:
        others = [server for name, server in servers.items() if name != home]
        return lambda cell: any(server.may_run(cell) for server in others)

    return {name: _order_queue(members, movable_from(name)) for name, members in assigned.items()}


# --------------------------------------------------------------------------- agents


class AgentError(RuntimeError):
    """A server agent call failed or could not be reached."""


class Agent(Protocol):
    def call(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class ShellAgent:
    """Run ``remote_agent.py`` on a server: in place on Parka, over ssh elsewhere."""

    def __init__(self, server: Server, timeout: float = 1800.0) -> None:
        self.server = server
        self.timeout = timeout

    def command(self, command: str) -> tuple[list[str], dict[str, str] | None]:
        agent = [self.server.python, '-m', 'scripts.campaign.remote_agent', command]
        if self.server.is_local:
            env = {**os.environ, 'PYTHONPATH': self.server.repo}
            return agent, env
        remote = (
            f'cd {shlex.quote(self.server.repo)} && '
            f'PYTHONPATH={shlex.quote(self.server.repo)} {shlex.join(agent)}'
        )
        ssh = [
            'ssh',
            '-o',
            'BatchMode=yes',
            '-o',
            'ConnectTimeout=30',
            '-o',
            'ServerAliveInterval=30',
            self.server.host,
            remote,
        ]
        return ssh, None

    def call(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        args, env = self.command(command)
        try:
            result = subprocess.run(  # nosec B603
                args,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.server.repo if self.server.is_local else None,
                env=env,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AgentError(f'{self.server.name} {command}: {error}') from error
        reply = None
        for line in reversed(result.stdout.splitlines()):
            try:
                reply = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if not isinstance(reply, dict) or 'error' in reply or result.returncode != 0:
            detail = reply.get('error') if isinstance(reply, dict) else result.stderr[-2000:]
            raise AgentError(f'{self.server.name} {command} exited {result.returncode}: {detail}')
        return reply


# --------------------------------------------------------------------------- coordinator


@dataclass
class ServerView:
    """What the coordinator knows about one server from its latest status reply."""

    events_offset: int = 0
    reachable: bool = False
    error: str | None = None
    last_seen: float | None = None
    state: dict[str, Any] | None = None
    launcher_alive: bool = False
    supervisor_alive: bool = False
    manifest_studies: int = 0
    manifest_end: bool = False
    disk_free_gib: dict[str, Any] = field(default_factory=dict)
    gpus: list[dict[str, Any]] = field(default_factory=list)
    completed: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)

    def finished(self, study: str) -> bool:
        return study in self.completed or study in self.failed

    @property
    def backlog(self) -> int:
        """Studies listed in the manifest that the launcher has not started yet."""
        if self.state is None:
            return max(self.manifest_studies - len(self.completed) - len(self.failed), 0)
        queued = sum(len(self.state.get(key, [])) for key in ('pending', 'warming', 'ready'))
        return queued + max(self.manifest_studies - int(self.state.get('known', 0)), 0)

    @property
    def running(self) -> dict[str, str]:
        return dict((self.state or {}).get('running', {}))


class Coordinator:
    def __init__(
        self,
        settings: Settings,
        cells: Sequence[Cell],
        followups: dict[str, Cell],
        agents: dict[str, Agent],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.servers = settings.servers
        self.cells = {cell.study: cell for cell in cells}
        self.followups = followups
        self.agents = agents
        self.clock = clock
        self.root = settings.coordinator_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / 'assignments.jsonl'
        self.groups: dict[tuple[Any, ...], list[str]] = collections.defaultdict(list)
        for cell in cells:
            self.groups[cell.group].append(cell.study)

        self.views = {name: ServerView() for name in self.servers}
        self.assigned: dict[str, str] = {}
        self.queue_reason: dict[str, str] = {}
        self.followup_handled: dict[str, str] = {}
        # Cache groups with studies assigned to a server whose caches were not removed yet.
        self.open_groups: dict[str, set[tuple[Any, ...]]] = {name: set() for name in self.servers}
        self.cleaned_at: dict[tuple[str, tuple[Any, ...]], float] = {}
        self.peak_cache_gib = dict.fromkeys(self.servers, 0.0)
        self.ended: set[str] = set()
        self.last_report = -math.inf
        self.queues = self._load_plan()
        self._replay()

    # ---------------------------------------------------------------- persistence

    def _load_plan(self) -> dict[str, collections.deque[str]]:
        path = self.root / 'plan.json'
        if path.exists():
            plan = json.loads(path.read_text())
            if set(plan) != set(self.servers) or sorted(
                study for queue in plan.values() for study in queue
            ) != sorted(self.cells):
                raise ValueError(f'{path} does not match the campaign cells and servers')
        else:
            plan = plan_homes(list(self.cells.values()), self.servers)
            _atomic_write(path, json.dumps(plan, indent=1))
        for queue in plan.values():
            self.queue_reason.update(dict.fromkeys(queue, 'home'))
        return {name: collections.deque(queue) for name, queue in plan.items()}

    def now_iso(self) -> str:
        return datetime.fromtimestamp(self.clock(), UTC).isoformat(timespec='seconds')

    def _record(self, record: dict[str, Any]) -> None:
        record = {'time': self.now_iso(), **record}
        with self.log_path.open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        self._apply(record)

    def _cell(self, study: str) -> Cell:
        return self.cells.get(study) or self.followups[study]

    def _apply(self, record: dict[str, Any]) -> None:
        kind = record['type']
        if kind == 'assign':
            server = record['server']
            self.assigned[record['study']] = server
            if record.get('main'):
                self.followup_handled[record['main']] = record['study']
            if record['study'] in self.cells or record['study'] in self.followups:
                self.open_groups[server].add(self._cell(record['study']).group)
                self.peak_cache_gib[server] = max(
                    self.peak_cache_gib[server], self.open_cache_gib(server)
                )
        elif kind == 'move':
            moving = set(record['studies'])
            for queue in self.queues.values():
                kept = [study for study in queue if study not in moving]
                queue.clear()
                queue.extend(kept)
            self.queues[record['to']].extend(record['studies'])
            self.queue_reason.update(dict.fromkeys(record['studies'], f'steal:{record["from"]}'))
        elif kind == 'followup_skip':
            self.followup_handled[record['study']] = f'skipped: {record["reason"]}'
        elif kind == 'cleanup':
            group = tuple(record['group'])
            self.open_groups[record['server']].discard(group)
            self.cleaned_at[(record['server'], group)] = datetime.fromisoformat(
                record['time']
            ).timestamp()
        elif kind == 'end':
            self.ended.add(record['server'])

    def _group_gib(self, group: tuple[Any, ...]) -> float:
        return self.cells[self.groups[group][0]].cache_gib

    def open_cache_gib(self, name: str) -> float:
        return sum(self._group_gib(group) for group in self.open_groups[name])

    def _fits(
        self, name: str, group: tuple[Any, ...], opening: set[tuple[Any, ...]] = frozenset()
    ) -> bool:
        """Whether the server's cache budget has room for this group (always for the first)."""
        open_groups = self.open_groups[name] | opening
        if group in open_groups or not open_groups:
            return True
        used = sum(self._group_gib(other) for other in open_groups)
        return used + self._group_gib(group) <= self.servers[name].cache_budget_gib

    def _replay(self) -> None:
        if not self.log_path.exists():
            return
        for line in self.log_path.read_text().splitlines():
            if line.strip():
                self._apply(json.loads(line))

    # ---------------------------------------------------------------- agent helpers

    def _payload(self, server: Server, **extra: Any) -> dict[str, Any]:
        return {
            'run_root': server.run_root,
            'data_root': server.data_root,
            'configs': [self.settings.main_config, self.settings.followup_config],
            **extra,
        }

    def _call(self, name: str, command: str, **extra: Any) -> dict[str, Any] | None:
        try:
            return self.agents[name].call(command, self._payload(self.servers[name], **extra))
        except AgentError as error:
            view = self.views[name]
            view.error = str(error)
            print(f'[coordinator] {utc_now()} {error}', flush=True)
            return None

    # ---------------------------------------------------------------- one cycle

    def reconcile(self) -> None:
        """Adopt studies found in server manifests but missing from the log (a lost reply)."""
        for name in self.servers:
            reply = self._call(name, 'manifest')
            if reply is None:
                raise AgentError(f'Cannot read the {name} manifest; refusing to start blind')
            for study in reply['studies']:
                owner = self.assigned.get(study)
                if owner is None:
                    cell = self.followups.get(study)
                    self._record(
                        {
                            'type': 'assign',
                            'study': study,
                            'server': name,
                            'kind': 'main' if study in self.cells else 'followup',
                            'reason': 'reconciled',
                            **({'main': cell.main} if cell is not None else {}),
                        }
                    )
                elif owner != name:
                    print(f'[coordinator] WARNING {study} is listed on {owner} and {name}')
            if reply['end']:
                self.ended.add(name)

    def poll(self) -> None:
        for name in self.servers:
            view = self.views[name]
            reply = self._call(name, 'status', events_offset=view.events_offset)
            if reply is None:
                view.reachable = False
                continue
            if reply['events_reset']:
                view.completed.clear()
                view.failed.clear()
            for event in reply['events']:
                study = event.get('study')
                if event.get('event') == 'completed' and study:
                    view.completed[study] = event
                    view.failed.pop(study, None)
                elif event.get('event') == 'failed' and study and study not in view.completed:
                    view.failed[study] = event.get('error', '')
            view.events_offset = reply['events_offset']
            view.reachable = True
            view.error = None
            view.last_seen = self.clock()
            view.state = reply['state']
            view.launcher_alive = reply['launcher_alive']
            view.supervisor_alive = reply['supervisor_alive']
            view.manifest_studies = reply['manifest']['studies']
            view.manifest_end = reply['manifest']['end']
            view.disk_free_gib = reply['disk_free_gib']
            view.gpus = reply['gpus']

    def queue_followups(self) -> None:
        for name, view in self.views.items():
            ready = [
                study
                for study in view.completed
                if study in self.cells
                and self.assigned.get(study) == name
                and study not in self.followup_handled
            ]
            if not view.reachable or not ready:
                continue
            for study in ready:
                if self.cells[study].followup is None:
                    self._record({'type': 'followup_skip', 'study': study, 'reason': 'none'})
            ready = [study for study in ready if study not in self.followup_handled]
            if not ready:
                continue
            reply = self._call(name, 'followup', studies=ready)
            if reply is None:
                continue
            for main, target in reply['followups'].items():
                self._assign_followup(name, main, target, 'best main trial')
            for main, reason in reply['skipped'].items():
                target = self.cells[main].followup if main in self.cells else None
                if reason == 'already queued' and target is not None:
                    self._assign_followup(name, main, target, 'reconciled')
                else:
                    self._record({'type': 'followup_skip', 'study': main, 'reason': reason})

    def _assign_followup(self, server: str, main: str, target: str, reason: str) -> None:
        if self.assigned.get(target) == server:
            self.followup_handled[main] = target
            return
        self._record(
            {
                'type': 'assign',
                'study': target,
                'server': server,
                'kind': 'followup',
                'main': main,
                'reason': reason,
            }
        )

    def feed(self) -> None:
        for name, server in self.servers.items():
            view = self.views[name]
            if not view.reachable or name in self.ended:
                continue
            slots = int((view.state or {}).get('workers') or server.slots)
            backlog = view.backlog
            if backlog >= self.settings.low_water * slots:
                continue
            want = math.ceil(self.settings.high_water * slots - backlog)
            batch: list[str] = []
            opening: set[tuple[Any, ...]] = set()
            while len(batch) < want:
                study = self._next_study(name, opening)
                if study is None:
                    break
                batch.append(study)
            if not batch:
                continue
            if self._call(name, 'append', studies=batch) is None:
                self.queues[name].extendleft(reversed(batch))
                continue
            for study in batch:
                self._record(
                    {
                        'type': 'assign',
                        'study': study,
                        'server': name,
                        'kind': 'main',
                        'reason': self.queue_reason.get(study, 'home'),
                    }
                )
            view.manifest_studies += len(batch)

    def _next_study(self, name: str, opening: set[tuple[Any, ...]]) -> str | None:
        """The first queued study whose cache group fits the server's disk budget.

        ``opening`` holds the groups this feeding batch already opens. With nothing left to
        queue, the server steals.
        """
        queue = self.queues[name]
        while queue and queue[0] in self.assigned:
            queue.popleft()
        if not queue:
            if self._steal(name, opening):
                return self._next_study(name, opening)
            return None
        for index, study in enumerate(queue):
            if study in self.assigned:
                continue
            group = self.cells[study].group
            if self._fits(name, group, opening):
                del queue[index]
                opening.add(group)
                return study
        return None

    def _steal(self, thief: str, opening: set[tuple[Any, ...]]) -> bool:
        """Move the movable rest of one cache group from the busiest server to the thief.

        Groups come from the back of the victim's queue (the work it would reach last), and groups
        the victim has not started go first, so a cache is rarely built twice.
        """
        victims = sorted(
            (name for name in self.servers if name != thief and self.queues[name]),
            key=lambda name: (-self.remaining_seconds(name), name),
        )
        for victim in victims:
            for allow_started in (False, True):
                if self._steal_from(thief, victim, opening, allow_started=allow_started):
                    return True
        return False

    def _steal_from(
        self,
        thief: str,
        victim: str,
        opening: set[tuple[Any, ...]],
        *,
        allow_started: bool,
    ) -> bool:
        server = self.servers[thief]
        now = self.clock()
        victim_left = self.remaining_seconds(victim)
        seen: set[tuple[Any, ...]] = set()
        for study in reversed(self.queues[victim]):
            group = self.cells[study].group
            if group in seen:
                continue
            seen.add(group)
            members = self.groups[group]
            if not allow_started and any(self.assigned.get(m) == victim for m in members):
                continue
            # The launcher trusts a cache it warmed for hours; never refill one it removed.
            if now - self.cleaned_at.get((thief, group), -math.inf) < FOLLOW_WARM_TTL_SECONDS:
                continue
            if not self._fits(thief, group, opening):
                continue
            movable = [
                member
                for member in self.queues[victim]
                if self.cells[member].group == group
                and member not in self.assigned
                and server.may_run(self.cells[member])
            ]
            if not movable:
                continue
            costs = [self.cells[m].cost * server.fold_time_factor for m in movable]
            if max(max(costs), sum(costs) / server.slots) >= victim_left:
                continue
            self._record(
                {
                    'type': 'move',
                    'from': victim,
                    'to': thief,
                    'group': list(group),
                    'studies': movable,
                }
            )
            return True
        return False

    def cleanup(self) -> None:
        """Remove a group's caches from a server once its share of the group is settled.

        Nothing of the group may be left in the server's queue; studies of the group still queued
        elsewhere would rebuild the cache if they were ever stolen back.
        """
        if not self.settings.cleanup:
            return
        for name, view in self.views.items():
            if not view.reachable:
                continue
            queued = {
                self.cells[study].group
                for study in self.queues[name]
                if study not in self.assigned
            }
            for group in sorted(self.open_groups[name] - queued):
                mine = [m for m in self.groups[group] if self.assigned.get(m) == name]
                if not all(self._settled(name, study) for study in mine):
                    continue
                reply = self._call(name, 'gc', studies=mine or self.groups[group][:1])
                if reply is not None:
                    self._record(
                        {
                            'type': 'cleanup',
                            'server': name,
                            'group': list(group),
                            'removed': len(reply['removed']),
                            'freed_gib': reply['freed_gib'],
                        }
                    )

    def _settled(self, name: str, study: str) -> bool:
        """A main study and its follow-up are finished on this server and need no caches."""
        view = self.views[name]
        if not view.finished(study) or study in view.running:
            return False
        if study in view.failed:
            return True
        target = self.followup_handled.get(study)
        if target is None:
            return False
        if target.startswith('skipped'):
            return True
        return view.finished(target) and target not in view.running

    def finish_servers(self) -> None:
        """Append ``#END`` once a server can never receive another study."""
        if any(study not in self.assigned for study in self.cells):
            return
        for name, view in self.views.items():
            if name in self.ended or not view.reachable:
                continue
            mine = [study for study, owner in self.assigned.items() if owner == name]
            mains = [study for study in mine if study in self.cells]
            if not all(view.finished(study) for study in mains):
                continue
            if not all(
                study in self.followup_handled for study in mains if study in view.completed
            ):
                continue
            if self._call(name, 'append', studies=[], end=True) is not None:
                self._record({'type': 'end', 'server': name})

    def done(self) -> bool:
        return all(
            name in self.ended and (view.state or {}).get('finished')
            for name, view in self.views.items()
        )

    # ---------------------------------------------------------------- projections

    def speed_correction(self, name: str) -> float:
        """Observed / predicted wall time of finished studies on this server."""
        server = self.servers[name]
        ratios = []
        for study, event in self.views[name].completed.items():
            cell = self.cells.get(study) or self.followups.get(study)
            if cell is None or not event.get('seconds') or not event.get('trials'):
                continue
            predicted = cell.fold_seconds * server.fold_time_factor * cell.folds * event['trials']
            ratios.append(float(event['seconds']) / predicted)
        return statistics.median(ratios) if len(ratios) >= 5 else 1.0

    def remaining_seconds(self, name: str) -> float:
        """Projected wall seconds until this server finishes its share, at full occupancy."""
        server = self.servers[name]
        view = self.views[name]
        now = self.clock()
        running = view.running
        total = 0.0
        for study, owner in self.assigned.items():
            if owner != name or view.finished(study):
                continue
            cell = self.cells.get(study) or self.followups.get(study)
            if cell is None:
                continue
            predicted = cell.cost * server.fold_time_factor
            if study in running:
                elapsed = now - datetime.fromisoformat(running[study]).timestamp()
                total += max(predicted - elapsed, 0.1 * predicted)
            else:
                total += predicted
        for study in self.queues[name]:
            if study not in self.assigned:
                total += self.cells[study].cost * server.fold_time_factor
        for study, owner in self.assigned.items():
            cell = self.cells.get(study)
            if owner == name and cell is not None and cell.followup:
                if study not in self.followup_handled:
                    total += self.followups[cell.followup].cost * server.fold_time_factor
        for study in self.queues[name]:
            cell = self.cells[study]
            if study not in self.assigned and cell.followup:
                total += self.followups[cell.followup].cost * server.fold_time_factor
        slots = int((view.state or {}).get('workers') or server.slots)
        return total * self.speed_correction(name) / max(slots, 1)

    # ---------------------------------------------------------------- reports

    def status(self) -> dict[str, Any]:
        now = self.clock()
        servers = {}
        for name, view in self.views.items():
            remaining = self.remaining_seconds(name)
            mine = [study for study, owner in self.assigned.items() if owner == name]
            servers[name] = {
                'reachable': view.reachable,
                'error': view.error,
                'launcher_alive': view.launcher_alive,
                'supervisor_alive': view.supervisor_alive,
                'ended': name in self.ended,
                'workers': (view.state or {}).get('workers'),
                'running': len(view.running),
                'backlog': view.backlog,
                'queued': sum(1 for study in self.queues[name] if study not in self.assigned),
                'assigned': len(mine),
                'completed': len(view.completed),
                'failed': len(view.failed),
                'disk_free_gib': view.disk_free_gib,
                'open_cache_groups': len(self.open_groups[name]),
                'open_cache_gib': round(self.open_cache_gib(name), 1),
                'peak_cache_gib': round(self.peak_cache_gib[name], 1),
                'gpu_used_mib': [gpu['used_mib'] for gpu in view.gpus],
                'speed_correction': round(self.speed_correction(name), 3),
                'remaining_hours': round(remaining / 3600, 1),
                'projected_finish': datetime.fromtimestamp(now + remaining, UTC).isoformat(
                    timespec='minutes'
                ),
            }
        main_done = self._finished(self.cells)
        follow_done = self._finished(self.followups)
        skips = collections.Counter(
            value.removeprefix('skipped: ')
            for value in self.followup_handled.values()
            if value.startswith('skipped')
        )
        skips.pop('none', None)
        return {
            'updated_at': self.now_iso(),
            'campaign': self.settings.campaign,
            'main': {
                'total': len(self.cells),
                'assigned': sum(1 for study in self.cells if study in self.assigned),
                **main_done,
            },
            'followup': {
                'total': len(self.followups),
                'assigned': sum(1 for study in self.followups if study in self.assigned),
                'not_queued': dict(skips),
                **follow_done,
            },
            'servers': servers,
            'projected_finish': max(entry['projected_finish'] for entry in servers.values()),
        }

    def _finished(self, cells: dict[str, Cell]) -> dict[str, int]:
        complete = partial = failed = 0
        for study, cell in cells.items():
            view = self.views.get(self.assigned.get(study, ''))
            if view is None:
                continue
            event = view.completed.get(study)
            if event is not None:
                if int(event.get('complete', 0)) >= cell.trials:
                    complete += 1
                else:
                    partial += 1
            elif study in view.failed:
                failed += 1
        return {'complete': complete, 'partial': partial, 'failed': failed}

    def coverage_markdown(self, status: dict[str, Any]) -> str:
        lines = [
            f'# {self.settings.campaign} coverage ({status["updated_at"]})',
            '',
            f'Projected finish: {status["projected_finish"]} UTC',
            '',
            '| ratio | model | main complete | main partial | main failed | follow-up complete |',
            '|---|---|---|---|---|---|',
        ]
        table: dict[tuple[float, str], collections.Counter[str]] = collections.defaultdict(
            collections.Counter
        )
        for study, cell in self.cells.items():
            counts = table[(cell.ratio, cell.model)]
            counts['total'] += 1
            counts[self._outcome(study, cell)] += 1
            if cell.followup:
                counts['follow_total'] += 1
                follow = self.followups[cell.followup]
                if self._outcome(cell.followup, follow) == 'complete':
                    counts['follow_complete'] += 1
        for (ratio, model), counts in sorted(
            table.items(), key=lambda item: (-item[0][0], item[0][1])
        ):
            follow = (
                f'{counts["follow_complete"]}/{counts["follow_total"]}'
                if counts['follow_total']
                else 'n/a'
            )
            lines.append(
                f'| {ratio} | {model} | {counts["complete"]}/{counts["total"]} | '
                f'{counts["partial"]} | {counts["failed"]} | {follow} |'
            )
        lines += [
            '',
            '| server | reachable | launcher | running | backlog | queued | done | '
            'failed | disk free (GiB) | remaining (h) | finish (UTC) |',
            '|---|---|---|---|---|---|---|---|---|---|---|',
        ]
        for name, entry in status['servers'].items():
            lines.append(
                f'| {name} | {entry["reachable"]} | {entry["launcher_alive"]} | '
                f'{entry["running"]} | {entry["backlog"]} | {entry["queued"]} | '
                f'{entry["completed"]} | {entry["failed"]} | '
                f'{entry["disk_free_gib"].get("data_root")} | {entry["remaining_hours"]} | '
                f'{entry["projected_finish"]} |'
            )
        return '\n'.join(lines) + '\n'

    def _outcome(self, study: str, cell: Cell) -> str:
        view = self.views.get(self.assigned.get(study, ''))
        if view is None:
            return 'unassigned'
        event = view.completed.get(study)
        if event is not None:
            return 'complete' if int(event.get('complete', 0)) >= cell.trials else 'partial'
        return 'failed' if study in view.failed else 'open'

    def report(self) -> dict[str, Any]:
        status = self.status()
        _atomic_write(self.root / 'status.json', json.dumps(status, indent=1))
        _atomic_write(self.root / 'coverage.md', self.coverage_markdown(status))
        with (self.root / 'status_history.jsonl').open('a') as handle:
            summary = {key: status[key] for key in ('updated_at', 'main', 'followup')}
            summary['projected_finish'] = status['projected_finish']
            handle.write(json.dumps(summary) + '\n')
        self.last_report = self.clock()
        return status

    def cycle(self, *, feed: bool = True) -> None:
        self.poll()
        self.queue_followups()
        if feed:
            self.feed()
        self.cleanup()
        self.finish_servers()
        if self.clock() - self.last_report >= self.settings.report_seconds:
            self.report()

    def run(self, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self.reconcile()
        while True:
            self.cycle()
            if self.done():
                self.report()
                print(f'[coordinator] {utc_now()} campaign finished', flush=True)
                return
            sleep(self.settings.poll_seconds)


# --------------------------------------------------------------------------- collection


def merge_exports(
    coordinator: Coordinator, snapshots: dict[str, Path], destination: Path
) -> dict[str, Any]:
    """Merge server exports, keeping each study's rows from the server it is assigned to."""
    from scripts.optuna_search import OptunaSearchConfig, _best_rows

    summary: dict[str, Any] = {}
    for config_path, cells in (
        (coordinator.settings.main_config, coordinator.cells),
        (coordinator.settings.followup_config, coordinator.followups),
    ):
        config = OptunaSearchConfig.from_yaml(REPO_ROOT / config_path)
        stem = Path(config_path).stem
        frames = []
        for name, snapshot in snapshots.items():
            path = snapshot / stem / 'trials.csv'
            if not path.exists():
                continue
            trials = pd.read_csv(path)
            owned = {s for s, owner in coordinator.assigned.items() if owner == name}
            frames.append(trials.loc[trials['study_name'].isin(owned)])
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        target = destination / stem
        target.mkdir(parents=True, exist_ok=True)
        merged.to_csv(target / 'trials.csv', index=False)
        complete = (
            merged.loc[merged['state'] == 'COMPLETE'].groupby('study_name').size()
            if not merged.empty
            else pd.Series(dtype=int)
        )
        if not merged.empty:
            _best_rows(merged, config.direction).to_csv(target / 'best_trials.csv', index=False)
        short = sorted(study for study in cells if complete.get(study, 0) < config.n_trials)
        summary[stem] = {
            'studies': len(cells),
            'with_all_trials_complete': len(cells) - len(short),
            'incomplete': short,
        }
    _atomic_write(destination / 'validation.json', json.dumps(summary, indent=1))
    return summary


def collect(coordinator: Coordinator) -> dict[str, Any]:
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    snapshots: dict[str, Path] = {}
    for name, server in coordinator.servers.items():
        reply = coordinator._call(name, 'export', stamp=stamp, jobs_per_gpu=server.jobs_per_gpu)
        if reply is None:
            raise AgentError(f'Export failed on {name}: {coordinator.views[name].error}')
        local = coordinator.root / 'servers' / name / stamp
        local.parent.mkdir(parents=True, exist_ok=True)
        if server.is_local:
            shutil.copytree(reply['snapshot'], local)
        else:
            subprocess.run(  # nosec B603 B607
                ['rsync', '-a', f'{server.host}:{reply["snapshot"]}/', f'{local}/'],
                check=True,
                timeout=7200,
            )
        snapshots[name] = local
    return merge_exports(coordinator, snapshots, coordinator.root / 'merged' / stamp)


# --------------------------------------------------------------------------- CLI


def load_cells(settings: Settings) -> tuple[list[Cell], dict[str, Cell]]:
    costs = CostModel.from_csv(
        settings.cost_table,
        cap=settings.fold_seconds_cap,
        default=settings.default_fold_seconds,
    )
    caches = CacheSizes.from_csv(settings.cache_table, default=settings.default_fold_cache_gib)
    return campaign_cells(settings, costs, caches)


def build(settings: Settings, agents: dict[str, Agent] | None = None) -> Coordinator:
    cells, followups = load_cells(settings)
    if agents is None:
        agents = {name: ShellAgent(server) for name, server in settings.servers.items()}
    return Coordinator(settings, cells, followups, agents)


def plan_summary(settings: Settings) -> str:
    cells, followups = load_cells(settings)
    by_study = {cell.study: cell for cell in cells}
    lines = [f'{len(cells)} main studies, {len(followups)} follow-ups']
    for name, queue in plan_homes(cells, settings.servers).items():
        server = settings.servers[name]
        hours = (
            sum(
                by_study[s].cost
                + (followups[by_study[s].followup].cost if by_study[s].followup else 0.0)
                for s in queue
            )
            * server.fold_time_factor
            / server.slots
            / 3600
        )
        datasets = sorted({by_study[s].dataset for s in queue})
        ratios = sorted({by_study[s].ratio for s in queue}, reverse=True)
        lines.append(
            f'{name}: {len(queue)} studies, {server.slots} slots, ~{hours:.0f} h at full '
            f'occupancy; ratios {ratios}; datasets {datasets}'
        )
    return '\n'.join(lines)


def preflight(settings: Settings, names: Sequence[str]) -> dict[str, Any]:
    """Run every server's preflight; packages are reported as differences from the first one."""
    replies = {}
    for name in names:
        server = settings.servers[name]
        payload = {
            'run_root': server.run_root,
            'data_root': server.data_root,
            'configs': [settings.main_config, settings.followup_config],
            'env': server.env,
        }
        try:
            replies[name] = ShellAgent(server, timeout=3600).call('preflight', payload)
        except AgentError as error:
            replies[name] = {'error': str(error)}
    reference = next((reply['packages'] for reply in replies.values() if 'packages' in reply), [])
    for reply in replies.values():
        packages = reply.pop('packages', None)
        if packages is not None:
            reply['packages_missing'] = sorted(set(reference) - set(packages))
            reply['packages_extra'] = sorted(set(packages) - set(reference))
    return replies


def simulate_campaign(settings: Settings, root: Path) -> dict[str, Any]:
    """Run the coordinator against simulated servers and report when each one finishes."""
    from scripts.campaign.simulate import SimulatedCluster

    cells, followups = load_cells(settings)
    cluster = SimulatedCluster(settings.servers, {cell.study: cell for cell in cells}, followups)
    coordinator = Coordinator(
        replace(settings, coordinator_root=root),
        cells,
        followups,
        cluster.agents(),
        clock=cluster.clock,
    )
    coordinator.run(sleep=cluster.advance)
    moves = collections.Counter()
    for line in coordinator.log_path.read_text().splitlines():
        record = json.loads(line)
        if record['type'] == 'move':
            moves[f'{record["from"]}->{record["to"]}'] += len(record['studies'])
    finish = {}
    for name, sim in cluster.servers.items():
        last = max((event['time'] for event in sim.events), default=None)
        studies = sum(1 for owner in coordinator.assigned.values() if owner == name)
        finish[name] = {
            'studies': studies,
            'finish': last,
            'peak_cache_gib': round(coordinator.peak_cache_gib[name], 1),
        }
    return {'servers': finish, 'moved_studies': dict(moves)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default=str(REPO_ROOT / 'scripts/campaign/servers.yaml'))
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan', help='Print the home placement and projected hours')
    plan.add_argument(
        '--simulate',
        type=Path,
        metavar='DIR',
        help='Also simulate the campaign with work stealing, writing coordinator files to DIR',
    )
    run = commands.add_parser('run', help='Feed, follow up, clean up and report until done')
    run.add_argument('--once', action='store_true', help='Run a single cycle')
    run.add_argument('--no-feed', action='store_true', help='Do not append new main studies')
    commands.add_parser('status', help='Poll every server and write the reports')
    for name, description in (
        ('start', 'Start server supervisors'),
        ('stop', 'Stop server supervisors'),
        ('preflight', 'Check packages, GPUs, W&B, HF data, grouping and disk on each server'),
    ):
        sub = commands.add_parser(name, help=description)
        sub.add_argument('--servers', nargs='+')
    commands.add_parser('collect', help='Export, copy and merge every server result')
    table = commands.add_parser('cost-table', help='Build the fold-seconds table from ledgers')
    table.add_argument('--ledger', nargs='+', required=True, type=Path)
    table.add_argument('--since', default='')
    table.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == 'cost-table':
        frame = build_cost_table(args.ledger, args.since)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(f'{len(frame)} rows, {int(frame["count"].sum())} folds -> {args.out}')
        return 0

    settings = Settings.from_yaml(args.settings)
    if args.command == 'plan':
        print(plan_summary(settings))
        if args.simulate:
            print(json.dumps(simulate_campaign(settings, args.simulate), indent=1))
        return 0
    if args.command == 'preflight':
        print(json.dumps(preflight(settings, args.servers or list(settings.servers)), indent=1))
        return 0
    coordinator = build(settings)
    if args.command in ('start', 'stop'):
        for name in args.servers or list(settings.servers):
            server = settings.servers[name]
            extra = {}
            if args.command == 'start':
                extra = {
                    'python': server.python,
                    'gpus': list(server.gpus),
                    'jobs_per_gpu': server.jobs_per_gpu,
                    'warmup_jobs': server.warmup_jobs,
                    'min_free_gpu_mib': server.min_free_gpu_mib,
                    'oom_retries': server.oom_retries,
                    'oom_min_free_gpu_mib': server.oom_min_free_gpu_mib,
                    'env': server.env,
                }
            print(name, coordinator._call(name, args.command, **extra), flush=True)
        return 0
    if args.command == 'status':
        coordinator.poll()
        print(json.dumps(coordinator.report(), indent=1))
        return 0
    if args.command == 'collect':
        print(json.dumps(collect(coordinator), indent=1, default=str)[:20000])
        return 0
    if args.once:
        coordinator.reconcile()
        coordinator.cycle(feed=not args.no_feed)
        coordinator.report()
        return 0
    coordinator.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())

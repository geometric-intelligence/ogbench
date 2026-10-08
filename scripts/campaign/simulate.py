"""Simulated servers for the campaign coordinator.

Each simulated server behaves like a follow-mode launcher behind ``remote_agent.py``: it runs
manifest studies in order on its slots, a study taking its cost times the server's fold-time
factor. The real ``Coordinator`` drives it on a simulated clock, which projects the campaign's
finish (``coordinator.py plan --simulate``) and tests the coordinator end to end.
"""

from __future__ import annotations

import collections
import heapq
import itertools
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from scripts.campaign.coordinator import Cell, Server

EPOCH = 1_900_000_000.0


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(EPOCH + seconds, UTC).isoformat(timespec='seconds')


@dataclass
class SimulatedServer:
    server: Server
    manifest: list[str] = field(default_factory=list)
    end: bool = False
    pending: collections.deque[str] = field(default_factory=collections.deque)
    running: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    cleaned: list[list[str]] = field(default_factory=list)
    down: bool = False

    @property
    def finished(self) -> bool:
        return self.end and not self.pending and not self.running


class SimulatedCluster:
    """Simulated servers sharing one clock; ``agents()`` plugs them into a Coordinator."""

    def __init__(
        self,
        servers: dict[str, Server],
        cells: dict[str, Cell],
        followups: dict[str, Cell],
        *,
        failing: frozenset[str] = frozenset(),
    ) -> None:
        self.servers = {name: SimulatedServer(server) for name, server in servers.items()}
        self.cells = {**cells, **followups}
        self.failing = failing
        self.now = 0.0
        self._finishing: list[tuple[float, int, str, str]] = []
        self._order = itertools.count()

    def clock(self) -> float:
        return EPOCH + self.now

    def agents(self) -> dict[str, _SimulatedAgent]:
        return {name: _SimulatedAgent(self, name) for name in self.servers}

    def advance(self, seconds: float) -> None:
        horizon = self.now + seconds
        self._start_all()
        while self._finishing and self._finishing[0][0] <= horizon:
            finish, _, name, study = heapq.heappop(self._finishing)
            self.now = finish
            sim = self.servers[name]
            started = sim.running.pop(study)
            cell = self.cells[study]
            if study in self.failing:
                summary = {'trials': cell.trials, 'complete': 0, 'failed': cell.trials}
            else:
                summary = {'trials': cell.trials, 'complete': cell.trials, 'failed': 0}
            sim.events.append(
                {
                    'event': 'completed',
                    'study': study,
                    'seconds': round(finish - started),
                    'time': _iso(finish),
                    **summary,
                }
            )
            self._start(name)
        self.now = horizon

    def _start_all(self) -> None:
        for name in self.servers:
            self._start(name)

    def _start(self, name: str) -> None:
        sim = self.servers[name]
        while sim.pending and len(sim.running) < sim.server.slots:
            study = sim.pending.popleft()
            sim.running[study] = self.now
            duration = self.cells[study].seconds_on(sim.server.fold_time_factor)
            heapq.heappush(self._finishing, (self.now + duration, next(self._order), name, study))
            sim.events.append({'event': 'started', 'study': study, 'time': _iso(self.now)})

    def _append(self, name: str, studies: list[str], end: bool) -> list[str]:
        sim = self.servers[name]
        new = [study for study in dict.fromkeys(studies) if study not in sim.manifest]
        if sim.end and new:
            raise ValueError(f'{name} manifest already ended')
        sim.manifest.extend(new)
        sim.pending.extend(new)
        sim.end = sim.end or end
        self._start(name)
        return new

    def handle(self, name: str, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        sim = self.servers[name]
        if sim.down:
            from scripts.campaign.coordinator import AgentError

            raise AgentError(f'{name} is unreachable')
        if command == 'manifest':
            return {'studies': list(sim.manifest), 'end': sim.end}
        if command == 'append':
            new = self._append(name, payload.get('studies', []), bool(payload.get('end')))
            return {'appended': new, 'studies': len(sim.manifest), 'end': sim.end}
        if command == 'status':
            offset = int(payload.get('events_offset', 0))
            return {
                'manifest': {'studies': len(sim.manifest), 'end': sim.end},
                'state': {
                    'workers': sim.server.slots,
                    'known': len(sim.manifest),
                    'pending': list(sim.pending),
                    'warming': [],
                    'ready': [],
                    'running': {study: _iso(start) for study, start in sim.running.items()},
                    'finished': sim.finished,
                },
                'launcher_alive': not sim.finished,
                'supervisor_alive': not sim.finished,
                'events': sim.events[offset:],
                'events_offset': len(sim.events),
                'events_reset': False,
                'disk_free_gib': {'data_root': 100.0, 'run_root': 100.0},
                'gpus': [],
            }
        if command == 'followup':
            followups, skipped = {}, {}
            for main in payload['studies']:
                cell = self.cells[main]
                if cell.followup is None:
                    skipped[main] = 'no follow-up cell'
                elif cell.followup in sim.manifest:
                    skipped[main] = 'already queued'
                elif main in self.failing:
                    skipped[main] = 'no complete trial'
                else:
                    followups[main] = cell.followup
            appended = self._append(name, list(followups.values()), end=False)
            return {'followups': followups, 'appended': appended, 'skipped': skipped}
        if command == 'gc':
            sim.cleaned.append(list(payload['studies']))
            return {'removed': [], 'missing': [], 'freed_gib': 0.0}
        raise ValueError(f'Simulated servers do not support {command}')


@dataclass
class _SimulatedAgent:
    cluster: SimulatedCluster
    name: str

    def call(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.cluster.handle(self.name, command, payload)

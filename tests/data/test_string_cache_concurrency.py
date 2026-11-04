"""Tests for process-safe STRING cache artifacts."""

from __future__ import annotations

import gzip
import json
import multiprocessing
import os
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from ogbench.data.adjacency import string as string_module
from ogbench.data.adjacency.string import STRINGAdjacencyBuilder


def _increment_under_lock(lock_path: str, counter_path: str, repeats: int) -> None:
    """Increment a shared counter while holding the production file lock."""
    for _ in range(repeats):
        with string_module._exclusive_file_lock(lock_path):
            path = Path(counter_path)
            value = int(path.read_text())
            time.sleep(0.001)
            path.write_text(str(value + 1))


def _gzip_bytes(content: bytes) -> bytes:
    return gzip.compress(content)


def test_valid_gzip_rejects_truncated_stream(tmp_path: Path) -> None:
    valid = tmp_path / 'valid.gz'
    valid.write_bytes(_gzip_bytes(b'complete payload'))
    truncated = tmp_path / 'truncated.gz'
    truncated.write_bytes(valid.read_bytes()[:-4])

    assert string_module._valid_gzip(str(valid))
    assert not string_module._valid_gzip(str(truncated))


def test_atomic_json_dump_replaces_existing_document(tmp_path: Path) -> None:
    path = tmp_path / 'cache.json'
    path.write_text('{"stale": true}')

    string_module._atomic_json_dump({'fresh': [1, 2, 3]}, str(path))

    assert json.loads(path.read_text()) == {'fresh': [1, 2, 3]}
    assert list(tmp_path.glob('cache.json.tmp-*')) == []


def test_download_repairs_truncated_gzip_and_reuses_completed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'artifact.txt.gz'
    path.write_bytes(b'not a gzip stream')
    payload = _gzip_bytes(b'header\nrow\n')
    response = Mock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [payload[:5], payload[5:]]
    request = Mock(return_value=response)
    monkeypatch.setattr(string_module.requests, 'get', request)
    builder = STRINGAdjacencyBuilder(cache_dir=str(tmp_path))

    builder._ensure_gzip_download(str(path), 'https://example.test/artifact', 'artifact')
    builder._ensure_gzip_download(str(path), 'https://example.test/artifact', 'artifact')

    assert path.read_bytes() == payload
    assert string_module._valid_gzip(str(path))
    assert Path(f'{path}.complete').exists()
    request.assert_called_once()


def test_corrupt_interaction_json_is_rebuilt_from_bulk_file(tmp_path: Path) -> None:
    builder = STRINGAdjacencyBuilder(cache_dir=str(tmp_path))
    ids_hash = 'testhash'
    cache = tmp_path / f'interactions_{builder.species}_{ids_hash}.json'
    cache.write_text('{broken')
    bulk = tmp_path / f'{builder.species}.protein.links.v12.0.txt.gz'
    with gzip.open(bulk, 'wt') as handle:
        handle.write('protein1 protein2 combined_score\n')
        handle.write('9606.A 9606.B 750\n')
    Path(f'{bulk}.complete').touch()

    interactions = builder._fetch_interactions_locked(['9606.A', '9606.B'], ids_hash)

    assert interactions == [{'stringId_A': '9606.A', 'stringId_B': '9606.B', 'score': 0.75}]
    assert json.loads(cache.read_text()) == interactions


@pytest.mark.skipif(os.name != 'posix', reason='fcntl locks require POSIX')
@pytest.mark.filterwarnings('ignore:This process .* is multi-threaded:DeprecationWarning')
def test_file_lock_serializes_processes(tmp_path: Path) -> None:
    counter = tmp_path / 'counter.txt'
    counter.write_text('0')
    lock = tmp_path / 'counter.lock'
    context = multiprocessing.get_context('fork')
    processes = [
        context.Process(
            target=_increment_under_lock,
            args=(str(lock), str(counter), 10),
        )
        for _ in range(4)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    assert counter.read_text() == '40'

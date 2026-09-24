"""STRING PPI-based adjacency matrix builder."""

import fcntl
import hashlib
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from ogbench.data.adjacency.base import AbstractAdjacencyBuilder

logger = logging.getLogger(__name__)


@contextmanager
def _exclusive_file_lock(path: str) -> Iterator[None]:
    """Serialize access to shared STRING cache artifacts across processes."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _valid_gzip(path: str) -> bool:
    """Fully validate a gzip stream, including its end-of-stream marker."""
    import gzip

    try:
        with gzip.open(path, 'rb') as handle:
            while handle.read(1024 * 1024):
                pass
    except (EOFError, OSError):
        return False
    return True


def _atomic_json_dump(value: Any, path: str) -> None:
    temporary = f'{path}.tmp-{os.getpid()}'
    try:
        with open(temporary, 'w') as handle:
            json.dump(value, handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class STRINGAdjacencyBuilder(AbstractAdjacencyBuilder):
    """Build adjacency matrix using STRING protein-protein interaction network.

    Nodes are identified by node_id (e.g. SomaID or Entrez Gene ID).
    Edges are derived from STRING interactions via a node_id → STRING identifier mapping.

    Assumptions:
    - map_df has columns 'node_id' and 'string_id'
    - 'string_id' may be pipe-delimited for multi-protein complex aptamers
    - 'string_id' can be any format STRING accepts (UniProt, Entrez, gene symbol, etc.)
    - Medium confidence threshold: combined score >= 400 (out of 1000)
    - Returns continuous scores normalized to [0, 1] — binarization handled upstream
    - Nodes with no STRING mapping or no interactions become isolated nodes
    - Results are cached to avoid repeated API calls across folds/runs
    """

    STRING_API = 'https://string-db.org/api/json'
    CALLER_ID = 'ogbench_ppi_graph'

    def __init__(
        self,
        species: int = 9606,
        cache_dir: str = 'temp_data/string_cache',
        string_data_dir: str | None = None,
    ) -> None:
        """
        Args:
            species: NCBI taxonomy ID (9606 = human)
            cache_dir: Directory to cache STRING API responses
            string_data_dir: Optional path to a directory containing pre-downloaded
                STRING bulk files (e.g. ``9606.protein.aliases.v12.0.txt.gz``).
                When set, files are read from here instead of downloading.
        """
        self.species = species
        self.cache_dir = cache_dir
        self.string_data_dir = string_data_dir
        os.makedirs(cache_dir, exist_ok=True)

    def build(self, node_features: pd.DataFrame, map_df: pd.DataFrame | None = None) -> np.ndarray:
        """Build adjacency matrix using STRING PPI.

        Args:
            node_features: DataFrame of shape (n_samples, n_nodes).
                           Columns are node_ids matching 'node_id' in map_df.
            map_df: DataFrame with columns 'node_id' and 'string_id'.
                    'string_id' can be any identifier format STRING accepts.
                    Pipe-delimited values (e.g. 'P02671|P02675') are supported for complexes.

        Returns:
            Symmetric adjacency matrix of shape (n_nodes, n_nodes).
            Values are STRING combined scores normalized to [0, 1].
            Zero means no interaction above threshold.
        """
        if map_df is None:
            raise ValueError('map_df is required for STRING adjacency builder')

        node_ids = list(node_features.columns)
        n = len(node_ids)

        # 1. Load node_id → list of STRING identifiers
        #    Handles pipe-delimited complexes: "P02671|P02675|P02679" → ["P02671", "P02675", "P02679"]
        node_to_ids = self._load_mapping(node_ids, map_df)
        logger.info('Mapped %d/%d nodes to STRING identifiers', len(node_to_ids), n)

        # 2. Collect all unique identifiers across all nodes
        all_ids = list({i for ids in node_to_ids.values() for i in ids})
        logger.info('Querying STRING for %d unique identifiers', len(all_ids))

        # 3. Map identifiers → STRING internal IDs via API
        #    STRING auto-detects identifier format (UniProt, Entrez, symbol, etc.)
        #    limit=1 takes the single best hit per query
        id_to_string_id = self._map_to_string_ids(all_ids)
        logger.info('STRING ID mapping: %d/%d resolved', len(id_to_string_id), len(all_ids))

        # 4. Fetch all interactions among the resolved STRING IDs
        string_ids = list(id_to_string_id.values())
        interactions = self._fetch_interactions(string_ids)
        logger.info('Retrieved %d interactions', len(interactions))

        # 5. Build reverse map: STRING internal ID → list of original identifiers
        string_id_to_id: dict[str, list[str]] = {}
        for orig_id, string_id in id_to_string_id.items():
            if string_id not in string_id_to_id:
                string_id_to_id[string_id] = []
            string_id_to_id[string_id].append(orig_id)

        # 6. Build identifier-pair → normalized score lookup
        #    Key is always sorted tuple (a, b) with a < b for consistent lookup
        #    If multiple STRING edges map to the same pair, keep the max score
        id_interactions: dict[tuple[str, str], float] = {}
        for item in interactions:
            sid_a = item.get('stringId_A', '')
            sid_b = item.get('stringId_B', '')
            score = item.get('score', 0)
            for ia in string_id_to_id.get(sid_a, []):
                for ib in string_id_to_id.get(sid_b, []):
                    if ia != ib:
                        key = (ia, ib) if ia < ib else (ib, ia)
                        id_interactions[key] = max(
                            id_interactions.get(key, 0.0), score
                        )  # handles multiple edges between the same pair of nodes

        # 7. Build node-level adjacency matrix
        # Convert interaction lookup to a DataFrame for vectorized merge
        if not id_interactions:
            return np.zeros((n, n), dtype=np.float32)

        interactions_df = pd.DataFrame(
            [(ia, ib, score) for (ia, ib), score in id_interactions.items()],
            columns=['id_a', 'id_b', 'score'],
        )

        # Explode node → identifier mapping into a flat DataFrame
        node_id_map = pd.DataFrame(
            [
                (ident, idx)
                for idx, node in enumerate(node_ids)
                for ident in node_to_ids.get(node, [])
            ],
            columns=['identifier', 'node_idx'],
        )

        # Join interactions to node indices via identifier
        merged = interactions_df.merge(
            node_id_map.rename(columns={'identifier': 'id_a', 'node_idx': 'idx_a'}),
            on='id_a',
        ).merge(
            node_id_map.rename(columns={'identifier': 'id_b', 'node_idx': 'idx_b'}),
            on='id_b',
        )

        # For complex aptamers: keep max score per node pair
        merged = merged[merged['idx_a'] != merged['idx_b']]
        merged = merged.groupby(['idx_a', 'idx_b'])['score'].max().reset_index()

        # Fill adjacency matrix
        adj = np.zeros((n, n), dtype=np.float32)
        adj[merged['idx_a'].values, merged['idx_b'].values] = merged['score'].values
        adj[merged['idx_b'].values, merged['idx_a'].values] = merged['score'].values  # symmetric

        n_edges = int((adj > 0).sum() // 2)
        isolated = int((adj.sum(axis=1) == 0).sum())
        logger.info(
            'Adjacency matrix: %d nodes, %d edges, %d isolated nodes',
            n,
            n_edges,
            isolated,
        )

        return adj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_mapping(self, node_ids: list[str], map_df: pd.DataFrame) -> dict[str, list[str]]:
        """Load node_id → list of STRING identifiers from map_df.

        Only returns entries for node_ids present in node_features columns. Splits pipe-delimited
        complex entries into individual identifiers.
        """
        map_df['node_id'] = map_df['node_id'].astype(str)
        map_df = map_df[map_df['node_id'].isin(set(node_ids))]

        result: dict[str, list[str]] = {}
        for _, row in map_df.iterrows():
            raw = str(row['string_id']) if pd.notna(row['string_id']) else ''
            ids = [x.strip() for x in raw.split('|') if x.strip()]
            if ids:
                result[row['node_id']] = ids

        unmapped = len(node_ids) - len(result)
        if unmapped > 0:
            logger.warning('%d nodes have no STRING mapping — will be isolated', unmapped)

        return result

    def _map_to_string_ids(self, identifiers: list[str]) -> dict[str, str]:
        lock_path = os.path.join(self.cache_dir, f'string_id_map_{self.species}.lock')
        with _exclusive_file_lock(lock_path):
            return self._map_to_string_ids_locked(identifiers)

    def _map_to_string_ids_locked(self, identifiers: list[str]) -> dict[str, str]:
        """Map identifiers to STRING internal IDs using local alias file.

        Downloads STRING alias file once — no API dependency.
        """
        import gzip

        cache_file = os.path.join(self.cache_dir, f'string_id_map_{self.species}.json')

        cached: dict[str, str] = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning('Ignoring corrupt STRING ID cache: %s', cache_file)

        to_query = [x for x in identifiers if x not in cached]

        if to_query:
            alias_name = f'{self.species}.protein.aliases.v12.0.txt.gz'
            alias_file = os.path.join(self.cache_dir, alias_name)
            if not os.path.exists(alias_file) and self.string_data_dir:
                local = os.path.join(self.string_data_dir, alias_name)
                if os.path.exists(local):
                    alias_file = local
                    logger.info('Using local alias file: %s', alias_file)
            if not os.path.exists(alias_file):
                url = f'https://stringdb-downloads.org/download/protein.aliases.v12.0/{alias_name}'
                self._ensure_gzip_download(alias_file, url, 'STRING alias file')
            elif os.path.dirname(alias_file) == self.cache_dir:
                url = f'https://stringdb-downloads.org/download/protein.aliases.v12.0/{alias_name}'
                self._ensure_gzip_download(alias_file, url, 'STRING alias file')

            query_set = set(to_query)
            logger.info('Mapping %d identifiers from alias file...', len(to_query))

            with gzip.open(alias_file, 'rt') as f:
                next(f)  # skip header: string_protein_id alias source
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    string_id, alias = parts[0], parts[1]
                    if alias in query_set and alias not in cached:
                        cached[alias] = string_id

            # Mark permanently unmappable identifiers with sentinel
            for x in to_query:
                if x not in cached:
                    cached[x] = ''

            _atomic_json_dump(cached, cache_file)

            resolved = sum(1 for x in to_query if cached.get(x))
            logger.info('Resolved %d/%d identifiers', resolved, len(to_query))

        return {x: cached[x] for x in identifiers if cached.get(x)}

    def _fetch_interactions(self, string_ids: list[str]) -> list[dict]:
        # Per-dataset cache — avoids re-parsing bulk file on repeated runs
        ids_hash = hashlib.md5(
            json.dumps(sorted(string_ids)).encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]
        lock_path = os.path.join(self.cache_dir, f'interactions_{self.species}_{ids_hash}.lock')
        with _exclusive_file_lock(lock_path):
            return self._fetch_interactions_locked(string_ids, ids_hash)

    def _fetch_interactions_locked(self, string_ids: list[str], ids_hash: str) -> list[dict]:
        """Read or construct one interaction cache while holding its lock."""
        cache_file = os.path.join(
            self.cache_dir,
            f'interactions_{self.species}_{ids_hash}.json',
        )

        if os.path.exists(cache_file):
            try:
                logger.info('Loading interactions from cache...')
                with open(cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning('Ignoring corrupt interaction cache: %s', cache_file)

        bulk_name = f'{self.species}.protein.links.v12.0.txt.gz'
        bulk_file = os.path.join(self.cache_dir, bulk_name)
        if not os.path.exists(bulk_file) and self.string_data_dir:
            local = os.path.join(self.string_data_dir, bulk_name)
            if os.path.exists(local):
                bulk_file = local
                logger.info('Using local bulk file: %s', bulk_file)
        if not os.path.exists(bulk_file):
            url = f'https://stringdb-downloads.org/download/protein.links.v12.0/{bulk_name}'
            self._ensure_gzip_download(bulk_file, url, 'STRING bulk interaction file (~100MB)')
        elif os.path.dirname(bulk_file) == self.cache_dir:
            url = f'https://stringdb-downloads.org/download/protein.links.v12.0/{bulk_name}'
            self._ensure_gzip_download(bulk_file, url, 'STRING bulk interaction file (~100MB)')

        # Parse bulk file and filter to our proteins
        import gzip

        string_id_set = set(string_ids)
        interactions = []

        logger.info('Filtering STRING bulk file for our proteins...')
        with gzip.open(bulk_file, 'rt') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) != 3:
                    continue
                sid_a, sid_b, score_str = parts
                if sid_a in string_id_set and sid_b in string_id_set:
                    score = int(score_str)
                    interactions.append(
                        {
                            'stringId_A': sid_a,
                            'stringId_B': sid_b,
                            'score': score / 1000.0,  # normalize to [0, 1]
                        }
                    )

        logger.info('Found %d interactions.', len(interactions))

        # Cache the filtered result so we never parse the bulk file again for this config
        _atomic_json_dump(interactions, cache_file)

        return interactions

    def _ensure_gzip_download(self, path: str, url: str, label: str) -> None:
        """Validate or atomically download a shared gzip cache artifact."""
        marker = f'{path}.complete'
        with _exclusive_file_lock(f'{path}.download.lock'):
            if os.path.exists(path):
                if os.path.exists(marker) or _valid_gzip(path):
                    Path(marker).touch()
                    return
                logger.warning('Removing truncated %s: %s', label, path)
                os.remove(path)
                if os.path.exists(marker):
                    os.remove(marker)

            temporary = f'{path}.tmp-{os.getpid()}'
            logger.info('Downloading %s...', label)
            try:
                response = requests.get(url, timeout=300, stream=True)
                response.raise_for_status()
                with open(temporary, 'wb') as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)
                if not _valid_gzip(temporary):
                    raise EOFError(f'Downloaded {label} is not a complete gzip stream')
                os.replace(temporary, path)
                Path(marker).touch()
                logger.info('%s download complete.', label)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)

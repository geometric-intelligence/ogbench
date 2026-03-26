"""STRING PPI-based adjacency matrix builder."""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import requests

from ogbench.data.adjacency.base import AbstractAdjacencyBuilder


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
        score_threshold: int = 400,
        species: int = 9606,
        cache_dir: str = 'temp_data/string_cache',
    ) -> None:
        """
        Args:
            score_threshold: STRING combined score threshold (400=medium, 700=high, 900=highest)
            species: NCBI taxonomy ID (9606 = human)
            cache_dir: Directory to cache STRING API responses
        """
        self.score_threshold = score_threshold
        self.species = species
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def build(self, node_features: pd.DataFrame, map_df: pd.DataFrame) -> np.ndarray:
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
        node_ids = list(node_features.columns)
        n = len(node_ids)

        # 1. Load node_id → list of STRING identifiers
        #    Handles pipe-delimited complexes: "P02671|P02675|P02679" → ["P02671", "P02675", "P02679"]
        node_to_ids = self._load_mapping(node_ids, map_df)
        print(f'  Mapped {len(node_to_ids)}/{n} nodes to STRING identifiers')

        # 2. Collect all unique identifiers across all nodes
        all_ids = list({i for ids in node_to_ids.values() for i in ids})
        print(
            f'  Querying STRING for {len(all_ids)} unique identifiers '
            f'(score >= {self.score_threshold})...'
        )

        # 3. Map identifiers → STRING internal IDs via API
        #    STRING auto-detects identifier format (UniProt, Entrez, symbol, etc.)
        #    limit=1 takes the single best hit per query
        id_to_string_id = self._map_to_string_ids(all_ids)
        print(f'  STRING ID mapping: {len(id_to_string_id)}/{len(all_ids)} resolved')

        # 4. Fetch all interactions among the resolved STRING IDs
        string_ids = list(id_to_string_id.values())
        interactions = self._fetch_interactions(string_ids)
        print(f'  Retrieved {len(interactions)} interactions')

        # 5. Build reverse map: STRING internal ID → original identifier
        string_id_to_id = {v: k for k, v in id_to_string_id.items()}

        # 6. Build identifier-pair → normalized score lookup
        #    Key is always sorted tuple (a, b) with a < b for consistent lookup
        #    If multiple STRING edges map to the same pair, keep the max score
        id_interactions: dict[tuple[str, str], float] = {}
        for item in interactions:
            sid_a = item.get('stringId_A', '')
            sid_b = item.get('stringId_B', '')
            score = item.get('score', 0) / 1000.0  # normalize to [0, 1]
            ia = string_id_to_id.get(sid_a)
            ib = string_id_to_id.get(sid_b)
            if ia and ib and ia != ib:
                key = (ia, ib) if ia < ib else (ib, ia)
                id_interactions[key] = max(id_interactions.get(key, 0.0), score)

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
                (node, ident, idx)
                for idx, node in enumerate(node_ids)
                for ident in node_to_ids.get(node, [])
            ],
            columns=['node_id', 'identifier', 'node_idx'],
        )

        # Join interactions to node indices via identifier
        merged = interactions_df.merge(
            node_id_map.rename(
                columns={'node_id': 'node_a', 'identifier': 'id_a', 'node_idx': 'idx_a'}
            ),
            on='id_a',
        ).merge(
            node_id_map.rename(
                columns={'node_id': 'node_b', 'identifier': 'id_b', 'node_idx': 'idx_b'}
            ),
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
        print(f'  Adjacency matrix: {n} nodes, {n_edges} edges, {isolated} isolated nodes')

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
            print(f'  Warning: {unmapped} nodes have no STRING mapping — will be isolated')

        return result

    def _map_to_string_ids(self, identifiers: list[str]) -> dict[str, str]:
        """Map identifiers to STRING internal IDs via STRING API.

        Results cached to disk to avoid redundant API calls.
        """
        cache_file = os.path.join(self.cache_dir, f'string_id_map_{self.species}.json')

        cached: dict[str, str] = {}
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cached = json.load(f)

        to_query = [x for x in identifiers if x not in cached]

        if to_query:
            batch_size = 100
            for i in range(0, len(to_query), batch_size):
                batch = to_query[i : i + batch_size]
                try:
                    r = requests.post(
                        f'{self.STRING_API}/get_string_ids',
                        data={
                            'identifiers': '\r'.join(batch),
                            'species': self.species,
                            'limit': 1,
                            'caller_identity': self.CALLER_ID,
                        },
                        timeout=30,
                    )
                    r.raise_for_status()
                    for item in r.json():
                        query = item.get('queryItem', '')
                        string_id = item.get('stringId', '')
                        if query and string_id:
                            cached[query] = string_id
                except requests.RequestException as e:
                    print(f'  Warning: STRING ID mapping batch {i // batch_size} failed: {e}')

            with open(cache_file, 'w') as f:
                json.dump(cached, f)

        return {x: cached[x] for x in identifiers if x in cached}

    def _fetch_interactions(self, string_ids: list[str]) -> list[dict]:
        ids_hash = hashlib.md5(
            json.dumps(sorted(string_ids)).encode(), usedforsecurity=False
        ).hexdigest()[:12]
        cache_file = os.path.join(
            self.cache_dir, f'interactions_{self.species}_{self.score_threshold}_{ids_hash}.json'
        )

        if os.path.exists(cache_file):
            print('  Loading interactions from cache...')
            with open(cache_file) as f:
                return json.load(f)

        # Download full STRING interaction file for this species
        # Much more reliable than API for large protein sets
        bulk_file = os.path.join(self.cache_dir, f'{self.species}.protein.links.v12.0.txt.gz')
        if not os.path.exists(bulk_file):
            print('  Downloading STRING bulk interaction file (one-time download ~100MB)...')
            url = f'https://stringdb-downloads.org/download/protein.links.v12.0/{self.species}.protein.links.v12.0.txt.gz'
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()
            with open(bulk_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print('  Download complete.')

        # Parse bulk file and filter to our proteins + score threshold
        import gzip

        string_id_set = set(string_ids)
        interactions = []

        print('  Filtering STRING bulk file for our proteins...')
        with gzip.open(bulk_file, 'rt') as f:
            next(f)  # skip header: protein1 protein2 combined_score
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) != 3:
                    continue
                sid_a, sid_b, score_str = parts
                if sid_a in string_id_set and sid_b in string_id_set:
                    score = int(score_str)
                    if score >= self.score_threshold:
                        interactions.append(
                            {
                                'stringId_A': sid_a,
                                'stringId_B': sid_b,
                                'score': score / 1000.0,
                            }
                        )

        print(f'  Found {len(interactions)} interactions above threshold {self.score_threshold}')

        with open(cache_file, 'w') as f:
            json.dump(interactions, f)

        return interactions

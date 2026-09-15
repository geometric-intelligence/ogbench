"""Gene / marker identity features injected before message passing.

Supports:
- ``genept``: pretrained GenePT (or compatible) embedding lookup
- ``learnable``: ``nn.Embedding`` over shared node indices

Selected via Hydra group ``gene_identity={disabled,genept,learnable}``.
Identity is concatenated (or added) onto ``data.x`` *before* the configured
feature encoder, so GNN backbones remain node-identity-aware at the input to MP.
"""

from __future__ import annotations

import logging
import os
import os.path as osp
import pickle  # nosec B403
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf, open_dict

from ogbench.nn.encoders.base import AbstractFeatureEncoder

logger = logging.getLogger(__name__)


def is_gene_identity_enabled(gene_cfg: DictConfig | dict | None) -> bool:
    """Return True when a non-disabled gene-identity mode is configured."""
    if gene_cfg is None:
        return False
    mode = gene_cfg.get('mode', 'disabled')
    return mode is not None and str(mode).lower() not in {'disabled', 'none', 'null', ''}


class GeneIdentityBank(nn.Module):
    """Fixed or fine-tunable [num_nodes, dim] identity table, with optional projection."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        *,
        trainable: bool = False,
        project_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embeddings.ndim != 2:
            raise ValueError(f'embeddings must be 2-D [N, D], got {tuple(embeddings.shape)}')
        if trainable:
            self.embeddings = nn.Parameter(embeddings.clone())
        else:
            self.register_buffer('embeddings', embeddings.clone())

        in_dim = int(embeddings.size(-1))
        if project_dim is not None and int(project_dim) > 0 and int(project_dim) != in_dim:
            self.proj = nn.Linear(in_dim, int(project_dim), bias=False)
            self.out_dim = int(project_dim)
        else:
            self.proj = None
            self.out_dim = in_dim

    def forward(self) -> torch.Tensor:
        """Return [num_nodes, out_dim] identity matrix."""
        emb = self.embeddings
        if self.proj is not None:
            emb = self.proj(emb)
        return emb


class LearnableGeneIdentityBank(nn.Module):
    """Learnable identity embedding indexed by shared node order."""

    def __init__(self, num_nodes: int, embed_dim: int) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError(f'num_nodes must be positive, got {num_nodes}')
        if embed_dim <= 0:
            raise ValueError(f'embed_dim must be positive, got {embed_dim}')
        self.emb = nn.Embedding(int(num_nodes), int(embed_dim))
        self.out_dim = int(embed_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self) -> torch.Tensor:
        return self.emb.weight


class GeneIdentityFeatureEncoder(AbstractFeatureEncoder):
    """Wrap a feature encoder, injecting gene identity into ``data.x`` first."""

    def __init__(
        self,
        base_encoder: nn.Module,
        bank: nn.Module,
        *,
        combine: str = 'concat',
        add_in_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.bank = bank
        self.combine = str(combine).lower()
        if self.combine not in {'concat', 'add'}:
            raise ValueError(f"combine must be 'concat' or 'add', got {combine!r}")

        self.add_proj: nn.Linear | None = None
        if self.combine == 'add':
            if add_in_channels is None:
                raise ValueError("combine='add' requires add_in_channels (feature dim of data.x)")
            self.add_proj = nn.Linear(int(bank.out_dim), int(add_in_channels), bias=False)

        # Mirror base encoder attrs used elsewhere
        self.in_channels = getattr(base_encoder, 'in_channels', None)
        self.out_channels = getattr(base_encoder, 'out_channels', None)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(combine={self.combine}, '
            f'bank_out_dim={getattr(self.bank, "out_dim", "?")}, '
            f'base={self.base_encoder})'
        )

    @staticmethod
    def _expand_bank(emb: torch.Tensor, num_rows: int) -> torch.Tensor:
        """Tile a [N, D] bank to match batched node rows [B*N, D]."""
        n_nodes = emb.size(0)
        if num_rows == n_nodes:
            return emb
        if num_rows % n_nodes != 0:
            raise ValueError(
                f'Cannot align gene-identity bank (N={n_nodes}) with feature rows '
                f'(rows={num_rows}). Graphs must share a fixed node order.'
            )
        batch_size = num_rows // n_nodes
        return emb.unsqueeze(0).expand(batch_size, -1, -1).reshape(num_rows, -1)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        if not hasattr(data, 'x') or data.x is None:
            raise AttributeError('GeneIdentityFeatureEncoder expects data.x')

        emb = self.bank()
        if emb.device != data.x.device or emb.dtype != data.x.dtype:
            emb = emb.to(device=data.x.device, dtype=data.x.dtype)
        emb = self._expand_bank(emb, data.x.size(0))

        if self.combine == 'concat':
            data.x = torch.cat([data.x, emb], dim=-1)
        else:
            assert self.add_proj is not None
            data.x = data.x + self.add_proj(emb)

        return self.base_encoder(data)


def _selected_data_path(dataset: Any) -> str:
    """Resolve ``selected_data.parquet`` from an HFOmics (or wrapped) dataset."""
    raw = dataset
    # Unwrap common loaders / preprocessors
    for _ in range(4):
        if hasattr(raw, 'raw_dir') and osp.isdir(raw.raw_dir):
            candidate = osp.join(raw.raw_dir, 'selected_data.parquet')
            if osp.isfile(candidate):
                return candidate
        if hasattr(raw, 'dataset'):
            raw = raw.dataset
            continue
        break
    raise FileNotFoundError(
        'Could not locate selected_data.parquet for gene-identity alignment. '
        'Gene identity requires an HFOmicsDataset (or wrapper) with processed raw_dir.'
    )


def load_node_ids(dataset: Any) -> list[str]:
    """Return node IDs in graph order from ``selected_data.parquet`` columns."""
    path = _selected_data_path(dataset)
    # Read schema only when possible; fall back to full load
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
        return [str(name) for name in schema.names]
    except Exception:
        df = pd.read_parquet(path)
        return [str(c) for c in df.columns]


def load_id_map(dataset: Any) -> pd.DataFrame | None:
    """Load ``{data_name}_map.parquet`` (node_id, string_id) when available."""
    raw = dataset
    data_name = getattr(raw, 'data_name', None)
    hf_repo_id = getattr(raw, 'hf_repo_id', 'geometric-intelligence/ogbench')
    revision = getattr(raw, 'revision', None)
    for _ in range(4):
        if data_name is None and hasattr(raw, 'dataset'):
            raw = raw.dataset
            data_name = getattr(raw, 'data_name', data_name)
            hf_repo_id = getattr(raw, 'hf_repo_id', hf_repo_id)
            revision = getattr(raw, 'revision', revision)
            continue
        break

    if not data_name:
        return None

    try:
        kwargs: dict[str, Any] = {
            'repo_id': hf_repo_id,
            'repo_type': 'dataset',
            'filename': f'{data_name}_map.parquet',
        }
        if revision is not None:
            kwargs['revision'] = revision
        map_file = hf_hub_download(**kwargs)  # nosec
        map_df = pd.read_parquet(map_file)
        if 'node_id' not in map_df.columns:
            logger.warning('Map file %s missing node_id column; ignoring', map_file)
            return None
        map_df = map_df.copy()
        map_df['node_id'] = map_df['node_id'].astype(str)
        return map_df
    except Exception as exc:
        logger.info('No gene ID map loaded for %s (%s)', data_name, exc)
        return None


def _normalize_key(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null'}:
        return None
    return text.upper()


def _candidate_keys_for_node(node_id: str, string_ids: list[str] | None) -> list[str]:
    keys: list[str] = []
    for raw in [node_id, *(string_ids or [])]:
        key = _normalize_key(raw)
        if key is not None and key not in keys:
            keys.append(key)
        # Pipe-delimited multi-protein aptamers
        if raw is not None and '|' in str(raw):
            for part in str(raw).split('|'):
                part_key = _normalize_key(part)
                if part_key is not None and part_key not in keys:
                    keys.append(part_key)
    return keys


def _optional_mygene_symbols(ids: list[str]) -> dict[str, str]:
    """Best-effort Entrez / UniProt → symbol via mygene (optional dependency)."""
    try:
        import mygene
    except ImportError:
        logger.warning(
            'mygene is not installed; Entrez/UniProt IDs will not be resolved to '
            'HGNC symbols. Install with `pip install mygene` or provide symbol_map_path.'
        )
        return {}

    mg = mygene.MyGeneInfo()
    # Keep order stable; query in chunks
    resolved: dict[str, str] = {}
    chunk_size = 1000
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        try:
            hits = mg.querymany(
                chunk,
                scopes='entrezgene,uniprot,symbol,alias',
                fields='symbol',
                species='human',
                as_dataframe=False,
                returnall=False,
                verbose=False,
            )
        except Exception as exc:
            logger.warning('mygene lookup failed: %s', exc)
            break
        for hit in hits:
            query = str(hit.get('query', ''))
            if hit.get('notfound'):
                continue
            symbol = hit.get('symbol')
            if symbol:
                resolved[query.upper()] = str(symbol).upper()
    return resolved


def load_genept_dict(path: str) -> dict[str, np.ndarray]:
    """Load a GenePT-style ``{SYMBOL: vector}`` pickle."""
    if not osp.isfile(path):
        raise FileNotFoundError(
            f'GenePT embeddings not found at {path}. '
            'Download from https://zenodo.org/records/10833191 '
            '(e.g. GenePT_gene_embedding_ada_text.pickle) or run '
            '`python scripts/download_genept.py`.'
        )
    with open(path, 'rb') as f:
        payload = pickle.load(f)  # nosec B301
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f'GenePT file must be a non-empty dict, got {type(payload)}')

    # Normalize keys to uppercase strings; values to 1-D float arrays
    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        norm = _normalize_key(key)
        if norm is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out[norm] = arr
    if not out:
        raise ValueError(f'No usable gene embeddings found in {path}')
    return out


def build_genept_matrix(
    node_ids: list[str],
    genept: dict[str, np.ndarray],
    *,
    map_df: pd.DataFrame | None = None,
    symbol_map_path: str | None = None,
    resolve_entrez: bool = True,
    missing: str = 'zero',
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Align GenePT vectors to ``node_ids`` graph order.

    Returns
    -------
    tensor
        Float tensor ``[N, D]``.
    stats
        Coverage / resolution metadata for logging.
    """
    missing = str(missing).lower()
    if missing not in {'zero', 'error'}:
        raise ValueError("missing must be 'zero' or 'error'")

    # Optional explicit node_id → symbol table
    explicit: dict[str, str] = {}
    if symbol_map_path:
        if not osp.isfile(symbol_map_path):
            raise FileNotFoundError(f'symbol_map_path not found: {symbol_map_path}')
        if symbol_map_path.endswith('.parquet'):
            sm = pd.read_parquet(symbol_map_path)
        else:
            sm = pd.read_csv(symbol_map_path)
        if 'node_id' not in sm.columns or 'symbol' not in sm.columns:
            raise ValueError('symbol_map_path must have columns node_id and symbol')
        for _, row in sm.iterrows():
            nid = _normalize_key(row['node_id'])
            sym = _normalize_key(row['symbol'])
            if nid and sym:
                # Keep original node_id casing variants via str map on raw ids too
                explicit[str(row['node_id'])] = sym
                explicit[nid] = sym

    node_to_string: dict[str, list[str]] = {}
    if map_df is not None and 'string_id' in map_df.columns:
        for _, row in map_df.iterrows():
            nid = str(row['node_id'])
            raw = row['string_id']
            if pd.isna(raw):
                continue
            node_to_string[nid] = [str(raw)]

    # First pass: gather unresolved entrez/uniprot-looking keys for mygene
    pending_bio_ids: list[str] = []
    per_node_candidates: list[list[str]] = []
    for nid in node_ids:
        cands = []
        if nid in explicit:
            cands.append(explicit[nid])
        if _normalize_key(nid) in explicit:
            cands.append(explicit[_normalize_key(nid)])  # type: ignore[index]
        cands.extend(_candidate_keys_for_node(nid, node_to_string.get(nid)))
        # de-dupe
        seen: set[str] = set()
        ordered: list[str] = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        per_node_candidates.append(ordered)
        for c in ordered:
            if c not in genept and c not in pending_bio_ids:
                pending_bio_ids.append(c)

    mygene_map: dict[str, str] = {}
    if resolve_entrez and pending_bio_ids:
        mygene_map = _optional_mygene_symbols(pending_bio_ids)

    sample_vec = next(iter(genept.values()))
    dim = int(sample_vec.shape[0])
    matrix = np.zeros((len(node_ids), dim), dtype=np.float32)
    matched = 0
    matched_via: dict[str, int] = {'direct': 0, 'mygene': 0, 'symbol_map': 0}
    explicit_symbols = set(explicit.values())

    for i, cands in enumerate(per_node_candidates):
        found = None
        via = 'direct'
        for c in cands:
            if c in genept:
                found = genept[c]
                via = (
                    'symbol_map'
                    if c in explicit_symbols
                    and (
                        explicit.get(node_ids[i]) == c
                        or explicit.get(_normalize_key(node_ids[i]) or '') == c
                    )
                    else 'direct'
                )
                break
            mapped = mygene_map.get(c)
            if mapped is not None and mapped in genept:
                found = genept[mapped]
                via = 'mygene'
                break
        if found is None:
            if missing == 'error':
                raise KeyError(f'No GenePT embedding for node {node_ids[i]!r} (tried {cands})')
            continue
        matrix[i] = np.asarray(found, dtype=np.float32).reshape(-1)
        matched += 1
        matched_via[via] += 1

    stats = {
        'num_nodes': len(node_ids),
        'matched': matched,
        'coverage': matched / max(len(node_ids), 1),
        'embed_dim': dim,
        'matched_via': matched_via,
    }
    return torch.from_numpy(matrix), stats


def _resolve_encoder_name(model_cfg: DictConfig) -> str:
    name = OmegaConf.select(model_cfg, 'feature_encoder.encoder_name')
    if name:
        return str(name)
    target = str(OmegaConf.select(model_cfg, 'feature_encoder._target_') or '')
    return target.rsplit('.', 1)[-1]


def _bump_channels_for_concat(cfg: DictConfig, emb_dim: int, num_nodes: int) -> None:
    """Update Hydra model dims so concat identity fits the feature encoder."""
    model_resolved = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_resolved, dict)
    fe = model_resolved.get('feature_encoder') or {}
    encoder_name = _resolve_encoder_name(cfg.model)
    with open_dict(cfg.model.feature_encoder):
        if encoder_name == 'FlatEncoder':
            out_channels = int(fe['out_channels'])
            cfg.model.feature_encoder.out_channels = out_channels + int(num_nodes) * int(emb_dim)
        else:
            in_channels = fe.get('in_channels')
            if isinstance(in_channels, list):
                new_in = list(in_channels)
                new_in[0] = int(new_in[0]) + int(emb_dim)
                cfg.model.feature_encoder.in_channels = new_in
            elif in_channels is not None:
                cfg.model.feature_encoder.in_channels = int(in_channels) + int(emb_dim)


def build_gene_identity_bank(
    gene_cfg: DictConfig,
    dataset: Any,
    num_nodes: int,
) -> tuple[nn.Module, dict[str, Any]]:
    """Construct the identity bank and return ``(bank, info)``."""
    mode = str(gene_cfg.get('mode', 'disabled')).lower()
    info: dict[str, Any] = {'mode': mode}

    if mode == 'learnable':
        embed_dim = int(gene_cfg.get('embed_dim', 32))
        bank: nn.Module = LearnableGeneIdentityBank(num_nodes, embed_dim)
        info.update({'embed_dim': embed_dim, 'out_dim': bank.out_dim})
        return bank, info

    if mode == 'genept':
        path = gene_cfg.get('embeddings_path')
        if path is None:
            raise ValueError('gene_identity.embeddings_path is required for mode=genept')
        path = os.path.expanduser(str(path))
        genept = load_genept_dict(path)
        node_ids = load_node_ids(dataset)
        if len(node_ids) != int(num_nodes):
            logger.warning(
                'selected_data columns (%d) != graph num_nodes (%d); using column count',
                len(node_ids),
                num_nodes,
            )
            num_nodes = len(node_ids)

        map_df = load_id_map(dataset)
        symbol_map_path = gene_cfg.get('symbol_map_path')
        matrix, stats = build_genept_matrix(
            node_ids,
            genept,
            map_df=map_df,
            symbol_map_path=str(symbol_map_path) if symbol_map_path else None,
            resolve_entrez=bool(gene_cfg.get('resolve_entrez', True)),
            missing=str(gene_cfg.get('missing', 'zero')),
        )
        if matrix.size(0) != int(num_nodes) and int(num_nodes) > 0:
            # Prefer graph size if columns somehow disagree after sync
            if matrix.size(0) > int(num_nodes):
                matrix = matrix[: int(num_nodes)]
            elif matrix.size(0) < int(num_nodes):
                pad = torch.zeros(int(num_nodes) - matrix.size(0), matrix.size(1))
                matrix = torch.cat([matrix, pad], dim=0)

        project_dim = gene_cfg.get('project_dim')
        project_dim = int(project_dim) if project_dim is not None else None
        bank = GeneIdentityBank(
            matrix,
            trainable=bool(gene_cfg.get('trainable', False)),
            project_dim=project_dim,
        )
        info.update(stats)
        info['out_dim'] = bank.out_dim
        info['embeddings_path'] = path
        return bank, info

    raise ValueError(f'Unknown gene_identity.mode={mode!r}')


def _dataset_num_nodes(dataset: Any) -> int:
    """Read the actual shared node count from a loaded graph."""
    if len(dataset) == 0:
        raise ValueError('Gene identity requires a non-empty dataset')

    sample = dataset[0]
    num_nodes = getattr(sample, 'num_nodes', None)
    if num_nodes is None and getattr(sample, 'x', None) is not None:
        num_nodes = sample.x.size(0)
    if num_nodes is None:
        raise ValueError('Could not determine num_nodes from the loaded dataset')
    return int(num_nodes)


def setup_gene_identity(cfg: DictConfig, dataset: Any) -> nn.Module | None:
    """Mutate model channel config for gene identity; return bank or None.

    Call after ``sync_num_nodes_from_dataset`` and before ``hydra.utils.instantiate(cfg.model)``.
    """
    gene_cfg = cfg.get('gene_identity')
    if not is_gene_identity_enabled(gene_cfg):
        return None

    num_nodes = _dataset_num_nodes(dataset)
    configured_nodes = OmegaConf.select(cfg, 'dataset.parameters.num_nodes')
    if configured_nodes is not None and int(configured_nodes) != num_nodes:
        raise ValueError(
            'Gene identity requires dataset.parameters.num_nodes to match the loaded graph: '
            f'configured={configured_nodes}, actual={num_nodes}'
        )

    bank, info = build_gene_identity_bank(gene_cfg, dataset, num_nodes)
    combine = str(gene_cfg.get('combine', 'concat')).lower()

    if combine == 'concat':
        _bump_channels_for_concat(cfg, int(bank.out_dim), num_nodes)

    # Stash runtime info for logging / wrap step
    with open_dict(cfg):
        cfg.gene_identity._runtime = {
            'out_dim': int(bank.out_dim),
            'combine': combine,
            'coverage': info.get('coverage'),
            'matched': info.get('matched'),
            'num_nodes': info.get('num_nodes', num_nodes),
            'mode': info.get('mode'),
        }

    cov = info.get('coverage')
    if cov is not None:
        logger.info(
            'Gene identity [%s]: coverage=%.1f%% (%s/%s), out_dim=%s, combine=%s',
            info.get('mode'),
            100.0 * float(cov),
            info.get('matched'),
            info.get('num_nodes'),
            bank.out_dim,
            combine,
        )
    else:
        logger.info(
            'Gene identity [%s]: out_dim=%s, combine=%s, num_nodes=%s',
            info.get('mode'),
            bank.out_dim,
            combine,
            num_nodes,
        )
    return bank


def wrap_feature_encoder_with_gene_identity(
    feature_encoder: nn.Module,
    bank: nn.Module,
    gene_cfg: DictConfig,
    *,
    add_in_channels: int | None = None,
) -> GeneIdentityFeatureEncoder:
    """Wrap an instantiated feature encoder with gene-identity injection."""
    combine = str(gene_cfg.get('combine', 'concat')).lower()
    if combine == 'add' and add_in_channels is None:
        # For add mode channels are not bumped, so encoder in_channels == data.x dim.
        in_ch = getattr(feature_encoder, 'in_channels', None)
        if isinstance(in_ch, list | tuple) and in_ch:
            add_in_channels = int(in_ch[0])
        elif isinstance(in_ch, int):
            add_in_channels = int(in_ch)

    return GeneIdentityFeatureEncoder(
        feature_encoder,
        bank,
        combine=combine,
        add_in_channels=add_in_channels,
    )


def apply_gene_identity_to_model(
    model: nn.Module,
    bank: nn.Module | None,
    cfg: DictConfig,
) -> nn.Module:
    """Attach gene-identity wrapper to ``model.feature_encoder`` when enabled."""
    if bank is None:
        return model
    gene_cfg = cfg.gene_identity
    combine = str(gene_cfg.get('combine', 'concat')).lower()
    add_in_channels = None
    if combine == 'add':
        # Feature dim before identity: dataset num_features (+ PE already in encoder in_ch)
        in_ch = getattr(model.feature_encoder, 'in_channels', None)
        if isinstance(in_ch, list | tuple) and in_ch:
            add_in_channels = int(in_ch[0])
        elif isinstance(in_ch, int):
            add_in_channels = int(in_ch)
        else:
            add_in_channels = int(OmegaConf.select(cfg, 'dataset.parameters.num_features') or 1)

    model.feature_encoder = wrap_feature_encoder_with_gene_identity(
        model.feature_encoder,
        bank,
        gene_cfg,
        add_in_channels=add_in_channels,
    )
    return model

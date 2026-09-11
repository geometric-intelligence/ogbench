"""Learnable node-identity features injected before message passing."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch_geometric
from omegaconf import DictConfig, OmegaConf, open_dict

from ogbench.nn.encoders.base import AbstractFeatureEncoder

logger = logging.getLogger(__name__)


def is_gene_identity_enabled(gene_cfg: DictConfig | dict | None) -> bool:
    """Return whether learnable node identity is enabled."""
    if gene_cfg is None:
        return False
    mode = str(gene_cfg.get('mode', 'disabled')).lower()
    return mode not in {'disabled', 'none', 'null', ''}


class LearnableGeneIdentityBank(nn.Module):
    """Learnable identity embedding indexed by shared node order."""

    def __init__(self, num_nodes: int, embed_dim: int) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError(f'num_nodes must be positive, got {num_nodes}')
        if embed_dim <= 0:
            raise ValueError(f'embed_dim must be positive, got {embed_dim}')

        self.emb = nn.Embedding(num_nodes, embed_dim)
        self.out_dim = embed_dim
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self) -> torch.Tensor:
        """Return the identity table with shape ``[num_nodes, embed_dim]``."""
        return self.emb.weight


class GeneIdentityFeatureEncoder(AbstractFeatureEncoder):
    """Wrap a feature encoder and inject node identity into ``data.x`` first."""

    def __init__(
        self,
        base_encoder: nn.Module,
        bank: LearnableGeneIdentityBank,
        *,
        combine: str = 'concat',
        add_in_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.bank = bank
        self.combine = combine.lower()
        if self.combine not in {'concat', 'add'}:
            raise ValueError(f"combine must be 'concat' or 'add', got {combine!r}")

        self.add_proj: nn.Linear | None = None
        if self.combine == 'add':
            if add_in_channels is None:
                raise ValueError("combine='add' requires the input feature dimension")
            self.add_proj = nn.Linear(bank.out_dim, add_in_channels, bias=False)

        # Mirror attributes consumed by model/readout configuration.
        self.in_channels = getattr(base_encoder, 'in_channels', None)
        self.out_channels = getattr(base_encoder, 'out_channels', None)

    @staticmethod
    def _expand_bank(embeddings: torch.Tensor, num_rows: int) -> torch.Tensor:
        """Tile one shared identity table across a batch of fixed-order graphs."""
        num_nodes = embeddings.size(0)
        if num_rows == num_nodes:
            return embeddings
        if num_rows % num_nodes != 0:
            raise ValueError(
                f'Cannot align gene-identity bank with {num_nodes} nodes to '
                f'{num_rows} feature rows. Graphs must share a fixed node order.'
            )
        batch_size = num_rows // num_nodes
        return embeddings.unsqueeze(0).expand(batch_size, -1, -1).reshape(num_rows, -1)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Inject identity features, then apply the configured feature encoder."""
        if not hasattr(data, 'x') or data.x is None:
            raise AttributeError('GeneIdentityFeatureEncoder expects data.x')

        embeddings = self.bank()
        embeddings = embeddings.to(device=data.x.device, dtype=data.x.dtype)
        embeddings = self._expand_bank(embeddings, data.x.size(0))

        if self.combine == 'concat':
            data.x = torch.cat([data.x, embeddings], dim=-1)
        else:
            assert self.add_proj is not None
            data.x = data.x + self.add_proj(embeddings)

        return self.base_encoder(data)


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


def _resolve_encoder_name(model_cfg: DictConfig) -> str:
    name = OmegaConf.select(model_cfg, 'feature_encoder.encoder_name')
    if name:
        return str(name)
    target = str(OmegaConf.select(model_cfg, 'feature_encoder._target_') or '')
    return target.rsplit('.', 1)[-1]


def _bump_channels_for_concat(cfg: DictConfig, embed_dim: int, num_nodes: int) -> None:
    """Adjust feature-encoder dimensions for appended identity channels."""
    model = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model, dict)
    feature_encoder = model.get('feature_encoder') or {}
    encoder_name = _resolve_encoder_name(cfg.model)

    with open_dict(cfg.model.feature_encoder):
        if encoder_name == 'FlatEncoder':
            in_channels = int(feature_encoder['in_channels'])
            cfg.model.feature_encoder.out_channels = num_nodes * (in_channels + embed_dim)
            return

        in_channels = feature_encoder.get('in_channels')
        if isinstance(in_channels, list):
            updated = list(in_channels)
            updated[0] = int(updated[0]) + embed_dim
            cfg.model.feature_encoder.in_channels = updated
        elif in_channels is not None:
            cfg.model.feature_encoder.in_channels = int(in_channels) + embed_dim


def setup_gene_identity(
    cfg: DictConfig,
    dataset: Any,
) -> LearnableGeneIdentityBank | None:
    """Build the configured identity bank and adjust model channel dimensions."""
    gene_cfg = cfg.get('gene_identity')
    if not is_gene_identity_enabled(gene_cfg):
        return None

    mode = str(gene_cfg.get('mode')).lower()
    if mode != 'learnable':
        raise ValueError(f"Unsupported gene_identity.mode={mode!r}; expected 'learnable'")

    num_nodes = _dataset_num_nodes(dataset)
    configured_nodes = OmegaConf.select(cfg, 'dataset.parameters.num_nodes')
    if configured_nodes is not None and int(configured_nodes) != num_nodes:
        raise ValueError(
            'Gene identity requires dataset.parameters.num_nodes to match the loaded graph: '
            f'configured={configured_nodes}, actual={num_nodes}'
        )

    embed_dim = int(gene_cfg.get('embed_dim', 32))
    bank = LearnableGeneIdentityBank(num_nodes, embed_dim)
    combine = str(gene_cfg.get('combine', 'concat')).lower()
    if combine == 'concat':
        _bump_channels_for_concat(cfg, embed_dim, num_nodes)
    elif combine != 'add':
        raise ValueError(f"combine must be 'concat' or 'add', got {combine!r}")

    with open_dict(cfg.gene_identity):
        cfg.gene_identity._runtime = {
            'mode': mode,
            'combine': combine,
            'embed_dim': embed_dim,
            'num_nodes': num_nodes,
        }
    logger.info(
        'Gene identity [%s]: embed_dim=%d, combine=%s, num_nodes=%d',
        mode,
        embed_dim,
        combine,
        num_nodes,
    )
    return bank


def apply_gene_identity_to_model(
    model: nn.Module,
    bank: LearnableGeneIdentityBank | None,
    cfg: DictConfig,
) -> nn.Module:
    """Wrap ``model.feature_encoder`` when learnable identity is enabled."""
    if bank is None:
        return model

    combine = str(cfg.gene_identity.get('combine', 'concat')).lower()
    add_in_channels = None
    if combine == 'add':
        in_channels = getattr(model.feature_encoder, 'in_channels', None)
        if isinstance(in_channels, list | tuple) and in_channels:
            add_in_channels = int(in_channels[0])
        elif isinstance(in_channels, int):
            add_in_channels = in_channels
        else:
            add_in_channels = int(OmegaConf.select(cfg, 'dataset.parameters.num_features') or 1)

    model.feature_encoder = GeneIdentityFeatureEncoder(
        model.feature_encoder,
        bank,
        combine=combine,
        add_in_channels=add_in_channels,
    )
    return model

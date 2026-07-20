"""Shared logic for validation top-K ensembles (seed-averaged).

By default **no graph lock**: top buckets can mix adjacency / ratio / sampling / threshold;
each bucket uses its own Hydra config to build the test ``DataLoader`` and model. Soft voting
requires identical test **labels** (and logits length) across those buckets for a given seed;
if not, that seed is skipped.

Optional ``graph_lock=True`` reproduces the older behaviour (restrict to the graph slice of the
single best mean-val bucket, one shared loader).

Used by ``ensemble_val_topk_test.py`` and ``export_ensemble_topk_aggregated.py``.
Callers must run ``rootutils.setup_root(...)`` before importing ``ogbench.run`` / composing configs.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from hydra import compose, initialize_config_dir
from lightning import LightningModule
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

from generate_rerun_sh import (  # noqa: E402
    _OVERRIDE_KEYS_DROP,
    _override_key,
    hydra_overrides_from_wandb_run,
)
from narrow_schema import EXPECTED_SEEDS, canonical_model_name  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

_HYDRA_VALUE_COALESCE: tuple[tuple[str, str], ...] = (
    ("model.model_name", "model.value.model_name"),
    ("dataset.loader.parameters.data_name", "dataset.value.loader.parameters.data_name"),
    ("dataset.loader.parameters.adjacency_method", "dataset.value.loader.parameters.adjacency_method"),
    ("dataset.loader.parameters.node_sample_ratio", "dataset.value.loader.parameters.node_sample_ratio"),
    ("dataset.loader.parameters.method", "dataset.value.loader.parameters.method"),
    ("seed", "seed.value"),
)

_SHORT_FROM_LONG: tuple[tuple[str, str], ...] = (
    ("data_name", "dataset.loader.parameters.data_name"),
    ("adjacency_method", "dataset.loader.parameters.adjacency_method"),
    ("node_sample_ratio", "dataset.loader.parameters.node_sample_ratio"),
    ("sampling_method", "dataset.loader.parameters.method"),
    ("readout_name", "model.readout.readout_name"),
)

FINGERPRINT_KEY_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("data_name", "dataset.loader.parameters.data_name"),
    ("model_name", "model.model_name"),
    ("adjacency_method", "dataset.loader.parameters.adjacency_method"),
    ("node_sample_ratio", "dataset.loader.parameters.node_sample_ratio"),
    ("sampling_method", "dataset.loader.parameters.method"),
    ("readout_name", "model.readout.readout_name"),
    ("dataset.loader.parameters.adjacency_threshold",),
    ("model.readout.hidden_dim",),
    ("model.readout.num_nodes",),
    ("dataset.parameters.num_nodes",),
    ("model.backbone.num_nodes",),
    ("model.backbone_wrapper.num_nodes",),
    ("model.backbone.hidden_channels",),
    ("model.backbone.heads",),
    ("model.backbone.num_layers",),
    ("model.backbone.num_heads",),
    ("model.backbone.in_channels",),
    ("model.backbone.out_channels",),
    ("model.feature_encoder.in_channels",),
    ("model.feature_encoder.out_channels",),
    ("model.readout.out_channels",),
    ("model.encodings",),
    ("transforms.CombinedPSEs.encodings",),
    ("transforms.PrecomputeKhops.num_layers",),
    ("loss.dataset_loss.class_weights",),
    ("dataset.parameters.class_weights",),
    ("dataset.split_params.data_split_dir",),
    ("dataset.parameters.num_samples",),
    ("dataset.parameters.full_num_nodes",),
    ("optimizer.parameters.lr",),
)

GRAPH_FINGERPRINT_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("data_name", "dataset.loader.parameters.data_name"),
    ("adjacency_method", "dataset.loader.parameters.adjacency_method"),
    ("node_sample_ratio", "dataset.loader.parameters.node_sample_ratio"),
    ("sampling_method", "dataset.loader.parameters.method"),
    ("dataset.loader.parameters.adjacency_threshold",),
)


def _resolve_fingerprint_columns(df: pd.DataFrame, candidates: tuple[tuple[str, ...], ...]) -> list[str]:
    cols: list[str] = []
    for group in candidates:
        for c in group:
            if c in df.columns:
                cols.append(c)
                break
    return cols


def _prune_run_unique_group_cols(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    n = len(df)
    if n == 0:
        return cols, []
    keep, dropped = [], []
    for c in cols:
        if df[c].nunique(dropna=False) >= n:
            dropped.append(c)
        else:
            keep.append(c)
    return keep, dropped


def _compose_bucket_key_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if not cols:
        return pd.Series(["<no-bucket-cols>"] * len(df), index=df.index, dtype=object)
    parts: list[pd.Series] = []
    for c in cols:
        if c not in df.columns:
            parts.append(pd.Series(["<missing>"] * len(df), index=df.index, dtype=object))
            continue
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            s = s.round(10)
        parts.append(s.astype(str).where(s.notna(), "<NA>"))
    out = parts[0]
    for p in parts[1:]:
        out = out + "||" + p
    return out


def normalize_results_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for plain, value in _HYDRA_VALUE_COALESCE:
        if value not in out.columns:
            continue
        if plain in out.columns:
            out[plain] = out[plain].where(out[plain].notna(), out[value])
        else:
            out[plain] = out[value]
        out = out.drop(columns=[value], errors="ignore")
    for short, long in _SHORT_FROM_LONG:
        if short not in out.columns and long in out.columns:
            out[short] = out[long]
    if "model.model_name" in out.columns:
        out["model_name"] = out["model.model_name"].map(canonical_model_name)
    elif "model_name" not in out.columns:
        raise ValueError("CSV must expose model name via model.model_name or model_name.")
    else:
        out["model_name"] = out["model_name"].map(canonical_model_name)
    if "data_name" not in out.columns:
        raise ValueError("CSV must expose dataset.loader.parameters.data_name or data_name.")
    out["data_name"] = out["data_name"].astype(str).str.strip().str.lower()
    if "seed" not in out.columns:
        raise ValueError("CSV must include a seed column (Hydra seed or seed.value).")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    return out


def _wandb_summary_dict(run) -> dict:
    try:
        return dict(run.summary) if run.summary else {}
    except Exception:
        return {}


def _filter_inference_overrides(overrides: list[str]) -> list[str]:
    drop_keys = _OVERRIDE_KEYS_DROP | {"train", "test", "ckpt_path"}
    out: list[str] = []
    for o in overrides:
        k = _override_key(o)
        if k in drop_keys:
            continue
        out.append(o)
    return out


def _inference_tail(*, work_dir: Path, wandb_offline: bool) -> list[str]:
    output_dir = Path(tempfile.mkdtemp(prefix="ensemble_val_topk_"))
    return [
        "train=false",
        "test=false",
        "logger.wandb.offline=true" if wandb_offline else "logger.wandb.offline=false",
        f"paths.output_dir={output_dir.resolve().as_posix()}",
        f"paths.work_dir={work_dir.resolve().as_posix()}",
    ]


def compose_cfg(overrides: list[str], *, config_dir: Path) -> DictConfig:
    import ogbench.run  # noqa: F401

    cdir = Path(config_dir).resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(cdir), job_name="ensemble_topk"):
        cfg = compose(config_name="train", overrides=overrides)
    return cfg


def build_datamodule(cfg: DictConfig):
    import hydra.utils
    import lightning as L

    from ogbench.data.preprocessor import PreProcessor
    from ogbench.dataloader import TBDataloader

    L.seed_everything(cfg.seed, workers=True)
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    transform_config = cfg.get("transforms", None)
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    dataset_train, dataset_val, dataset_test = preprocessor.load_dataset_splits(cfg.dataset.split_params)
    if cfg.dataset.parameters.task_level not in ("node", "graph"):
        raise ValueError("Unsupported task_level")
    return TBDataloader(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
        **cfg.dataset.get("dataloader_params", {}),
    )


def instantiate_tb_model(cfg: DictConfig) -> LightningModule:
    import hydra.utils

    return hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )


def _align_tensor_to_target(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """Match checkpoint tensor to parameter shape (e.g. SAGN BN buffers ``[1,2,64]`` vs ``[2,64]``)."""
    if value.shape == target.shape:
        return value
    v = value
    while v.ndim > target.ndim and v.shape[0] == 1:
        v = v.squeeze(0)
    return v if v.shape == target.shape else None


def load_checkpoint_weights(model: LightningModule, ckpt_path: Path, *, device: torch.device) -> None:
    import ogbench.run  # noqa: F401

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_sd = model.state_dict()

    def _apply_state(state: dict) -> tuple:
        return model.load_state_dict(state, strict=False)

    try:
        incompatible = _apply_state(sd)
    except RuntimeError as e:
        err_l = str(e).lower()
        if "size mismatch" not in err_l and "error(s) in loading state_dict" not in err_l:
            raise
        filtered: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k not in model_sd:
                continue
            tgt = model_sd[k]
            if not isinstance(v, torch.Tensor) or not isinstance(tgt, torch.Tensor):
                continue
            aligned = _align_tensor_to_target(v, tgt)
            if aligned is not None:
                filtered[k] = aligned
        if not filtered:
            raise
        print(
            "WARN checkpoint load: retried after squeezing leading singleton dims on "
            f"{len(sd) - len(filtered)} key(s) (e.g. SAGN BatchNorm buffers).",
            flush=True,
        )
        incompatible = _apply_state(filtered)

    if incompatible.missing_keys:
        print(f"WARN checkpoint load: missing {len(incompatible.missing_keys)} key(s)", flush=True)
    if incompatible.unexpected_keys:
        print(f"WARN checkpoint load: unexpected {len(incompatible.unexpected_keys)} key(s)", flush=True)


@torch.no_grad()
def collect_logits_labels(
    model: LightningModule,
    loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    model.state_str = "Test"
    logits_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        batch["model_state"] = "Test"
        model.state_str = "Test"
        out = model.forward(batch)
        out = model.process_outputs(out, batch)
        logits_chunks.append(out["logits"].detach().float().cpu())
        labels_chunks.append(out["labels"].detach().cpu())
    logits = torch.cat(logits_chunks, dim=0)
    labels = torch.cat(labels_chunks, dim=0)
    return logits, labels


def ensemble_test_metrics_from_logits(logits_stack: torch.Tensor, labels: np.ndarray) -> dict[str, float]:
    """logits_stack: (K, N, C) — soft vote then classification metrics."""
    probs = torch.softmax(logits_stack, dim=-1).mean(dim=0).numpy()
    y = labels.astype(np.int64)
    pred = probs.argmax(axis=-1)
    out: dict[str, float] = {
        "best_test_f1_macro": float(f1_score(y, pred, average="macro")),
        "best_test_f1_weighted": float(f1_score(y, pred, average="weighted")),
        "best_test_accuracy": float(accuracy_score(y, pred)),
    }
    n_classes = probs.shape[1]
    try:
        if n_classes == 2:
            out["best_test_auroc"] = float(roc_auc_score(y, probs[:, 1]))
        else:
            out["best_test_auroc"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        out["best_test_auroc"] = float("nan")
    return out


def softmax_average_f1(logits_stack: torch.Tensor, labels: np.ndarray) -> float:
    return ensemble_test_metrics_from_logits(logits_stack, labels)["best_test_f1_macro"]


def _fetch_run_from_wandb(
    api: wandb.Api,
    path_prefix: str,
    run_id: str,
    *,
    val_f1: float,
) -> dict | None:
    run = api.run(f"{path_prefix}/{run_id}")
    summary = _wandb_summary_dict(run)
    ckpt_raw = summary.get("checkpoint")
    overrides_raw = hydra_overrides_from_wandb_run(run)
    if not ckpt_raw or not overrides_raw:
        print(f"SKIP {run_id}: missing checkpoint or Hydra args in W&B.", flush=True)
        return None
    ckpt_path = Path(str(ckpt_raw).strip()).expanduser()
    if not ckpt_path.is_file():
        print(f"SKIP {run_id}: checkpoint not on disk: {ckpt_path}", flush=True)
        return None
    test_f1 = None
    for k in ("best_test/f1_macro", "best_test_f1_macro"):
        if k in summary and summary[k] is not None:
            try:
                test_f1 = float(summary[k])
            except (TypeError, ValueError):
                pass
            break
    return {
        "run_id": run_id,
        "name": getattr(run, "name", "") or "",
        "ckpt_path": ckpt_path,
        "overrides": list(overrides_raw),
        "val_f1": val_f1,
        "test_f1_logged": test_f1,
    }


def _std_ddof1(arr: np.ndarray) -> float:
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


@dataclass
class EnsembleRunConfig:
    csv_path: Path
    entity: str
    project: str
    model: str
    dataset: str
    ks: tuple[int, ...]
    min_seeds_per_bucket: int
    max_buckets: int
    wandb_offline: bool
    extra_override: tuple[str, ...]
    work_dir: Path
    config_dir: Path
    graph_lock: bool = False
    verbose: bool = True


def run_ensemble(cfg: EnsembleRunConfig, api: wandb.Api | None = None) -> tuple[list[dict[str, object]], str]:
    """Returns ``(rows, message)``. Each row is one ``ensemble_K`` value with aggregated metrics.

    On failure ``rows`` is empty and ``message`` explains why.
    """
    if not cfg.csv_path.is_file():
        return [], f"CSV not found: {cfg.csv_path}"

    df = pd.read_csv(cfg.csv_path, low_memory=False)
    df = normalize_results_df(df)

    m_name = cfg.model.strip().lower()
    d_name = cfg.dataset.strip().lower()
    work = df.loc[(df["model_name"] == m_name) & (df["data_name"] == d_name)].copy()
    if work.empty:
        return [], f"No rows for model={cfg.model!r} dataset={cfg.dataset!r}"

    val_col = "best_val_f1_macro"
    if val_col not in work.columns:
        return [], f"CSV missing {val_col!r}"
    work["_val"] = pd.to_numeric(work[val_col], errors="coerce")
    train_col = "best_train_f1_macro"
    if train_col in work.columns:
        work["_train"] = pd.to_numeric(work[train_col], errors="coerce")
    else:
        work["_train"] = np.nan

    work = work.dropna(subset=["_val", "seed"])

    fp_cols = _resolve_fingerprint_columns(work, FINGERPRINT_KEY_CANDIDATES)
    fp_cols, _ = _prune_run_unique_group_cols(work, fp_cols)
    work["_fp_bucket"] = _compose_bucket_key_frame(work, fp_cols)

    global_stats = (
        work.groupby("_fp_bucket", dropna=False)
        .agg(val_mean=("_val", "mean"), n_seeds=("seed", lambda s: s.nunique()))
        .reset_index()
    )
    global_stats = global_stats.loc[global_stats["n_seeds"] >= int(cfg.min_seeds_per_bucket)]
    if global_stats.empty:
        return [], "No buckets after min-seeds filter (global)."

    global_stats = global_stats.sort_values("val_mean", ascending=False).reset_index(drop=True)
    graph_cols = _resolve_fingerprint_columns(work, GRAPH_FINGERPRINT_CANDIDATES)
    graph_cols = [c for c in graph_cols if c in work.columns]
    if cfg.graph_lock and graph_cols:
        winner_bucket = str(global_stats.iloc[0]["_fp_bucket"])
        winner_rows = work.loc[work["_fp_bucket"].astype(str) == winner_bucket]
        ref_graph_key = _compose_bucket_key_frame(winner_rows.iloc[[0]], graph_cols).iloc[0]
        work["_graph_key"] = _compose_bucket_key_frame(work, graph_cols)
        n_before = len(work)
        work = work.loc[work["_graph_key"] == ref_graph_key].copy()
        if cfg.verbose:
            print(
                f"[{m_name}/{d_name}] graph lock ON → {len(work)} rows ({n_before - len(work)} dropped) "
                f"graph_key={ref_graph_key!r}",
                flush=True,
            )
    elif cfg.verbose:
        print(
            f"[{m_name}/{d_name}] graph lock OFF — ranking buckets over all graph variants "
            f"({len(work)} row(s)).",
            flush=True,
        )

    work["_fp_bucket"] = _compose_bucket_key_frame(work, fp_cols)
    bucket_stats = (
        work.groupby("_fp_bucket", dropna=False)
        .agg(
            val_mean=("_val", "mean"),
            val_std=("_val", "std"),
            n_seeds=("seed", "nunique"),
            n_runs=("run_id", "count"),
        )
        .reset_index()
    )
    bucket_stats = bucket_stats.loc[bucket_stats["n_seeds"] >= int(cfg.min_seeds_per_bucket)]
    bucket_stats = bucket_stats.sort_values("val_mean", ascending=False).reset_index(drop=True)
    if bucket_stats.empty:
        return [], "No buckets after min-seeds filter."

    ordered_buckets = bucket_stats["_fp_bucket"].astype(str).tolist()[: int(cfg.max_buckets)]
    if len(ordered_buckets) < min(cfg.ks):
        return [], f"Only {len(ordered_buckets)} bucket(s); need ≥ min(ks)={min(cfg.ks)}."

    need_buckets = list(ordered_buckets)
    seed_sets: list[set[float]] = []
    for b in need_buckets:
        seeds_b = work.loc[work["_fp_bucket"].astype(str) == str(b), "seed"].dropna().unique()
        seed_sets.append({float(x) for x in seeds_b})
    seeds_common = set.intersection(*seed_sets) if seed_sets else set()
    if not seeds_common:
        return [], "Empty seed intersection across top buckets."

    seeds_sorted = sorted(seeds_common)
    if len(seeds_sorted) < int(cfg.min_seeds_per_bucket):
        return [], f"Intersection has {len(seeds_sorted)} seed(s); need ≥ {cfg.min_seeds_per_bucket}."

    path_prefix = f"{cfg.entity}/{cfg.project}"
    api = api or wandb.Api()

    try:
        return _run_ensemble_inference_phase(
            cfg=cfg,
            api=api,
            path_prefix=path_prefix,
            work=work,
            m_name=m_name,
            d_name=d_name,
            ordered_buckets=ordered_buckets,
            seeds_sorted=seeds_sorted,
        )
    except Exception as e:
        err = str(e).replace("\n", " ")[:800]
        if cfg.verbose:
            traceback.print_exc()
        return [], f"SKIP ({type(e).__name__}): {err}"


def _run_ensemble_inference_phase(
    *,
    cfg: EnsembleRunConfig,
    api: wandb.Api,
    path_prefix: str,
    work: pd.DataFrame,
    m_name: str,
    d_name: str,
    ordered_buckets: list[str],
    seeds_sorted: list[float],
) -> tuple[list[dict[str, object]], str]:
    key_run: dict[tuple[str, float], dict] = {}
    for b in ordered_buckets:
        sub = work.loc[work["_fp_bucket"].astype(str) == str(b)]
        seed_float = pd.to_numeric(sub["seed"], errors="coerce")
        for s in seeds_sorted:
            row_match = sub.loc[np.isclose(seed_float, float(s), rtol=0.0, atol=1e-9)]
            if row_match.empty:
                continue
            row = row_match.sort_values("_val", ascending=False).iloc[0]
            rid = str(row["run_id"])
            val_f1 = float(row["_val"])
            rec = _fetch_run_from_wandb(api, path_prefix, rid, val_f1=val_f1)
            if rec is not None:
                key_run[(str(b), float(s))] = rec

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Per-seed: per-K dict of test metrics + mean val/train over K members
    per_seed_k: dict[float, dict[int, dict[str, float]]] = {}

    n_buck = len(ordered_buckets)

    for s in seeds_sorted:
        b0 = ordered_buckets[0]
        ref_key = (str(b0), float(s))
        if ref_key not in key_run:
            if cfg.verbose:
                print(f"[{m_name}/{d_name}] SKIP seed={s}: no checkpoint for rank-1 bucket.", flush=True)
            continue

        logits_by_bucket: dict[str, torch.Tensor] = {}
        ref_labels: torch.Tensor | None = None
        skip_seed = False
        vals_k: dict[int, list[float]] = {k: [] for k in cfg.ks}
        trains_k: dict[int, list[float]] = {k: [] for k in cfg.ks}

        loader_shared = None
        ref_dm = None
        if cfg.graph_lock:
            ref_rec = key_run[ref_key]
            ref_overrides = _filter_inference_overrides(ref_rec["overrides"]) + _inference_tail(
                work_dir=cfg.work_dir, wandb_offline=bool(cfg.wandb_offline)
            ) + list(cfg.extra_override)
            ref_cfg = compose_cfg(ref_overrides, config_dir=cfg.config_dir)
            ref_dm = build_datamodule(ref_cfg)
            ref_dm.setup("test")
            loader_shared = ref_dm.test_dataloader()

        for bi, b in enumerate(ordered_buckets):
            rk = (str(b), float(s))
            if rk not in key_run:
                skip_seed = True
                break
            rec = key_run[rk]
            ov = _filter_inference_overrides(rec["overrides"]) + _inference_tail(
                work_dir=cfg.work_dir,
                wandb_offline=bool(cfg.wandb_offline),
            ) + list(cfg.extra_override)
            model_cfg = compose_cfg(ov, config_dir=cfg.config_dir)
            if cfg.graph_lock:
                assert loader_shared is not None
                loader = loader_shared
            else:
                datamodule = build_datamodule(model_cfg)
                datamodule.setup("test")
                loader = datamodule.test_dataloader()
            model = instantiate_tb_model(model_cfg)
            load_checkpoint_weights(model, rec["ckpt_path"], device=device)
            model.to(device)
            logits, labels = collect_logits_labels(model, loader, device)
            logits_by_bucket[str(b)] = logits
            if ref_labels is None:
                ref_labels = labels
            elif not torch.equal(labels, ref_labels):
                short_b = (b[:56] + "…") if len(b) > 56 else b
                if cfg.verbose:
                    print(
                        f"[{m_name}/{d_name}] SKIP seed={s}: test labels differ from rank-1 bucket "
                        f"(bucket {short_b!r}) — cannot soft-vote across graphs.",
                        flush=True,
                    )
                skip_seed = True
                break
            # CSV val/train for this run
            sub = work.loc[work["_fp_bucket"].astype(str) == str(b)]
            sf = pd.to_numeric(sub["seed"], errors="coerce")
            row_one = sub.loc[np.isclose(sf, float(s), rtol=0.0, atol=1e-9)].sort_values("_val", ascending=False).iloc[0]
            v_run = float(row_one["_val"])
            t_run = float(row_one["_train"]) if pd.notna(row_one["_train"]) else float("nan")
            for k in cfg.ks:
                if bi < k:
                    vals_k[k].append(v_run)
                    trains_k[k].append(t_run)
            del model
            if not cfg.graph_lock:
                del datamodule
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if ref_dm is not None:
            del ref_dm

        if skip_seed or ref_labels is None or len(logits_by_bucket) < n_buck:
            if cfg.verbose and not skip_seed:
                print(f"[{m_name}/{d_name}] incomplete logits seed={s}", flush=True)
            continue

        labels_np = ref_labels.numpy()
        bucket_order = [str(x) for x in ordered_buckets]
        stack_full = torch.stack([logits_by_bucket[b] for b in bucket_order], dim=0)

        seed_out: dict[int, dict[str, float]] = {}
        for k in cfg.ks:
            if k > stack_full.shape[0]:
                continue
            sub = stack_full[:k]
            mets = ensemble_test_metrics_from_logits(sub, labels_np)
            seed_out[k] = dict(mets)
            seed_out[k]["_val_mean_k"] = float(np.mean(vals_k[k])) if vals_k[k] else float("nan")
            seed_out[k]["_train_mean_k"] = float(np.nanmean(trains_k[k])) if trains_k[k] else float("nan")
        per_seed_k[float(s)] = seed_out

    if not per_seed_k:
        return [], "No seed completed inference."

    # Reference axis row (best bucket, first seed that completed)
    ref_seed = min(per_seed_k.keys())
    axis_src = work.loc[
        (work["_fp_bucket"].astype(str) == ordered_buckets[0])
        & (np.isclose(pd.to_numeric(work["seed"], errors="coerce"), ref_seed, rtol=0.0, atol=1e-9))
    ]
    if axis_src.empty:
        axis_src = work.loc[work["_fp_bucket"].astype(str) == ordered_buckets[0]]
    axis_row = axis_src.iloc[0]

    def _cell(name: str, fallback: str = "") -> str:
        if name in axis_row.index and pd.notna(axis_row[name]):
            return str(axis_row[name]).strip()
        return fallback

    readout = _cell("readout_name") or _cell("model.readout.readout_name", "NoReadOut")
    axis = {
        "data_name": _cell("data_name", d_name),
        "model_name": _cell("model_name", m_name),
        "adjacency_method": _cell("adjacency_method"),
        "node_sample_ratio": axis_row["node_sample_ratio"] if "node_sample_ratio" in axis_row.index else "",
        "sampling_method": _cell("sampling_method"),
        "readout_name": readout,
        "_bucket_key": str(ordered_buckets[0]),
    }

    rows_out: list[dict[str, object]] = []
    n_seeds_used = len(per_seed_k)

    for k in sorted(set(cfg.ks)):
        if k > len(ordered_buckets):
            continue
        # collect arrays over seeds for each metric
        test_macro, test_w, test_acc, test_auroc = [], [], [], []
        val_agg, train_agg = [], []
        for s in sorted(per_seed_k.keys()):
            d = per_seed_k[s].get(k)
            if d is None:
                continue
            test_macro.append(d["best_test_f1_macro"])
            test_w.append(d["best_test_f1_weighted"])
            test_acc.append(d["best_test_accuracy"])
            test_auroc.append(d["best_test_auroc"])
            val_agg.append(d["_val_mean_k"])
            train_agg.append(d["_train_mean_k"])

        if len(test_macro) < int(cfg.min_seeds_per_bucket):
            if cfg.verbose:
                print(
                    f"[{m_name}/{d_name}] K={k}: only {len(test_macro)} seed(s) completed; skip.",
                    flush=True,
                )
            continue

        arr_m = np.array(test_macro, dtype=np.float64)
        arr_w = np.array(test_w, dtype=np.float64)
        arr_a = np.array(test_acc, dtype=np.float64)
        arr_r = np.array(test_auroc, dtype=np.float64)
        arr_v = np.array(val_agg, dtype=np.float64)
        arr_t = np.array(train_agg, dtype=np.float64)

        row = {
            **axis,
            "ensemble_K": int(k),
            "n_runs_seeds": int(n_seeds_used),
            "best_val_f1_macro_mean": float(np.mean(arr_v)),
            "best_val_f1_macro_std": _std_ddof1(arr_v),
            "best_test_f1_macro_mean": float(np.mean(arr_m)),
            "best_test_f1_macro_std": _std_ddof1(arr_m),
            "best_train_f1_macro_mean": float(np.nanmean(arr_t)),
            "best_train_f1_macro_std": _std_ddof1(arr_t[np.isfinite(arr_t)]),
            "best_test_f1_weighted_mean": float(np.mean(arr_w)),
            "best_test_f1_weighted_std": _std_ddof1(arr_w),
            "best_test_accuracy_mean": float(np.mean(arr_a)),
            "best_test_accuracy_std": _std_ddof1(arr_a),
            "best_test_auroc_mean": float(np.nanmean(arr_r)),
            "best_test_auroc_std": _std_ddof1(arr_r[np.isfinite(arr_r)]),
        }
        rows_out.append(row)

    msg = f"ok {len(rows_out)} K-row(s) for {m_name}/{d_name} (seeds={n_seeds_used})"
    return rows_out, msg


def print_cli_summary(cfg: EnsembleRunConfig, rows: list[dict], ks: tuple[int, ...]) -> None:
    """Console summary for ``ensemble_val_topk_test`` (macro F1 only)."""
    by_k = {int(r["ensemble_K"]): r for r in rows}
    print("\n=== Summary: mean ± std over seeds (soft vote) ===", flush=True)
    for k in sorted(set(ks)):
        r = by_k.get(k)
        if r is None:
            print(f"  K={k}: (no row)", flush=True)
            continue
        print(
            f"  K={k}: test macro-F1 mean={r['best_test_f1_macro_mean']:.6f} "
            f"std={r['best_test_f1_macro_std']:.6f}  n_seeds={r['n_runs_seeds']}",
            flush=True,
        )
    print(f"\nExpected seeds in project: {EXPECTED_SEEDS}", flush=True)

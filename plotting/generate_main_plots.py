# Aggregate per-run CSV (full W&B config rows from load_results.py) across seeds, then plot.
# Raw: plotting/final_results_hyperparams_neurips.csv
# Out: lean plotting/aggregated_final_results_neurips.csv (dataset, model, adjacency, method, ratio, readout + metrics)
# Figures: plotting/plots/*.pdf and plotting/plots/*.png

from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(_DIR, "plots")
# Used for every `savefig` in this module (PDF rasterization + PNG pixel density).
SAVEFIG_DPI = 600


class PngDestReplaceBlocked(Exception):
    """Temp PNG was written; ``os.replace`` could not overwrite the final path (often a busy/locked PNG on Windows)."""


def _os_error_replace_dest_busy(e: BaseException) -> bool:
    """True if ``os.replace(tmp, dest)`` likely failed because ``dest`` is busy, not because ``tmp`` is bad."""
    if isinstance(e, PermissionError):
        return True
    if isinstance(e, OSError):
        if getattr(e, "errno", None) in (13, 22):
            return True
        if getattr(e, "winerror", None) == 5:
            return True
    return False


def _os_error_direct_save_dest_busy(e: BaseException) -> bool:
    """True when ``savefig(..., fname=dest)`` failed only because ``dest`` is busy (retry / lower DPI may help)."""
    if isinstance(e, PermissionError):
        return True
    if isinstance(e, OSError):
        if getattr(e, "errno", None) == 13:
            return True
        if getattr(e, "winerror", None) == 5:
            return True
    return False


def _replace_file_resilient(src: str, dst: str) -> None:
    """``os.replace(src, dst)`` with retries for transient Windows locks (WinError 5 / PermissionError).

    On failure, ``src`` is removed if it still exists so callers do not leave stray temp PNGs.
    """
    last: OSError | None = None
    for attempt in range(18):
        try:
            os.replace(src, dst)
            return
        except OSError as e:
            last = e
            if not _os_error_replace_dest_busy(e):
                break
            time.sleep(0.1 * (attempt + 1))
    if os.name == "nt" and os.path.isfile(dst):
        try:
            os.remove(dst)
            os.rename(src, dst)
            return
        except OSError:
            pass
    if os.path.isfile(src):
        try:
            os.unlink(src)
        except OSError:
            pass
    if last is not None:
        raise last
    raise OSError(f"Could not replace {dst!r} with {src!r}")


if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from narrow_schema import EXPECTED_SEEDS, canonical_model_name

RUN_META_COLS = frozenset({"run_id", "run_name", "state"})
PER_RUN_METRIC_COLS = ("best_val_f1_macro", "best_test_f1_macro", "best_train_f1_macro")
# When present in the per-run CSV, aggregate mean/std across seeds like the core F1 columns.
OPTIONAL_EXTRA_TEST_METRICS = ("best_test_f1_weighted", "best_test_accuracy", "best_test_auroc")
SEED_COL = "seed"
BUCKET_KEY_COL = "_bucket_key"
PERFECT_TRAIN_F1_THRESHOLD = 0.98
VAL_SELECTION_TIE_TOL = 1e-12

# Hydra `foo` vs `foo.value` → one column (prefer non-.value) so grouping stays stable.
_HYDRA_VALUE_COALESCE: tuple[tuple[str, str], ...] = (
    ("model.model_name", "model.value.model_name"),
    ("dataset.loader.parameters.data_name", "dataset.value.loader.parameters.data_name"),
    ("dataset.loader.parameters.adjacency_method", "dataset.value.loader.parameters.adjacency_method"),
    ("dataset.loader.parameters.node_sample_ratio", "dataset.value.loader.parameters.node_sample_ratio"),
    ("dataset.loader.parameters.method", "dataset.value.loader.parameters.method"),
    ("model.readout.readout_name", "model.value.readout.readout_name"),
    ("experiment", "experiment.value"),
)

# Short plot-axis columns (filled from long keys when missing).
_SHORT_FROM_LONG: tuple[tuple[str, str], ...] = (
    ("data_name", "dataset.loader.parameters.data_name"),
    ("adjacency_method", "dataset.loader.parameters.adjacency_method"),
    ("node_sample_ratio", "dataset.loader.parameters.node_sample_ratio"),
    ("sampling_method", "dataset.loader.parameters.method"),
    ("readout_name", "model.readout.readout_name"),
)

DEFAULT_INPUT_CSV = os.path.join(_DIR, "final_results_hyperparams_neurips.csv")
DEFAULT_OUTPUT_CSV = os.path.join(_DIR, "aggregated_final_results_neurips.csv")
DEFAULT_BASELINE_AGG_CSV = os.path.join(_DIR, "baseline_aggregated_gnn_features_neurips.csv")

C_DATA = "data_name"
C_ADJ = "adjacency_method"
C_RATIO = "node_sample_ratio"
C_METHOD = "sampling_method"
C_MODEL = "model_name"
C_READOUT = "readout_name"
C_N = "n_runs_seeds"
VAL_F1 = "best_val_f1_macro_mean"
TEST_F1 = "best_test_f1_macro_mean"
TEST_F1_STD = "best_test_f1_macro_std"
TRAIN_F1 = "best_train_f1_macro_mean"
TRAIN_F1_STD = "best_train_f1_macro_std"
VAL_F1_STD = "best_val_f1_macro_std"

DATASETS = ["motrpac", "addneuromed", "parkinsons", "brca"]
# Capacity-gap diagnostic uses random-train macro-F1 ~= 1 / num_classes.
DATASET_NUM_CLASSES = {
    "motrpac": 2,
    "addneuromed": 3,
    "parkinsons": 2,
    "brca": 2,
}

CANONICAL_MODEL_ORDER = [
    "mlp",
    "gin",
    "gcn",
    "gatv2",
    "sage",
    "gps",
    "sagn",
    "chebnet",
    "gatv4",
]

MODEL_DISPLAY_NAMES = {
    "mlp": "MLP",
    "gin": "GIN",
    "gcn": "GCN",
    "gatv2": "GATv2",
    "sage": "SAGE",
    "gps": "GPS",
    "sagn": "SAGN",
    "chebnet": "ChebNet",
    "gatv4": "MLA-GNN",
}

DATASET_DISPLAY_NAMES = {
    "motrpac": "Heritage",
    "addneuromed": "Addneuromed",
    "parkinsons": "Parkinsons",
    "brca": "BRCA",
}

# Raw `adjacency_method` values from configs → plot / legend labels
ADJ_METHOD_DISPLAY_NAMES: dict[str, str] = {
    "string": "PPI",
    "wgcna": "Co-expression",
}


def _display_adjacency_method(name: object) -> str:
    k = str(name).strip().lower()
    return ADJ_METHOD_DISPLAY_NAMES.get(k, str(name).strip())


# Classical message-passing GNNs (subset used for the simplified mega-style plot)
STANDARD_MPNN_MODELS = ("gin", "gcn", "gatv2", "sage")
# Placeholder in bar-slot lists: one grouped bar replacing all of STANDARD_MPNN_MODELS
_SLOT_MPNN_CLUSTER = "__MPNN_CLUSTER__"

# Readout comparison (Omics vs none) — same semantics as tutorials/final_results.ipynb
READOUT_COMPARE_ORDER = ("OmicsReadOut", "NoReadOut")
READOUT_HATCHES = {"OmicsReadOut": "", "NoReadOut": "."}
READOUT_ALPHAS = {"OmicsReadOut": 0.9, "NoReadOut": 0.6}

MODEL_COLORS = {
    "mlp": "#5B9BD5",
    "gin": "#D62828",
    "gcn": "#E85D04",
    "gatv2": "#F48C06",
    "sage": "#FAA307",
    "gps": "#20BFC3",
    "sagn": "#7B2CBF",
    "chebnet": "#2D6A4F",
    "gatv4": "#52B788",
}

# Validation-rank vs test (pooled): archetype groups for the line + std band figure.
# Each entry: legend label, canonical model keys, line/fill color.
VAL_RANK_ARCHETYPE_GROUPS: tuple[tuple[str, frozenset[str], str], ...] = (
    ("MLP", frozenset({"mlp"}), "#d4ac0d"),
    ("MPNN (GCN, GIN, GATv2, SAGE)", frozenset({"gcn", "gin", "gatv2", "sage"}), "#3498db"),
    ("Graph transformers (GPS)", frozenset({"gps"}), "#e67e22"),
    ("SAGN", frozenset({"sagn"}), "#16a085"),
    ("Omics-specific (ChebNet, MLA-GNN)", frozenset({"chebnet", "gatv4"}), "#8e44ad"),
)
VAL_RANK_ARCHETYPE_N_BINS = 28
# Best-test reference for MLP archetype (ribbon stays gold); others match ribbon color.
VAL_RANK_ARCHETYPE_MLP_BEST_HLINE = "#D62828"
# Zoomed val-rank figures: restrict x-axis and data to validation ranks 1..N (pooled ordering).
VAL_RANK_ZOOM_TOP_N = 100


def _place_val_rank_figure_title(
    fig: mpl.figure.Figure,
    axes_grid: np.ndarray,
    title: str,
    *,
    fontsize: int = 13,
    pad_frac: float = 0.02,
) -> None:
    """Draw the main title just above the top row of axes.

    ``fig.suptitle`` uses figure y-coords that do not follow ``tight_layout`` / ``bbox_inches='tight'``
    the same way as axes, which often leaves a large empty band; anchoring with ``fig.text`` avoids that.
    """
    flat = axes_grid.ravel()
    y1 = max(ax.get_position().y1 for ax in flat)
    y = min(float(y1) + pad_frac, 0.995)
    fig.text(0.5, y, title, ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


def _val_rank_best_test_short_label(archetype_title: str) -> str:
    """Compact legend tag for per-archetype best-test horizontal lines."""
    if archetype_title == "MLP":
        return "MLP"
    if archetype_title.startswith("MPNN"):
        return "MPNN"
    if "GPS" in archetype_title or "transformers" in archetype_title.lower():
        return "GPS"
    if archetype_title == "SAGN":
        return "SAGN"
    if "Omics" in archetype_title:
        return "Omics"
    return archetype_title


# Sklearn baselines: horizontal reference lines (no error bars). Solid vs dotted, per-dataset y from
# best val-F1 row; test F1 macro as the reference level (train−val plot uses the same row for diff).
BASELINE_MODEL_ORDER = ("svm", "elastic_net")
BASELINE_DISPLAY_NAMES = {"svm": "SVM", "elastic_net": "Elastic Net"}
# Overview plots: horizontal reference lines
BASELINE_HLINE_STYLE = {"svm": "-", "elastic_net": ":"}
BASELINE_HLINE_COLOR = {"svm": "#3A3A3A", "elastic_net": "#777777"}
# Method × ratio grid: bars (distinct grays)
BASELINE_BAR_COLORS = {"svm": "#5C5C5C", "elastic_net": "#B5B5B5"}

_baseline_agg_cache: pd.DataFrame | None = None


def _load_baseline_agg(csv_path: str | None = None) -> pd.DataFrame:
    """Aggregated gnn_features baselines; empty DataFrame if file missing or invalid."""
    global _baseline_agg_cache
    path = csv_path or DEFAULT_BASELINE_AGG_CSV
    if _baseline_agg_cache is not None and csv_path is None:
        return _baseline_agg_cache
    if not os.path.isfile(path):
        if csv_path is None:
            _baseline_agg_cache = pd.DataFrame()
        return pd.DataFrame()
    raw = pd.read_csv(path, low_memory=False)
    need = [C_DATA, C_MODEL, C_METHOD, VAL_F1, TEST_F1]
    if any(c not in raw.columns for c in need):
        if csv_path is None:
            _baseline_agg_cache = pd.DataFrame()
        return pd.DataFrame()
    out = raw.copy()
    out[C_DATA] = out[C_DATA].astype(str).str.strip().str.lower()
    out[C_MODEL] = out[C_MODEL].astype(str).str.strip().str.lower()
    out[C_METHOD] = out[C_METHOD].astype(str).str.strip().str.lower()
    for c in (VAL_F1, TEST_F1, TRAIN_F1, TRAIN_F1_STD, VAL_F1_STD, TEST_F1_STD):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[out[C_DATA].isin(DATASETS)]
    if csv_path is None:
        _baseline_agg_cache = out
    return out


def _baseline_ratio_matches(r_csv: object, r_plot: object) -> bool:
    """Align baseline `node_sample_ratio` with GNN ratio ticks (float vs `full`)."""
    if r_csv is None or r_plot is None:
        return False
    if isinstance(r_csv, float) and np.isnan(r_csv):
        return False
    if isinstance(r_plot, float) and np.isnan(r_plot):
        return False
    sc = str(r_csv).strip().lower()
    sp = str(r_plot).strip().lower()
    if sc in ("", "nan", "none") or sp in ("", "nan", "none"):
        return False
    if sc == "full" or sp == "full":
        return sc == sp
    try:
        return round(float(r_csv), 4) == round(float(r_plot), 4)
    except (TypeError, ValueError):
        return sc == sp


def _baseline_mu_sig_per_ratio(
    df_b: pd.DataFrame, dataset: str, method: str, r_plot: object, model_key: str
) -> tuple[float, float]:
    """One aggregated baseline row for (dataset, method, ratio, svm|elastic_net)."""
    if df_b.empty:
        return (float("nan"), float("nan"))
    sub = df_b[
        (df_b[C_DATA] == dataset) & (df_b[C_METHOD] == method) & (df_b[C_MODEL] == model_key)
    ]
    for _, row in sub.iterrows():
        if _baseline_ratio_matches(row[C_RATIO], r_plot):
            mu = float(row[TEST_F1]) if pd.notna(row[TEST_F1]) else float("nan")
            sig = (
                float(row[TEST_F1_STD])
                if TEST_F1_STD in row.index and pd.notna(row[TEST_F1_STD])
                else float("nan")
            )
            return (mu, sig)
    return (float("nan"), float("nan"))


def _baseline_val_mu_per_ratio(
    df_b: pd.DataFrame, dataset: str, method: str, r_plot: object, model_key: str
) -> float:
    """Validation F1 mean for one aggregated baseline row (dataset, method, ratio, model)."""
    if df_b.empty:
        return float("nan")
    sub = df_b[
        (df_b[C_DATA] == dataset) & (df_b[C_METHOD] == method) & (df_b[C_MODEL] == model_key)
    ]
    for _, row in sub.iterrows():
        if _baseline_ratio_matches(row[C_RATIO], r_plot):
            return float(row[VAL_F1]) if pd.notna(row[VAL_F1]) else float("nan")
    return float("nan")


def _baseline_row_best_overall(df_b: pd.DataFrame, dataset: str, model_key: str) -> pd.Series | None:
    """One row: max val F1 over (ratio × method) for this dataset and baseline model."""
    sub = df_b[(df_b[C_DATA] == dataset) & (df_b[C_MODEL] == model_key)]
    if sub.empty:
        return None
    sub_ok = sub.dropna(subset=[VAL_F1, TEST_F1])
    if sub_ok.empty:
        return None
    return sub_ok.loc[sub_ok[VAL_F1].idxmax()]


def _draw_baseline_test_f1_hlines(ax, df_b: pd.DataFrame, dataset: str, *, zorder: float = 4.5) -> None:
    """Full-width horizontal lines at test F1 for best-val baseline configs (per dataset)."""
    if df_b.empty:
        return
    for bk in BASELINE_MODEL_ORDER:
        br = _baseline_row_best_overall(df_b, dataset, bk)
        if br is None or pd.isna(br[TEST_F1]):
            continue
        y = float(br[TEST_F1])
        ax.axhline(
            y,
            color=BASELINE_HLINE_COLOR[bk],
            linestyle=BASELINE_HLINE_STYLE[bk],
            linewidth=2.4,
            zorder=zorder,
            clip_on=True,
        )


def _draw_baseline_train_minus_val_hlines(ax, df_b: pd.DataFrame, dataset: str, *, zorder: float = 4.5) -> None:
    """Horizontal lines at (train−val) F1 macro for the same best-val row; skipped if train is missing."""
    if df_b.empty:
        return
    for bk in BASELINE_MODEL_ORDER:
        br = _baseline_row_best_overall(df_b, dataset, bk)
        diff = _baseline_train_minus_val_from_row(br)
        if diff is None:
            continue
        ax.axhline(
            diff,
            color=BASELINE_HLINE_COLOR[bk],
            linestyle=BASELINE_HLINE_STYLE[bk],
            linewidth=2.4,
            zorder=zorder,
            clip_on=True,
        )


def _baseline_train_minus_val_from_row(br: pd.Series) -> float | None:
    if br is None or pd.isna(br.get(VAL_F1)) or pd.isna(br.get(TRAIN_F1)):
        return None
    return float(br[TRAIN_F1]) - float(br[VAL_F1])


def _baseline_legend_line_handles(*, compact_labels: bool = False) -> list[Line2D]:
    """Line2D handles for baseline hline legend. Use compact_labels when a parent title already says Baseline."""
    svm_l = "SVM" if compact_labels else "SVM baseline"
    en_l = "Elastic Net" if compact_labels else "Elastic Net baseline"
    return [
        Line2D(
            [0],
            [0],
            color=BASELINE_HLINE_COLOR["svm"],
            linestyle=BASELINE_HLINE_STYLE["svm"],
            linewidth=2.4,
            label=svm_l,
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE_HLINE_COLOR["elastic_net"],
            linestyle=BASELINE_HLINE_STYLE["elastic_net"],
            linewidth=2.4,
            label=en_l,
        ),
    ]


def _baseline_legend_trainval_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=BASELINE_HLINE_COLOR["svm"],
            linestyle=BASELINE_HLINE_STYLE["svm"],
            linewidth=2.4,
            label="SVM baseline (train−val, same best-val row)",
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE_HLINE_COLOR["elastic_net"],
            linestyle=BASELINE_HLINE_STYLE["elastic_net"],
            linewidth=2.4,
            label="Elastic Net baseline (train−val, same best-val row)",
        ),
    ]


def _strict_oversized_seed_groups() -> bool:
    """If False (env OGBENCH_RELAX_OVERSIZED_GROUPS=1), skip the >3-runs bucket check."""
    return os.environ.get("OGBENCH_RELAX_OVERSIZED_GROUPS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def _format_oversized_group(idx: object, group_cols: list[str]) -> str:
    if isinstance(idx, tuple) and len(idx) == len(group_cols):
        pairs = list(zip(group_cols, idx))[:12]
        body = "\n      ".join(f"{k}={v!r}" for k, v in pairs)
        if len(group_cols) > 12:
            body += "\n      ..."
        return body
    return repr(idx)


def assert_no_oversized_seed_groups(
    n_runs: pd.Series,
    n_seeds: pd.Series,
    *,
    group_cols: list[str],
) -> None:
    """Raise if any config fingerprint has > EXPECTED_SEEDS runs after (config+seed) dedupe."""
    if n_runs.empty:
        return
    bad_eq = n_runs != n_seeds
    if bad_eq.any():
        idx = bad_eq[bad_eq].index[0]
        raise ValueError(
            f"Group size != distinct {SEED_COL} values after deduplicating rows — data bug or "
            f"duplicate {SEED_COL} with conflicting metrics in the same bucket.\n"
            f"  size={int(n_runs.loc[idx])} nunique({SEED_COL})={int(n_seeds.loc[idx])}\n"
            f"  {_format_oversized_group(idx, group_cols)}"
        )
    over = n_runs > EXPECTED_SEEDS
    if not over.any():
        return
    n_bad = int(over.sum())
    examples = "\n\n".join(
        f"  — size={int(n_runs.loc[idx])} distinct_seeds={int(n_seeds.loc[idx])}\n"
        f"      {_format_oversized_group(idx, group_cols)}"
        for idx in list(over[over].index[:5])
    )
    raise ValueError(
        f"{n_bad} config bucket(s) have MORE than {EXPECTED_SEEDS} runs after grouping on "
        f"the experiment fingerprint keys (FINGERPRINT_KEY_CANDIDATES).\n\n"
        f"That usually means two different experiments share the same flattened config keys "
        f"(rare) or duplicate W&B rows.\n\n"
        f"To bypass this check (not recommended): export OGBENCH_RELAX_OVERSIZED_GROUPS=1\n\n"
        f"First buckets (up to 5):\n\n{examples}"
    )


def _models_to_diagnose(df: pd.DataFrame, diagnose_models: Sequence[str] | None) -> tuple[str, ...]:
    """All canonical models in `df` (stable order), or an explicit subset."""
    if diagnose_models is not None:
        return tuple(diagnose_models)
    found = {str(x) for x in df[C_MODEL].dropna().unique()}
    ordered = [m for m in CANONICAL_MODEL_ORDER if m in found]
    ordered.extend(sorted(found - set(ordered)))
    return tuple(ordered)


def _normalize_seed_column(df: pd.DataFrame) -> None:
    if "seed" not in df.columns and "seed.value" in df.columns:
        df["seed"] = df["seed.value"]
        df.drop(columns=["seed.value"], inplace=True)
    elif "seed.value" in df.columns:
        s = pd.to_numeric(df["seed"], errors="coerce")
        v = pd.to_numeric(df["seed.value"], errors="coerce")
        df["seed"] = s.where(s.notna(), v)
        df.drop(columns=["seed.value"], inplace=True)


def _coalesce_hydra_value_variants(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for plain, value in _HYDRA_VALUE_COALESCE:
        if value not in out.columns:
            continue
        if plain in out.columns:
            out[plain] = out[plain].where(out[plain].notna(), out[value])
        else:
            out[plain] = out[value]
        out = out.drop(columns=[value], errors="ignore")
    return out


def _mirror_short_axis_columns(df: pd.DataFrame) -> None:
    for short, long in _SHORT_FROM_LONG:
        if short not in df.columns and long in df.columns:
            df[short] = df[long]


def _drop_redundant_long_axis_columns(df: pd.DataFrame) -> None:
    for short, long in _SHORT_FROM_LONG:
        if short in df.columns and long in df.columns:
            df.drop(columns=[long], inplace=True)
    if C_MODEL in df.columns and "model.model_name" in df.columns:
        df.drop(columns=["model.model_name"], inplace=True)


def _ensure_canonical_model_column(df: pd.DataFrame) -> None:
    if "model.model_name" in df.columns:
        raw = df["model.model_name"]
    elif C_MODEL in df.columns:
        raw = df[C_MODEL]
    else:
        raise ValueError(
            "Per-run CSV must include 'model_name' and/or 'model.model_name' (from W&B config)."
        )
    df[C_MODEL] = raw.map(canonical_model_name)


# One hyperparameter experiment = one tuple of column values (first existing name wins per slot).
# A pure "all columns minus blacklist" groupby is unique per row on flattened W&B CSVs (hundreds of
# low-cardinality fields still combine to a unique tuple), so aggregation must use this compact key.
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


def _resolve_fingerprint_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for candidates in FINGERPRINT_KEY_CANDIDATES:
        for c in candidates:
            if c in df.columns:
                cols.append(c)
                break
    return cols


def _prune_run_unique_group_cols(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    """Drop fingerprint keys where every row has a distinct value (cannot group seeds)."""
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
    """Stable string key for a full hyperparameter bucket."""
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


def _build_lean_aggregated(
    wide: pd.DataFrame,
    *,
    metric_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    cols = tuple(metric_cols) if metric_cols is not None else PER_RUN_METRIC_COLS
    lean = pd.DataFrame(index=wide.index)
    for short, long in _SHORT_FROM_LONG:
        if short in wide.columns:
            lean[short] = wide[short]
        elif long in wide.columns:
            lean[short] = wide[long]
        else:
            lean[short] = np.nan
    lean[C_MODEL] = wide[C_MODEL] if C_MODEL in wide.columns else np.nan
    if BUCKET_KEY_COL in wide.columns:
        lean[BUCKET_KEY_COL] = wide[BUCKET_KEY_COL]
    lean["n_runs_seeds"] = wide["n_runs_seeds"]
    for m in cols:
        lean[f"{m}_mean"] = wide[f"{m}_mean"]
        lean[f"{m}_std"] = wide[f"{m}_std"]
    col_order = [
        C_DATA,
        C_MODEL,
        C_ADJ,
        C_RATIO,
        C_METHOD,
        "readout_name",
        BUCKET_KEY_COL,
        "n_runs_seeds",
    ]
    for m in cols:
        col_order.extend([f"{m}_mean", f"{m}_std"])
    return lean[[c for c in col_order if c in lean.columns]]


def _verbose_aggregation_report(
    df: pd.DataFrame,
    group_cols: list[str],
    n_runs: pd.Series,
    n_seeds: pd.Series,
    *,
    diagnose_models: Sequence[str] | None,
    strict_oversized: bool,
) -> None:
    """Global bucket stats, oversized warning, per-model kept vs dropped config counts."""
    k = EXPECTED_SEEDS
    models = _models_to_diagnose(df, diagnose_models)
    eligible = (n_runs == k) & (n_seeds == k)
    n_buckets = int(len(n_runs))
    n_elig = int(eligible.sum())
    over = n_runs > k
    n_over = int(over.sum())
    under_or_wrong = (~eligible) & (~over)
    bad_eq = n_runs != n_seeds
    n_mismatch = int(bad_eq.sum())

    print(
        f"\n{'=' * 72}\nAggregation (group = experiment fingerprint keys; see "
        f"FINGERPRINT_KEY_CANDIDATES)\n{'=' * 72}"
    )
    print(f"Group columns: {len(group_cols)}")
    print(f"Per-run rows: {len(df)}")
    print(f"Config buckets: {n_buckets}")
    print(f"  Kept (exactly {k} rows and {k} distinct seeds): {n_elig}")
    print(f"  Dropped buckets: {n_buckets - n_elig}")

    dropped = ~eligible
    if dropped.any():
        sub_ns = n_seeds[dropped]
        sub_nr = n_runs[dropped]
        print("\n  Dropped buckets: counts by (distinct seeds, row count):")
        ct = (
            pd.DataFrame({"distinct_seeds": sub_ns.values, "rows_in_bucket": sub_nr.values})
            .value_counts()
            .sort_index()
        )
        print(ct.to_string())

    if n_over > 0:
        runs_in_over = int(n_runs[over].sum())
        print(
            f"\n*** WARNING: {n_over} bucket(s) have MORE than {k} runs "
            f"({runs_in_over} per-run rows) ***"
        )
        print(
            "  Unusual for a full-config groupby: check duplicate W&B rows or inconsistent CSV columns."
        )
        if not strict_oversized:
            print("  (Strict check skipped: OGBENCH_RELAX_OVERSIZED_GROUPS is set.)")
        for idx in list(over[over].index[:3]):
            print(f"  example bucket: {_format_oversized_group(idx, group_cols)}")

    if n_mismatch > 0:
        print(
            f"\n  Note: {n_mismatch} bucket(s) where row count != distinct seeds "
            f"(duplicate {SEED_COL} with different metrics?)."
        )

    print(f"\n--- Per model (eligibility = {k} runs & {k} seeds per bucket) ---")
    for m in models:
        sub = df[df[C_MODEL] == m]
        if sub.empty:
            print(f"  {m}: no rows (check model_name / canonical_model_name)")
            continue
        g = sub.groupby(group_cols, dropna=False)
        sz = g.size()
        nu = g[SEED_COL].nunique()
        ok = (sz == k) & (nu == k)
        n_g = len(sz)
        n_ok = int(ok.sum())
        dropped_runs = int(sz[~ok].sum()) if (~ok).any() else 0
        print(
            f"  {m}: per_run_rows={len(sub)} buckets={n_g} kept={n_ok} dropped_buckets={n_g - n_ok} "
            f"per_run_rows_in_dropped_buckets={dropped_runs}"
        )


def _verbose_plot_prep_report(agg: pd.DataFrame, diagnose_models: Sequence[str] | None) -> None:
    """What load_prepared_df keeps: finite metrics + dataset whitelist."""
    need = [C_DATA, C_ADJ, C_RATIO, C_METHOD, C_MODEL, VAL_F1, TEST_F1, TRAIN_F1]
    models = _models_to_diagnose(agg, diagnose_models)
    print(f"\n{'=' * 72}\nPlot prep (same filters as load_prepared_df)\n{'=' * 72}")
    for m in models:
        base = agg[agg[C_MODEL] == m]
        if base.empty:
            print(f"  {m}: no rows in aggregated CSV")
            continue
        miss = [c for c in need if c not in base.columns]
        if miss:
            print(f"  {m}: missing columns {miss}")
            continue
        m_ok = base[VAL_F1].notna() & base[TEST_F1].notna() & base[TRAIN_F1].notna()
        after_m = base.loc[m_ok]
        ds = after_m[C_DATA].isin(DATASETS)
        after_d = after_m.loc[ds]
        print(
            f"  {m}: agg_rows={len(base)} after_finite_metrics={len(after_m)} "
            f"plot_datasets={len(after_d)} (dropped_other_ds={len(after_m) - len(after_d)})"
        )


def prepare_per_run_df_for_fingerprint_grouping(
    df: pd.DataFrame,
    *,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Finished per-run rows, numeric seed/metrics, fingerprint dedupe — same front matter as aggregation export.

    Returns ``(df, group_cols)`` ready for ``df.groupby(group_cols)``. Adds ``BUCKET_KEY_COL`` per row.
    """
    df = df.copy()

    if "state" in df.columns:
        n0 = len(df)
        fin = df["state"].astype(str).str.lower().eq("finished")
        df = df.loc[fin].copy()
        if verbose:
            print(f"Kept {len(df)} / {n0} rows with state=='finished'")
    elif verbose:
        print("No 'state' column; not filtering by finished.")

    miss_m = [m for m in PER_RUN_METRIC_COLS if m not in df.columns]
    if miss_m:
        raise ValueError(
            f"Per-run CSV missing metric columns {miss_m}. Expected {list(PER_RUN_METRIC_COLS)}."
        )

    _normalize_seed_column(df)
    if SEED_COL not in df.columns:
        raise ValueError(
            f"Per-run CSV must include {SEED_COL!r} or seed.value (from W&B / Hydra)."
        )

    df = _coalesce_hydra_value_variants(df)
    _mirror_short_axis_columns(df)

    if "model.model_name" in df.columns:
        raw_model = df["model.model_name"].copy()
    elif C_MODEL in df.columns:
        raw_model = df[C_MODEL].copy()
    else:
        raw_model = None

    _ensure_canonical_model_column(df)

    if "run_id" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=["run_id"], keep="last")
        if verbose and len(df) < n_before:
            print(f"Dropped {n_before - len(df)} duplicate run_id (keep=last)")

    if verbose and raw_model is not None:
        n_bad = int(df[C_MODEL].isna().sum())
        if n_bad:
            print(f"Unmapped model_name: dropped {n_bad} rows; raw counts (top 15):")
            print(raw_model[df[C_MODEL].isna()].astype(str).value_counts().head(15).to_string())

    df = df.dropna(subset=[C_MODEL])

    n_seed_before = len(df)
    df[SEED_COL] = pd.to_numeric(df[SEED_COL], errors="coerce")
    df = df.dropna(subset=[SEED_COL])
    if verbose and len(df) < n_seed_before:
        print(f"Dropped {n_seed_before - len(df)} rows (non-numeric or missing seed)")

    for m in PER_RUN_METRIC_COLS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    for m in OPTIONAL_EXTRA_TEST_METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    for col in ("node_sample_ratio", C_RATIO):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _drop_redundant_long_axis_columns(df)

    group_cols = _resolve_fingerprint_columns(df)
    group_cols, dropped_unique = _prune_run_unique_group_cols(df, group_cols)
    if verbose and dropped_unique:
        print(
            f"Pruned {len(dropped_unique)} fingerprint key(s) with nunique >= n_rows ({len(df)}): "
            f"{dropped_unique}"
        )
    if not group_cols:
        raise ValueError(
            "No fingerprint columns resolved. Check CSV vs FINGERPRINT_KEY_CANDIDATES in generate_main_plots.py."
        )
    if verbose:
        print(f"Grouping on {len(group_cols)} fingerprint columns.")

    dedupe_subset = group_cols + [SEED_COL]
    n_before_cfg = len(df)
    if "run_id" in df.columns:
        df = df.sort_values("run_id", kind="mergesort")
    df = df.drop_duplicates(subset=dedupe_subset, keep="last")
    if verbose and len(df) < n_before_cfg:
        print(
            f"Deduped {n_before_cfg - len(df)} rows (same fingerprint + {SEED_COL}, keep=last by run_id)"
        )

    df[BUCKET_KEY_COL] = _compose_bucket_key_frame(df, group_cols)
    return df, group_cols


def export_aggregated_final_results(
    input_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    *,
    verbose: bool = True,
    diagnose_models: Sequence[str] | None = None,
    best_val_f1_per_plot_slice: bool = False,
) -> pd.DataFrame:
    """Finished runs; groupby fingerprint keys; keep 3-seed buckets; lean CSV. Plots call _pick_best_row by val F1.

    If best_val_f1_per_plot_slice=True, keep one row per (data, model, adjacency, ratio, method, readout) with max val F1.
    """
    input_csv = input_csv or DEFAULT_INPUT_CSV
    output_csv = output_csv or DEFAULT_OUTPUT_CSV

    df = pd.read_csv(input_csv, low_memory=False)
    if verbose:
        print(f"Read {len(df)} per-run rows from {input_csv}")

    df, group_cols = prepare_per_run_df_for_fingerprint_grouping(df, verbose=verbose)

    metric_cols = list(PER_RUN_METRIC_COLS)
    for c in OPTIONAL_EXTRA_TEST_METRICS:
        if c in df.columns and df[c].notna().any() and c not in metric_cols:
            metric_cols.append(c)

    gb = df.groupby(group_cols, dropna=False)
    n_runs = gb.size()
    n_seeds_distinct = gb[SEED_COL].nunique()
    strict = _strict_oversized_seed_groups()

    if verbose:
        _verbose_aggregation_report(
            df,
            group_cols,
            n_runs,
            n_seeds_distinct,
            diagnose_models=diagnose_models,
            strict_oversized=strict,
        )

    if strict:
        assert_no_oversized_seed_groups(n_runs, n_seeds_distinct, group_cols=group_cols)
    elif verbose:
        over = (n_runs > EXPECTED_SEEDS) | (n_seeds_distinct > EXPECTED_SEEDS)
        if over.any():
            print(
                f"(Relaxed mode) Would-have-failed strict check: {int(over.sum())} bucket(s) "
                f"> {EXPECTED_SEEDS} runs or seeds."
            )

    wide = gb.agg({m: ["mean", "std"] for m in metric_cols})
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide[BUCKET_KEY_COL] = _compose_bucket_key_frame(wide, group_cols)
    wide["n_runs_seeds"] = n_runs.values
    ok = (wide["n_runs_seeds"] == EXPECTED_SEEDS) & (n_seeds_distinct.values == EXPECTED_SEEDS)
    pre = len(wide)
    wide = wide.loc[ok].copy()

    if verbose:
        print(f"\nLean aggregated rows: {len(wide)} / {pre} config buckets before 3-seed filter")

    out = _build_lean_aggregated(wide, metric_cols=metric_cols)

    if best_val_f1_per_plot_slice:
        slice_cols = [
            c for c in (C_DATA, C_MODEL, C_ADJ, C_RATIO, C_METHOD, "readout_name") if c in out.columns
        ]
        pre_slice = len(out)
        out = (
            out.sort_values(VAL_F1, ascending=False, na_position="last")
            .drop_duplicates(subset=slice_cols, keep="first")
            .sort_values(slice_cols, kind="mergesort")
            .reset_index(drop=True)
        )
        if verbose and len(out) < pre_slice:
            print(
                f"Kept best val F1 per plot slice ({', '.join(slice_cols)}): "
                f"{len(out)} rows (dropped {pre_slice - len(out)} duplicate slice(s))"
            )

    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _display_model(m: str) -> str:
    if m in BASELINE_DISPLAY_NAMES:
        return BASELINE_DISPLAY_NAMES[m]
    return MODEL_DISPLAY_NAMES.get(m, m)


def _display_dataset(d: str) -> str:
    return DATASET_DISPLAY_NAMES.get(d, str(d).title())


def _model_color(m: str) -> str:
    return MODEL_COLORS.get(m, "#BBBBBB")


def _bar_facecolor(model_key: str, *, has_value: bool) -> tuple:
    """Model color at full opacity, or a faded tint when there is no metric (not generic gray)."""
    r, g, b, _ = to_rgba(_model_color(model_key))
    if has_value:
        return (r, g, b, 1.0)
    return (r, g, b, 0.28)


def _save_figure_pdf_png(fig, basename_with_ext: str, *, dpi: int = SAVEFIG_DPI) -> None:
    """Write PDF and PNG under PLOTS_DIR (extension in basename_with_ext is ignored)."""
    stem = os.path.splitext(os.path.basename(basename_with_ext))[0] or "figure"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    base = os.path.normpath(os.path.join(PLOTS_DIR, stem))
    pdf_path = f"{base}.pdf"
    png_path = f"{base}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)

    def _write_png_to_path(dest_png: str, png_dpi: int) -> None:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".png", dir=PLOTS_DIR, prefix=".tmp_savefig_"
        )
        tmp_path = tmp.name
        tmp.close()
        try:
            fig.savefig(tmp_path, bbox_inches="tight", dpi=png_dpi, format="png")
        except BaseException:
            if os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise
        try:
            _replace_file_resilient(tmp_path, dest_png)
        except OSError as e:
            if _os_error_replace_dest_busy(e):
                raise PngDestReplaceBlocked(str(e)) from e
            raise

    def _write_png_direct(dest_png: str, png_dpi: int) -> None:
        """Last resort: overwrite in place (no temp+replace). Sometimes succeeds when replace does not."""
        try:
            fig.savefig(dest_png, bbox_inches="tight", dpi=png_dpi, format="png")
        except OSError as e:
            # Windows + Pillow sometimes fail opening ``dest_png`` with EINVAL (22) even when a
            # sibling temp path works; write via ``.wip.png`` then replace (with lock retries).
            if os.name != "nt" or getattr(e, "errno", None) != 22:
                raise
            wip = f"{dest_png}.wip.png"
            fig.savefig(wip, bbox_inches="tight", dpi=png_dpi, format="png")
            try:
                _replace_file_resilient(wip, dest_png)
            except OSError:
                if os.path.isfile(wip):
                    try:
                        os.unlink(wip)
                    except OSError:
                        pass
                raise

    # Temp file + atomic replace avoids Windows EINVAL when the destination PNG is open elsewhere.
    # Do not treat errno 22 from PIL opening ``dest`` as "replace busy": that is often a non-retryable
    # Windows/Pillow issue; mis-classifying it caused useless retries then ``raise last_exc``.
    dpi_tries = [dpi]
    if dpi > 300:
        dpi_tries.append(min(dpi, 300))
    last_exc: BaseException | None = None
    for png_dpi in dpi_tries:
        for attempt in range(8):
            try:
                _write_png_to_path(png_path, png_dpi)
                return
            except PngDestReplaceBlocked as e:
                last_exc = e
                time.sleep(0.08 * (attempt + 1))
        try:
            _write_png_direct(png_path, png_dpi)
            return
        except OSError as e:
            last_exc = e
            if not _os_error_direct_save_dest_busy(e):
                raise
    if last_exc is not None:
        raise last_exc


def load_prepared_df(
    csv_path: str | None = None,
    *,
    required_seeds: int | None = None,
) -> pd.DataFrame:
    path = csv_path or DEFAULT_OUTPUT_CSV
    k = EXPECTED_SEEDS if required_seeds is None else required_seeds
    df = pd.read_csv(path, low_memory=False)
    df = df[df[C_N] == k].copy()
    for c in (C_DATA, C_ADJ, C_RATIO, C_METHOD, C_MODEL):
        if c not in df.columns:
            raise KeyError(f"Missing column {c!r} in {path}")
    df[C_MODEL] = df[C_MODEL].map(canonical_model_name)
    df = df.dropna(subset=[C_MODEL])
    df[C_METHOD] = df[C_METHOD].astype(str).str.lower()
    df[C_RATIO] = pd.to_numeric(df[C_RATIO], errors="coerce").round(2)
    for mcol in (VAL_F1, TEST_F1, TEST_F1_STD, TRAIN_F1, TRAIN_F1_STD, VAL_F1_STD):
        if mcol in df.columns:
            df[mcol] = pd.to_numeric(df[mcol], errors="coerce")
    need = [C_DATA, C_ADJ, C_RATIO, C_METHOD, C_MODEL, VAL_F1, TEST_F1, TRAIN_F1]
    df = df.dropna(subset=need)
    df = df[df[C_DATA].isin(DATASETS)]
    return df


def load_per_run_plot_df(
    csv_path: str | None = None,
    *,
    finished_only: bool = True,
) -> pd.DataFrame:
    """Prepared per-run rows for run-level diagnostics (no seed aggregation)."""
    path = csv_path or DEFAULT_INPUT_CSV
    df = pd.read_csv(path, low_memory=False)

    if finished_only and "state" in df.columns:
        fin = df["state"].astype(str).str.lower().eq("finished")
        df = df.loc[fin].copy()
    else:
        df = df.copy()

    _normalize_seed_column(df)
    df = _coalesce_hydra_value_variants(df)
    _mirror_short_axis_columns(df)
    _ensure_canonical_model_column(df)
    _drop_redundant_long_axis_columns(df)

    if "run_id" in df.columns:
        df = df.drop_duplicates(subset=["run_id"], keep="last")
    df = df.dropna(subset=[C_MODEL])

    for col in (SEED_COL, C_RATIO, *PER_RUN_METRIC_COLS):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for c in (C_METHOD, C_ADJ, C_DATA):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    if C_READOUT in df.columns:
        df[C_READOUT] = df[C_READOUT].astype(str).str.strip()

    need = [C_DATA, C_MODEL, VAL_F1.replace("_mean", ""), TEST_F1.replace("_mean", ""), TRAIN_F1.replace("_mean", "")]
    df = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    df = df[df[C_DATA].isin(DATASETS)].copy()
    group_cols = _resolve_fingerprint_columns(df)
    group_cols, _ = _prune_run_unique_group_cols(df, group_cols)
    if group_cols:
        df[BUCKET_KEY_COL] = _compose_bucket_key_frame(df, group_cols)
    return df


def _draw_heatmap_with_annotations(
    ax,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    title: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    fmt: str = ".2f",
    extra_text: np.ndarray | None = None,
) -> None:
    shown = np.ma.masked_invalid(matrix)
    im = ax.imshow(shown, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels([_display_dataset(d) for d in col_labels], rotation=35, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels([_display_model(m) for m in row_labels], fontsize=11)
    ax.tick_params(axis="both", length=0)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if np.isnan(val):
                txt = "NA"
            else:
                txt = format(float(val), fmt)
            if extra_text is not None and extra_text[i, j]:
                txt = f"{txt}\n{extra_text[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11)


def plot_val_test_run_correlation(
    df_runs: pd.DataFrame,
    out_name: str = "runlevel_val_test_correlation_by_model_dataset.pdf",
) -> None:
    """Pearson correlation between val and test across all runs per (model, dataset)."""
    val_col, test_col = "best_val_f1_macro", "best_test_f1_macro"
    dfc = df_runs.dropna(subset=[C_DATA, C_MODEL, val_col, test_col]).copy()
    if dfc.empty:
        print(f"Skipping {out_name}: no finite run-level val/test rows.")
        return

    models = _ordered_models(dfc)
    corr_mat = np.full((len(models), len(DATASETS)), np.nan, dtype=float)
    n_mat = np.zeros((len(models), len(DATASETS)), dtype=int)
    labels = np.empty((len(models), len(DATASETS)), dtype=object)
    labels[:, :] = ""

    for i, m in enumerate(models):
        for j, d in enumerate(DATASETS):
            sub = dfc[(dfc[C_MODEL] == m) & (dfc[C_DATA] == d)][[val_col, test_col]].dropna()
            n = len(sub)
            n_mat[i, j] = n
            labels[i, j] = f"n={n}"
            if n < 3:
                continue
            if sub[val_col].nunique() <= 1 or sub[test_col].nunique() <= 1:
                continue
            corr_mat[i, j] = float(sub[val_col].corr(sub[test_col], method="pearson"))

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(DATASETS) * 2.5), max(4.5, len(models) * 0.8 + 1.5)))
    _draw_heatmap_with_annotations(
        ax,
        corr_mat,
        models,
        DATASETS,
        title="Run-level Val/Test Correlation (Pearson r)",
        cbar_label="Pearson r",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        fmt=".2f",
        extra_text=labels,
    )
    plt.tight_layout()
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_perfect_train_diagnostics(
    df_runs: pd.DataFrame,
    out_name: str = "runlevel_perfect_train_diagnostics.pdf",
) -> None:
    """Quantify perfect-train prevalence and its association with validation/test performance."""
    val_col, test_col, train_col = "best_val_f1_macro", "best_test_f1_macro", "best_train_f1_macro"
    dfc = df_runs.dropna(subset=[C_DATA, C_MODEL, val_col, test_col, train_col]).copy()
    if dfc.empty:
        print(f"Skipping {out_name}: no finite run-level train/val/test rows.")
        return

    dfc["is_perfect_train"] = dfc[train_col] >= PERFECT_TRAIN_F1_THRESHOLD
    models = _ordered_models(dfc)
    frac_mat = np.full((len(models), len(DATASETS)), np.nan, dtype=float)
    delta_test_mat = np.full((len(models), len(DATASETS)), np.nan, dtype=float)
    frac_lbl = np.empty((len(models), len(DATASETS)), dtype=object)
    dtest_lbl = np.empty((len(models), len(DATASETS)), dtype=object)
    frac_lbl[:, :] = ""
    dtest_lbl[:, :] = ""

    for i, m in enumerate(models):
        for j, d in enumerate(DATASETS):
            sub = dfc[(dfc[C_MODEL] == m) & (dfc[C_DATA] == d)]
            n = len(sub)
            if n == 0:
                continue
            perf = sub[sub["is_perfect_train"]]
            non_perf = sub[~sub["is_perfect_train"]]
            frac = float(len(perf) / n)
            frac_mat[i, j] = frac
            frac_lbl[i, j] = f"{len(perf)}/{n}"
            if not perf.empty and not non_perf.empty:
                delta_test = float(perf[test_col].mean() - non_perf[test_col].mean())
                delta_test_mat[i, j] = delta_test
                dtest_lbl[i, j] = f"n={len(perf)}/{len(non_perf)}"
            else:
                dtest_lbl[i, j] = "insufficient split"

    fig, axes = plt.subplots(1, 2, figsize=(max(13, len(DATASETS) * 4), max(5, len(models) * 0.7 + 1.5)))
    _draw_heatmap_with_annotations(
        axes[0],
        frac_mat,
        models,
        DATASETS,
        title=f"Perfect-train rate (train F1 >= {PERFECT_TRAIN_F1_THRESHOLD:.3f})",
        cbar_label="Fraction of runs",
        vmin=0.0,
        vmax=1.0,
        cmap="YlOrRd",
        fmt=".2f",
        extra_text=frac_lbl,
    )
    _draw_heatmap_with_annotations(
        axes[1],
        delta_test_mat,
        models,
        DATASETS,
        title="Test F1 delta: perfect-train minus non-perfect",
        cbar_label="Delta test F1",
        vmin=-0.25,
        vmax=0.25,
        cmap="coolwarm",
        fmt="+.3f",
        extra_text=dtest_lbl,
    )
    plt.tight_layout()
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def _selection_group_columns(df_runs: pd.DataFrame) -> list[str]:
    """Hyperparameter identity columns for run-level selection analysis (excluding seed)."""
    cols = _resolve_fingerprint_columns(df_runs)
    cols, _ = _prune_run_unique_group_cols(df_runs, cols)
    cols = [c for c in cols if c in df_runs.columns]
    if cols:
        return cols
    fallback = [c for c in (C_DATA, C_MODEL, C_ADJ, C_RATIO, C_METHOD, C_READOUT) if c in df_runs.columns]
    return fallback


def plot_val_selection_regret(
    df_runs: pd.DataFrame,
    out_name: str = "runlevel_val_selection_regret_by_model_dataset.pdf",
) -> None:
    """How often best-val run misses best-test run, and by how much."""
    val_col, test_col = "best_val_f1_macro", "best_test_f1_macro"
    dfc = df_runs.dropna(subset=[C_DATA, C_MODEL, val_col, test_col]).copy()
    if dfc.empty:
        print(f"Skipping {out_name}: no finite run-level val/test rows.")
        return

    group_cols = _selection_group_columns(dfc)
    if not group_cols:
        print(f"Skipping {out_name}: could not resolve grouping columns for selection analysis.")
        return

    rows: list[dict[str, float | str | int]] = []
    for _, sub in dfc.groupby(group_cols, dropna=False):
        sub = sub[[C_DATA, C_MODEL, val_col, test_col]].dropna()
        if len(sub) < 2:
            continue
        idx_best_val = sub[val_col].idxmax()
        test_at_best_val = float(sub.loc[idx_best_val, test_col])
        best_test = float(sub[test_col].max())
        regret = best_test - test_at_best_val
        miss = regret > VAL_SELECTION_TIE_TOL
        rows.append(
            {
                C_DATA: str(sub[C_DATA].iloc[0]),
                C_MODEL: str(sub[C_MODEL].iloc[0]),
                "miss": float(miss),
                "regret": regret,
            }
        )
    if not rows:
        print(f"Skipping {out_name}: no multi-run config buckets for selection analysis.")
        return
    ev = pd.DataFrame(rows)
    models = _ordered_models(ev)

    miss_mat = np.full((len(models), len(DATASETS)), np.nan, dtype=float)
    regret_mat = np.full((len(models), len(DATASETS)), np.nan, dtype=float)
    miss_lbl = np.empty((len(models), len(DATASETS)), dtype=object)
    reg_lbl = np.empty((len(models), len(DATASETS)), dtype=object)
    miss_lbl[:, :] = ""
    reg_lbl[:, :] = ""

    for i, m in enumerate(models):
        for j, d in enumerate(DATASETS):
            sub = ev[(ev[C_MODEL] == m) & (ev[C_DATA] == d)]
            n = len(sub)
            if n == 0:
                continue
            miss_mat[i, j] = float(sub["miss"].mean())
            regret_mat[i, j] = float(sub["regret"].mean())
            miss_lbl[i, j] = f"n={n}"
            reg_lbl[i, j] = f"n={n}"

    finite_regret = regret_mat[np.isfinite(regret_mat)]
    reg_max = float(np.quantile(np.abs(finite_regret), 0.95)) if finite_regret.size else 0.1
    reg_max = max(reg_max, 0.02)

    fig, axes = plt.subplots(1, 2, figsize=(max(13, len(DATASETS) * 4), max(5, len(models) * 0.7 + 1.5)))
    _draw_heatmap_with_annotations(
        axes[0],
        miss_mat,
        models,
        DATASETS,
        title="Val-selection miss rate",
        cbar_label="Fraction of configs where best val != best test",
        vmin=0.0,
        vmax=1.0,
        cmap="YlGnBu",
        fmt=".2f",
        extra_text=miss_lbl,
    )
    _draw_heatmap_with_annotations(
        axes[1],
        regret_mat,
        models,
        DATASETS,
        title="Val-selection regret in test F1",
        cbar_label="Max test - test at best val",
        vmin=0.0,
        vmax=reg_max,
        cmap="magma",
        fmt=".3f",
        extra_text=reg_lbl,
    )
    plt.tight_layout()
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_capacity_gap_vs_generalization_efficiency(
    df: pd.DataFrame,
    out_name: str = "capacity_gap_vs_generalization_efficiency.pdf",
) -> None:
    """Scatter: effective capacity utilization vs generalization efficiency."""
    need = [C_MODEL, C_DATA, C_ADJ, VAL_F1, TEST_F1, TRAIN_F1]
    if any(c not in df.columns for c in need):
        print(f"Skipping {out_name}: missing required columns.")
        return

    adjs = sorted(df[C_ADJ].dropna().unique().tolist())
    models = _ordered_models(df)
    points: list[dict[str, object]] = []

    for m in models:
        for d in DATASETS:
            n_classes = int(DATASET_NUM_CLASSES.get(d, 2))
            random_baseline_train_f1 = 1.0 / float(max(2, n_classes))
            denom = max(1e-12, 1.0 - random_baseline_train_f1)
            for a in adjs:
                sub = df[(df[C_MODEL] == m) & (df[C_DATA] == d) & (df[C_ADJ] == a)]
                row = _pick_best_row(sub, require_test_f1=True, require_train_f1=True)
                if row is None:
                    continue
                train_f1 = float(row[TRAIN_F1])
                test_f1 = float(row[TEST_F1])
                if not np.isfinite(train_f1) or train_f1 <= 0:
                    continue
                cap_gap = (train_f1 - random_baseline_train_f1) / denom
                gen_eff = test_f1 / train_f1
                points.append(
                    {
                        C_MODEL: m,
                        C_DATA: d,
                        C_ADJ: a,
                        "capacity_gap": float(np.clip(cap_gap, 0.0, 1.5)),
                        "generalization_efficiency": float(gen_eff),
                    }
                )

    if not points:
        print(f"Skipping {out_name}: no finite points.")
        return
    p = pd.DataFrame(points)
    marker_map = {a: mk for a, mk in zip(adjs, ["o", "s", "^", "D", "P", "X"], strict=False)}

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    for a in adjs:
        for m in models:
            sub = p[(p[C_ADJ] == a) & (p[C_MODEL] == m)]
            if sub.empty:
                continue
            ax.scatter(
                sub["capacity_gap"],
                sub["generalization_efficiency"],
                s=85,
                c=[_model_color(m)],
                marker=marker_map[a],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.88,
                zorder=3,
            )

    for _, r in p.iterrows():
        ax.text(
            float(r["capacity_gap"]) + 0.008,
            float(r["generalization_efficiency"]) + 0.005,
            _display_dataset(str(r[C_DATA]))[:3],
            fontsize=8,
            alpha=0.72,
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.3, alpha=0.8)
    ax.set_xlabel("Effective capacity utilization", fontsize=14)
    ax.set_ylabel("Generalization efficiency (test F1 / train F1)", fontsize=14)
    ax.set_title("Capacity Gap vs Generalization Efficiency", fontsize=16, fontweight="bold", pad=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(-0.02, max(1.05, float(p["capacity_gap"].max()) + 0.05))
    ax.set_ylim(max(0.0, float(p["generalization_efficiency"].min()) - 0.05), max(1.08, float(p["generalization_efficiency"].max()) + 0.05))

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_model_color(m),
            markeredgecolor="black",
            markersize=8,
            label=_display_model(m),
        )
        for m in models
        if (p[C_MODEL] == m).any()
    ]
    adj_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[a],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="None",
            markersize=8,
            label=_display_adjacency_method(a),
        )
        for a in adjs
        if (p[C_ADJ] == a).any()
    ]
    lg1 = ax.legend(handles=model_handles, loc="lower left", title="Model", fontsize=10, title_fontsize=11)
    ax.add_artist(lg1)
    lg2 = ax.legend(handles=adj_handles, loc="lower right", title="Adjacency", fontsize=10, title_fontsize=11)
    ax.add_artist(lg2)

    plt.tight_layout()
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def _ordered_models(df: pd.DataFrame) -> list[str]:
    found = set(df[C_MODEL].dropna().unique())
    out = [m for m in CANONICAL_MODEL_ORDER if m in found]
    for m in sorted(found):
        if m not in out:
            out.append(m)
    return out


def _ordered_non_baseline_models(df: pd.DataFrame) -> list[str]:
    """Models present in `df`, canonical order, excluding sklearn baselines."""
    return [m for m in _ordered_models(df) if m not in BASELINE_MODEL_ORDER]


def _pick_best_row(
    sub: pd.DataFrame,
    *,
    require_test_f1: bool = True,
    require_train_f1: bool = False,
) -> pd.Series | None:
    if sub.empty:
        return None
    ok = sub[VAL_F1].notna()
    if require_test_f1:
        ok &= sub[TEST_F1].notna()
    if require_train_f1:
        ok &= sub[TRAIN_F1].notna()
    sub_ok = sub.loc[ok]
    if sub_ok.empty or sub_ok[VAL_F1].isna().all():
        return None
    return sub_ok.loc[sub_ok[VAL_F1].idxmax()]


def plot_overall_best_test_f1(df: pd.DataFrame, out_name: str = "best_overall_test_f1_macro.pdf") -> None:
    models = _ordered_models(df)
    df_b = _load_baseline_agg()
    # Wide row figure: font sizes scaled to this layout (not the tall 18×6 readout figure).
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15.5, 4.55))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        "Best Overall Model Performance by Test F1 Macro",
        fontsize=20,
        fontweight="bold",
        y=0.928,
    )

    plot_data: dict[str, dict[str, tuple[float, float]]] = {m: {} for m in models}
    for m in models:
        for d in DATASETS:
            sub = df[(df[C_MODEL] == m) & (df[C_DATA] == d)]
            row = _pick_best_row(sub)
            if row is not None:
                plot_data[m][d] = (float(row[TEST_F1]), float(row[TEST_F1_STD]))
            else:
                plot_data[m][d] = (np.nan, np.nan)

    ylims = {}
    for d in DATASETS:
        vals = []
        for m in models:
            mu, sig = plot_data[m][d]
            if not np.isnan(mu):
                s = 0.0 if np.isnan(sig) else sig
                vals.extend([mu - s, mu + s])
        if not df_b.empty:
            for bk in BASELINE_MODEL_ORDER:
                br = _baseline_row_best_overall(df_b, d, bk)
                if br is not None and pd.notna(br[TEST_F1]):
                    vals.append(float(br[TEST_F1]))
        # Tight y-range so bars stay visually prominent on a short row figure.
        ylims[d] = (min(vals) - 0.04, max(vals) + 0.06) if vals else (0.0, 1.0)

    for j, d in enumerate(DATASETS):
        ax = axes[0, j]
        means, stds, labels, colors = [], [], [], []
        for m in models:
            mu, sig = plot_data[m][d]
            if not np.isnan(mu):
                means.append(mu)
                stds.append(0.0 if np.isnan(sig) else sig)
                labels.append(_display_model(m))
                colors.append(_model_color(m))
        if means:
            x = np.arange(len(means))
            ax.bar(
                x,
                means,
                yerr=stds,
                width=0.56,
                capsize=4,
                color=colors,
                edgecolor="black",
                linewidth=1.5,
                zorder=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=19)
        _draw_baseline_test_f1_hlines(ax, df_b, d)
        ax.set_title(_display_dataset(d), fontsize=23, fontweight="bold", pad=5)
        ax.tick_params(axis="y", labelsize=16)
        ax.tick_params(axis="x", labelsize=19)
        if j == 0:
            ax.set_ylabel("F1 Macro Score", fontsize=21)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(ylims[d])

    if not df_b.empty:
        fig.legend(
            handles=_baseline_legend_line_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.882),
            ncol=2,
            fontsize=20,
            frameon=True,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        fig.subplots_adjust(wspace=0.175)
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        fig.subplots_adjust(wspace=0.175)
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_test_f1_by_adjacency(
    df: pd.DataFrame,
    out_name: str = "best_overall_test_f1_macro_by_adjmethod.pdf",
    *,
    grid_2x2: bool = False,
) -> None:
    adjs = sorted(df[C_ADJ].dropna().unique().tolist())
    if not adjs:
        print(f"Skipping {out_name}: no {C_ADJ!r} values.")
        return
    if len(adjs) < 2:
        adjs = list(df[C_ADJ].dropna().unique())
    a1, a2 = adjs[0], adjs[1] if len(adjs) > 1 else adjs[0]

    models = _ordered_models(df)
    hatch_styles = {a1: "", a2: "///"}
    hatch_names = {a1: _display_adjacency_method(a1), a2: _display_adjacency_method(a2)}

    plot_data: dict = {m: {d: {} for d in DATASETS} for m in models}
    for m in models:
        for d in DATASETS:
            for a in (a1, a2):
                sub = df[(df[C_MODEL] == m) & (df[C_DATA] == d) & (df[C_ADJ] == a)]
                row = _pick_best_row(sub)
                if row is not None:
                    plot_data[m][d][a] = (float(row[TEST_F1]), float(row[TEST_F1_STD]))
                else:
                    plot_data[m][d][a] = (np.nan, np.nan)

    df_b = _load_baseline_agg()
    ylims = {}
    for d in DATASETS:
        vals = []
        for m in models:
            for a in (a1, a2):
                mu, sig = plot_data[m][d][a]
                if not np.isnan(mu):
                    s = 0.0 if np.isnan(sig) else sig
                    vals.extend([mu - s, mu + s])
        if not df_b.empty:
            for bk in BASELINE_MODEL_ORDER:
                br = _baseline_row_best_overall(df_b, d, bk)
                if br is not None and pd.notna(br[TEST_F1]):
                    vals.append(float(br[TEST_F1]))
        ylims[d] = (min(vals) - 0.05, max(vals) + 0.05) if vals else (0.0, 1.0)

    if grid_2x2:
        fig, axes = plt.subplots(2, 2, figsize=(max(12, len(models) * 3), 9))
    else:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(max(12, len(models) * 3), 5))
        axes = np.atleast_2d(axes)
    fig.suptitle(
        "Best Overall Model Performance by Test F1 Macro",
        fontsize=26,
        fontweight="bold",
        y=0.90 if grid_2x2 else 0.98,
    )

    bar_w = 0.35
    gap = 0.15
    gw = 2 * bar_w + gap
    tick_label_fs = 22 if grid_2x2 else 20

    for j, d in enumerate(DATASETS):
        if grid_2x2:
            plot_row, plot_col = divmod(j, 2)
            ax = axes[plot_row, plot_col]
        else:
            plot_row = 0
            ax = axes[0, j]
        x_positions = []
        means, stds, colors, hatches = [], [], [], []
        xtick_locs, xtick_labels = [], []
        g = 0
        for m in models:
            for i, a in enumerate((a1, a2)):
                mu, sig = plot_data[m][d][a]
                pos = g * gw + i * bar_w
                x_positions.append(pos)
                has = not np.isnan(mu)
                means.append(float(mu) if has else 0.0)
                stds.append(0.0 if (not has or np.isnan(sig)) else float(sig))
                colors.append(_bar_facecolor(m, has_value=has))
                hatches.append(hatch_styles[a])
            xtick_locs.append(g * gw + bar_w)
            xtick_labels.append(_display_model(m))
            g += 1

        for x, mu, sig, col, h in zip(x_positions, means, stds, colors, hatches):
            ax.bar(
                x,
                mu,
                yerr=sig,
                width=bar_w,
                capsize=5,
                color=col,
                edgecolor="black",
                linewidth=1.5,
                hatch=h,
                zorder=3,
            )
        _draw_baseline_test_f1_hlines(ax, df_b, d)
        ax.set_xticks(xtick_locs)
        if grid_2x2 and plot_row == 0:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=tick_label_fs)
        _title_pad = 6 if grid_2x2 else 10
        ax.set_title(_display_dataset(d), fontsize=24, fontweight="bold", pad=_title_pad)
        ax.tick_params(axis="both", labelsize=tick_label_fs)
        if not grid_2x2 and j == 0:
            ax.set_ylabel("F1 Macro Score", fontsize=20)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(ylims[d])

    if grid_2x2:
        fig.supylabel("F1 Macro Score", fontsize=23, x=0.035, y=0.4)

    hatch_handles = [
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_styles[a1],
            label=hatch_names[a1],
            linewidth=1.5,
        ),
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_styles[a2],
            label=hatch_names[a2],
            linewidth=1.5,
        ),
    ]
    _leg_fs = 21
    _leg_title_fs = 21
    _leg_kw = {
        "loc": "upper center",
        "fontsize": _leg_fs,
        "title_fontsize": _leg_title_fs,
        "frameon": True,
        "ncol": 2,
    }
    if grid_2x2:
        adj_bbox = (0.35, 0.86) if not df_b.empty else (0.5, 0.805)
        base_bbox = (0.65, 0.86)
        _tight_top = 0.795 if not df_b.empty else 0.855
        _tight_left = 0.028
    else:
        adj_bbox = (0.35, 0.93) if not df_b.empty else (0.5, 0.83)
        base_bbox = (0.65, 0.93)
        _tight_top = 0.82 if not df_b.empty else 0.88
        _tight_left = 0.0
    adj_leg = fig.legend(
        handles=hatch_handles,
        title="Edge Construction",
        bbox_to_anchor=adj_bbox,
        **_leg_kw,
    )
    adj_leg.get_title().set_fontweight("bold")
    if not df_b.empty:
        fig.add_artist(adj_leg)
        base_leg = fig.legend(
            handles=_baseline_legend_line_handles(),
            title="Baselines",
            bbox_to_anchor=base_bbox,
            **_leg_kw,
        )
        base_leg.get_title().set_fontweight("bold")
    plt.tight_layout(rect=[_tight_left, 0, 1, _tight_top])
    if grid_2x2:
        fig.subplots_adjust(hspace=0.21)
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_val_f1_by_adjacency(
    df: pd.DataFrame,
    out_name: str = "best_overall_val_f1_macro_by_adjmethod.pdf",
    *,
    grid_2x2: bool = False,
) -> None:
    """Same layout as `plot_test_f1_by_adjacency`, but bar height/error are validation F1."""
    adjs = sorted(df[C_ADJ].dropna().unique().tolist())
    if not adjs:
        print(f"Skipping {out_name}: no {C_ADJ!r} values.")
        return
    if len(adjs) < 2:
        adjs = list(df[C_ADJ].dropna().unique())
    a1, a2 = adjs[0], adjs[1] if len(adjs) > 1 else adjs[0]

    models = _ordered_models(df)
    hatch_styles = {a1: "", a2: "///"}
    hatch_names = {a1: _display_adjacency_method(a1), a2: _display_adjacency_method(a2)}

    plot_data: dict = {m: {d: {} for d in DATASETS} for m in models}
    for m in models:
        for d in DATASETS:
            for a in (a1, a2):
                sub = df[(df[C_MODEL] == m) & (df[C_DATA] == d) & (df[C_ADJ] == a)]
                row = _pick_best_row(sub, require_test_f1=False)
                if row is not None:
                    plot_data[m][d][a] = (float(row[VAL_F1]), float(row[VAL_F1_STD]))
                else:
                    plot_data[m][d][a] = (np.nan, np.nan)

    ylims = {}
    for d in DATASETS:
        vals = []
        for m in models:
            for a in (a1, a2):
                mu, sig = plot_data[m][d][a]
                if not np.isnan(mu):
                    s = 0.0 if np.isnan(sig) else sig
                    vals.extend([mu - s, mu + s])
        ylims[d] = (min(vals) - 0.05, max(vals) + 0.05) if vals else (0.0, 1.0)

    if grid_2x2:
        fig, axes = plt.subplots(2, 2, figsize=(max(12, len(models) * 3), 9))
    else:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(max(12, len(models) * 3), 5))
        axes = np.atleast_2d(axes)
    fig.suptitle(
        "Best Overall Model Performance by Validation F1 Macro",
        fontsize=26,
        fontweight="bold",
        y=0.90 if grid_2x2 else 0.98,
    )

    bar_w = 0.35
    gap = 0.15
    gw = 2 * bar_w + gap

    for j, d in enumerate(DATASETS):
        if grid_2x2:
            plot_row, plot_col = divmod(j, 2)
            ax = axes[plot_row, plot_col]
        else:
            plot_row = 0
            ax = axes[0, j]
        x_positions = []
        means, stds, colors, hatches = [], [], [], []
        xtick_locs, xtick_labels = [], []
        g = 0
        for m in models:
            for i, a in enumerate((a1, a2)):
                mu, sig = plot_data[m][d][a]
                pos = g * gw + i * bar_w
                x_positions.append(pos)
                has = not np.isnan(mu)
                means.append(float(mu) if has else 0.0)
                stds.append(0.0 if (not has or np.isnan(sig)) else float(sig))
                colors.append(_bar_facecolor(m, has_value=has))
                hatches.append(hatch_styles[a])
            xtick_locs.append(g * gw + bar_w)
            xtick_labels.append(_display_model(m))
            g += 1

        for x, mu, sig, col, h in zip(x_positions, means, stds, colors, hatches):
            ax.bar(
                x,
                mu,
                yerr=sig,
                width=bar_w,
                capsize=5,
                color=col,
                edgecolor="black",
                linewidth=1.5,
                hatch=h,
                zorder=3,
            )
        ax.set_xticks(xtick_locs)
        if grid_2x2 and plot_row == 0:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=20)
        _title_pad = 6 if grid_2x2 else 10
        ax.set_title(_display_dataset(d), fontsize=24, fontweight="bold", pad=_title_pad)
        ax.tick_params(axis="both", labelsize=20)
        if not grid_2x2 and j == 0:
            ax.set_ylabel("F1 Macro Score", fontsize=20)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(ylims[d])

    if grid_2x2:
        fig.supylabel("F1 Macro Score", fontsize=23, x=0.035, y=0.4)

    hatch_handles = [
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_styles[a1],
            label=hatch_names[a1],
            linewidth=1.5,
        ),
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_styles[a2],
            label=hatch_names[a2],
            linewidth=1.5,
        ),
    ]
    _leg_fs = 21
    _leg_title_fs = 21
    _leg_kw = {
        "loc": "upper center",
        "fontsize": _leg_fs,
        "title_fontsize": _leg_title_fs,
        "frameon": True,
        "ncol": 2,
    }
    if grid_2x2:
        adj_bbox = (0.5, 0.86)
        _tight_top = 0.855
        _tight_left = 0.028
    else:
        adj_bbox = (0.5, 0.83)
        _tight_top = 0.88
        _tight_left = 0.0
    adj_leg = fig.legend(
        handles=hatch_handles,
        title="Edge Construction",
        bbox_to_anchor=adj_bbox,
        **_leg_kw,
    )
    adj_leg.get_title().set_fontweight("bold")
    plt.tight_layout(rect=[_tight_left, 0, 1, _tight_top])
    if grid_2x2:
        fig.subplots_adjust(hspace=0.21)
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def _subset_runs_for_agg_row(df_runs: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Per-run rows matching one selected aggregated hyperparameter row."""
    if BUCKET_KEY_COL in df_runs.columns and BUCKET_KEY_COL in row.index and pd.notna(row[BUCKET_KEY_COL]):
        return df_runs[df_runs[BUCKET_KEY_COL].astype(str) == str(row[BUCKET_KEY_COL])].copy()
    sub = df_runs[
        (df_runs[C_MODEL] == row[C_MODEL]) & (df_runs[C_DATA] == row[C_DATA]) & (df_runs[C_ADJ] == row[C_ADJ])
    ].copy()
    for col in (C_METHOD, C_READOUT):
        if col in df_runs.columns and col in row.index and pd.notna(row[col]):
            sub = sub[sub[col].astype(str) == str(row[col])]
    if C_RATIO in df_runs.columns and C_RATIO in row.index and pd.notna(row[C_RATIO]):
        rr = pd.to_numeric(sub[C_RATIO], errors="coerce")
        sub = sub.loc[np.isclose(rr, float(row[C_RATIO]), rtol=0.0, atol=1e-8)]
    return sub


def print_best_wandb_runs_by_adjacency(df_agg: pd.DataFrame, df_runs: pd.DataFrame) -> None:
    """Print selected aggregated metrics and corresponding W&B runs (same selection as adjacency plots)."""
    req_agg = [C_MODEL, C_DATA, C_ADJ, VAL_F1, VAL_F1_STD, TEST_F1, TEST_F1_STD]
    miss_agg = [c for c in req_agg if c not in df_agg.columns]
    if miss_agg:
        print(f"Skipping W&B best-run printout: aggregated dataframe missing columns {miss_agg}.")
        return

    req_run = [C_MODEL, C_DATA, C_ADJ]
    miss_run = [c for c in req_run if c not in df_runs.columns]
    if miss_run:
        print(f"Skipping W&B best-run printout: per-run dataframe missing columns {miss_run}.")
        return

    agg = df_agg.dropna(subset=[C_MODEL, C_DATA, C_ADJ, VAL_F1]).copy()
    if agg.empty:
        print("Skipping W&B best-run printout: no finite aggregated rows.")
        return

    adjs = sorted(agg[C_ADJ].dropna().unique().tolist())
    models = _ordered_models(agg)
    print("\nBest hyperparameter setting by validation mean (aggregated over seeds):")
    for m in models:
        print(f"\nMODEL: {_display_model(m)}")
        for d in DATASETS:
            for a in adjs:
                sub = agg[(agg[C_MODEL] == m) & (agg[C_DATA] == d) & (agg[C_ADJ] == a)]
                row = _pick_best_row(sub, require_test_f1=False)
                if row is None:
                    continue

                val_mu = float(row[VAL_F1])
                val_sd = float(row[VAL_F1_STD]) if pd.notna(row[VAL_F1_STD]) else float("nan")
                tst_mu = float(row[TEST_F1]) if pd.notna(row[TEST_F1]) else float("nan")
                tst_sd = float(row[TEST_F1_STD]) if pd.notna(row[TEST_F1_STD]) else float("nan")

                val_txt = f"{val_mu:.4f}±{val_sd:.4f}" if not np.isnan(val_sd) else f"{val_mu:.4f}"
                tst_txt = f"{tst_mu:.4f}±{tst_sd:.4f}" if not np.isnan(tst_mu) and not np.isnan(tst_sd) else (
                    f"{tst_mu:.4f}" if not np.isnan(tst_mu) else "NA"
                )

                details = []
                if C_METHOD in row.index and pd.notna(row[C_METHOD]):
                    details.append(f"method={row[C_METHOD]}")
                if C_RATIO in row.index and pd.notna(row[C_RATIO]):
                    details.append(f"ratio={float(row[C_RATIO]):.2f}")
                if C_READOUT in row.index and pd.notna(row[C_READOUT]):
                    details.append(f"readout={row[C_READOUT]}")
                cfg_txt = ", ".join(details) if details else "no-extra-hparams"

                run_rows = _subset_runs_for_agg_row(df_runs, row)
                run_rows = run_rows.copy()
                if SEED_COL in run_rows.columns:
                    run_rows["_seed_sort"] = pd.to_numeric(run_rows[SEED_COL], errors="coerce")
                    run_rows = run_rows.sort_values(["_seed_sort", "run_id"], kind="mergesort")
                elif "run_id" in run_rows.columns:
                    run_rows = run_rows.sort_values("run_id", kind="mergesort")

                run_items: list[str] = []
                for _, rr in run_rows.iterrows():
                    rn = str(rr.get("run_name", "")).strip() or "<no-run-name>"
                    rid = str(rr.get("run_id", "")).strip()
                    sd = rr.get(SEED_COL, np.nan)
                    seed_txt = f"seed={int(sd)}" if pd.notna(sd) else "seed=?"
                    rid_txt = f", run_id={rid}" if rid else ""
                    run_items.append(f"{seed_txt}: {rn}{rid_txt}")
                runs_txt = " ; ".join(run_items) if run_items else "<no matching per-run rows>"

                print(
                    f"  dataset={_display_dataset(d):<12} adj={_display_adjacency_method(a):<13} "
                    f"val_mean±std={val_txt} | test_mean±std={tst_txt} | {cfg_txt}"
                )
                print(f"    wandb_runs: {runs_txt}")


def plot_train_minus_val(df: pd.DataFrame, out_name: str = "train_val_difference_f1_macro_by_adjmethod.pdf") -> None:
    adjs = sorted(df[C_ADJ].dropna().unique().tolist())
    if not adjs:
        print(f"Skipping {out_name}: no {C_ADJ!r} values.")
        return
    if len(adjs) < 2:
        adjs = list(df[C_ADJ].dropna().unique())
    a1, a2 = adjs[0], adjs[1] if len(adjs) > 1 else adjs[0]

    models = _ordered_models(df)
    hatch_styles = {a1: "", a2: "///"}
    hatch_names = {a1: _display_adjacency_method(a1), a2: _display_adjacency_method(a2)}

    plot_data: dict = {m: {d: {} for d in DATASETS} for m in models}
    for m in models:
        for d in DATASETS:
            for a in (a1, a2):
                sub = df[(df[C_MODEL] == m) & (df[C_DATA] == d) & (df[C_ADJ] == a)]
                row = _pick_best_row(sub, require_test_f1=False, require_train_f1=True)
                if row is None:
                    plot_data[m][d][a] = (np.nan, np.nan)
                    continue
                t_mu = float(row[TRAIN_F1])
                v_mu = float(row[VAL_F1])
                diff = t_mu - v_mu
                ts = float(row[TRAIN_F1_STD]) if not np.isnan(row[TRAIN_F1_STD]) else np.nan
                vs = float(row[VAL_F1_STD]) if not np.isnan(row[VAL_F1_STD]) else np.nan
                if not np.isnan(ts) and not np.isnan(vs):
                    ds = np.sqrt(ts**2 + vs**2)
                else:
                    ds = np.nan
                plot_data[m][d][a] = (diff, ds)

    df_b = _load_baseline_agg()
    ylims = {}
    for d in DATASETS:
        vals = []
        for m in models:
            for a in (a1, a2):
                mu, sig = plot_data[m][d][a]
                if not np.isnan(mu):
                    s = 0.0 if np.isnan(sig) else sig
                    vals.extend([mu - s, mu + s])
        if not df_b.empty:
            for bk in BASELINE_MODEL_ORDER:
                br = _baseline_row_best_overall(df_b, d, bk)
                diff = _baseline_train_minus_val_from_row(br)
                if diff is not None:
                    vals.append(diff)
        if vals:
            ylim = max(abs(min(vals)), abs(max(vals))) + 0.02
            ylims[d] = (-ylim, ylim)
        else:
            ylims[d] = (-0.3, 0.3)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(max(12, len(models) * 3), 5))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        "Difference: Train F1 Macro - Val F1 Macro\n(Positive = Train Better)",
        fontsize=26,
        fontweight="bold",
        y=0.995,
    )

    bar_w = 0.35
    gap = 0.15
    gw = 2 * bar_w + gap

    for j, d in enumerate(DATASETS):
        ax = axes[0, j]
        x_positions, means, stds, colors, hatches = [], [], [], [], []
        xtick_locs, xtick_labels = [], []
        g = 0
        for m in models:
            for i, a in enumerate((a1, a2)):
                mu, sig = plot_data[m][d][a]
                x_positions.append(g * gw + i * bar_w)
                has = not np.isnan(mu)
                means.append(float(mu) if has else 0.0)
                stds.append(0.0 if (not has or np.isnan(sig)) else float(sig))
                colors.append(_bar_facecolor(m, has_value=has))
                hatches.append(hatch_styles[a])
            xtick_locs.append(g * gw + bar_w)
            xtick_labels.append(_display_model(m))
            g += 1

        for x, mu, sig, col, h in zip(x_positions, means, stds, colors, hatches):
            ax.bar(
                x,
                mu,
                yerr=sig,
                width=bar_w,
                capsize=5,
                color=col,
                edgecolor="black",
                linewidth=1.5,
                hatch=h,
                zorder=3,
            )
        ax.axhline(0, color="red", linestyle="-", linewidth=2.5, alpha=0.8)
        _draw_baseline_train_minus_val_hlines(ax, df_b, d)
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=20)
        ax.set_title(_display_dataset(d), fontsize=24, fontweight="bold", pad=10)
        ax.tick_params(axis="both", labelsize=20)
        if j == 0:
            ax.set_ylabel("Train F1 - Val F1", fontsize=20)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(ylims[d])

    hatch_handles = [
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_styles[a1],
            label=hatch_names[a1],
            linewidth=1.5,
        ),
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_styles[a2],
            label=hatch_names[a2],
            linewidth=1.5,
        ),
    ]
    ref_handles = [
        Line2D([0], [0], color="red", linestyle="-", linewidth=2.5, label="Zero gap"),
    ]
    bl_handles = _baseline_legend_trainval_handles() if not df_b.empty else []
    all_tv = hatch_handles + ref_handles + bl_handles
    fig.legend(
        handles=all_tv,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.82),
        ncol=min(len(all_tv), 5),
        fontsize=13 if not df_b.empty else 18,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.80 if not df_b.empty else 0.88])
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def _resolve_ratio_column_values(df: pd.DataFrame, targets: Sequence[float]) -> list:
    """Pick actual `C_RATIO` values from *df* matching *targets* (float-tolerant), in target order."""
    present = sorted(df[C_RATIO].dropna().unique().tolist())
    out: list = []
    for t in targets:
        for r in present:
            try:
                fr = float(r)
            except (TypeError, ValueError):
                continue
            if np.isclose(fr, float(t), rtol=0.0, atol=1e-5):
                out.append(r)
                break
    return out


def plot_mega_method_dataset_ratio(
    df: pd.DataFrame,
    out_name: str = "best_performance_by_dataset_method_ratio.pdf",
    *,
    ratios_only: Sequence[float] | None = None,
    highlight_best_val_per_dataset_model: bool = False,
) -> None:
    """Rows = datasets, columns = sampling methods; bars = GNN models + baseline SVM/EN grays × node ratios.

    If *highlight_best_val_per_dataset_model*, each (dataset, model) bar that attains the highest
    validation F1 over the method × ratio grid is hatched; legend includes ``best val``.
    """
    models = _ordered_models(df)
    df_b = _load_baseline_agg()
    methods = sorted(df[C_METHOD].dropna().unique().tolist())
    if ratios_only is not None:
        ratios = _resolve_ratio_column_values(df, ratios_only)
        if len(ratios) != len(ratios_only):
            print(
                f"Skipping {out_name}: need ratio column values matching {tuple(ratios_only)!r}; "
                f"found {ratios!r} in data."
            )
            return
    else:
        ratios = sorted(df[C_RATIO].dropna().unique().tolist())
    if not methods or not ratios:
        print(f"Skipping {out_name}: need at least one {C_METHOD!r} and one {C_RATIO!r}.")
        return

    dfc = df.dropna(subset=[VAL_F1, TEST_F1])
    idx = dfc.groupby([C_METHOD, C_DATA, C_RATIO, C_MODEL], dropna=False)[VAL_F1].idxmax()
    df_cell = dfc.loc[idx].copy()

    plot_tensor: dict = {}
    for meth in methods:
        plot_tensor[meth] = {}
        for d in DATASETS:
            plot_tensor[meth][d] = {}
            for r in ratios:
                plot_tensor[meth][d][r] = {}
                for m in models:
                    row = df_cell[
                        (df_cell[C_METHOD] == meth)
                        & (df_cell[C_DATA] == d)
                        & (df_cell[C_RATIO] == r)
                        & (df_cell[C_MODEL] == m)
                    ]
                    if len(row) >= 1:
                        rr = row.iloc[0]
                        plot_tensor[meth][d][r][m] = (
                            float(rr[TEST_F1]),
                            float(rr[TEST_F1_STD]),
                            float(rr[VAL_F1]),
                        )
                    else:
                        plot_tensor[meth][d][r][m] = (np.nan, np.nan, np.nan)

    plot_keys: list[str] = list(models) + (list(BASELINE_MODEL_ORDER) if not df_b.empty else [])
    n_keys = max(len(plot_keys), 1)

    best_val_cell: dict[tuple[str, str], tuple[object, object]] = {}
    if highlight_best_val_per_dataset_model:
        for d in DATASETS:
            for m in models:
                best_v = float("-inf")
                best_mr: tuple[object, object] | None = None
                for meth in methods:
                    for r in ratios:
                        cell = plot_tensor[meth][d][r].get(m)
                        if cell is None:
                            continue
                        v_mu = cell[2]
                        if np.isnan(v_mu):
                            continue
                        if float(v_mu) > best_v + VAL_SELECTION_TIE_TOL:
                            best_v = float(v_mu)
                            best_mr = (meth, r)
                if best_mr is not None:
                    best_val_cell[(d, m)] = best_mr
            if not df_b.empty:
                for bk in BASELINE_MODEL_ORDER:
                    best_v = float("-inf")
                    best_mr = None
                    for meth in methods:
                        for r in ratios:
                            v_mu = _baseline_val_mu_per_ratio(df_b, d, meth, r, bk)
                            if np.isnan(v_mu):
                                continue
                            if float(v_mu) > best_v + VAL_SELECTION_TIE_TOL:
                                best_v = float(v_mu)
                                best_mr = (meth, r)
                    if best_mr is not None:
                        best_val_cell[(d, bk)] = best_mr

    row_mins: dict[str, float] = {}
    row_maxs: dict[str, float] = {}
    for d in DATASETS:
        lo, hi = float("inf"), 0.3
        for meth in methods:
            for r in ratios:
                for m in models:
                    cell = plot_tensor[meth][d][r].get(m, (np.nan, np.nan, np.nan))
                    mu, sig = cell[0], cell[1]
                    if not np.isnan(mu):
                        s = 0.0 if np.isnan(sig) else sig
                        lo = min(lo, mu - s)
                        hi = max(hi, mu + s)
                if not df_b.empty:
                    for bk in BASELINE_MODEL_ORDER:
                        mu, sig = _baseline_mu_sig_per_ratio(df_b, d, meth, r, bk)
                        if not np.isnan(mu):
                            s = 0.0 if np.isnan(sig) else sig
                            lo = min(lo, mu - s)
                            hi = max(hi, mu + s)
        row_mins[d] = max(0.0, lo - 0.05) if lo < float("inf") else 0.0
        row_maxs[d] = hi + 0.05

    fig, axes = plt.subplots(len(DATASETS), len(methods), figsize=(16, 10))
    if len(DATASETS) == 1 and len(methods) == 1:
        axes = np.array([[axes]])
    elif len(DATASETS) == 1:
        axes = np.atleast_2d(axes)
    elif len(methods) == 1:
        axes = np.reshape(axes, (-1, 1))

    bar_w = 0.9 / n_keys
    group_sp = 1.1

    for i, d in enumerate(DATASETS):
        for j, meth in enumerate(methods):
            ax = axes[i, j]
            x_groups = np.arange(len(ratios)) * group_sp
            x_by_key = {
                k: x_groups + (ii - (n_keys - 1) / 2) * bar_w for ii, k in enumerate(plot_keys)
            }
            labeled = {k: False for k in plot_keys}
            for m in models:
                for ri, r in enumerate(ratios):
                    cell = plot_tensor[meth][d][r][m]
                    mu, sig = cell[0], cell[1]
                    if np.isnan(mu):
                        continue
                    lbl = _display_model(m) if not labeled[m] else None
                    labeled[m] = True
                    is_best_val = highlight_best_val_per_dataset_model and best_val_cell.get(
                        (d, m)
                    ) == (meth, r)
                    ax.bar(
                        x_by_key[m][ri],
                        mu,
                        yerr=sig,
                        width=bar_w,
                        capsize=2,
                        label=lbl,
                        color=_model_color(m),
                        edgecolor="black",
                        linewidth=1.4 if is_best_val else 0.5,
                        alpha=0.9,
                        hatch="///" if is_best_val else "",
                        error_kw={"elinewidth": 1, "capthick": 1},
                        zorder=3,
                    )
            if not df_b.empty:
                for bk in BASELINE_MODEL_ORDER:
                    for ri, r in enumerate(ratios):
                        mu, sig = _baseline_mu_sig_per_ratio(df_b, d, meth, r, bk)
                        if np.isnan(mu):
                            continue
                        lbl = _display_model(bk) if not labeled[bk] else None
                        labeled[bk] = True
                        is_best_val = highlight_best_val_per_dataset_model and best_val_cell.get(
                            (d, bk)
                        ) == (meth, r)
                        ax.bar(
                            x_by_key[bk][ri],
                            mu,
                            yerr=0.0 if np.isnan(sig) else sig,
                            width=bar_w,
                            capsize=2,
                            label=lbl,
                            color=BASELINE_BAR_COLORS[bk],
                            edgecolor="black",
                            linewidth=1.4 if is_best_val else 0.5,
                            alpha=0.95,
                            hatch="///" if is_best_val else "",
                            error_kw={"elinewidth": 1, "capthick": 1},
                            zorder=3,
                        )
            ax.set_xticks(x_groups)
            ax.set_xticklabels([str(x) for x in ratios], fontsize=18)
            ax.tick_params(axis="both", labelsize=18)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)
            ax.set_ylim(row_mins[d], row_maxs[d])
            if j > 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            if i == 0:
                ax.set_title(meth.replace("_", " ").title(), fontsize=18, fontweight="bold", pad=10)
            if j == 0:
                ax.text(
                    -0.30,
                    0.5,
                    _display_dataset(d),
                    transform=ax.transAxes,
                    fontsize=20,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                )
                ax.text(
                    -0.17,
                    0.5,
                    "F1 Macro Score",
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                )

    _mega_title = "Best Model Performance by Dataset, Node Selection Method and Sampling Ratio"
    #if ratios_only is not None:
    #    _mega_title += f"\n(node sample ratios {', '.join(str(float(x)) for x in ratios_only)} only)"
    fig.suptitle(
        _mega_title,
        fontsize=20,
        fontweight="bold",
        y=0.97,
        x=0.55,
    )
    # Legend must list every model that appears in any panel — not only axes[0,0], which can omit e.g. SAGN.
    models_in_figure = [
        m
        for m in models
        if any(
            not np.isnan(plot_tensor[meth][d][r][m][0])  # test mean
            for meth in methods
            for d in DATASETS
            for r in ratios
        )
    ]
    uh = [
        mpatches.Patch(facecolor=_model_color(m), edgecolor="black", linewidth=0.5, alpha=0.9)
        for m in models_in_figure
    ]
    ul = [_display_model(m) for m in models_in_figure]
    baselines_in_figure: list[str] = []
    if not df_b.empty:
        for bk in BASELINE_MODEL_ORDER:
            if any(
                not np.isnan(_baseline_mu_sig_per_ratio(df_b, d, meth, r, bk)[0])
                for meth in methods
                for d in DATASETS
                for r in ratios
            ):
                baselines_in_figure.append(bk)
    uh_b = [
        mpatches.Patch(
            facecolor=BASELINE_BAR_COLORS[bk], edgecolor="black", linewidth=0.5, alpha=0.95
        )
        for bk in baselines_in_figure
    ]
    ul_b = [_display_model(bk) for bk in baselines_in_figure]
    leg_handles = list(uh) + list(uh_b)
    leg_labels = list(ul) + list(ul_b)
    if highlight_best_val_per_dataset_model:
        leg_handles.append(
            mpatches.Patch(
                facecolor="#DCDCDC",
                edgecolor="black",
                hatch="///",
                linewidth=1.0,
            )
        )
        leg_labels.append("best val")
    ncol_leg = max(1, min(len(leg_handles), 12))
    fig.legend(
        leg_handles,
        leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.94),
        ncol=ncol_leg,
        fontsize=11,
        frameon=True,
    )
    plt.tight_layout(rect=[0.06, 0.05, 1, 0.95])
    fig.supxlabel("Node Sampling Ratio", fontsize=18, y=0.035, x=0.55)
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_mega_method_dataset_ratio_08_and_10(
    df: pd.DataFrame,
    out_name: str = "best_performance_by_dataset_method_ratio_r08_r10.pdf",
) -> None:
    """Same grid as `plot_mega_method_dataset_ratio`, but only node sample ratios 0.8 and 1.0."""
    plot_mega_method_dataset_ratio(df, out_name=out_name, ratios_only=(0.8, 1.0))


def plot_mega_method_dataset_ratio_collapsed_graph_based(
    df: pd.DataFrame,
    out_name: str = "best_performance_by_dataset_method_ratio_collapsed_graphbased.pdf",
) -> None:
    """Copy of mega ratio plot with GNNs collapsed to one Graph-based bar (best mean per cell)."""
    models = _ordered_models(df)
    df_b = _load_baseline_agg()
    methods = sorted(df[C_METHOD].dropna().unique().tolist())
    ratios = sorted(df[C_RATIO].dropna().unique().tolist())
    if not methods or not ratios:
        print(f"Skipping {out_name}: need at least one {C_METHOD!r} and one {C_RATIO!r}.")
        return

    dfc = df.dropna(subset=[VAL_F1, TEST_F1])
    idx = dfc.groupby([C_METHOD, C_DATA, C_RATIO, C_MODEL], dropna=False)[VAL_F1].idxmax()
    df_cell = dfc.loc[idx].copy()

    gnn_models = [m for m in models if m != "mlp"]
    if not gnn_models:
        print(f"Skipping {out_name}: no graph-based models present.")
        return

    mlp_tensor: dict = {}
    graph_tensor: dict = {}
    for meth in methods:
        mlp_tensor[meth] = {}
        graph_tensor[meth] = {}
        for d in DATASETS:
            mlp_tensor[meth][d] = {}
            graph_tensor[meth][d] = {}
            for r in ratios:
                mlp_row = df_cell[
                    (df_cell[C_METHOD] == meth)
                    & (df_cell[C_DATA] == d)
                    & (df_cell[C_RATIO] == r)
                    & (df_cell[C_MODEL] == "mlp")
                ]
                if len(mlp_row) >= 1:
                    rr = mlp_row.iloc[0]
                    mlp_tensor[meth][d][r] = (
                        float(rr[TEST_F1]),
                        float(rr[TEST_F1_STD]) if pd.notna(rr[TEST_F1_STD]) else float("nan"),
                    )
                else:
                    mlp_tensor[meth][d][r] = (float("nan"), float("nan"))

                best_mu = float("-inf")
                best_sig = float("nan")
                for gm in gnn_models:
                    g_row = df_cell[
                        (df_cell[C_METHOD] == meth)
                        & (df_cell[C_DATA] == d)
                        & (df_cell[C_RATIO] == r)
                        & (df_cell[C_MODEL] == gm)
                    ]
                    if len(g_row) == 0:
                        continue
                    mu = float(g_row.iloc[0][TEST_F1])
                    if np.isnan(mu):
                        continue
                    if mu > best_mu:
                        best_mu = mu
                        best_sig = (
                            float(g_row.iloc[0][TEST_F1_STD])
                            if pd.notna(g_row.iloc[0][TEST_F1_STD])
                            else float("nan")
                        )
                graph_tensor[meth][d][r] = (
                    (best_mu, best_sig) if best_mu > float("-inf") else (float("nan"), float("nan"))
                )

    plot_keys: list[str] = ["mlp", "graph_based"] + (list(BASELINE_MODEL_ORDER) if not df_b.empty else [])
    n_keys = max(len(plot_keys), 1)

    row_mins: dict[str, float] = {}
    row_maxs: dict[str, float] = {}
    for d in DATASETS:
        lo, hi = float("inf"), 0.3
        for meth in methods:
            for r in ratios:
                for mu, sig in (mlp_tensor[meth][d][r], graph_tensor[meth][d][r]):
                    if not np.isnan(mu):
                        s = 0.0 if np.isnan(sig) else sig
                        lo = min(lo, mu - s)
                        hi = max(hi, mu + s)
                if not df_b.empty:
                    for bk in BASELINE_MODEL_ORDER:
                        mu, sig = _baseline_mu_sig_per_ratio(df_b, d, meth, r, bk)
                        if not np.isnan(mu):
                            s = 0.0 if np.isnan(sig) else sig
                            lo = min(lo, mu - s)
                            hi = max(hi, mu + s)
        row_mins[d] = max(0.0, lo - 0.05) if lo < float("inf") else 0.0
        row_maxs[d] = hi + 0.05

    fig, axes = plt.subplots(len(DATASETS), len(methods), figsize=(16, 10))
    if len(DATASETS) == 1 and len(methods) == 1:
        axes = np.array([[axes]])
    elif len(DATASETS) == 1:
        axes = np.atleast_2d(axes)
    elif len(methods) == 1:
        axes = np.reshape(axes, (-1, 1))

    bar_w = 0.9 / n_keys
    group_sp = 1.1

    for i, d in enumerate(DATASETS):
        for j, meth in enumerate(methods):
            ax = axes[i, j]
            x_groups = np.arange(len(ratios)) * group_sp
            x_by_key = {
                k: x_groups + (ii - (n_keys - 1) / 2) * bar_w for ii, k in enumerate(plot_keys)
            }

            for key in ("mlp", "graph_based"):
                for ri, r in enumerate(ratios):
                    mu, sig = mlp_tensor[meth][d][r] if key == "mlp" else graph_tensor[meth][d][r]
                    if np.isnan(mu):
                        continue
                    ax.bar(
                        x_by_key[key][ri],
                        mu,
                        yerr=0.0 if np.isnan(sig) else sig,
                        width=bar_w,
                        capsize=2,
                        color=_model_color("mlp") if key == "mlp" else "#D62828",
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.9,
                        error_kw={"elinewidth": 1, "capthick": 1},
                        zorder=3,
                    )
            if not df_b.empty:
                for bk in BASELINE_MODEL_ORDER:
                    for ri, r in enumerate(ratios):
                        mu, sig = _baseline_mu_sig_per_ratio(df_b, d, meth, r, bk)
                        if np.isnan(mu):
                            continue
                        ax.bar(
                            x_by_key[bk][ri],
                            mu,
                            yerr=0.0 if np.isnan(sig) else sig,
                            width=bar_w,
                            capsize=2,
                            color=BASELINE_BAR_COLORS[bk],
                            edgecolor="black",
                            linewidth=0.5,
                            alpha=0.95,
                            error_kw={"elinewidth": 1, "capthick": 1},
                            zorder=3,
                        )
            ax.set_xticks(x_groups)
            ax.set_xticklabels([str(x) for x in ratios], fontsize=18)
            ax.tick_params(axis="both", labelsize=18)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)
            ax.set_ylim(row_mins[d], row_maxs[d])
            if j > 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            if i == 0:
                ax.set_title(meth.replace("_", " ").title(), fontsize=18, fontweight="bold", pad=10)
            if j == 0:
                ax.text(
                    -0.30,
                    0.5,
                    _display_dataset(d),
                    transform=ax.transAxes,
                    fontsize=20,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                )
                ax.text(
                    -0.17,
                    0.5,
                    "F1 Macro Score",
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                )

    fig.suptitle(
        "Best Model Performance by Dataset, Node Selection Method and Sampling Ratio",
        fontsize=20,
        fontweight="bold",
        y=0.97,
        x=0.55,
    )

    legend_handles: list[mpatches.Patch] = [
        mpatches.Patch(facecolor=_model_color("mlp"), edgecolor="black", linewidth=0.5, alpha=0.9),
        mpatches.Patch(facecolor="#D62828", edgecolor="black", linewidth=0.5, alpha=0.9),
    ]
    legend_labels = ["MLP", "Graph-based"]
    if not df_b.empty:
        legend_handles.extend(
            mpatches.Patch(
                facecolor=BASELINE_BAR_COLORS[bk], edgecolor="black", linewidth=0.5, alpha=0.95
            )
            for bk in BASELINE_MODEL_ORDER
        )
        legend_labels.extend(_display_model(bk) for bk in BASELINE_MODEL_ORDER)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.95),
        ncol=max(1, min(len(legend_handles), 8)),
        fontsize=16,
        frameon=True,
    )
    plt.tight_layout(rect=[0.06, 0.05, 1, 0.95])
    fig.supxlabel("Node Sampling Ratio", fontsize=18, y=0.035, x=0.55)
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def _bar_slots_collapsed_standard_mpnn(models: Sequence[str]) -> list[str]:
    """Preserve `_ordered_models` order but replace the first contiguous block of standard MPNNs with one slot."""
    slots: list[str] = []
    cluster_placed = False
    for m in models:
        if m in STANDARD_MPNN_MODELS:
            if not cluster_placed:
                slots.append(_SLOT_MPNN_CLUSTER)
                cluster_placed = True
        else:
            slots.append(m)
    return slots


def plot_dataset_method_ratio_representative_mpnn(
    df: pd.DataFrame,
    out_name: str = "best_standard_mpnn_by_dataset_method_ratio.pdf",
) -> None:
    """Same grid as the full mega plot: all architectures shown, but GIN/GCN/GATv2/SAGE are merged into
    a single bar per dataset (the one with highest mean test F1 over methods × ratios; same val-F1
    picks per cell). That bar uses that model's color (can differ by dataset row).
    """
    models = _ordered_models(df)
    if not models:
        print(f"Skipping {out_name}: no models in dataframe.")
        return

    methods = sorted(df[C_METHOD].dropna().unique().tolist())
    ratios = sorted(df[C_RATIO].dropna().unique().tolist())
    if not methods or not ratios:
        print(f"Skipping {out_name}: need at least one {C_METHOD!r} and one {C_RATIO!r}.")
        return

    mpnn_in_data = [m for m in STANDARD_MPNN_MODELS if m in models]
    slots = _bar_slots_collapsed_standard_mpnn(models)
    if not mpnn_in_data:
        slots = list(models)

    df_b = _load_baseline_agg()

    dfc = df.dropna(subset=[VAL_F1, TEST_F1])
    idx = dfc.groupby([C_METHOD, C_DATA, C_RATIO, C_MODEL], dropna=False)[VAL_F1].idxmax()
    df_cell = dfc.loc[idx].copy()

    plot_tensor: dict = {}
    for meth in methods:
        plot_tensor[meth] = {}
        for d in DATASETS:
            plot_tensor[meth][d] = {}
            for r in ratios:
                plot_tensor[meth][d][r] = {}
                for m in models:
                    row = df_cell[
                        (df_cell[C_METHOD] == meth)
                        & (df_cell[C_DATA] == d)
                        & (df_cell[C_RATIO] == r)
                        & (df_cell[C_MODEL] == m)
                    ]
                    if len(row) >= 1:
                        rr = row.iloc[0]
                        plot_tensor[meth][d][r][m] = (float(rr[TEST_F1]), float(rr[TEST_F1_STD]))
                    else:
                        plot_tensor[meth][d][r][m] = (np.nan, np.nan)

    representative: dict[str, str] = {}
    if mpnn_in_data:
        for d in DATASETS:
            best_m: str | None = None
            best_mean = float("-inf")
            for m in mpnn_in_data:
                vals: list[float] = []
                for meth in methods:
                    for r in ratios:
                        mu, _ = plot_tensor[meth][d][r][m]
                        if not np.isnan(mu):
                            vals.append(float(mu))
                if not vals:
                    continue
                avg = float(np.mean(vals))
                if avg > best_mean:
                    best_mean = avg
                    best_m = m
            if best_m is not None:
                representative[d] = best_m

    bar_slots = list(slots) + (list(BASELINE_MODEL_ORDER) if not df_b.empty else [])

    def _cell_mu_sig(d: str, meth: str, r: object, slot: str) -> tuple[float, float]:
        if slot in BASELINE_MODEL_ORDER:
            return _baseline_mu_sig_per_ratio(df_b, d, meth, r, slot)
        if slot == _SLOT_MPNN_CLUSTER:
            m = representative.get(d)
            if m is None:
                return (float("nan"), float("nan"))
            return plot_tensor[meth][d][r][m]
        return plot_tensor[meth][d][r][slot]

    row_mins: dict[str, float] = {}
    row_maxs: dict[str, float] = {}
    for d in DATASETS:
        lo, hi = float("inf"), 0.3
        for meth in methods:
            for r in ratios:
                for slot in bar_slots:
                    mu, sig = _cell_mu_sig(d, meth, r, slot)
                    if not np.isnan(mu):
                        s = 0.0 if np.isnan(sig) else sig
                        lo = min(lo, mu - s)
                        hi = max(hi, mu + s)
        row_mins[d] = max(0.0, lo - 0.05) if lo < float("inf") else 0.0
        row_maxs[d] = hi + 0.05

    fig, axes = plt.subplots(len(DATASETS), len(methods), figsize=(16, 10))
    if len(DATASETS) == 1 and len(methods) == 1:
        axes = np.array([[axes]])
    elif len(DATASETS) == 1:
        axes = np.atleast_2d(axes)
    elif len(methods) == 1:
        axes = np.reshape(axes, (-1, 1))

    n_bar = max(len(bar_slots), 1)
    bar_w = 0.72 / n_bar
    group_sp = 1.1

    for i, d in enumerate(DATASETS):
        for j, meth in enumerate(methods):
            ax = axes[i, j]
            x_groups = np.arange(len(ratios)) * group_sp
            for si, slot in enumerate(bar_slots):
                x_off = x_groups + (si - (n_bar - 1) / 2) * bar_w
                for ri, r in enumerate(ratios):
                    mu, sig = _cell_mu_sig(d, meth, r, slot)
                    if np.isnan(mu):
                        continue
                    if slot in BASELINE_MODEL_ORDER:
                        face = BASELINE_BAR_COLORS[slot]
                    elif slot == _SLOT_MPNN_CLUSTER:
                        m_key = representative.get(d)
                        if m_key is None:
                            continue
                        face = _model_color(m_key)
                    else:
                        face = _model_color(slot)
                    ax.bar(
                        x_off[ri],
                        mu,
                        yerr=0.0 if np.isnan(sig) else sig,
                        width=bar_w,
                        capsize=2,
                        color=face,
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.9,
                        error_kw={"elinewidth": 1, "capthick": 1},
                        zorder=3,
                    )
            ax.set_xticks(x_groups)
            ax.set_xticklabels([str(x) for x in ratios], fontsize=18)
            ax.tick_params(axis="both", labelsize=18)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)
            ax.set_ylim(row_mins[d], row_maxs[d])
            if j > 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            if i == 0:
                ax.set_title(meth.replace("_", " ").title(), fontsize=18, fontweight="bold", pad=10)
            if j == 0:
                d_title = _display_dataset(d)
                if d in representative:
                    d_title = f"{d_title}\n({_display_model(representative[d])})"
                ax.text(
                    -0.30,
                    0.5,
                    d_title,
                    transform=ax.transAxes,
                    fontsize=20,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                )
                ax.text(
                    -0.17,
                    0.5,
                    "F1 Macro Score",
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                )

    fig.suptitle(
        "Model performance by dataset, method, and node ratio\n"
        "(standard MPNNs collapsed to the best GIN/GCN/GATv2/SAGE per dataset by mean test F1; bar color = that model)",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    # Legend: non-MPNN models that appear anywhere, then the four standard MPNNs (for cluster bar colors).
    def _slot_has_finite_data(slot: str) -> bool:
        for d in DATASETS:
            for meth in methods:
                for r in ratios:
                    mu, _ = _cell_mu_sig(d, meth, r, slot)
                    if not np.isnan(mu):
                        return True
        return False

    uh: list[mpatches.Patch] = []
    ul: list[str] = []
    seen_labels: set[str] = set()
    cluster_legend_done = False
    for slot in bar_slots:
        if slot in BASELINE_MODEL_ORDER:
            if not _slot_has_finite_data(slot):
                continue
            lab = _display_model(slot)
            if lab in seen_labels:
                continue
            seen_labels.add(lab)
            uh.append(
                mpatches.Patch(
                    facecolor=BASELINE_BAR_COLORS[slot],
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.95,
                )
            )
            ul.append(lab)
            continue
        if slot == _SLOT_MPNN_CLUSTER:
            if cluster_legend_done or not _slot_has_finite_data(_SLOT_MPNN_CLUSTER):
                continue
            cluster_legend_done = True
            for m in STANDARD_MPNN_MODELS:
                if m not in models:
                    continue
                lab = _display_model(m)
                if lab in seen_labels:
                    continue
                seen_labels.add(lab)
                uh.append(
                    mpatches.Patch(
                        facecolor=_model_color(m), edgecolor="black", linewidth=0.5, alpha=0.9
                    )
                )
                ul.append(lab)
            continue
        if not _slot_has_finite_data(slot):
            continue
        lab = _display_model(slot)
        if lab in seen_labels:
            continue
        seen_labels.add(lab)
        uh.append(
            mpatches.Patch(facecolor=_model_color(slot), edgecolor="black", linewidth=0.5, alpha=0.9)
        )
        ul.append(lab)

    ncol = max(1, min(len(uh), 12))
    fig.legend(
        uh,
        ul,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.94),
        ncol=ncol,
        fontsize=10,
        frameon=True,
    )
    plt.tight_layout(rect=[0.06, 0, 1, 0.90])
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_readout_effect(
    df: pd.DataFrame,
    out_name: str = "effect_of_readout_test_f1_macro.pdf",
) -> None:
    """Per dataset: grouped bars (OmicsReadOut vs NoReadOut), best row by val F1 per model; MLP excluded."""
    if C_READOUT not in df.columns:
        print(f"Skipping {out_name}: no {C_READOUT!r} column.")
        return

    dfc = df[df[C_READOUT].astype(str).isin(READOUT_COMPARE_ORDER)].copy()
    if dfc.empty:
        print(f"Skipping {out_name}: no rows with readout in {READOUT_COMPARE_ORDER!r}.")
        return

    found = {str(x) for x in dfc[C_MODEL].dropna().unique()}
    models = [m for m in CANONICAL_MODEL_ORDER if m in found and m != "mlp"]
    for m in sorted(found):
        if m != "mlp" and m not in models:
            models.append(m)
    if not models:
        print(f"Skipping {out_name}: no non-MLP models with readout data.")
        return

    df_b = _load_baseline_agg()
    plot_data: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for model in models:
        plot_data[model] = {}
        for dataset in DATASETS:
            plot_data[model][dataset] = {}
            for readout in READOUT_COMPARE_ORDER:
                sub = dfc[
                    (dfc[C_MODEL] == model) & (dfc[C_DATA] == dataset) & (dfc[C_READOUT].astype(str) == readout)
                ]
                row = _pick_best_row(sub)
                if row is not None:
                    mu = float(row[TEST_F1])
                    sig = float(row[TEST_F1_STD]) if pd.notna(row[TEST_F1_STD]) else float("nan")
                    plot_data[model][dataset][readout] = (mu, sig)
                else:
                    plot_data[model][dataset][readout] = (float("nan"), float("nan"))

    dataset_ylims: dict[str, tuple[float, float]] = {}
    for dataset in DATASETS:
        vals: list[float] = []
        for model in models:
            for readout in READOUT_COMPARE_ORDER:
                mu, sig = plot_data[model][dataset][readout]
                if not np.isnan(mu):
                    s = 0.0 if np.isnan(sig) else sig
                    vals.extend([mu - s, mu + s])
        if not df_b.empty:
            for bk in BASELINE_MODEL_ORDER:
                br = _baseline_row_best_overall(df_b, dataset, bk)
                if br is not None and pd.notna(br.get(TEST_F1)):
                    vals.append(float(br[TEST_F1]))
        dataset_ylims[dataset] = (min(vals) - 0.05, max(vals) + 0.05) if vals else (0.0, 1.0)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 5.15))
    axes = np.atleast_2d(axes)
    if axes.shape[1] != len(DATASETS):
        axes = axes.reshape(1, -1)

    bar_w = 0.35
    x_positions = np.arange(len(models))

    for j, dataset in enumerate(DATASETS):
        ax = axes[0, j]
        for m_idx, model in enumerate(models):
            mu_o, sig_o = plot_data[model][dataset]["OmicsReadOut"]
            mu_n, sig_n = plot_data[model][dataset]["NoReadOut"]
            has_o = not np.isnan(mu_o)
            has_n = not np.isnan(mu_n)

            if has_o and has_n:
                for r_idx, readout in enumerate(READOUT_COMPARE_ORDER):
                    mu, sig = plot_data[model][dataset][readout]
                    offset = (r_idx - 0.5) * bar_w
                    ax.bar(
                        x_positions[m_idx] + offset,
                        mu,
                        yerr=0.0 if np.isnan(sig) else sig,
                        width=bar_w,
                        capsize=3,
                        color=_model_color(model),
                        edgecolor="black",
                        linewidth=1.5,
                        hatch=READOUT_HATCHES[readout],
                        alpha=READOUT_ALPHAS[readout],
                        error_kw={"elinewidth": 1, "capthick": 1},
                    )
            elif has_o or has_n:
                readout = "OmicsReadOut" if has_o else "NoReadOut"
                mu, sig = plot_data[model][dataset][readout]
                ax.bar(
                    x_positions[m_idx],
                    mu,
                    yerr=0.0 if np.isnan(sig) else sig,
                    width=bar_w * 1.5,
                    capsize=3,
                    color=_model_color(model),
                    edgecolor="black",
                    linewidth=1.5,
                    hatch=READOUT_HATCHES[readout],
                    alpha=READOUT_ALPHAS[readout],
                    error_kw={"elinewidth": 1, "capthick": 1},
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([_display_model(m) for m in models], rotation=45, ha="right", fontsize=19)
        ax.tick_params(axis="y", labelsize=16)
        ax.tick_params(axis="x", labelsize=19)
        ax.set_title(_display_dataset(dataset), fontsize=23, fontweight="bold", pad=5)
        if j == 0:
            ax.set_ylabel("F1 Macro Score", fontsize=21)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        _draw_baseline_test_f1_hlines(ax, df_b, dataset)
        ax.set_ylim(dataset_ylims[dataset])

    fig.suptitle(
        "Effect of Readout on Model Performance",
        fontsize=23,
        fontweight="bold",
        y=0.995,
    )
    legend_handles = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="", alpha=0.9, label="MLP Readout"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch=".", alpha=0.6, label="Vanilla Readout"),
    ]
    if not df_b.empty:
        legend_handles.extend(_baseline_legend_line_handles())
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=min(len(legend_handles), 4),
        fontsize=20,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_adjacency_effect_readout_style(
    df: pd.DataFrame,
    out_name: str = "effect_of_edge_construction_test_f1_macro.pdf",
) -> None:
    """Per dataset: grouped bars (two adjacency methods), best row by val F1 per model; MLP excluded.

    Layout, fonts, bar width, and legend style match `plot_readout_effect` (1×4) exactly.
    """
    adjs = sorted(df[C_ADJ].dropna().unique().tolist())
    if not adjs:
        print(f"Skipping {out_name}: no {C_ADJ!r} values.")
        return
    if len(adjs) < 2:
        adjs = list(df[C_ADJ].dropna().unique())
    a1, a2 = adjs[0], adjs[1] if len(adjs) > 1 else adjs[0]
    same_adj = a1 == a2

    found = {str(x) for x in df[C_MODEL].dropna().unique()}
    models = [m for m in CANONICAL_MODEL_ORDER if m in found and m != "mlp"]
    for m in sorted(found):
        if m != "mlp" and m not in models:
            models.append(m)
    if not models:
        print(f"Skipping {out_name}: no non-MLP models.")
        return

    df_b = _load_baseline_agg()
    plot_data: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for model in models:
        plot_data[model] = {}
        for dataset in DATASETS:
            plot_data[model][dataset] = {}
            for a in (a1, a2):
                sub = df[(df[C_MODEL] == model) & (df[C_DATA] == dataset) & (df[C_ADJ] == a)]
                row = _pick_best_row(sub)
                if row is not None:
                    mu = float(row[TEST_F1])
                    sig = float(row[TEST_F1_STD]) if pd.notna(row[TEST_F1_STD]) else float("nan")
                    plot_data[model][dataset][str(a)] = (mu, sig)
                else:
                    plot_data[model][dataset][str(a)] = (float("nan"), float("nan"))

    k1, k2 = str(a1), str(a2)
    hatch_by_key = {k1: "", k2: "///"}
    alpha_by_key = {k1: 0.9, k2: 0.6}

    dataset_ylims: dict[str, tuple[float, float]] = {}
    for dataset in DATASETS:
        vals: list[float] = []
        for model in models:
            for k in (k1, k2):
                mu, sig = plot_data[model][dataset][k]
                if not np.isnan(mu):
                    s = 0.0 if np.isnan(sig) else sig
                    vals.extend([mu - s, mu + s])
        if not df_b.empty:
            for bk in BASELINE_MODEL_ORDER:
                br = _baseline_row_best_overall(df_b, dataset, bk)
                if br is not None and pd.notna(br.get(TEST_F1)):
                    vals.append(float(br[TEST_F1]))
        dataset_ylims[dataset] = (min(vals) - 0.05, max(vals) + 0.05) if vals else (0.0, 1.0)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 5.15))
    axes = np.atleast_2d(axes)
    if axes.shape[1] != len(DATASETS):
        axes = axes.reshape(1, -1)

    bar_w = 0.35
    x_positions = np.arange(len(models))

    for j, dataset in enumerate(DATASETS):
        ax = axes[0, j]
        for m_idx, model in enumerate(models):
            mu_1, sig_1 = plot_data[model][dataset][k1]
            mu_2, sig_2 = plot_data[model][dataset][k2]
            has_1 = not np.isnan(mu_1)
            has_2 = not np.isnan(mu_2)

            if same_adj:
                if has_1:
                    ax.bar(
                        x_positions[m_idx],
                        mu_1,
                        yerr=0.0 if np.isnan(sig_1) else sig_1,
                        width=bar_w * 1.5,
                        capsize=3,
                        color=_model_color(model),
                        edgecolor="black",
                        linewidth=1.5,
                        hatch="",
                        alpha=alpha_by_key[k1],
                        error_kw={"elinewidth": 1, "capthick": 1},
                    )
            elif has_1 and has_2:
                for r_idx, k in enumerate((k1, k2)):
                    mu, sig = plot_data[model][dataset][k]
                    offset = (r_idx - 0.5) * bar_w
                    ax.bar(
                        x_positions[m_idx] + offset,
                        mu,
                        yerr=0.0 if np.isnan(sig) else sig,
                        width=bar_w,
                        capsize=3,
                        color=_model_color(model),
                        edgecolor="black",
                        linewidth=1.5,
                        hatch=hatch_by_key[k],
                        alpha=alpha_by_key[k],
                        error_kw={"elinewidth": 1, "capthick": 1},
                    )
            elif has_1 or has_2:
                k = k1 if has_1 else k2
                mu, sig = plot_data[model][dataset][k]
                ax.bar(
                    x_positions[m_idx],
                    mu,
                    yerr=0.0 if np.isnan(sig) else sig,
                    width=bar_w * 1.5,
                    capsize=3,
                    color=_model_color(model),
                    edgecolor="black",
                    linewidth=1.5,
                    hatch=hatch_by_key[k],
                    alpha=alpha_by_key[k],
                    error_kw={"elinewidth": 1, "capthick": 1},
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([_display_model(m) for m in models], rotation=45, ha="right", fontsize=19)
        ax.tick_params(axis="y", labelsize=16)
        ax.tick_params(axis="x", labelsize=19)
        ax.set_title(_display_dataset(dataset), fontsize=23, fontweight="bold", pad=5)
        if j == 0:
            ax.set_ylabel("F1 Macro Score", fontsize=21)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        _draw_baseline_test_f1_hlines(ax, df_b, dataset)
        ax.set_ylim(dataset_ylims[dataset])

    fig.suptitle(
        "Effect of Edge Construction on Model Performance",
        fontsize=23,
        fontweight="bold",
        y=0.995,
    )
    legend_handles = [
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_by_key[k1],
            alpha=alpha_by_key[k1],
            label=_display_adjacency_method(a1),
        ),
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_by_key[k2],
            alpha=alpha_by_key[k2],
            label=_display_adjacency_method(a2),
        ),
    ]
    if not df_b.empty:
        legend_handles.extend(_baseline_legend_line_handles())
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=min(len(legend_handles), 4),
        fontsize=20,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_readout_and_adjacency_stacked(
    df: pd.DataFrame,
    out_name: str = "effect_of_readout_and_edge_construction_test_f1_macro_stacked.pdf",
) -> None:
    """Two rows (readout, then edge construction), 1×4 datasets, shared x; model names on bottom row only; MLP omitted."""
    if C_READOUT not in df.columns:
        print(f"Skipping {out_name}: no {C_READOUT!r} column.")
        return

    dfc = df[df[C_READOUT].astype(str).isin(READOUT_COMPARE_ORDER)].copy()
    if dfc.empty:
        print(f"Skipping {out_name}: no rows with readout in {READOUT_COMPARE_ORDER!r}.")
        return

    adjs = sorted(df[C_ADJ].dropna().unique().tolist())
    if not adjs:
        print(f"Skipping {out_name}: no {C_ADJ!r} values.")
        return
    if len(adjs) < 2:
        adjs = list(df[C_ADJ].dropna().unique())
    a1, a2 = adjs[0], adjs[1] if len(adjs) > 1 else adjs[0]
    same_adj = a1 == a2
    k1, k2 = str(a1), str(a2)
    hatch_by_key = {k1: "", k2: "///"}
    alpha_by_key = {k1: 0.9, k2: 0.6}

    found_all = {str(x) for x in df[C_MODEL].dropna().unique()}
    models = [m for m in CANONICAL_MODEL_ORDER if m in found_all and m != "mlp"]
    for m in sorted(found_all):
        if m != "mlp" and m not in models:
            models.append(m)
    if not models:
        print(f"Skipping {out_name}: no non-MLP models.")
        return

    df_b = _load_baseline_agg()

    readout_data: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for model in models:
        readout_data[model] = {}
        for dataset in DATASETS:
            readout_data[model][dataset] = {}
            for readout in READOUT_COMPARE_ORDER:
                sub = dfc[
                    (dfc[C_MODEL] == model) & (dfc[C_DATA] == dataset) & (dfc[C_READOUT].astype(str) == readout)
                ]
                row = _pick_best_row(sub)
                if row is not None:
                    mu = float(row[TEST_F1])
                    sig = float(row[TEST_F1_STD]) if pd.notna(row[TEST_F1_STD]) else float("nan")
                    readout_data[model][dataset][readout] = (mu, sig)
                else:
                    readout_data[model][dataset][readout] = (float("nan"), float("nan"))

    adj_data: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for model in models:
        adj_data[model] = {}
        for dataset in DATASETS:
            adj_data[model][dataset] = {}
            for a in (a1, a2):
                sub = df[(df[C_MODEL] == model) & (df[C_DATA] == dataset) & (df[C_ADJ] == a)]
                row = _pick_best_row(sub)
                if row is not None:
                    mu = float(row[TEST_F1])
                    sig = float(row[TEST_F1_STD]) if pd.notna(row[TEST_F1_STD]) else float("nan")
                    adj_data[model][dataset][str(a)] = (mu, sig)
                else:
                    adj_data[model][dataset][str(a)] = (float("nan"), float("nan"))

    ylims_readout: dict[str, tuple[float, float]] = {}
    ylims_adj: dict[str, tuple[float, float]] = {}
    for dataset in DATASETS:
        vals_r: list[float] = []
        for model in models:
            for readout in READOUT_COMPARE_ORDER:
                mu, sig = readout_data[model][dataset][readout]
                if not np.isnan(mu):
                    s = 0.0 if np.isnan(sig) else sig
                    vals_r.extend([mu - s, mu + s])
        if not df_b.empty:
            for bk in BASELINE_MODEL_ORDER:
                br = _baseline_row_best_overall(df_b, dataset, bk)
                if br is not None and pd.notna(br.get(TEST_F1)):
                    vals_r.append(float(br[TEST_F1]))
        ylims_readout[dataset] = (min(vals_r) - 0.05, max(vals_r) + 0.05) if vals_r else (0.0, 1.0)

        vals_a: list[float] = []
        for model in models:
            for k in (k1, k2):
                mu, sig = adj_data[model][dataset][k]
                if not np.isnan(mu):
                    s = 0.0 if np.isnan(sig) else sig
                    vals_a.extend([mu - s, mu + s])
        if not df_b.empty:
            for bk in BASELINE_MODEL_ORDER:
                br = _baseline_row_best_overall(df_b, dataset, bk)
                if br is not None and pd.notna(br.get(TEST_F1)):
                    vals_a.append(float(br[TEST_F1]))
        ylims_adj[dataset] = (min(vals_a) - 0.05, max(vals_a) + 0.05) if vals_a else (0.0, 1.0)

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(18, 9.45), sharex=True, sharey=False)
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)
    bar_w = 0.35
    x_positions = np.arange(len(models))

    for j, dataset in enumerate(DATASETS):
        ax_top = axes[0, j]
        for m_idx, model in enumerate(models):
            mu_o, sig_o = readout_data[model][dataset]["OmicsReadOut"]
            mu_n, sig_n = readout_data[model][dataset]["NoReadOut"]
            has_o = not np.isnan(mu_o)
            has_n = not np.isnan(mu_n)
            if has_o and has_n:
                for r_idx, readout in enumerate(READOUT_COMPARE_ORDER):
                    mu, sig = readout_data[model][dataset][readout]
                    offset = (r_idx - 0.5) * bar_w
                    ax_top.bar(
                        x_positions[m_idx] + offset,
                        mu,
                        yerr=0.0 if np.isnan(sig) else sig,
                        width=bar_w,
                        capsize=3,
                        color=_model_color(model),
                        edgecolor="black",
                        linewidth=1.5,
                        hatch=READOUT_HATCHES[readout],
                        alpha=READOUT_ALPHAS[readout],
                        error_kw={"elinewidth": 1, "capthick": 1},
                    )
            elif has_o or has_n:
                readout = "OmicsReadOut" if has_o else "NoReadOut"
                mu, sig = readout_data[model][dataset][readout]
                ax_top.bar(
                    x_positions[m_idx],
                    mu,
                    yerr=0.0 if np.isnan(sig) else sig,
                    width=bar_w * 1.5,
                    capsize=3,
                    color=_model_color(model),
                    edgecolor="black",
                    linewidth=1.5,
                    hatch=READOUT_HATCHES[readout],
                    alpha=READOUT_ALPHAS[readout],
                    error_kw={"elinewidth": 1, "capthick": 1},
                )

        ax_top.set_xticks(x_positions)
        ax_top.set_xticklabels([])
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="y", labelsize=16)
        ax_top.set_title(_display_dataset(dataset), fontsize=23, fontweight="bold", pad=5)
        if j == 0:
            ax_top.set_ylabel("F1 Macro Score", fontsize=21)
        ax_top.set_xlim(-0.5, len(models) - 0.5)
        ax_top.grid(axis="y", alpha=0.3, linestyle="--")
        ax_top.set_axisbelow(True)
        _draw_baseline_test_f1_hlines(ax_top, df_b, dataset)
        ax_top.set_ylim(ylims_readout[dataset])

        ax_bot = axes[1, j]
        for m_idx, model in enumerate(models):
            mu_1, sig_1 = adj_data[model][dataset][k1]
            mu_2, sig_2 = adj_data[model][dataset][k2]
            has_1 = not np.isnan(mu_1)
            has_2 = not np.isnan(mu_2)
            if same_adj:
                if has_1:
                    ax_bot.bar(
                        x_positions[m_idx],
                        mu_1,
                        yerr=0.0 if np.isnan(sig_1) else sig_1,
                        width=bar_w * 1.5,
                        capsize=3,
                        color=_model_color(model),
                        edgecolor="black",
                        linewidth=1.5,
                        hatch="",
                        alpha=alpha_by_key[k1],
                        error_kw={"elinewidth": 1, "capthick": 1},
                    )
            elif has_1 and has_2:
                for r_idx, k in enumerate((k1, k2)):
                    mu, sig = adj_data[model][dataset][k]
                    offset = (r_idx - 0.5) * bar_w
                    ax_bot.bar(
                        x_positions[m_idx] + offset,
                        mu,
                        yerr=0.0 if np.isnan(sig) else sig,
                        width=bar_w,
                        capsize=3,
                        color=_model_color(model),
                        edgecolor="black",
                        linewidth=1.5,
                        hatch=hatch_by_key[k],
                        alpha=alpha_by_key[k],
                        error_kw={"elinewidth": 1, "capthick": 1},
                    )
            elif has_1 or has_2:
                k = k1 if has_1 else k2
                mu, sig = adj_data[model][dataset][k]
                ax_bot.bar(
                    x_positions[m_idx],
                    mu,
                    yerr=0.0 if np.isnan(sig) else sig,
                    width=bar_w * 1.5,
                    capsize=3,
                    color=_model_color(model),
                    edgecolor="black",
                    linewidth=1.5,
                    hatch=hatch_by_key[k],
                    alpha=alpha_by_key[k],
                    error_kw={"elinewidth": 1, "capthick": 1},
                )

        ax_bot.set_xticks(x_positions)
        ax_bot.set_xticklabels([_display_model(m) for m in models], rotation=45, ha="right", fontsize=19)
        ax_bot.tick_params(axis="y", labelsize=16)
        ax_bot.tick_params(axis="x", labelsize=19)
        if j == 0:
            ax_bot.set_ylabel("F1 Macro Score", fontsize=21)
        ax_bot.set_xlim(-0.5, len(models) - 0.5)
        ax_bot.grid(axis="y", alpha=0.3, linestyle="--")
        ax_bot.set_axisbelow(True)
        _draw_baseline_test_f1_hlines(ax_bot, df_b, dataset)
        ax_bot.set_ylim(ylims_adj[dataset])

    fig.suptitle(
        "Readout and Edge Construction Effects on Model Performance",
        fontsize=23,
        fontweight="bold",
        y=0.98,
    )
    legend_handles = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="", alpha=0.9, label="MLP Readout"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch=".", alpha=0.6, label="Vanilla Readout"),
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_by_key[k1],
            alpha=alpha_by_key[k1],
            label=_display_adjacency_method(a1),
        ),
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_by_key[k2],
            alpha=alpha_by_key[k2],
            label=_display_adjacency_method(a2),
        ),
    ]
    if not df_b.empty:
        legend_handles.extend(_baseline_legend_line_handles())
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(len(legend_handles), 3),
        fontsize=20,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.subplots_adjust(hspace=0.16, wspace=0.22)
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_readout_adjacency_compact(
    df: pd.DataFrame,
    out_name: str = "effect_of_readout_and_edge_construction_test_f1_macro_2x4.pdf",
) -> None:
    """Per dataset: two readout groups, each with two edge-construction bars; models collapsed to best test mean."""
    if C_READOUT not in df.columns:
        print(f"Skipping {out_name}: no {C_READOUT!r} column.")
        return
    adjs = sorted(df[C_ADJ].dropna().unique().tolist())
    if not adjs:
        print(f"Skipping {out_name}: no {C_ADJ!r} values.")
        return
    if len(adjs) < 2:
        adjs = list(df[C_ADJ].dropna().unique())
    a1, a2 = adjs[0], adjs[1] if len(adjs) > 1 else adjs[0]

    dfc = df[df[C_READOUT].astype(str).isin(READOUT_COMPARE_ORDER)].copy()
    if dfc.empty:
        print(f"Skipping {out_name}: no rows with readout in {READOUT_COMPARE_ORDER!r}.")
        return

    adj_colors = {a1: "#5B9BD5", a2: "#F28E2B"}
    adj_names = {a1: _display_adjacency_method(a1), a2: _display_adjacency_method(a2)}
    readout_titles = {"OmicsReadOut": "MLP", "NoReadOut": "Vanilla"}
    df_b = _load_baseline_agg()

    # Collapse all models by selecting the row with highest test-mean for each
    # (dataset, readout, adjacency) combination.
    plot_data: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for d in DATASETS:
        plot_data[d] = {}
        for readout in READOUT_COMPARE_ORDER:
            plot_data[d][readout] = {}
            for a in (a1, a2):
                sub = dfc[
                    (dfc[C_DATA] == d)
                    & (dfc[C_READOUT].astype(str) == readout)
                    & (dfc[C_ADJ] == a)
                    & (dfc[C_MODEL] != "mlp")
                ]
                best = _pick_best_row(sub)
                if best is None:
                    plot_data[d][readout][a] = (float("nan"), float("nan"))
                    continue
                mu = float(best[TEST_F1]) if pd.notna(best.get(TEST_F1)) else float("nan")
                sig = float(best[TEST_F1_STD]) if pd.notna(best.get(TEST_F1_STD)) else float("nan")
                plot_data[d][readout][a] = (mu, sig)

    ylims: dict[str, tuple[float, float]] = {}
    for d in DATASETS:
        vals: list[float] = []
        for readout in READOUT_COMPARE_ORDER:
            for a in (a1, a2):
                mu, sig = plot_data[d][readout][a]
                if not np.isnan(mu):
                    s = 0.0 if np.isnan(sig) else sig
                    vals.extend([mu - s, mu + s])
        if not df_b.empty:
            for bk in BASELINE_MODEL_ORDER:
                br = _baseline_row_best_overall(df_b, d, bk)
                if br is not None and pd.notna(br.get(TEST_F1)):
                    vals.append(float(br[TEST_F1]))
        # Match plot_overall_best_test_f1 y-range padding so bar heights read similarly.
        ylims[d] = (min(vals) - 0.06, max(vals) + 0.09) if vals else (0.0, 1.0)

    ncols = len(DATASETS)
    # Extra width + tight_layout rect reserve the right margin for stacked legends (baseline + edge construction).
    fig_w = 12.25 if not df_b.empty else 12.0
    # Slightly shorter than best-overall row (4.85) for a tighter vertical footprint.
    fig_h = 4.35
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h), sharey=False)
    axes = np.atleast_1d(axes)

    bar_w = 0.22
    group_centers = np.arange(len(READOUT_COMPARE_ORDER), dtype=float) * 0.78
    adj_offsets = np.array([-bar_w / 2, bar_w / 2], dtype=float)

    for j, d in enumerate(DATASETS):
        ax = axes[j]
        for ro_i, readout in enumerate(READOUT_COMPARE_ORDER):
            for a_i, a in enumerate((a1, a2)):
                mu, sig = plot_data[d][readout][a]
                if np.isnan(mu):
                    continue
                x = group_centers[ro_i] + adj_offsets[a_i]
                ax.bar(
                    x,
                    mu,
                    yerr=0.0 if np.isnan(sig) else sig,
                    width=bar_w,
                    capsize=5,
                    color=adj_colors[a],
                    edgecolor="black",
                    linewidth=1.5,
                    alpha=0.9,
                    error_kw={"elinewidth": 1, "capthick": 1},
                    zorder=3,
                )
        _draw_baseline_test_f1_hlines(ax, df_b, d)
        ax.set_xticks(group_centers)
        ax.set_xticklabels([readout_titles[r] for r in READOUT_COMPARE_ORDER], fontsize=14)
        ax.tick_params(axis="y", labelsize=18)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(ylims[d])
        ax.set_xlim(group_centers[0] - 0.34, group_centers[-1] + 0.34)
        ax.set_title(_display_dataset(d), fontsize=22, fontweight="bold", pad=5)
        if j == 0:
            ax.set_ylabel("F1 Macro Score", fontsize=20)

    adj_leg_handles = [
        mpatches.Patch(facecolor=adj_colors[a1], edgecolor="black", label=adj_names[a1]),
        mpatches.Patch(facecolor=adj_colors[a2], edgecolor="black", label=adj_names[a2]),
    ]
    edge_legend_title = "Edge Construction"
    baseline_legend_title = "Baseline"
    legend_title_fs = 16
    # Upper stack: baseline then edge construction. ``upper left`` + shared x aligns left edges;
    # ``upper right`` + shared x only aligns right edges, so different legend widths looked staggered.
    leg_left_x = 0.705
    leg_baseline_y = 0.7
    # Below baseline block (ncol=1 stacks SVM / Elastic Net vertically).
    leg_edge_y = 0.45

    if not df_b.empty:
        leg_baseline = fig.legend(
            handles=_baseline_legend_line_handles(compact_labels=True),
            title=baseline_legend_title,
            loc="upper left",
            bbox_to_anchor=(leg_left_x, leg_baseline_y),
            ncol=1,
            fontsize=17,
            frameon=False,
        )
        leg_baseline.get_title().set_fontsize(legend_title_fs)
        fig.add_artist(leg_baseline)
        leg_edge = fig.legend(
            handles=adj_leg_handles,
            title=edge_legend_title,
            loc="upper left",
            bbox_to_anchor=(leg_left_x, leg_edge_y),
            ncol=1,
            fontsize=17,
            frameon=False,
        )
        leg_edge.get_title().set_fontsize(legend_title_fs)
        tight_top = 0.88
        tight_rect = [0, 0.05, 0.70, tight_top]
    else:
        leg_edge = fig.legend(
            handles=adj_leg_handles,
            title=edge_legend_title,
            loc="upper left",
            bbox_to_anchor=(leg_left_x, 0.88),
            ncol=1,
            fontsize=17,
            frameon=True,
        )
        leg_edge.get_title().set_fontsize(legend_title_fs)
        tight_top = 0.91
        tight_rect = [0, 0.05, 0.78, tight_top]
    fig.suptitle(
        "Graph-based Model Performance By Readout and Edge Construction Type",
        fontsize=20,
        fontweight="bold",
        x=0.45,
        y=0.85,
    )
    plt.tight_layout(rect=tight_rect, w_pad=0.12)
    fig.subplots_adjust(wspace=0.4)
    fig.supxlabel("Readout type", fontsize=20, x=0.4, y=0.02)
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_train_val_test_trajectories(
    df: pd.DataFrame,
    out_name: str = "train_val_test_trajectory_best_models.pdf",
) -> None:
    """Val-selected trajectories per dataset for best GNN, MLP, and baselines."""
    need = [C_DATA, C_MODEL, VAL_F1, TEST_F1, TRAIN_F1]
    if any(c not in df.columns for c in need):
        print(f"Skipping {out_name}: missing required aggregated columns.")
        return

    df_b = _load_baseline_agg()
    x = np.array([0, 1, 2], dtype=float)
    xticks = ["Train", "Val", "Test"]

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 4.8), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)

    style = {
        "best_gnn": dict(color="#D62828", marker="o", linewidth=2.4, markersize=7, label="Best GNN"),
        "mlp": dict(color=_model_color("mlp"), marker="^", linewidth=2.2, markersize=7, label="MLP"),
        "svm": dict(color=BASELINE_BAR_COLORS["svm"], marker="s", linewidth=2.0, markersize=6, label="SVM"),
        "elastic_net": dict(
            color=BASELINE_BAR_COLORS["elastic_net"],
            marker="D",
            linewidth=2.0,
            markersize=6,
            label="Elastic Net",
        ),
    }
    shown_labels: set[str] = set()

    def _pick_best_row_by_metric(
        sub: pd.DataFrame,
        metric_col: str,
        *,
        require_train_f1: bool = False,
    ) -> pd.Series | None:
        if sub.empty:
            return None
        ok = sub[metric_col].notna() & sub[VAL_F1].notna() & sub[TEST_F1].notna()
        if require_train_f1:
            ok &= sub[TRAIN_F1].notna()
        sub_ok = sub.loc[ok]
        if sub_ok.empty or sub_ok[metric_col].isna().all():
            return None
        return sub_ok.loc[sub_ok[metric_col].idxmax()]

    sel_metric = VAL_F1
    for j, d in enumerate(DATASETS):
        ax = axes[j]

        # Best GNN among non-MLP, selected on validation performance.
        gnn_models = [m for m in _ordered_models(df) if m != "mlp"]
        best_gnn_row: pd.Series | None = None
        for m in gnn_models:
            row = _pick_best_row_by_metric(
                df[(df[C_DATA] == d) & (df[C_MODEL] == m)],
                sel_metric,
                require_train_f1=True,
            )
            if row is None:
                continue
            if best_gnn_row is None or float(row[sel_metric]) > float(best_gnn_row[sel_metric]):
                best_gnn_row = row
        if best_gnn_row is not None:
            y = np.array(
                [float(best_gnn_row[TRAIN_F1]), float(best_gnn_row[VAL_F1]), float(best_gnn_row[TEST_F1])],
                dtype=float,
            )
            yerr = np.array(
                [
                    float(best_gnn_row[TRAIN_F1_STD]) if pd.notna(best_gnn_row.get(TRAIN_F1_STD, np.nan)) else 0.0,
                    float(best_gnn_row[VAL_F1_STD]) if pd.notna(best_gnn_row.get(VAL_F1_STD, np.nan)) else 0.0,
                    float(best_gnn_row[TEST_F1_STD]) if pd.notna(best_gnn_row.get(TEST_F1_STD, np.nan)) else 0.0,
                ],
                dtype=float,
            )
            kw = dict(style["best_gnn"])
            if kw["label"] in shown_labels:
                kw["label"] = None
            ax.errorbar(x, y, yerr=yerr, capsize=4, **kw)
            shown_labels.add(style["best_gnn"]["label"])

        # MLP.
        mlp_row = _pick_best_row_by_metric(
            df[(df[C_DATA] == d) & (df[C_MODEL] == "mlp")],
            sel_metric,
            require_train_f1=True,
        )
        if mlp_row is not None:
            y = np.array([float(mlp_row[TRAIN_F1]), float(mlp_row[VAL_F1]), float(mlp_row[TEST_F1])], dtype=float)
            yerr = np.array(
                [
                    float(mlp_row[TRAIN_F1_STD]) if pd.notna(mlp_row.get(TRAIN_F1_STD, np.nan)) else 0.0,
                    float(mlp_row[VAL_F1_STD]) if pd.notna(mlp_row.get(VAL_F1_STD, np.nan)) else 0.0,
                    float(mlp_row[TEST_F1_STD]) if pd.notna(mlp_row.get(TEST_F1_STD, np.nan)) else 0.0,
                ],
                dtype=float,
            )
            kw = dict(style["mlp"])
            if kw["label"] in shown_labels:
                kw["label"] = None
            ax.errorbar(x, y, yerr=yerr, capsize=4, **kw)
            shown_labels.add(style["mlp"]["label"])

        # Baselines, also selected on validation performance.
        if not df_b.empty:
            for bk in BASELINE_MODEL_ORDER:
                sub_b = df_b[(df_b[C_DATA] == d) & (df_b[C_MODEL] == bk)].dropna(subset=[VAL_F1, TEST_F1])
                if sub_b.empty:
                    continue
                br = sub_b.loc[sub_b[sel_metric].idxmax()] if sub_b[sel_metric].notna().any() else None
                if br is None:
                    continue
                y = np.array(
                    [
                        float(br[TRAIN_F1]) if pd.notna(br.get(TRAIN_F1)) else np.nan,
                        float(br[VAL_F1]) if pd.notna(br.get(VAL_F1)) else np.nan,
                        float(br[TEST_F1]) if pd.notna(br.get(TEST_F1)) else np.nan,
                    ],
                    dtype=float,
                )
                yerr = np.array(
                    [
                        float(br[TRAIN_F1_STD]) if pd.notna(br.get(TRAIN_F1_STD)) else 0.0,
                        float(br[VAL_F1_STD]) if pd.notna(br.get(VAL_F1_STD)) else 0.0,
                        float(br[TEST_F1_STD]) if pd.notna(br.get(TEST_F1_STD)) else 0.0,
                    ],
                    dtype=float,
                )
                kw = dict(style[bk])
                if kw["label"] in shown_labels:
                    kw["label"] = None
                finite = np.isfinite(y)
                ax.errorbar(x[finite], y[finite], yerr=yerr[finite], capsize=4, **kw)
                shown_labels.add(style[bk]["label"])

        ax.set_xticks(x)
        ax.set_xticklabels(xticks, fontsize=10)
        ax.set_title(_display_dataset(d), fontsize=14, fontweight="bold", pad=8)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlim(-0.15, 2.15)
        ax.set_ylim(0.35, 1.02)
        ax.tick_params(axis="y", labelsize=10)
        if j == 0:
            ax.set_ylabel("Val-selected\nF1 Macro", fontsize=12)

    handles = [
        Line2D(
            [0],
            [0],
            color=style[k]["color"],
            marker=style[k]["marker"],
            linewidth=style[k]["linewidth"],
            markersize=style[k]["markersize"],
            label=style[k]["label"],
        )
        for k in ("best_gnn", "mlp", "svm", "elastic_net")
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, fontsize=11, frameon=True)
    fig.suptitle(
        "Train-Val-Test Trajectories: Best models selected on validation F1\n"
        "Baselines show Val/Test when train metric is unavailable",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def _pooled_val_rank_test_frame(
    df: pd.DataFrame,
    dataset: str,
    models: list[str],
) -> pd.DataFrame | None:
    """All non-baseline configs for one dataset, sorted by val F1 desc, with pooled ``val_rank`` 1..N."""
    rows: list[dict[str, float | str]] = []
    for m in models:
        sub = df[(df[C_DATA] == dataset) & (df[C_MODEL] == m)]
        for _, row in sub.iterrows():
            v, t = row[VAL_F1], row[TEST_F1]
            if pd.notna(v) and pd.notna(t):
                rows.append({"model": m, "val": float(v), "test": float(t)})
    if not rows:
        return None
    pool = pd.DataFrame(rows).sort_values("val", ascending=False, kind="mergesort").reset_index(drop=True)
    pool["val_rank"] = np.arange(1, len(pool) + 1, dtype=float)
    return pool


def _binned_test_mean_std_along_val_rank(
    pool: pd.DataFrame,
    model_keys: frozenset[str],
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Equal-width bins on validation rank; mean test F1 and sample SD of test F1 within each bin."""
    sub = pool[pool["model"].isin(model_keys)]
    if sub.empty:
        return None
    max_r = float(pool["val_rank"].max())
    if max_r < 1:
        return None
    edges = np.linspace(0.5, max_r + 0.5, int(n_bins) + 1)
    centers: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for i in range(int(n_bins)):
        lo, hi = float(edges[i]), float(edges[i + 1])
        chunk = sub[(sub["val_rank"] > lo) & (sub["val_rank"] <= hi)]
        if chunk.empty:
            continue
        centers.append((lo + hi) / 2.0)
        means.append(float(chunk["test"].mean()))
        stds.append(float(chunk["test"].std(ddof=1)) if len(chunk) > 1 else 0.0)
    if not centers:
        return None
    return (
        np.array(centers, dtype=float),
        np.array(means, dtype=float),
        np.array(stds, dtype=float),
    )


def plot_validation_rank_vs_test_f1_pooled(
    df: pd.DataFrame,
    out_name: str = "val_rank_vs_test_f1_macro_pooled_dl.pdf",
    *,
    max_val_rank: float | None = None,
) -> None:
    """Test F1 vs pooled validation rank; single-color scatter + best achievable test (red line).

    If ``max_val_rank`` is set (e.g. 100), only ranks ``1..max_val_rank`` are shown and the x-axis
    is fixed to that range; the horizontal line remains the best test F1 over the full pooled list.
    """
    need = [C_DATA, C_MODEL, VAL_F1, TEST_F1]
    if any(c not in df.columns for c in need):
        print(f"Skipping {out_name}: missing required columns.")
        return

    models = _ordered_non_baseline_models(df)
    if not models:
        print(f"Skipping {out_name}: no non-baseline models in dataframe.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    for ax, d in zip(axes.flat, DATASETS):
        pool = _pooled_val_rank_test_frame(df, d, models)
        if pool is None:
            ax.set_title(_display_dataset(d), fontsize=12, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Validation rank", fontsize=11)
            ax.set_ylabel("Test F1 macro", fontsize=11)
            continue

        vis = pool if max_val_rank is None else pool.loc[pool["val_rank"] <= max_val_rank]
        if vis.empty:
            ax.set_title(_display_dataset(d), fontsize=12, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Validation rank", fontsize=11)
            ax.set_ylabel("Test F1 macro", fontsize=11)
            continue

        val_ranks = vis["val_rank"].to_numpy()
        test_f1s = vis["test"].to_numpy()
        ax.scatter(val_ranks, test_f1s, alpha=0.5, s=10, color="steelblue", zorder=2)
        best_test = float(np.max(pool["test"].to_numpy()))
        ax.axhline(
            best_test,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Best test F1",
            zorder=3,
        )
        ax.set_xlabel("Validation rank", fontsize=11)
        ax.set_ylabel("Test F1 macro", fontsize=11)
        ax.set_title(_display_dataset(d), fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        if max_val_rank is not None:
            ax.set_xlim(0.5, float(max_val_rank) + 0.5)

    zoom_suffix = f", top {int(max_val_rank)} ranks" if max_val_rank is not None else ""
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _place_val_rank_figure_title(fig, axes, "Test F1 vs validation rank" + zoom_suffix)
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def plot_validation_rank_vs_test_f1_pooled_by_model(
    df: pd.DataFrame,
    out_name: str = "val_rank_vs_test_f1_macro_pooled_dl_by_model.pdf",
    *,
    max_val_rank: float | None = None,
) -> None:
    """Pooled val-rank plot: per archetype, binned mean ± SD + best achievable test F1 in that archetype (hline).

    If ``max_val_rank`` is set, binning and curves use only pooled ranks ``1..max_val_rank``; the
    best-test horizontal lines still use the full pooled ranking (global best per archetype).
    """
    need = [C_DATA, C_MODEL, VAL_F1, TEST_F1]
    if any(c not in df.columns for c in need):
        print(f"Skipping {out_name}: missing required columns.")
        return

    models = _ordered_non_baseline_models(df)
    if not models:
        print(f"Skipping {out_name}: no non-baseline models in dataframe.")
        return

    model_set = set(models)
    n_bins = VAL_RANK_ARCHETYPE_N_BINS

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    for ax, d in zip(axes.flat, DATASETS):
        pool = _pooled_val_rank_test_frame(df, d, models)
        if pool is None:
            ax.set_title(_display_dataset(d), fontsize=12, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Validation rank", fontsize=11)
            ax.set_ylabel("Test F1 macro", fontsize=11)
            continue

        pool_vis = pool if max_val_rank is None else pool.loc[pool["val_rank"] <= max_val_rank]
        if pool_vis.empty:
            ax.set_title(_display_dataset(d), fontsize=12, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Validation rank", fontsize=11)
            ax.set_ylabel("Test F1 macro", fontsize=11)
            continue

        for label, mkeys, color in VAL_RANK_ARCHETYPE_GROUPS:
            keys = mkeys & model_set
            if not keys:
                continue
            binned = _binned_test_mean_std_along_val_rank(pool_vis, keys, n_bins=n_bins)
            if binned is None:
                continue
            xc, ym, ys = binned
            y_low = np.clip(ym - ys, 0.0, 1.0)
            y_high = np.clip(ym + ys, 0.0, 1.0)
            ax.fill_between(xc, y_low, y_high, color=color, alpha=0.22, linewidth=0, zorder=2)
            ax.plot(xc, ym, color=color, linewidth=2.0, label=label, zorder=3)

        for label, mkeys, color in VAL_RANK_ARCHETYPE_GROUPS:
            keys = mkeys & model_set
            if not keys:
                continue
            sub_g = pool[pool["model"].isin(keys)]
            if sub_g.empty:
                continue
            best_in_group = float(sub_g["test"].max())
            hcolor = VAL_RANK_ARCHETYPE_MLP_BEST_HLINE if label == "MLP" else color
            ax.axhline(
                best_in_group,
                color=hcolor,
                linestyle=":",
                linewidth=2.2,
                zorder=4,
                label=f"Best test ({_val_rank_best_test_short_label(label)})",
            )
        ax.set_xlabel("Validation rank", fontsize=11)
        ax.set_ylabel("Test F1 macro", fontsize=11)
        ax.set_title(_display_dataset(d), fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7, ncol=1)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        if max_val_rank is not None:
            ax.set_xlim(0.5, float(max_val_rank) + 0.5)

    zoom_suffix = f", top {int(max_val_rank)} ranks" if max_val_rank is not None else ""
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _place_val_rank_figure_title(
        fig, axes, "Test F1 vs validation rank (archetype groups)" + zoom_suffix
    )
    _save_figure_pdf_png(fig, out_name)
    plt.close(fig)


def run_all_plots(df: pd.DataFrame | None = None) -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["CMU Serif", "DejaVu Serif", "Times New Roman"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["savefig.dpi"] = SAVEFIG_DPI

    if df is None:
        df = load_prepared_df()
    if df.empty:
        print("No rows to plot (empty aggregated / filtered dataframe) — skipping PDFs.")
        return
    plot_overall_best_test_f1(df)
    plot_test_f1_by_adjacency(
        df,
        out_name="best_overall_test_f1_macro_by_adjmethod_2x2.pdf",
        grid_2x2=True,
    )
    plot_mega_method_dataset_ratio(df)
    plot_mega_method_dataset_ratio(
        df,
        out_name="best_performance_by_dataset_method_ratio_best_val.pdf",
        highlight_best_val_per_dataset_model=True,
    )
    plot_mega_method_dataset_ratio_collapsed_graph_based(df)
    plot_readout_adjacency_compact(
        df,
        out_name="effect_of_readout_and_adjacency_test_f1_macro_2x4.pdf",
    )
    plot_readout_effect(df)
    plot_adjacency_effect_readout_style(df)
    plot_readout_and_adjacency_stacked(df)
    plot_train_val_test_trajectories(df)
    plot_validation_rank_vs_test_f1_pooled(df)
    plot_validation_rank_vs_test_f1_pooled(
        df,
        f"val_rank_vs_test_f1_macro_pooled_dl_top{int(VAL_RANK_ZOOM_TOP_N)}.pdf",
        max_val_rank=float(VAL_RANK_ZOOM_TOP_N),
    )
    plot_validation_rank_vs_test_f1_pooled_by_model(df)
    plot_validation_rank_vs_test_f1_pooled_by_model(
        df,
        f"val_rank_vs_test_f1_macro_pooled_dl_by_model_top{int(VAL_RANK_ZOOM_TOP_N)}.pdf",
        max_val_rank=float(VAL_RANK_ZOOM_TOP_N),
    )
    print(f"Saved 14 figures (PDF + PNG) under {PLOTS_DIR}")


def _cli_verbose() -> bool:
    """Verbose aggregation + plot-prep report is default; use --quiet or OGBENCH_PLOT_QUIET=1 to silence."""
    if "--quiet" in sys.argv:
        return False
    if os.environ.get("OGBENCH_PLOT_QUIET", "").strip().lower() in ("1", "true", "yes"):
        return False
    ve = os.environ.get("OGBENCH_PLOT_VERBOSE", "").strip().lower()
    if ve in ("0", "false", "no"):
        return False
    return True


if __name__ == "__main__":
    verbose = _cli_verbose()
    agg = export_aggregated_final_results(verbose=verbose)
    print(f"Wrote {len(agg)} rows to {DEFAULT_OUTPUT_CSV}")
    if verbose:
        _verbose_plot_prep_report(agg, diagnose_models=None)
    run_all_plots(agg)

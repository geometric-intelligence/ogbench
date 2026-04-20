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
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from narrow_schema import EXPECTED_SEEDS, canonical_model_name

RUN_META_COLS = frozenset({"run_id", "run_name", "state"})
PER_RUN_METRIC_COLS = ("best_val_f1_macro", "best_test_f1_macro", "best_train_f1_macro")
SEED_COL = "seed"

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


def _baseline_legend_line_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=BASELINE_HLINE_COLOR["svm"],
            linestyle=BASELINE_HLINE_STYLE["svm"],
            linewidth=2.4,
            label="SVM baseline",
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE_HLINE_COLOR["elastic_net"],
            linestyle=BASELINE_HLINE_STYLE["elastic_net"],
            linewidth=2.4,
            label="Elastic Net baseline",
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


def _build_lean_aggregated(wide: pd.DataFrame) -> pd.DataFrame:
    lean = pd.DataFrame(index=wide.index)
    for short, long in _SHORT_FROM_LONG:
        if short in wide.columns:
            lean[short] = wide[short]
        elif long in wide.columns:
            lean[short] = wide[long]
        else:
            lean[short] = np.nan
    lean[C_MODEL] = wide[C_MODEL] if C_MODEL in wide.columns else np.nan
    lean["n_runs_seeds"] = wide["n_runs_seeds"]
    for m in PER_RUN_METRIC_COLS:
        lean[f"{m}_mean"] = wide[f"{m}_mean"]
        lean[f"{m}_std"] = wide[f"{m}_std"]
    col_order = [
        C_DATA,
        C_MODEL,
        C_ADJ,
        C_RATIO,
        C_METHOD,
        "readout_name",
        "n_runs_seeds",
    ]
    for m in PER_RUN_METRIC_COLS:
        col_order.extend([f"{m}_mean", f"{m}_std"])
    return lean[col_order]


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

    wide = gb.agg({m: ["mean", "std"] for m in PER_RUN_METRIC_COLS})
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["n_runs_seeds"] = n_runs.values
    ok = (wide["n_runs_seeds"] == EXPECTED_SEEDS) & (n_seeds_distinct.values == EXPECTED_SEEDS)
    pre = len(wide)
    wide = wide.loc[ok].copy()

    if verbose:
        print(f"\nLean aggregated rows: {len(wide)} / {pre} config buckets before 3-seed filter")

    out = _build_lean_aggregated(wide)

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
            os.replace(tmp_path, dest_png)
        except BaseException:
            if os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise

    # Temp file + atomic replace avoids Windows EINVAL when the destination PNG is open elsewhere.
    dpi_tries = [dpi]
    if dpi > 300:
        dpi_tries.append(min(dpi, 300))
    last_exc: OSError | None = None
    for png_dpi in dpi_tries:
        for attempt in range(4):
            try:
                _write_png_to_path(png_path, png_dpi)
                return
            except OSError as e:
                last_exc = e
                if getattr(e, "errno", None) != 22:
                    raise
                time.sleep(0.05 * (attempt + 1))
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


def _ordered_models(df: pd.DataFrame) -> list[str]:
    found = set(df[C_MODEL].dropna().unique())
    out = [m for m in CANONICAL_MODEL_ORDER if m in found]
    for m in sorted(found):
        if m not in out:
            out.append(m)
    return out


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
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12, 5))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        "Best Overall Model Performance by Test F1 Macro\n(Selected by Val F1 Macro)",
        fontsize=26,
        fontweight="bold",
        y=0.995,
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
        ylims[d] = (min(vals) - 0.05, max(vals) + 0.05) if vals else (0.0, 1.0)

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
                width=0.6,
                capsize=5,
                color=colors,
                edgecolor="black",
                linewidth=1.5,
                zorder=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=20)
        _draw_baseline_test_f1_hlines(ax, df_b, d)
        ax.set_title(_display_dataset(d), fontsize=24, fontweight="bold", pad=10)
        ax.tick_params(axis="both", labelsize=20)
        if j == 0:
            ax.set_ylabel("F1 Macro Score", fontsize=20)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(ylims[d])

    if not df_b.empty:
        fig.legend(
            handles=_baseline_legend_line_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=2,
            fontsize=14,
            frameon=True,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.86])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.92])
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
) -> None:
    """Rows = datasets, columns = sampling methods; bars = GNN models + baseline SVM/EN grays × node ratios."""
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
                        plot_tensor[meth][d][r][m] = (float(rr[TEST_F1]), float(rr[TEST_F1_STD]))
                    else:
                        plot_tensor[meth][d][r][m] = (np.nan, np.nan)

    plot_keys: list[str] = list(models) + (list(BASELINE_MODEL_ORDER) if not df_b.empty else [])
    n_keys = max(len(plot_keys), 1)

    row_mins: dict[str, float] = {}
    row_maxs: dict[str, float] = {}
    for d in DATASETS:
        lo, hi = float("inf"), 0.3
        for meth in methods:
            for r in ratios:
                for m in models:
                    mu, sig = plot_tensor[meth][d][r].get(m, (np.nan, np.nan))
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
                    mu, sig = plot_tensor[meth][d][r][m]
                    if np.isnan(mu):
                        continue
                    lbl = _display_model(m) if not labeled[m] else None
                    labeled[m] = True
                    ax.bar(
                        x_by_key[m][ri],
                        mu,
                        yerr=sig,
                        width=bar_w,
                        capsize=2,
                        label=lbl,
                        color=_model_color(m),
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
                        lbl = _display_model(bk) if not labeled[bk] else None
                        labeled[bk] = True
                        ax.bar(
                            x_by_key[bk][ri],
                            mu,
                            yerr=0.0 if np.isnan(sig) else sig,
                            width=bar_w,
                            capsize=2,
                            label=lbl,
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
            not np.isnan(plot_tensor[meth][d][r][m][0])
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
    ncol_leg = max(1, min(len(uh) + len(uh_b), 12))
    fig.legend(
        uh + uh_b,
        ul + ul_b,
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
        dataset_ylims[dataset] = (min(vals) - 0.05, max(vals) + 0.05) if vals else (0.0, 1.0)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 6))
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
        ax.set_xticklabels([_display_model(m) for m in models], rotation=45, ha="right", fontsize=30)
        ax.tick_params(axis="both", labelsize=30)
        ax.set_title(_display_dataset(dataset), fontsize=34, fontweight="bold", pad=10)
        if j == 0:
            ax.set_ylabel("F1 Macro Score", fontsize=30)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(dataset_ylims[dataset])

    fig.suptitle("Effect of Readout on Model Performance", fontsize=36, fontweight="bold", y=0.995)
    legend_handles = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="", alpha=0.9, label="MLP Readout"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch=".", alpha=0.6, label="Vanilla Readout"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        fontsize=28,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
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
    plot_test_f1_by_adjacency(df)
    plot_test_f1_by_adjacency(
        df,
        out_name="best_overall_test_f1_macro_by_adjmethod_2x2.pdf",
        grid_2x2=True,
    )
    plot_train_minus_val(df)
    plot_mega_method_dataset_ratio(df)
    plot_mega_method_dataset_ratio_08_and_10(df)
    plot_dataset_method_ratio_representative_mpnn(df)
    plot_readout_effect(df)
    print(f"Saved 8 figures (PDF + PNG) under {PLOTS_DIR}")


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

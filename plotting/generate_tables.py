# LaTeX tables from the same per-run export as ``generate_main_plots.py`` (W&B CSV).
# Raw: plotting/final_results_hyperparams_neurips.csv
# Reuses fingerprint grouping + 3-seed aggregation; optional columns
# ``best_test_f1_weighted``, ``best_test_accuracy``, ``best_test_auroc`` when present
# (exported by ``load_results.py`` after re-fetch).
# Best-config + resource table: test metrics from the main CSV winner buckets; GPU / params /
# train + per-epoch time from W&B ``best_model_reruns`` (tag ``best_rerun``), mean ± std over seeds.

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import generate_main_plots as gmp
from narrow_schema import EXPECTED_SEEDS, canonical_model_name
from winner_fingerprints import compute_winner_fingerprints

# Match ``load_results.wandb_username`` without importing ``load_results`` (that imports wandb).
DEFAULT_WANDB_ENTITY = "bioshape-lab"

# Row order in the paper tables
TABLE_READOUT_ORDER = ("NoReadOut", "OmicsReadOut")
# Raw ``adjacency_method`` values (see ``ADJ_METHOD_DISPLAY_NAMES`` in generate_main_plots)
TABLE_ADJ_ORDER = ("string", "wgcna")

OPTIONAL_EXTRA_TEST_METRICS = gmp.OPTIONAL_EXTRA_TEST_METRICS

TABLE_METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("$F_{macro}$", "best_test_f1_macro_mean", "best_test_f1_macro_std"),
    ("$F_{weighted}$", "best_test_f1_weighted_mean", "best_test_f1_weighted_std"),
    ("Accuracy", "best_test_accuracy_mean", "best_test_accuracy_std"),
    ("AUROC", "best_test_auroc_mean", "best_test_auroc_std"),
)

MEAN_TIE_TOL = 1e-9

BEST_RERUN_TAG_DEFAULT = "best_rerun"
DEFAULT_BEST_RERUN_WANDB_PROJECT = "best_model_reruns"
DATASET_TAG_SET = frozenset(str(d).lower() for d in gmp.DATASETS)
IGNORE_BEST_RERUN_TAGS = frozenset({"hpsearch", "best_rerun", "rerun"})

# Primary W&B keys (slash form as logged). ``_rerun_pick_metric`` also tries dotted keys and ``run.config``.
_RERUN_SUMMARY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("peak_mem_gb", ("GPU/peak_memory_allocated_GB",)),
    (
        "params_total",
        (
            "model/params/total",
            "model.params.total",
        ),
    ),
    ("train_time_s", ("PaperMetrics/total_train_time_s", "PaperMetrics.total_train_time_s")),
    (
        "per_epoch_time_s",
        ("PaperMetrics/per_epoch_time_s", "PaperMetrics.per_epoch_time_s"),
    ),
)


def _coerce_float_wandb(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    if isinstance(v, str):
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none"):
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")
    if isinstance(v, dict) and "value" in v:
        return _coerce_float_wandb(v.get("value"))
    return float("nan")


def _wandb_config_as_dict(cfg: Any) -> dict[str, Any]:
    """``wandb.Api().runs()`` often returns lazy runs with ``config`` empty, a string, or non-dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, str):
        import json

        try:
            out = json.loads(cfg)
        except json.JSONDecodeError:
            return {}
        return out if isinstance(out, dict) else {}
    return {}


def _mapping_pick_numeric(m: dict[str, Any], keys: tuple[str, ...]) -> float:
    for k in keys:
        if k not in m:
            continue
        f = _coerce_float_wandb(m[k])
        if not np.isnan(f):
            return f
    return float("nan")


def _scan_flat_for_param_total(flat: dict[str, Any]) -> float:
    """Last resort: any flat key that looks like ``*params*total*`` (excl. trainable splits)."""
    for fk, fv in flat.items():
        if not isinstance(fk, str):
            continue
        lk = fk.lower().replace("\\", "/")
        if "param" not in lk or "total" not in lk:
            continue
        if "trainable" in lk or "non_train" in lk:
            continue
        f = _coerce_float_wandb(fv)
        if not np.isnan(f) and f > 0:
            return f
    return float("nan")


def _rerun_pick_metric(run: Any, keys: tuple[str, ...]) -> float:
    """Read a scalar from W&B ``run.summary`` and/or ``run.config`` (hyperparams often live in config only)."""
    summary = getattr(run, "summary", None)
    if summary is not None:
        for k in keys:
            try:
                if hasattr(summary, "get"):
                    v = summary.get(k)
                elif k in summary:
                    v = summary[k]
                else:
                    v = None
            except (TypeError, KeyError):
                v = None
            if v is not None:
                f = _coerce_float_wandb(v)
                if not np.isnan(f):
                    return f
        try:
            sd = dict(summary)
        except Exception:
            sd = {}
        f = _mapping_pick_numeric(sd, keys)
        if not np.isnan(f):
            return f

    cfg = _wandb_config_as_dict(getattr(run, "config", None))
    if cfg:
        f = _mapping_pick_numeric(cfg, keys)
        if not np.isnan(f):
            return f
        flat = _flatten_hydra_config(cfg)
        f = _mapping_pick_numeric(flat, keys)
        if not np.isnan(f):
            return f
        for k in keys:
            kd = k.replace("/", ".")
            if kd in flat:
                f = _coerce_float_wandb(flat[kd])
                if not np.isnan(f):
                    return f
        if "model/params/total" in keys or "model.params.total" in keys:
            f = _scan_flat_for_param_total(flat)
            if not np.isnan(f):
                return f
    return float("nan")


def _flatten_hydra_config(config: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten_hydra_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _flatten_cfg_seed(config: dict[str, Any] | None) -> float:
    if not config:
        return float("nan")
    flat = _flatten_hydra_config(config)
    for key in ("seed", "seed.value"):
        if key in flat:
            try:
                return float(flat[key])
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def _model_data_from_wandb_tags(tags: object) -> tuple[str | None, str | None]:
    if not tags:
        return None, None
    low = [str(t).strip().lower() for t in tags if str(t).strip()]
    d_found = [t for t in low if t in DATASET_TAG_SET]
    if len(d_found) != 1:
        return None, None
    data_name = d_found[0]
    model_raw: str | None = None
    priority = {m: i for i, m in enumerate(gmp.CANONICAL_MODEL_ORDER)}
    best_pri = 10**9
    for t in low:
        if t in IGNORE_BEST_RERUN_TAGS or t == data_name:
            continue
        cm = canonical_model_name(t)
        if not cm:
            continue
        pr = priority.get(cm, 500)
        if pr < best_pri:
            best_pri = pr
            model_raw = t
    if model_raw is None:
        return None, None
    model = canonical_model_name(model_raw)
    if not model:
        return None, None
    return str(model), data_name


def fetch_best_rerun_resource_long(
    *,
    wandb_entity: str,
    wandb_project: str,
    rerun_tag: str = BEST_RERUN_TAG_DEFAULT,
    api_timeout: int = 120,
    per_page: int = 500,
    verbose: bool = True,
) -> pd.DataFrame:
    """One row per finished rerun with parsed model/data/seed and summary resource metrics."""
    import wandb  # optional: only needed when fetching reruns from the API

    path = f"{wandb_entity}/{wandb_project}"
    if verbose:
        print(f"Fetching best reruns from W&B {path} (tag {rerun_tag!r})…", flush=True)
    api = wandb.Api(timeout=api_timeout)
    try:
        runs = api.runs(path, per_page=per_page, lazy=False)
    except TypeError:
        runs = api.runs(path, per_page=per_page)
    rows: list[dict[str, Any]] = []
    tag_l = rerun_tag.strip().lower()
    for run in runs:
        tags = getattr(run, "tags", None) or []
        tags_l = [str(x).strip().lower() for x in tags]
        if tag_l not in tags_l:
            continue
        if str(getattr(run, "state", "") or "").lower() != "finished":
            continue
        model, data_name = _model_data_from_wandb_tags(tags)
        if not model or not data_name:
            if verbose:
                print(f"  skip run {run.id}: could not parse model/data from tags {tags!r}")
            continue
        run_full = api.run(f"{path}/{run.id}")
        cfg_dict = _wandb_config_as_dict(getattr(run_full, "config", None))
        rec: dict[str, Any] = {
            "run_id": str(run.id),
            gmp.C_MODEL: model,
            gmp.C_DATA: data_name,
            gmp.SEED_COL: _flatten_cfg_seed(cfg_dict),
        }
        for out_key, wb_keys in _RERUN_SUMMARY_ALIASES:
            rec[out_key] = _rerun_pick_metric(run_full, wb_keys)
        rows.append(rec)
    if verbose:
        n_par = sum(1 for r in rows if not np.isnan(float(r.get("params_total", np.nan))))
        print(
            f"  collected {len(rows)} finished run(s) with tag {rerun_tag!r}; "
            f"non-NaN params_total: {n_par}/{len(rows)}.",
            flush=True,
        )
    return pd.DataFrame(rows)


def load_best_rerun_resource_csv(path: str) -> pd.DataFrame:
    """CSV with ``model_name``, ``data_name``, and numeric resource columns.

    Accepts short names ``peak_mem_gb``, ``params_total``, ``train_time_s``, ``per_epoch_time_s``
    or the W&B paths used in ``_RERUN_SUMMARY_ALIASES``.
    """
    df = pd.read_csv(path, low_memory=False)
    need = {gmp.C_MODEL, gmp.C_DATA}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"CSV {path!r} missing columns {sorted(miss)}")
    out = df.copy()
    out[gmp.C_MODEL] = (
        out[gmp.C_MODEL].astype(str).str.strip().str.lower().map(canonical_model_name)
    )
    out[gmp.C_DATA] = out[gmp.C_DATA].astype(str).str.strip().str.lower()
    rename: dict[str, str] = {}
    for out_key, wb_keys in _RERUN_SUMMARY_ALIASES:
        if out_key in out.columns:
            continue
        for wk in wb_keys:
            if wk in out.columns:
                rename[wk] = out_key
                break
        else:
            tail = wb_keys[0].split("/")[-1]
            if tail in out.columns:
                rename[tail] = out_key
    out = out.rename(columns=rename)
    for out_key, _ in _RERUN_SUMMARY_ALIASES:
        if out_key not in out.columns:
            out[out_key] = np.nan
    if gmp.SEED_COL not in out.columns:
        out[gmp.SEED_COL] = np.nan
    return out


def aggregate_best_rerun_resources(df_long: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds per (model_name, data_name)."""
    empty_cols = [
        gmp.C_MODEL,
        gmp.C_DATA,
        "peak_mem_gb_mean",
        "peak_mem_gb_std",
        "params_total_mean",
        "params_total_std",
        "train_time_s_mean",
        "train_time_s_std",
        "per_epoch_time_s_mean",
        "per_epoch_time_s_std",
    ]
    if df_long.empty:
        return pd.DataFrame(columns=empty_cols)
    work = df_long.dropna(subset=[gmp.C_MODEL, gmp.C_DATA]).copy()
    num_cols = [k for k, _ in _RERUN_SUMMARY_ALIASES if k in work.columns]
    if not num_cols:
        return pd.DataFrame(columns=empty_cols)
    for c in num_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    gb = work.groupby([gmp.C_MODEL, gmp.C_DATA], dropna=False)
    wide = gb.agg({c: ["mean", "std"] for c in num_cols})
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.reset_index()


def _fmt_params_human(n: float) -> str:
    if n is None or (isinstance(n, float) and (np.isnan(n) or n < 0)):
        return "---"
    x = float(n)
    if x >= 1e6:
        s = f"{x / 1e6:.1f}M"
        return s.replace(".0M", "M") if s.endswith(".0M") else s
    if x >= 1e3:
        return f"{int(round(x / 1e3))}K"
    return str(int(round(x)))


def _fmt_resource_pm(mu: float, sig: float, *, decimals: int) -> str:
    if mu is None or (isinstance(mu, float) and np.isnan(mu)):
        return "---"
    m = float(mu)
    if sig is None or (isinstance(sig, float) and (np.isnan(sig) or sig < 1e-9)):
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} $\\pm$ {float(sig):.{decimals}f}"


def _latex_cell_styled(
    body: str,
    *,
    bold: bool,
    within: bool,
) -> str:
    if body == "---":
        return "---"
    if bold:
        return f"\\cellcolor{{gray!15}}\\textbf{{{body}}}"
    if within:
        return f"\\cellcolor{{purple!12}}{body}"
    return body


def _column_style_higher_n(mus: list[float], sigs: list[float]) -> tuple[list[bool], list[bool]]:
    n = len(mus)
    bold = [False] * n
    within = [False] * n
    finite = [i for i in range(n) if mus[i] is not None and not (isinstance(mus[i], float) and np.isnan(mus[i]))]
    if not finite:
        return bold, within
    best_i = max(finite, key=lambda i: mus[i])
    best_mu = mus[best_i]
    best_sig = sigs[best_i]
    band = (
        0.0
        if best_sig is None or (isinstance(best_sig, float) and np.isnan(best_sig))
        else float(best_sig)
    )
    winners = [i for i in finite if abs(mus[i] - best_mu) <= MEAN_TIE_TOL]
    for i in winners:
        bold[i] = True
    for i in finite:
        if bold[i]:
            continue
        if mus[i] >= best_mu - band - MEAN_TIE_TOL:
            within[i] = True
    return bold, within


def _column_style_lower_n(mus: list[float], sigs: list[float]) -> tuple[list[bool], list[bool]]:
    n = len(mus)
    bold = [False] * n
    within = [False] * n
    finite = [i for i in range(n) if mus[i] is not None and not (isinstance(mus[i], float) and np.isnan(mus[i]))]
    if not finite:
        return bold, within
    best_i = min(finite, key=lambda i: mus[i])
    best_mu = mus[best_i]
    best_sig = sigs[best_i]
    band = (
        0.0
        if best_sig is None or (isinstance(best_sig, float) and np.isnan(best_sig))
        else float(best_sig)
    )
    winners = [i for i in finite if abs(mus[i] - best_mu) <= MEAN_TIE_TOL]
    for i in winners:
        bold[i] = True
    for i in finite:
        if bold[i]:
            continue
        if mus[i] <= best_mu + band + MEAN_TIE_TOL:
            within[i] = True
    return bold, within


def _models_present_for_dataset(
    perf_rows: pd.DataFrame,
    res_rows: pd.DataFrame,
    dataset_key: str,
) -> list[str]:
    d = str(dataset_key).strip().lower()
    s1 = set()
    if not perf_rows.empty and gmp.C_MODEL in perf_rows.columns:
        s1 |= set(perf_rows.loc[perf_rows[gmp.C_DATA].astype(str).str.lower() == d, gmp.C_MODEL].dropna().unique())
    if not res_rows.empty and gmp.C_MODEL in res_rows.columns:
        s1 |= set(res_rows.loc[res_rows[gmp.C_DATA].astype(str).str.lower() == d, gmp.C_MODEL].dropna().unique())
    return [m for m in gmp.CANONICAL_MODEL_ORDER if m in s1]


def best_config_resource_table_tex(
    datasets: list[str],
    lean: pd.DataFrame,
    df_raw: pd.DataFrame,
    rerun_agg: pd.DataFrame,
    *,
    expected_seeds: int = EXPECTED_SEEDS,
    label_suffix: str = "best-configs-with-gpu",
) -> str:
    """Main-CSV test metrics for val-selected winner buckets; resource stats from ``rerun_agg``."""
    winners_df, group_cols, _ = compute_winner_fingerprints(
        df_raw, rank_by="val", expected_seeds=expected_seeds
    )
    if winners_df.empty:
        raise ValueError("No winner buckets (check per-run CSV and expected seed count).")

    wk = winners_df.copy()
    wk[gmp.BUCKET_KEY_COL] = gmp._compose_bucket_key_frame(wk, group_cols)
    if gmp.BUCKET_KEY_COL not in lean.columns:
        raise KeyError(
            f"Aggregated lean frame missing {gmp.BUCKET_KEY_COL!r}; cannot match winner buckets."
        )
    perf = lean.merge(wk[[gmp.BUCKET_KEY_COL]], on=gmp.BUCKET_KEY_COL, how="inner")
    if perf.empty:
        raise ValueError("Could not merge winner buckets into aggregated lean frame.")

    perf = perf.drop_duplicates(subset=[gmp.C_MODEL, gmp.C_DATA], keep="first")

    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\caption{Best configuration per model and dataset, selected by validation $F_{\mathrm{macro}}$; "
        r"test performance (mean $\pm$ std) comes from the original grid-search runs. "
        r"\# parameters and GPU memory are means over \texttt{best\_model\_reruns} seeds (single value per cell); "
        r"training and per-epoch times are mean $\pm$ std over seeds. "
        r"Best value per dataset and column is bold on gray; other entries within one std of that best are shaded purple.}",
        r"\begin{tabular*}{\textwidth}{@{}l l c c c c c c c c@{}}",
        r"\toprule",
        r" &  & \makecell{$F_{\mathrm{macro}}$ \\ $(\uparrow)$} & \makecell{$F_{\mathrm{weighted}}$ \\ $(\uparrow)$} "
        r"& \makecell{Accuracy \\ $(\uparrow)$} & \makecell{AUROC \\ $(\uparrow)$} "
        r"& \makecell{$\#$ Params. \\ $(\downarrow)$} & \makecell{GPU \\ memory \\ (GB) \\ $(\downarrow)$} "
        r"& \makecell{Training time \\ end-to-end \\ (s) \\ $(\downarrow)$} & \makecell{Time / \\ epoch \\ (s) \\ $(\downarrow)$} \\",
        r"\midrule",
    ]

    metric_specs: tuple[tuple[str, str, str], ...] = (
        ("$F_{macro}$", "best_test_f1_macro_mean", "best_test_f1_macro_std"),
        ("$F_{weighted}$", "best_test_f1_weighted_mean", "best_test_f1_weighted_std"),
        ("Accuracy", "best_test_accuracy_mean", "best_test_accuracy_std"),
        ("AUROC", "best_test_auroc_mean", "best_test_auroc_std"),
    )

    for dsi, dkey in enumerate(datasets):
        dkey = str(dkey).strip().lower()
        if dsi > 0:
            lines.append(r"\midrule")
        disp_ds = gmp._display_dataset(dkey)
        lines.append(rf"\rowcolor{{gray!20}} \multicolumn{{10}}{{l}}{{\textbf{{{disp_ds}}}}}\\")

        models = _models_present_for_dataset(perf, rerun_agg, dkey)
        if not models:
            continue

        n = len(models)
        res_field_mu = {
            "params": np.full(n, np.nan),
            "p_mem": np.full(n, np.nan),
            "tr": np.full(n, np.nan),
            "pe": np.full(n, np.nan),
        }
        res_field_sig = {k: np.full(n, np.nan) for k in res_field_mu}

        for mi, model in enumerate(models):
            rr = rerun_agg[
                (rerun_agg[gmp.C_MODEL] == model) & (rerun_agg[gmp.C_DATA].astype(str).str.lower() == dkey)
            ]
            if not rr.empty:
                r0 = rr.iloc[0]
                for key, mean_c, std_c in (
                    ("params", "params_total_mean", "params_total_std"),
                    ("p_mem", "peak_mem_gb_mean", "peak_mem_gb_std"),
                    ("tr", "train_time_s_mean", "train_time_s_std"),
                    ("pe", "per_epoch_time_s_mean", "per_epoch_time_s_std"),
                ):
                    if mean_c in r0.index:
                        v = r0[mean_c]
                        res_field_mu[key][mi] = float(v) if pd.notna(v) else np.nan
                    if std_c in r0.index:
                        v = r0[std_c]
                        res_field_sig[key][mi] = float(v) if pd.notna(v) else np.nan

        style_per_metric: list[tuple[list[bool], list[bool]]] = []
        for _, mcol, scol in metric_specs:
            mus = []
            sigs = []
            for mi, model in enumerate(models):
                pr = perf[(perf[gmp.C_MODEL] == model) & (perf[gmp.C_DATA].astype(str).str.lower() == dkey)]
                if pr.empty:
                    mus.append(float("nan"))
                    sigs.append(float("nan"))
                else:
                    row = pr.iloc[0]
                    mu_v = row[mcol] if mcol in row.index else np.nan
                    s_v = row[scol] if scol in row.index else np.nan
                    mus.append(float(mu_v) if pd.notna(mu_v) else float("nan"))
                    sigs.append(float(s_v) if pd.notna(s_v) else float("nan"))
            style_per_metric.append(_column_style_higher_n(mus, sigs))

        res_styles = {
            "params": _column_style_lower_n(res_field_mu["params"].tolist(), res_field_sig["params"].tolist()),
            "p_mem": _column_style_lower_n(res_field_mu["p_mem"].tolist(), res_field_sig["p_mem"].tolist()),
            "tr": _column_style_lower_n(res_field_mu["tr"].tolist(), res_field_sig["tr"].tolist()),
            "pe": _column_style_lower_n(res_field_mu["pe"].tolist(), res_field_sig["pe"].tolist()),
        }

        for mi, model in enumerate(models):
            m_cell = gmp._display_model(model)
            cells: list[str] = []

            for k, (_, mcol, scol) in enumerate(metric_specs):
                pr = perf[(perf[gmp.C_MODEL] == model) & (perf[gmp.C_DATA].astype(str).str.lower() == dkey)]
                if pr.empty:
                    cells.append("---")
                    continue
                row = pr.iloc[0]
                mu_v = row[mcol] if mcol in row.index else np.nan
                s_v = row[scol] if scol in row.index else np.nan
                mu = float(mu_v) if pd.notna(mu_v) else float("nan")
                sig = float(s_v) if pd.notna(s_v) else float("nan")
                bld, wth = style_per_metric[k][0][mi], style_per_metric[k][1][mi]
                body = _fmt_pm(mu, sig)
                cells.append(_latex_cell_styled(body, bold=bld, within=wth))

            rr = rerun_agg[
                (rerun_agg[gmp.C_MODEL] == model) & (rerun_agg[gmp.C_DATA].astype(str).str.lower() == dkey)
            ]
            r0 = rr.iloc[0] if not rr.empty else None

            def _res_cell(
                key: str,
                *,
                decimals: int,
                human_params: bool,
                mean_only: bool = False,
            ) -> str:
                mu = res_field_mu[key][mi]
                sig = res_field_sig[key][mi]
                bld, wth = res_styles[key][0][mi], res_styles[key][1][mi]
                if r0 is None or np.isnan(mu):
                    return "---"
                if human_params:
                    body = _fmt_params_human(mu)
                elif mean_only:
                    body = f"{float(mu):.{decimals}f}"
                else:
                    body = _fmt_resource_pm(mu, sig, decimals=decimals)
                return _latex_cell_styled(body, bold=bld, within=wth)

            cells.append(_res_cell("params", decimals=0, human_params=True))
            cells.append(_res_cell("p_mem", decimals=2, human_params=False, mean_only=True))
            cells.append(_res_cell("tr", decimals=2, human_params=False))
            cells.append(_res_cell("pe", decimals=2, human_params=False))

            lines.append(" & " + f"{m_cell} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            rf"\label{{tab:{label_suffix}}}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _coerce_optional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    for c in OPTIONAL_EXTRA_TEST_METRICS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def aggregate_per_run_for_tables(
    df_input: pd.DataFrame,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Same 3-seed fingerprint buckets as ``export_aggregated_final_results``, plus optional test metrics."""
    df = _coerce_optional_metrics(df_input.copy())
    df, group_cols = gmp.prepare_per_run_df_for_fingerprint_grouping(df, verbose=verbose)

    metric_cols = list(gmp.PER_RUN_METRIC_COLS)
    for c in OPTIONAL_EXTRA_TEST_METRICS:
        if c in df.columns and df[c].notna().any() and c not in metric_cols:
            metric_cols.append(c)

    for m in metric_cols:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    gb = df.groupby(group_cols, dropna=False)
    n_runs = gb.size()
    n_seeds_distinct = gb[gmp.SEED_COL].nunique()
    strict = gmp._strict_oversized_seed_groups()
    if strict:
        gmp.assert_no_oversized_seed_groups(n_runs, n_seeds_distinct, group_cols=group_cols)
    elif verbose:
        over = (n_runs > gmp.EXPECTED_SEEDS) | (n_seeds_distinct > gmp.EXPECTED_SEEDS)
        if over.any():
            print(
                f"(Relaxed mode) {int(over.sum())} bucket(s) would exceed "
                f"{gmp.EXPECTED_SEEDS} runs/seeds."
            )

    wide = gb.agg({m: ["mean", "std"] for m in metric_cols})
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide[gmp.BUCKET_KEY_COL] = gmp._compose_bucket_key_frame(wide, group_cols)
    wide["n_runs_seeds"] = n_runs.values
    k = gmp.EXPECTED_SEEDS
    ok = (wide["n_runs_seeds"] == k) & (n_seeds_distinct.values == k)
    wide = wide.loc[ok].copy()

    lean = pd.DataFrame(index=wide.index)
    for short, long in gmp._SHORT_FROM_LONG:
        if short in wide.columns:
            lean[short] = wide[short]
        elif long in wide.columns:
            lean[short] = wide[long]
        else:
            lean[short] = np.nan
    if gmp.C_MODEL in wide.columns:
        lean[gmp.C_MODEL] = wide[gmp.C_MODEL]
    elif "model.model_name" in wide.columns:
        lean[gmp.C_MODEL] = wide["model.model_name"].map(canonical_model_name)
    else:
        lean[gmp.C_MODEL] = np.nan
    if gmp.BUCKET_KEY_COL in wide.columns:
        lean[gmp.BUCKET_KEY_COL] = wide[gmp.BUCKET_KEY_COL]
    lean["n_runs_seeds"] = wide["n_runs_seeds"]
    for m in metric_cols:
        lean[f"{m}_mean"] = wide[f"{m}_mean"]
        lean[f"{m}_std"] = wide[f"{m}_std"]
    return lean


def _gnn_models_for_table(df: pd.DataFrame) -> list[str]:
    found = {str(x) for x in df[gmp.C_MODEL].dropna().unique()}
    models = [m for m in gmp.CANONICAL_MODEL_ORDER if m in found and m != "mlp"]
    for m in sorted(found):
        if m != "mlp" and m not in models:
            models.append(m)
    return models


def _fmt_pm(mu: float, sig: float) -> str:
    s_mu = f"{float(mu):.3f}"
    if sig is None or (isinstance(sig, float) and np.isnan(sig)):
        return s_mu
    return f"{s_mu} $\\pm$ {float(sig):.3f}"


def _latex_cell(mu: object, sig: object, *, bold: bool, within: bool) -> str:
    if mu is None or (isinstance(mu, (float, np.floating)) and np.isnan(mu)):
        return "---"
    m = float(mu)
    s = float(sig) if sig is not None and not (isinstance(sig, float) and np.isnan(sig)) else float("nan")
    body = _fmt_pm(m, s)
    if bold:
        return f"\\textbf{{{body}}}"
    if within:
        return f"\\withinstd {body}"
    return body


def _pair_style(
    mus: list[float],
    sigs: list[float],
) -> tuple[list[bool], list[bool]]:
    """Per pair of conditions (e.g. two readouts or two adjacencies): (bold, within_std) for one metric."""
    n = len(mus)
    bold = [False] * n
    within = [False] * n
    finite = [
        i
        for i in range(n)
        if mus[i] is not None and not (isinstance(mus[i], float) and np.isnan(mus[i]))
    ]
    if not finite:
        return bold, within
    best_i = max(finite, key=lambda i: mus[i])
    best_mu = mus[best_i]
    best_sig = sigs[best_i]
    band = 0.0 if best_sig is None or (isinstance(best_sig, float) and np.isnan(best_sig)) else float(best_sig)

    winners = [i for i in finite if abs(mus[i] - best_mu) <= MEAN_TIE_TOL]
    for i in winners:
        bold[i] = True
    for i in finite:
        if bold[i]:
            continue
        if mus[i] >= best_mu - band - MEAN_TIE_TOL:
            within[i] = True
    return bold, within


def readout_ablation_table_tex(
    dataset_key: str,
    lean: pd.DataFrame,
    *,
    label_suffix: str | None = None,
) -> str:
    if gmp.C_READOUT not in lean.columns:
        raise KeyError(f"Aggregated frame missing {gmp.C_READOUT!r}.")
    dkey = str(dataset_key).strip().lower()
    dfc = lean[lean[gmp.C_DATA].astype(str).str.strip().str.lower() == dkey].copy()
    if dfc.empty:
        raise ValueError(f"No rows for dataset {dataset_key!r} in aggregated frame.")

    models = _gnn_models_for_table(dfc)
    if not models:
        raise ValueError(f"No non-MLP models for dataset {dataset_key!r}.")

    disp_ds = gmp._display_dataset(dkey)
    label = label_suffix or f"{dkey}-readout-vs-noreadout"

    col_heads = " & ".join(h for h, _, _ in TABLE_METRIC_SPECS)
    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\caption{{Readout ablation on \textit{{{disp_ds}}}. For each (model, readout), we select the configuration with the highest validation $F_{{macro}}$ (\texttt{{val\_f1\_macro}}) and report test metrics (mean $\pm$ std) from \texttt{{summary.best\_test/*}}. For each metric within a model pair, the higher mean is bolded; the other cell is shaded blue if it is within one std of the bolded cell (mean $\ge$ best mean $-$ best std).}}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{2.4cm} l l Y Y Y Y}",
        r"\toprule",
        rf"Dataset & Model & Readout & {col_heads} \\",
        r"\midrule",
        rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{disp_ds}}}}}\\[-0.6ex]",
    ]

    for mi, model in enumerate(models):
        if mi > 0:
            lines.append(r"\midrule")
        rows_by_readout: dict[str, pd.Series | None] = {}
        for rname in TABLE_READOUT_ORDER:
            sub = dfc[(dfc[gmp.C_MODEL] == model) & (dfc[gmp.C_READOUT].astype(str) == rname)]
            rows_by_readout[rname] = gmp._pick_best_row(sub)

        for ri, readout in enumerate(TABLE_READOUT_ORDER):
            m_cell = gmp._display_model(model) if ri == 0 else ""
            r_cell = readout
            cells: list[str] = []
            for _, mcol, scol in TABLE_METRIC_SPECS:
                mus_pair: list[float] = []
                sigs_pair: list[float] = []
                for rname in TABLE_READOUT_ORDER:
                    rr = rows_by_readout[rname]
                    if rr is None:
                        mus_pair.append(float("nan"))
                        sigs_pair.append(float("nan"))
                    else:
                        mu_v = rr[mcol] if mcol in rr.index else np.nan
                        s_v = rr[scol] if scol in rr.index else np.nan
                        mus_pair.append(float(mu_v) if pd.notna(mu_v) else float("nan"))
                        sigs_pair.append(float(s_v) if pd.notna(s_v) else float("nan"))
                bld, wth = _pair_style(mus_pair, sigs_pair)
                cells.append(
                    _latex_cell(
                        mus_pair[ri],
                        sigs_pair[ri],
                        bold=bld[ri],
                        within=wth[ri],
                    )
                )

            lines.append(" & " + f"{m_cell} & {r_cell} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            rf"\label{{tab:{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def adjacency_ablation_table_tex(
    dataset_key: str,
    lean: pd.DataFrame,
    *,
    label_suffix: str | None = None,
) -> str:
    if gmp.C_ADJ not in lean.columns:
        raise KeyError(f"Aggregated frame missing {gmp.C_ADJ!r}.")
    dkey = str(dataset_key).strip().lower()
    dfc = lean[lean[gmp.C_DATA].astype(str).str.strip().str.lower() == dkey].copy()
    adj_norm = dfc[gmp.C_ADJ].astype(str).str.strip().str.lower()
    dfc = dfc.loc[adj_norm.isin(TABLE_ADJ_ORDER)].copy()
    if dfc.empty:
        raise ValueError(f"No rows for dataset {dataset_key!r} with adjacency in {TABLE_ADJ_ORDER!r}.")

    models = _gnn_models_for_table(dfc)
    if not models:
        raise ValueError(f"No non-MLP models for dataset {dataset_key!r}.")

    disp_ds = gmp._display_dataset(dkey)
    label = label_suffix or f"{dkey}-adjacency-ppi-vs-coexpression"

    col_heads = " & ".join(h for h, _, _ in TABLE_METRIC_SPECS)
    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\caption{{Adjacency ablation on \textit{{{disp_ds}}}. For each (model, adjacency method), we select the configuration with the highest validation $F_{{macro}}$ (\texttt{{val\_f1\_macro}}) and report test metrics (mean $\pm$ std) from \texttt{{summary.best\_test/*}}. PPI vs Co-expression. For each metric within a model pair, the higher mean is bolded; the other cell is shaded blue if it is within one std of the bolded cell (mean $\ge$ best mean $-$ best std).}}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{2.4cm} l l Y Y Y Y}",
        r"\toprule",
        rf"Dataset & Model & Adjacency & {col_heads} \\",
        r"\midrule",
        rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{disp_ds}}}}}\\[-0.6ex]",
    ]

    for mi, model in enumerate(models):
        if mi > 0:
            lines.append(r"\midrule")
        rows_by_adj: dict[str, pd.Series | None] = {}
        for adj_raw in TABLE_ADJ_ORDER:
            sub = dfc[
                (dfc[gmp.C_MODEL] == model)
                & (dfc[gmp.C_ADJ].astype(str).str.strip().str.lower() == adj_raw)
            ]
            rows_by_adj[adj_raw] = gmp._pick_best_row(sub)

        for ai, adj_raw in enumerate(TABLE_ADJ_ORDER):
            m_cell = gmp._display_model(model) if ai == 0 else ""
            adj_cell = gmp._display_adjacency_method(adj_raw)
            cells: list[str] = []
            for _, mcol, scol in TABLE_METRIC_SPECS:
                mus_pair: list[float] = []
                sigs_pair: list[float] = []
                for akey in TABLE_ADJ_ORDER:
                    rr = rows_by_adj[akey]
                    if rr is None:
                        mus_pair.append(float("nan"))
                        sigs_pair.append(float("nan"))
                    else:
                        mu_v = rr[mcol] if mcol in rr.index else np.nan
                        s_v = rr[scol] if scol in rr.index else np.nan
                        mus_pair.append(float(mu_v) if pd.notna(mu_v) else float("nan"))
                        sigs_pair.append(float(s_v) if pd.notna(s_v) else float("nan"))
                bld, wth = _pair_style(mus_pair, sigs_pair)
                cells.append(
                    _latex_cell(
                        mus_pair[ai],
                        sigs_pair[ai],
                        bold=bld[ai],
                        within=wth[ai],
                    )
                )

            lines.append(" & " + f"{m_cell} & {adj_cell} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            rf"\label{{tab:{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _datasets_to_emit(arg: str | None) -> list[str]:
    if not arg:
        return list(gmp.DATASETS)
    out = []
    for part in arg.split(","):
        p = part.strip().lower()
        if p:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "LaTeX tables: best config + resources (main CSV + best_model_reruns), "
            "readout ablation, adjacency ablation."
        )
    )
    ap.add_argument(
        "--per-run-csv",
        default=gmp.DEFAULT_INPUT_CSV,
        help="Per-run W&B export (default: plotting/final_results_hyperparams_neurips.csv)",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(_DIR, "tables"),
        help="Directory for .tex fragments",
    )
    ap.add_argument(
        "--datasets",
        default=None,
        help=f"Comma-separated subset (default: all {gmp.DATASETS})",
    )
    ap.add_argument("--quiet", action="store_true", help="Less console output during aggregation")
    ap.add_argument(
        "--only",
        choices=("all", "readout", "adjacency", "best_config"),
        default="all",
        help="Which table set to write (default: all). best_config writes one multi-dataset .tex file.",
    )
    ap.add_argument(
        "--expected-seeds",
        type=int,
        default=EXPECTED_SEEDS,
        metavar="K",
        help=f"Winner buckets must have K runs and K distinct seeds (default: {EXPECTED_SEEDS}).",
    )
    ap.add_argument(
        "--best-rerun-csv",
        default=None,
        help="Optional: per-seed resource metrics from W&B export (columns model_name, data_name, …). "
        "If set, skips live W&B fetch for reruns.",
    )
    ap.add_argument(
        "--skip-best-rerun-fetch",
        action="store_true",
        help="Do not call W&B for best_model_reruns; resource columns render as '---' unless --best-rerun-csv is set.",
    )
    ap.add_argument("--wandb-entity", default=None, help=f"Default: {DEFAULT_WANDB_ENTITY}")
    ap.add_argument(
        "--wandb-best-rerun-project",
        default=DEFAULT_BEST_RERUN_WANDB_PROJECT,
        metavar="NAME",
        help=f"Project with tagged replays (default: {DEFAULT_BEST_RERUN_WANDB_PROJECT!r}).",
    )
    ap.add_argument(
        "--best-rerun-tag",
        default=BEST_RERUN_TAG_DEFAULT,
        metavar="NAME",
        help=f"Only runs whose tags include this string (default: {BEST_RERUN_TAG_DEFAULT!r}).",
    )
    args = ap.parse_args()

    verbose = not args.quiet
    df_raw = pd.read_csv(args.per_run_csv, low_memory=False)
    if verbose:
        print(f"Read {len(df_raw)} rows from {args.per_run_csv}")
    lean = aggregate_per_run_for_tables(df_raw, verbose=verbose)
    if verbose:
        present = [c for c in OPTIONAL_EXTRA_TEST_METRICS if f"{c}_mean" in lean.columns]
        if present:
            print(f"Optional test metrics in aggregation: {present}")
        else:
            print(
                "No optional test-metric columns in export; F_weighted/Accuracy/AUROC cells will be '---'. "
                "Re-fetch W&B with updated load_results.py to populate them."
            )

    os.makedirs(args.out_dir, exist_ok=True)
    emit_readout = args.only in ("all", "readout")
    emit_adj = args.only in ("all", "adjacency")
    emit_best = args.only in ("all", "best_config")

    rerun_agg = aggregate_best_rerun_resources(pd.DataFrame())
    if emit_best:
        if args.best_rerun_csv:
            if verbose:
                print(f"Loading best reruns from {args.best_rerun_csv!r}")
            long_rr = load_best_rerun_resource_csv(args.best_rerun_csv)
            rerun_agg = aggregate_best_rerun_resources(long_rr)
        elif not args.skip_best_rerun_fetch:
            entity = args.wandb_entity or DEFAULT_WANDB_ENTITY
            long_rr = fetch_best_rerun_resource_long(
                wandb_entity=entity,
                wandb_project=str(args.wandb_best_rerun_project).strip(),
                rerun_tag=args.best_rerun_tag,
                verbose=verbose,
            )
            rerun_agg = aggregate_best_rerun_resources(long_rr)
        elif verbose:
            print("Skipping W&B fetch (--skip-best-rerun-fetch); resource cells will be '---' without --best-rerun-csv.")

    if emit_best:
        try:
            tex = best_config_resource_table_tex(
                _datasets_to_emit(args.datasets),
                lean,
                df_raw,
                rerun_agg,
                expected_seeds=args.expected_seeds,
            )
        except (ValueError, KeyError) as e:
            print(f"Skip best_config table: {e}")
        else:
            path = os.path.join(args.out_dir, "best_config_per_model_dataset.tex")
            with open(path, "w", encoding="utf-8") as f:
                f.write(tex)
            print(f"Wrote {path}")

    for ds in _datasets_to_emit(args.datasets):
        if emit_readout:
            try:
                tex = readout_ablation_table_tex(ds, lean)
            except (ValueError, KeyError) as e:
                print(f"Skip readout {ds}: {e}")
            else:
                path = os.path.join(args.out_dir, f"readout_ablation_{ds}.tex")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(tex)
                print(f"Wrote {path}")

        if emit_adj:
            try:
                tex = adjacency_ablation_table_tex(ds, lean)
            except (ValueError, KeyError) as e:
                print(f"Skip adjacency {ds}: {e}")
            else:
                path = os.path.join(args.out_dir, f"adjacency_ablation_{ds}.tex")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(tex)
                print(f"Wrote {path}")


if __name__ == "__main__":
    main()

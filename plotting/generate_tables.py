# LaTeX tables from the same per-run export as ``generate_main_plots.py`` (W&B CSV).
# Raw: plotting/final_results_hyperparams_neurips.csv
# Reuses fingerprint grouping + 3-seed aggregation; optional columns
# ``best_test_f1_weighted``, ``best_test_accuracy``, ``best_test_auroc`` when present
# (exported by ``load_results.py`` after re-fetch).

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import generate_main_plots as gmp

# Row order in the paper tables
TABLE_READOUT_ORDER = ("NoReadOut", "OmicsReadOut")
# Raw ``adjacency_method`` values (see ``ADJ_METHOD_DISPLAY_NAMES`` in generate_main_plots)
TABLE_ADJ_ORDER = ("string", "wgcna")

OPTIONAL_EXTRA_TEST_METRICS = ("best_test_f1_weighted", "best_test_accuracy", "best_test_auroc")

TABLE_METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("$F_{macro}$", "best_test_f1_macro_mean", "best_test_f1_macro_std"),
    ("$F_{weighted}$", "best_test_f1_weighted_mean", "best_test_f1_weighted_std"),
    ("Accuracy", "best_test_accuracy_mean", "best_test_accuracy_std"),
    ("AUROC", "best_test_auroc_mean", "best_test_auroc_std"),
)

MEAN_TIE_TOL = 1e-9


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
    lean[gmp.C_MODEL] = wide[gmp.C_MODEL] if gmp.C_MODEL in wide.columns else np.nan
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
        description="LaTeX ablation tables: readout (NoReadOut vs OmicsReadOut) and adjacency (PPI vs Co-expression)."
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
        choices=("all", "readout", "adjacency"),
        default="all",
        help="Which table set to write (default: all).",
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

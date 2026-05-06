"""Build a bash script to re-train **all seeds** of the best hyperparameter bucket per (model, dataset).

Buckets match ``export_aggregated_final_results`` (``FINGERPRINT_KEY_CANDIDATES`` + seed dedupe). Only
buckets with exactly ``--expected-seeds`` finished runs and that many distinct seeds are eligible.
The winner per ``(model_name, data_name)`` maximizes the **mean** of per-run ``best_val_f1_macro``
(default) or ``best_test_f1_macro`` (``--rank-by test``) across those seeds. The script lists one
Hydra replay per **seed** (same W&B ``args`` logic as ``generate_rerun_sh.py``). New runs log to
Hydra ``logger.wandb.project=best_model_reruns`` by default (override with ``--logger-wandb-project``);
W&B API still reads original runs from ``--wandb-project`` / ``load_results.wandb_project``.

Run from repo root::

    python plotting/generate_best_model_rerun_sh.py --report
    python plotting/generate_best_model_rerun_sh.py --out plotting/rerun_best_per_model_dataset.sh --n-gpus 8
    python plotting/generate_best_model_rerun_sh.py ... --rank-by test
    python plotting/generate_best_model_rerun_sh.py ... --expected-seeds 3

Default W&B tag is ``best_rerun`` (not ``rerun``) so these jobs are easy to filter from failed-run reruns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

import load_results as lr
from generate_main_plots import (
    C_DATA,
    C_MODEL,
    SEED_COL,
    _compose_bucket_key_frame,
    prepare_per_run_df_for_fingerprint_grouping,
)
from generate_rerun_sh import write_rerun_shell_for_missing_test_f1
from narrow_schema import EXPECTED_SEEDS

DEFAULT_BEST_RERUN_TAG = "best_rerun"
DEFAULT_LOGGER_WANDB_PROJECT = "best_model_reruns"


def best_run_ids_per_model_dataset(
    df: pd.DataFrame,
    *,
    rank_by: str = "val",
    expected_seeds: int = EXPECTED_SEEDS,
) -> tuple[list[str], pd.DataFrame]:
    """Return all ``run_id``s for winning buckets (one bucket per model × dataset; every seed in bucket).

    ``rank_by``: ``val`` → mean ``best_val_f1_macro``; ``test`` → mean ``best_test_f1_macro``.
    """
    rb = rank_by.lower().strip()
    if rb == "test":
        metric_col = lr.MISSING_TEST_F1_COL
    elif rb == "val":
        metric_col = "best_val_f1_macro"
    else:
        raise ValueError("rank_by must be 'test' or 'val'")

    if metric_col not in df.columns:
        raise KeyError(f"CSV missing metric column {metric_col!r}")

    df_prep, group_cols = prepare_per_run_df_for_fingerprint_grouping(df, verbose=False)

    if C_MODEL not in df_prep.columns or C_DATA not in df_prep.columns:
        raise KeyError(f"Prepared frame must include {C_MODEL!r} and {C_DATA!r}.")
    if "run_id" not in df_prep.columns:
        raise KeyError("CSV missing 'run_id'")

    g = df_prep.groupby(group_cols, dropna=False)
    bucket = pd.DataFrame(
        {
            "n_runs": g.size(),
            "n_seeds": g[SEED_COL].nunique(),
            "mean_metric": g[metric_col].mean(),
        }
    ).reset_index()

    ok = (
        (bucket["n_runs"] == expected_seeds)
        & (bucket["n_seeds"] == expected_seeds)
        & bucket["mean_metric"].notna()
    )
    bucket = bucket.loc[ok].copy()
    if bucket.empty:
        return [], pd.DataFrame(
            columns=[
                "model_name",
                "data_name",
                "mean_metric",
                "n_runs",
                "n_seeds",
                "seeds",
                "run_ids",
            ]
        )

    if C_MODEL not in bucket.columns or C_DATA not in bucket.columns:
        raise KeyError(
            f"Fingerprint group_cols must include {C_MODEL!r} and {C_DATA!r}; got {group_cols!r}."
        )

    bucket["_bk"] = _compose_bucket_key_frame(bucket, group_cols)
    winner_rows: list[pd.Series] = []
    for _, sub in bucket.groupby([C_MODEL, C_DATA], dropna=False):
        w = sub.sort_values(["mean_metric", "_bk"], ascending=[False, True], kind="mergesort").iloc[0]
        winner_rows.append(w)
    winners_df = pd.DataFrame(winner_rows).reset_index(drop=True)

    winners_df = winners_df.drop_duplicates(subset=group_cols, keep="first")
    selected = df_prep.merge(winners_df[group_cols], on=group_cols, how="inner")
    sort_cols = [C_MODEL, C_DATA, SEED_COL, "run_id"]
    selected = selected.sort_values([c for c in sort_cols if c in selected.columns], kind="mergesort")
    run_ids = selected["run_id"].astype(str).tolist()

    with_mean = selected.merge(
        winners_df[group_cols + ["mean_metric"]],
        on=group_cols,
        how="left",
    )

    def _seed_summary(s: pd.Series) -> str:
        u = pd.to_numeric(s, errors="coerce").dropna().unique().tolist()
        out: list[str] = []
        for x in sorted(u, key=lambda v: (float(v), str(v))):
            xf = float(x)
            out.append(str(int(xf)) if abs(xf - round(xf)) < 1e-9 else str(xf))
        return ",".join(out)

    tab = (
        with_mean.groupby([C_MODEL, C_DATA], as_index=False)
        .agg(
            mean_metric=("mean_metric", "first"),
            n_runs=("run_id", "count"),
            n_seeds=(SEED_COL, "nunique"),
            seeds=(SEED_COL, _seed_summary),
            run_ids=("run_id", lambda r: ",".join(r.astype(str))),
        )
        .rename(columns={C_MODEL: "model_name", C_DATA: "data_name"})
    )
    return run_ids, tab


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        default=None,
        help=f"Exported runs CSV (default: {lr.csv_filename})",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Print selection summary (counts + optional preview) and exit.",
    )
    p.add_argument(
        "--report-preview",
        type=int,
        default=0,
        metavar="N",
        help="With --report: print first N (model, dataset) winner rows.",
    )
    p.add_argument(
        "--out",
        metavar="PATH.sh",
        default=None,
        help="Write bash script here (required unless --report).",
    )
    p.add_argument(
        "--rank-by",
        choices=("test", "val"),
        default="val",
        help="Per-seed metric averaged within a bucket (default: val).",
    )
    p.add_argument(
        "--expected-seeds",
        type=int,
        default=EXPECTED_SEEDS,
        metavar="K",
        help=f"Only buckets with K runs and K distinct seeds (default: {EXPECTED_SEEDS}).",
    )
    p.add_argument(
        "--train-cmd",
        default="python -m ogbench",
        help="Launcher split for cluster format (default: python -m ogbench).",
    )
    p.add_argument(
        "--format",
        choices=("cluster", "legacy"),
        default="cluster",
        help="Same as generate_rerun_sh.py (default: cluster).",
    )
    p.add_argument(
        "--no-background",
        action="store_true",
        help="Cluster format only: omit trailing &.",
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Cluster format only: do not append wait after background jobs.",
    )
    p.add_argument(
        "--rerun-tag",
        default=DEFAULT_BEST_RERUN_TAG,
        metavar="NAME",
        help=f"Appended to logger.wandb.tags (default: {DEFAULT_BEST_RERUN_TAG!r}).",
    )
    p.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        metavar="N",
        help="cluster: trainer.devices=[i %% N]. legacy: CUDA_VISIBLE_DEVICES=i%%N. 0 = no pin.",
    )
    p.add_argument("--wandb-entity", default=None, help=f"Default: {lr.wandb_username}")
    p.add_argument(
        "--wandb-project",
        default=None,
        help=f"W&B API project to load historical runs (default: {lr.wandb_project}).",
    )
    p.add_argument(
        "--logger-wandb-project",
        default=DEFAULT_LOGGER_WANDB_PROJECT,
        metavar="NAME",
        help=(
            "Hydra override logger.wandb.project for emitted training commands "
            f"(default: {DEFAULT_LOGGER_WANDB_PROJECT!r}). Use \"\" to keep the original project from each run."
        ),
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=5,
        metavar="N",
        help="Log each N-th W&B fetch plus first and last (default: 5).",
    )
    args = p.parse_args()

    csv_path = args.csv or lr.csv_filename
    entity = args.wandb_entity or lr.wandb_username
    project = args.wandb_project or lr.wandb_project

    df = pd.read_csv(csv_path, low_memory=False)
    metric_name = "best_test_f1_macro" if args.rank_by == "test" else "best_val_f1_macro"

    try:
        df_prep, _ = prepare_per_run_df_for_fingerprint_grouping(df, verbose=False)
        n_pairs_any = (
            df_prep.loc[df_prep[C_MODEL].notna() & df_prep[C_DATA].notna()][[C_MODEL, C_DATA]]
            .drop_duplicates()
            .shape[0]
        )
    except (KeyError, ValueError) as e:
        print(f"CSV preparation error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        run_ids, winners = best_run_ids_per_model_dataset(
            df,
            rank_by=args.rank_by,
            expected_seeds=args.expected_seeds,
        )
    except (KeyError, ValueError) as e:
        print(f"Selection error: {e}", file=sys.stderr)
        sys.exit(1)

    n_pairs_with_winner = len(winners)

    if args.report:
        print(f"CSV: {csv_path}")
        hp = (args.logger_wandb_project or "").strip()
        hydra_proj_note = hp if hp else "(unchanged from each run)"
        print(
            f"rank_by={args.rank_by!r}  metric={metric_name!r}  "
            f"expected_seeds={args.expected_seeds}  (mean within bucket → max over buckets per pair)"
        )
        print(f"Hydra logger.wandb.project in emitted script: {hydra_proj_note!r}")
        print(f"Unique (model_name, data_name) after prep: {n_pairs_any:,}")
        print(f"(model_name, data_name) with a winning {args.expected_seeds}-seed bucket: {n_pairs_with_winner:,}")
        print(f"Total rerun commands (one per seed): {len(run_ids):,}")
        if n_pairs_with_winner < n_pairs_any:
            print(
                f"  ({n_pairs_any - n_pairs_with_winner:,} pair(s) have no eligible bucket: "
                f"need {args.expected_seeds} finished runs and {args.expected_seeds} distinct seeds.)"
            )
        if args.report_preview > 0 and not winners.empty:
            print(f"\nFirst {min(args.report_preview, len(winners))} winner row(s):")
            print(
                winners.head(args.report_preview).to_string(index=False),
                flush=True,
            )
        return

    if not args.out:
        p.error("--out PATH.sh is required (unless using --report)")

    if not run_ids:
        print(
            "No runs selected (no bucket with expected seed count and finite mean metric).",
            file=sys.stderr,
        )
        sys.exit(2)

    purpose = (
        f"# Re-run training: all seeds of the bucket with highest mean {metric_name} per "
        f"({C_MODEL}, {C_DATA}) (same fingerprint rules as aggregated export)."
    )
    print(f"CSV: {csv_path}")
    hydra_proj = (args.logger_wandb_project or "").strip()
    logger_override = hydra_proj if hydra_proj else None
    print(
        f"{len(run_ids):,} run(s) to fetch from W&B ({entity}/{project}); "
        f"Hydra logger.wandb.project={logger_override or '(unchanged from each run)'}…",
        flush=True,
    )

    n_ok, n_skip = write_rerun_shell_for_missing_test_f1(
        run_ids,
        args.out,
        wandb_entity=entity,
        wandb_project=project,
        train_cmd=args.train_cmd,
        n_gpus=args.n_gpus,
        api_timeout=120,
        progress_every=max(1, args.progress_every),
        output_format=args.format,
        background=not args.no_background,
        append_wait=not args.no_wait,
        rerun_tag=args.rerun_tag,
        purpose_header_comment=purpose,
        logger_wandb_project=logger_override,
    )
    print(f"Wrote {args.out!r}: {n_ok} command(s), {n_skip} skipped (see comments at end of script).")


if __name__ == "__main__":
    main()

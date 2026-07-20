"""Batch top-K val ensembles → CSV shaped like ``aggregated_final_results_neurips.csv``.

For every (model, dataset) in the canonical grid, ranks hyperparam buckets by **mean**
``best_val_f1_macro`` over seeds (≥ ``--min-seeds-per-bucket``). By default **no graph lock**:
top buckets may use different graph constructions; pass ``--graph-lock`` to restrict to the
best bucket's graph slice (shared test loader). Writes one row per (model, dataset, ``ensemble_K``).

Default models: MLP + all GNNs from the paper grid. Default ``--ks 3,5,10``,
``--min-seeds-per-bucket 3``, ``--max-buckets 10``.

Example::

    python plotting/export_ensemble_topk_aggregated.py \\
        --out-csv plotting/aggregated_ensemble_topk_neurips.csv \\
        --wandb-offline

Resume after interruption (skip pairs already present)::

    python plotting/export_ensemble_topk_aggregated.py --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import rootutils

REPO_ROOT = Path(__file__).resolve().parents[1]
_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

from ensemble_val_topk_core import (  # noqa: E402
    EnsembleRunConfig,
    REPO_ROOT as CORE_REPO_ROOT,
    run_ensemble,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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

DATASETS = ["motrpac", "addneuromed", "parkinsons", "brca"]

OUTPUT_COLUMNS = [
    "data_name",
    "model_name",
    "adjacency_method",
    "node_sample_ratio",
    "sampling_method",
    "readout_name",
    "_bucket_key",
    "ensemble_K",
    "n_runs_seeds",
    "best_val_f1_macro_mean",
    "best_val_f1_macro_std",
    "best_test_f1_macro_mean",
    "best_test_f1_macro_std",
    "best_train_f1_macro_mean",
    "best_train_f1_macro_std",
    "best_test_f1_weighted_mean",
    "best_test_f1_weighted_std",
    "best_test_accuracy_mean",
    "best_test_accuracy_std",
    "best_test_auroc_mean",
    "best_test_auroc_std",
]


def _pair_key(row: pd.Series) -> tuple:
    return (str(row["data_name"]), str(row["model_name"]), int(row["ensemble_K"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-csv",
        default=str(REPO_ROOT / "plotting/aggregated_ensemble_topk_neurips.csv"),
        help="Output path (same column layout as aggregated_final + ensemble_K).",
    )
    p.add_argument(
        "--csv",
        default=str(REPO_ROOT / "plotting/final_results_hyperparams_neurips.csv"),
        help="Per-run W&B export CSV.",
    )
    p.add_argument("--entity", default="bioshape-lab")
    p.add_argument("--project", default="bgbench_dataset_grid_search_final")
    p.add_argument(
        "--models",
        default=",".join(CANONICAL_MODEL_ORDER),
        help="Comma-separated canonical model names (subset of default grid).",
    )
    p.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated data_name values.",
    )
    p.add_argument("--ks", default="3,5,10", help="Comma-separated ensemble sizes.")
    p.add_argument(
        "--min-seeds-per-bucket",
        type=int,
        default=3,
        help="Min seeds per hyperparam bucket + min seeds in intersection for export rows.",
    )
    p.add_argument(
        "--max-buckets",
        type=int,
        default=10,
        help="Top this many val-mean buckets (≤ available); must be ≥ max(K) to evaluate largest K.",
    )
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--extra-override", action="append", default=[], help="Hydra override (repeatable).")
    p.add_argument(
        "--graph-lock",
        action="store_true",
        help="Restrict to the graph slice of the best mean-val bucket and use one shared test loader.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip (data_name, model_name, ensemble_K) triples already present in out-csv.",
    )
    args = p.parse_args()

    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    if not ks or any(k < 1 for k in ks):
        raise SystemExit("--ks must be positive integers.")
    if min(ks) > args.max_buckets:
        raise SystemExit("--max-buckets must be ≥ min(ks) (smallest ensemble size).")

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple] = set()
    if args.resume and out_path.is_file():
        prev = pd.read_csv(out_path, low_memory=False)
        if not prev.empty and all(c in prev.columns for c in ("data_name", "model_name", "ensemble_K")):
            for _, r in prev.iterrows():
                done.add(_pair_key(r))
        print(f"Resume: {len(done)} existing row key(s) in {out_path}", flush=True)

    work_dir = Path.cwd().resolve()
    config_dir = CORE_REPO_ROOT / "configs"
    if not config_dir.is_dir():
        raise SystemExit(f"Config dir not found: {config_dir}")

    import wandb

    api = wandb.Api()
    all_rows: list[dict] = []

    for model in models:
        for dataset in datasets:
            skip_all = all((dataset, model, int(k)) in done for k in ks)
            if skip_all:
                print(f"SKIP (resume) {model}/{dataset}", flush=True)
                continue
            cfg = EnsembleRunConfig(
                csv_path=Path(args.csv),
                entity=args.entity,
                project=args.project,
                model=model,
                dataset=dataset,
                ks=ks,
                min_seeds_per_bucket=int(args.min_seeds_per_bucket),
                max_buckets=int(args.max_buckets),
                wandb_offline=bool(args.wandb_offline),
                extra_override=tuple(args.extra_override or ()),
                work_dir=work_dir,
                config_dir=config_dir,
                graph_lock=bool(args.graph_lock),
                verbose=True,
            )
            rows, msg = run_ensemble(cfg, api=api)
            print(f"{model}/{dataset}: {msg}", flush=True)
            for row in rows:
                key = (str(row["data_name"]), str(row["model_name"]), int(row["ensemble_K"]))
                if args.resume and key in done:
                    continue
                all_rows.append(row)
                done.add(key)

            # Flush incrementally
            if all_rows:
                df_new = pd.DataFrame(all_rows)
                for c in OUTPUT_COLUMNS:
                    if c not in df_new.columns:
                        df_new[c] = None
                df_new = df_new[OUTPUT_COLUMNS]
                if out_path.is_file() and out_path.stat().st_size > 0:
                    df_old = pd.read_csv(out_path, low_memory=False)
                    df_out = pd.concat([df_old, df_new], ignore_index=True)
                    df_out = df_out.drop_duplicates(
                        subset=["data_name", "model_name", "ensemble_K"],
                        keep="last",
                    )
                else:
                    df_out = df_new
                df_out.to_csv(out_path, index=False)
                print(f"  wrote {len(df_out)} total rows → {out_path}", flush=True)
                all_rows.clear()

    print(f"Done. Output: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

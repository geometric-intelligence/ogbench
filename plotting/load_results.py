"""W&B → CSV: flattened config + run meta + F1. Baselines → separate CSV + gnn_features aggregation.

Rerun CLI scripts: ``plotting/generate_rerun_sh.py`` (rows missing test F1; CSV for ``run_id``,
Hydra args from W&B); ``plotting/generate_best_model_rerun_sh.py`` (mean val F1 over seeds per
fingerprint bucket; best bucket per model × dataset; rerun all seeds in that bucket; Hydra from W&B).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import wandb
from wandb.errors import CommError

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

wandb_username = "bioshape-lab"
wandb_project = "bgbench_dataset_grid_search_final"
csv_filename = "plotting/final_results_hyperparams_neurips.csv"

wandb_project_baselines = "ogbench_baselines_with_train_metrics"
csv_filename_baselines_runs = "plotting/baseline_runs_ogbench_final.csv"
csv_filename_baselines_agg_gnn_features = "plotting/baseline_aggregated_gnn_features_neurips.csv"

RUN_META_KEYS = ("run_id", "run_name", "state")
METRIC_KEYS = (
    "best_val_f1_macro",
    "best_test_f1_macro",
    "best_train_f1_macro",
    "best_test_f1_weighted",
    "best_test_accuracy",
    "best_test_auroc",
)
BASELINE_METRIC_KEYS = METRIC_KEYS
EXPECTED_BASELINE_SEEDS = 1
PER_RUN_F1_COLS = ("best_val_f1_macro", "best_test_f1_macro", "best_train_f1_macro")
MISSING_TEST_F1_COL = "best_test_f1_macro"


def runs_missing_best_test_f1_macro_mask(df: pd.DataFrame) -> pd.Series:
    """True where ``best_test_f1_macro`` is absent (no W&B ``best_test/f1_macro`` in export)."""
    if MISSING_TEST_F1_COL not in df.columns:
        raise KeyError(f"CSV missing {MISSING_TEST_F1_COL!r} (needed to spot runs without test metrics).")
    col = df[MISSING_TEST_F1_COL]
    num = pd.to_numeric(col, errors="coerce")
    str_empty = col.astype(str).str.strip().isin(("", "nan", "none", "None"))
    return num.isna() | str_empty


def flatten_config(config, parent_key="", sep="."):
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _cellify(val):
    if val is None:
        return None
    if hasattr(val, "item") and callable(getattr(val, "item", None)):
        try:
            if not isinstance(val, (bytes, str)):
                val = val.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(val, (list, dict, tuple)):
        return json.dumps(val, sort_keys=isinstance(val, dict), default=str)
    return val


def _extract_run_data(run):
    try:
        cfg = run.config.copy() if run.config else {}
        flat = flatten_config(cfg)
        row = {str(k): _cellify(v) for k, v in flat.items()}
        row["run_id"] = str(run.id)
        row["run_name"] = getattr(run, "name", None) or ""
        row["state"] = getattr(run, "state", None) or ""
        summary = run.summary or {}
        for out_name, wb_key in (
            ("best_val_f1_macro", "best_val/f1_macro"),
            ("best_test_f1_macro", "best_test/f1_macro"),
            ("best_train_f1_macro", "best_train/f1_macro"),
            ("best_test_f1_weighted", "best_test/f1_weighted"),
            ("best_test_accuracy", "best_test/accuracy"),
            ("best_test_auroc", "best_test/auroc"),
        ):
            v = summary.get(wb_key)
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[out_name] = v
            else:
                row[out_name] = None
        return ("success", row)
    except Exception as e:
        return ("failed", (run.id, str(e)))


def _extract_baseline_run_data(run):
    try:
        cfg = run.config.copy() if run.config else {}
        flat = flatten_config(cfg)
        row = {str(k): _cellify(v) for k, v in flat.items()}
        row["run_id"] = str(run.id)
        row["run_name"] = getattr(run, "name", None) or ""
        row["state"] = getattr(run, "state", None) or ""
        summary = run.summary or {}
        metric_key_candidates = {
            "best_val_f1_macro": ("val/f1_macro",),
            "best_test_f1_macro": ("test/f1_macro",),
            "best_train_f1_macro": ("train/f1_macro", "train/best_cv_score"),
        }
        for out_name, wb_keys in metric_key_candidates.items():
            v = None
            for wb_key in wb_keys:
                vv = summary.get(wb_key)
                if vv is not None:
                    v = vv
                    break
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[out_name] = v
            else:
                row[out_name] = None
        return ("success", row)
    except Exception as e:
        return ("failed", (run.id, str(e)))


def _is_retryable_error(e):
    if isinstance(e, CommError):
        return True
    error_str = str(e).lower()
    if hasattr(e, "response") and hasattr(e.response, "status_code"):
        return e.response.status_code in [429, 500, 502, 503, 504]
    patterns = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "server error",
        "try again",
        "connection",
        "reset",
    )
    return any(p in error_str for p in patterns)


def _ordered_columns(all_cols: set[str], metric_keys: tuple[str, ...] = METRIC_KEYS) -> list[str]:
    front = [c for c in RUN_META_KEYS if c in all_cols]
    metrics = [c for c in metric_keys if c in all_cols]
    rest = sorted(c for c in all_cols if c not in front + metrics)
    return front + rest + metrics


def _save_batch_to_csv(
    records, csv_filename, existing_columns=None, metric_keys: tuple[str, ...] = METRIC_KEYS
):
    if not records:
        return existing_columns

    df_new = pd.DataFrame(records)
    all_cols = set(df_new.columns)

    if os.path.exists(csv_filename):
        if existing_columns is None:
            df_head = pd.read_csv(csv_filename, nrows=0)
            existing_columns = df_head.columns.tolist()
        merged = list(
            dict.fromkeys(
                existing_columns + [c for c in _ordered_columns(all_cols, metric_keys) if c not in existing_columns]
            )
        )
        new_only = [c for c in merged if c not in existing_columns]
        if new_only:
            df_existing = pd.read_csv(csv_filename, low_memory=False)
            for col in new_only:
                df_existing[col] = None
            df_existing = df_existing[merged]
            df_existing.to_csv(csv_filename, index=False)
            existing_columns = merged
        for col in existing_columns:
            if col not in df_new.columns:
                df_new[col] = None
        df_new = df_new[existing_columns]
        df_new.to_csv(csv_filename, mode="a", header=False, index=False)
    else:
        cols = _ordered_columns(all_cols, metric_keys)
        for c in cols:
            if c not in df_new.columns:
                df_new[c] = None
        df_new = df_new[cols]
        df_new.to_csv(csv_filename, index=False)
        existing_columns = df_new.columns.tolist()

    return existing_columns


def load_results_dataframe_optimized(
    wandb_username,
    wandb_project,
    csv_filename="wandb_results.csv",
    force_load=False,
    save_csv=True,
    filters=None,
    batch_size=1000,
    per_page=500,
    max_retries=6,
    *,
    extract_run_data_fn=_extract_run_data,
    metric_keys_for_csv: tuple[str, ...] = METRIC_KEYS,
):
    if filters is None:
        filters = {}

    runs_path = f"{wandb_username}/{wandb_project}"
    print(f"\n{'=' * 80}\nFETCHING RUNS: {runs_path}\n{'=' * 80}")

    if force_load and os.path.exists(csv_filename):
        backup = csv_filename.replace(".csv", "_backup.csv")
        os.rename(csv_filename, backup)
        print(f"Backed up to {backup}")

    existing_run_ids = set()
    existing_columns = None
    if os.path.exists(csv_filename) and not force_load:
        try:
            df_existing = pd.read_csv(csv_filename, low_memory=False)
            existing_run_ids = set(df_existing["run_id"].astype(str))
            existing_columns = df_existing.columns.tolist()
            print(f"Resuming: {len(existing_run_ids)} run_ids in {csv_filename}")
        except Exception as e:
            print(f"Could not read existing CSV: {e}")

    def _fetch_with_retry():
        nonlocal existing_columns
        last_error = None

        for attempt in range(max_retries):
            records = []
            failed_runs = []
            new_count = skipped_count = total_iterated = saved_total = 0
            progress_interval = 100

            try:
                print(f"\nAttempt {attempt + 1}/{max_retries}: wandb.Api()…")
                api = wandb.Api(timeout=120)
                runs = api.runs(runs_path, filters=filters, per_page=per_page)
                print(f"Processing (batch_size={batch_size})…")

                for run in runs:
                    total_iterated += 1
                    run_id = str(run.id)
                    if run_id in existing_run_ids:
                        skipped_count += 1
                        if total_iterated % progress_interval == 0:
                            print(
                                f"  … {total_iterated} | new {new_count} | skip {skipped_count} | saved {saved_total}",
                                flush=True,
                            )
                        continue

                    status, result = extract_run_data_fn(run)
                    if status == "success":
                        records.append(result)
                        existing_run_ids.add(run_id)
                        new_count += 1
                    else:
                        print("Failed run: ", result)
                        failed_runs.append(result)

                    if total_iterated % progress_interval == 0:
                        print(
                            f"  … {total_iterated} | new {new_count} | skip {skipped_count} | saved {saved_total}",
                            flush=True,
                        )

                    if save_csv and len(records) >= batch_size:
                        existing_columns = _save_batch_to_csv(
                            records, csv_filename, existing_columns, metric_keys_for_csv
                        )
                        saved_total += len(records)
                        print(f"  Saved batch {len(records)} (total saved {saved_total})")
                        records = []

                if save_csv and records:
                    existing_columns = _save_batch_to_csv(
                        records, csv_filename, existing_columns, metric_keys_for_csv
                    )
                    saved_total += len(records)
                    print(f"Saved final batch {len(records)}")

                print(f"\n{'=' * 80}\nDONE: iterated {total_iterated} | new {new_count} | skip {skipped_count}")
                if failed_runs:
                    print(f"Extract failed: {len(failed_runs)}")
                print(f"{'=' * 80}\n")

                df = pd.read_csv(csv_filename, low_memory=False)
                print(f"DataFrame shape: {df.shape}")
                return df

            except Exception as e:
                last_error = e
                if save_csv and records:
                    existing_columns = _save_batch_to_csv(
                        records, csv_filename, existing_columns, metric_keys_for_csv
                    )
                    saved_total += len(records)
                    print(f"Saved progress before retry ({len(records)} runs)")
                    records = []

                if not _is_retryable_error(e) or attempt == max_retries - 1:
                    print(f"\nFatal after {attempt + 1} attempts: {e}")
                    raise

                delay = min(120.0, (2**attempt) * 10 + np.random.uniform(0, 3))
                print(f"Transient error, retry in {delay:.0f}s…\n")
                time.sleep(delay)

        raise last_error if last_error else RuntimeError("fetch failed")

    try:
        return _fetch_with_retry()
    except Exception as e:
        print(f"\nFetch incomplete: {e}")
        if os.path.exists(csv_filename):
            df = pd.read_csv(csv_filename, low_memory=False)
            print(f"Returning partial CSV shape: {df.shape}")
            return df
        return pd.DataFrame()


def load_baseline_runs_dataframe_optimized(
    wandb_username=wandb_username,
    wandb_project=wandb_project_baselines,
    csv_filename=csv_filename_baselines_runs,
    **kwargs,
):
    force_load = bool(kwargs.get("force_load", False))
    if os.path.exists(csv_filename) and not force_load:
        try:
            df_existing = pd.read_csv(csv_filename, low_memory=False)
            bt = (
                pd.to_numeric(df_existing["best_train_f1_macro"], errors="coerce")
                if "best_train_f1_macro" in df_existing.columns
                else pd.Series(dtype=float)
            )
            if int(bt.notna().sum()) == 0:
                print("Baseline CSV has no train F1; forcing reload.")
                kwargs = {**kwargs, "force_load": True}
        except Exception as e:
            print(f"Could not inspect baseline CSV: {e}")

    return load_results_dataframe_optimized(
        wandb_username,
        wandb_project,
        csv_filename=csv_filename,
        extract_run_data_fn=_extract_baseline_run_data,
        metric_keys_for_csv=BASELINE_METRIC_KEYS,
        **kwargs,
    )


def _series_first_existing(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    raise KeyError(f"None of: {names}")


def _mask_preprocessing_gnn_features(df: pd.DataFrame) -> pd.Series:
    keys = ("preprocessing", "preprocessing.value", "baseline_filter", "baseline_filter.value")
    m = pd.Series(False, index=df.index)
    found = False
    for k in keys:
        if k not in df.columns:
            continue
        found = True
        m |= df[k].astype(str).str.strip().str.lower().eq("gnn_features")
    if not found:
        raise ValueError("No preprocessing/baseline_filter column for gnn_features filter.")
    return m


def _normalize_baseline_ratio(x) -> object:
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ("", "nan", "none"):
        return np.nan
    if s == "full":
        return "full"
    try:
        return float(s)
    except ValueError:
        return s


def _canonical_baseline_model_name(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = str(raw).strip().lower()
    if not s or s in ("nan", "none"):
        return ""
    if s.endswith("_gnn_features"):
        s = s[: -len("_gnn_features")]
    return s


def export_baseline_gnn_features_aggregated(
    input_csv: str | None = None,
    output_csv: str | None = None,
    *,
    expected_seeds: int = EXPECTED_BASELINE_SEEDS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Aggregate baseline runs where preprocessing == gnn_features; require expected_seeds per slice."""
    input_csv = input_csv or csv_filename_baselines_runs
    output_csv = output_csv or csv_filename_baselines_agg_gnn_features

    df = pd.read_csv(input_csv, low_memory=False)
    if verbose:
        print(f"Read {len(df)} rows from {input_csv}")

    if "state" in df.columns:
        n0 = len(df)
        df = df.loc[df["state"].astype(str).str.lower().eq("finished")].copy()
        if verbose:
            print(f"Kept {len(df)} / {n0} finished")

    if "best_train_f1_macro" in df.columns and (
        "train/f1_macro" in df.columns or "train/best_cv_score" in df.columns
    ):
        bt = pd.to_numeric(df["best_train_f1_macro"], errors="coerce")
        train_f1 = (
            pd.to_numeric(df["train/f1_macro"], errors="coerce")
            if "train/f1_macro" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        cv = (
            pd.to_numeric(df["train/best_cv_score"], errors="coerce")
            if "train/best_cv_score" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        fill_t = bt.isna() & train_f1.notna()
        fill_c = bt.isna() & ~fill_t & cv.notna()
        if fill_t.any():
            df.loc[fill_t, "best_train_f1_macro"] = train_f1.loc[fill_t]
        if fill_c.any():
            df.loc[fill_c, "best_train_f1_macro"] = cv.loc[fill_c]
        if verbose and (fill_t.any() or fill_c.any()):
            print(f"Filled best_train_f1_macro: train {int(fill_t.sum())}, cv {int(fill_c.sum())}")

    miss = [c for c in PER_RUN_F1_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns {miss}")

    df = df.loc[_mask_preprocessing_gnn_features(df)].copy()
    if verbose:
        print(f"gnn_features rows: {len(df)}")
    if df.empty:
        raise ValueError("No gnn_features rows.")

    if "run_id" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=["run_id"], keep="last")
        if verbose and len(df) < n_before:
            print(f"Dedup run_id: {n_before - len(df)}")

    try:
        data_name = _series_first_existing(df, ("dataset", "dataset.value"))
    except KeyError:
        data_name = _series_first_existing(df, ("data_name", "data_name.value"))
    baseline_name = _series_first_existing(df, ("baseline_name", "baseline_name.value"))
    method = _series_first_existing(df, ("method", "method.value"))
    node_ratio = _series_first_existing(df, ("node_sample_ratio", "node_sample_ratio.value"))
    seed_s = _series_first_existing(df, ("seed", "seed.value"))

    work = pd.DataFrame(
        {
            "data_name": data_name.astype(str).str.strip().str.lower(),
            "model_name": baseline_name.map(_canonical_baseline_model_name),
            "sampling_method": method.astype(str).str.strip().str.lower(),
            "node_sample_ratio": node_ratio.map(_normalize_baseline_ratio),
            "seed": pd.to_numeric(seed_s, errors="coerce"),
        }
    )
    for c in PER_RUN_F1_COLS:
        work[c] = pd.to_numeric(df[c].values, errors="coerce")

    work = work.dropna(subset=["data_name", "model_name", "sampling_method", "seed"])
    work = work.loc[work["model_name"].astype(str).str.len() > 0]
    work = work.loc[~work["data_name"].isin(("", "nan"))]
    n_before = len(work)
    work = work.dropna(subset=["node_sample_ratio"])
    if verbose and len(work) < n_before:
        print(f"Dropped {n_before - len(work)} missing node_sample_ratio")

    dedupe_keys = ["data_name", "model_name", "node_sample_ratio", "sampling_method", "seed"]
    work = work.sort_values("seed", kind="mergesort").drop_duplicates(subset=dedupe_keys, keep="last")

    gcols = ["data_name", "model_name", "node_sample_ratio", "sampling_method"]
    gb = work.groupby(gcols, dropna=False)
    n_runs = gb.size()
    n_seeds = gb["seed"].nunique()

    wide = gb.agg({m: ["mean", "std"] for m in PER_RUN_F1_COLS})
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    if expected_seeds == 1:
        for m in PER_RUN_F1_COLS:
            std_col = f"{m}_std"
            if std_col in wide.columns:
                wide[std_col] = 0.0
    wide = wide.reset_index()
    wide["n_runs_seeds"] = n_runs.values

    ok = (wide["n_runs_seeds"] == expected_seeds) & (n_seeds.values == expected_seeds)
    pre = len(wide)
    wide = wide.loc[ok].copy()
    if verbose:
        print(f"Rows with {expected_seeds} runs & seeds: {len(wide)} / {pre}")

    val_mean_col = "best_val_f1_macro_mean"
    slice_cols = list(gcols)
    pre_best = len(wide)
    wide = (
        wide.sort_values(val_mean_col, ascending=False, na_position="last")
        .drop_duplicates(subset=slice_cols, keep="first")
        .sort_values(slice_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    if verbose and len(wide) < pre_best:
        print(f"Best val slice dedupe: dropped {pre_best - len(wide)}")

    out_cols = slice_cols + ["n_runs_seeds"] + [
        "best_val_f1_macro_mean",
        "best_val_f1_macro_std",
        "best_test_f1_macro_mean",
        "best_test_f1_macro_std",
        "best_train_f1_macro_mean",
        "best_train_f1_macro_std",
    ]
    out = wide[out_cols]
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_csv, index=False)
    if verbose:
        print(f"Wrote {len(out)} → {output_csv}")
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fetch W&B (GNN grid or baselines) or aggregate baselines only.")
    p.add_argument("--baselines", action="store_true", help="Fetch baseline project to baseline runs CSV")
    p.add_argument("--aggregate-baselines-only", action="store_true", help="Build aggregated gnn_features CSV only")
    p.add_argument("--force-load", action="store_true", help="Backup CSV and refetch all runs")
    args = p.parse_args()

    if args.aggregate_baselines_only:
        export_baseline_gnn_features_aggregated(verbose=True)
    elif args.baselines:
        df = load_baseline_runs_dataframe_optimized(
            force_load=args.force_load,
            save_csv=True,
            batch_size=1000,
            per_page=500,
            max_retries=6,
        )
        print(f"Loaded {len(df)} baseline runs")
        export_baseline_gnn_features_aggregated(verbose=True)
    else:
        df = load_results_dataframe_optimized(
            wandb_username,
            wandb_project,
            csv_filename=csv_filename,
            force_load=args.force_load,
            save_csv=True,
            batch_size=1000,
            per_page=500,
            max_retries=6,
        )
        print(f"Loaded {len(df)} runs")
        try:
            m = runs_missing_best_test_f1_macro_mask(df)
            print(f"Rows missing {MISSING_TEST_F1_COL} in CSV: {int(m.sum()):,} / {len(df):,}")
        except KeyError:
            pass

import os
import json
import ast
import time
import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional
from wandb.errors import CommError
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# ---------------------------------------------------------------------
# Helper to flatten nested config dicts
# ---------------------------------------------------------------------
def flatten_config(config, parent_key='', sep='.'):
    """Flatten a nested dictionary by joining keys with a separator."""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------
# Thread-local storage for wandb API instances (reuse across fetches)
# ---------------------------------------------------------------------
_thread_local = threading.local()


def _get_thread_api(timeout=180):
    """Get or create a thread-local wandb API instance."""
    if not hasattr(_thread_local, 'api'):
        _thread_local.api = wandb.Api(timeout=timeout)
    return _thread_local.api


def _is_retryable_error(e):
    """Check if an exception is retryable (rate limit, timeout, server errors)."""
    if isinstance(e, CommError):
        return True
    error_str = str(e).lower()
    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
        return e.response.status_code in [429, 500, 502, 503, 504]
    retryable_patterns = ['429', '500', '502', '503', '504', 'timeout', 'timed out', 
                          'server error', 'try again', 'connection', 'reset']
    return any(p in error_str for p in retryable_patterns)


# ---------------------------------------------------------------------
# Extract data from a run object (already loaded, no additional API call)
# ---------------------------------------------------------------------
def _extract_run_data(run):
    """Extract data from a wandb Run object.
    
    The run object is already loaded from iteration - we just extract its data.
    Accessing config/summary may trigger lazy loading but is faster than re-fetching.
    """
    try:
        cfg = run.config.copy() if run.config else {}
        cfg_flat = flatten_config(cfg)
        
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
        }
        
        # Add all summary metrics (only simple types)
        if run.summary:
            for key, value in run.summary.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    row[f"summary.{key}"] = value
        
        # Add all config parameters
        row.update(cfg_flat)
        return ("success", row)
        
    except Exception as e:
        return ("failed", (run.id, str(e)))


# ---------------------------------------------------------------------
# Load runs from W&B or from a cached CSV - STREAMING APPROACH
# ---------------------------------------------------------------------
def load_results_dataframe(
    wandb_username,
    wandb_project,
    csv_filename="wandb_results.csv",
    force_load=False,
    save_csv=True,
    filters=None,
    batch_size=500,
    per_page=50,
    fetch_recent_only=True,
    early_stop_threshold=100,
):
    """Load results from W&B with efficient streaming and incremental saves.
    
    Key features:
    - Streams runs directly from API (no separate ID collection phase)
    - Extracts data from run objects during iteration (efficient)
    - Saves to CSV every batch_size runs (resumable)
    - Skips already-fetched runs automatically
    - On error: saves progress and retries, properly resuming
    - NEW: fetch_recent_only mode fetches newest runs first and stops early
    
    Args:
        wandb_username: W&B username/entity
        wandb_project: W&B project name
        csv_filename: Output CSV filename
        force_load: If True, reload all runs even if CSV exists
        save_csv: If True, save results to CSV incrementally
        filters: Optional filters for W&B API
        batch_size: Runs to process before saving to CSV (default: 500)
        per_page: Runs per API page request (default: 50, lower = more stable)
        fetch_recent_only: If True, fetch newest runs first and stop early when
                          hitting existing runs (default: True). Much faster for
                          incremental updates.
        early_stop_threshold: When fetch_recent_only=True, stop after hitting this
                             many consecutive runs that are already in CSV (default: 100)
    """
    if filters is None:
        filters = {}

    runs_path = f"{wandb_username}/{wandb_project}"
    max_retries = 10
    retry_delay = 5.0
    
    # Main retry loop - each attempt is a fresh start with updated CSV
    for attempt in range(max_retries):
        # Load existing CSV to get already-fetched run IDs
        existing_run_ids = set()
        existing_columns = None
        
        if os.path.exists(csv_filename) and not force_load:
            try:
                df_existing = pd.read_csv(csv_filename, low_memory=False)
                existing_run_ids = set(df_existing['run_id'].astype(str))
                existing_columns = df_existing.columns.tolist()
                print(f"▶ CSV has {len(existing_run_ids)} runs already saved")
            except Exception as e:
                print(f"⚠ Could not read existing CSV: {e}")
        elif force_load and os.path.exists(csv_filename) and attempt == 0:
            backup_name = csv_filename.replace('.csv', '_backup.csv')
            os.rename(csv_filename, backup_name)
            print(f"▶ Force reload: backed up existing CSV to {backup_name}")
        
        # Get total count for progress bar
        if attempt == 0:
            print(f"▶ Connecting to W&B project: {runs_path}")
        
        try:
            api = wandb.Api(timeout=300)
            total_runs = len(api.runs(runs_path, filters=filters))
            new_to_fetch = max(0, total_runs - len(existing_run_ids))
            
            if attempt == 0:
                print(f"▶ Total runs in project: {total_runs}")
                print(f"▶ New runs to fetch: {new_to_fetch}")
            
            if new_to_fetch == 0:
                print(f"▶ All runs already saved! Loading CSV...")
                df = pd.read_csv(csv_filename, low_memory=False)
                print(f"▶ DataFrame shape: {df.shape}")
                return df
                
        except Exception as e:
            print(f"⚠ Could not get run count: {e}")
            total_runs = None
            new_to_fetch = None
        
        # Stream and process runs
        # When fetch_recent_only=True, sort by newest first for faster incremental updates
        order_str = "-created_at" if fetch_recent_only else None
        mode_str = "newest-first" if fetch_recent_only else "default order"
        print(f"\n▶ Attempt {attempt + 1}/{max_retries}: Streaming runs ({mode_str}, batch={batch_size}, page={per_page})...")
        if fetch_recent_only:
            print(f"   Early stop after {early_stop_threshold} consecutive existing runs")
        
        records = []
        failed_runs = []
        iter_count = 0
        new_count = 0
        saved_this_attempt = 0
        consecutive_existing = 0  # Track consecutive runs already in CSV
        early_stopped = False
        
        pbar = tqdm(desc="Fetching", unit="runs", total=new_to_fetch if fetch_recent_only else total_runs)
        
        try:
            api = wandb.Api(timeout=300)
            runs = api.runs(runs_path, filters=filters, per_page=per_page, order=order_str)
            
            for run in runs:
                run_id = str(run.id)
                iter_count += 1
                
                # Skip if already in CSV
                if run_id in existing_run_ids:
                    consecutive_existing += 1
                    pbar.set_postfix({
                        "new": new_count,
                        "consec_exist": consecutive_existing,
                        "saved": saved_this_attempt
                    })
                    
                    # Early stop if we've hit many consecutive existing runs
                    if fetch_recent_only and consecutive_existing >= early_stop_threshold:
                        early_stopped = True
                        print(f"\n▶ Early stop: hit {consecutive_existing} consecutive existing runs")
                        break
                    continue
                
                # Reset consecutive counter when we find a new run
                consecutive_existing = 0
                pbar.update(1)
                
                # Extract data from run object
                status, result = _extract_run_data(run)
                
                if status == "success":
                    records.append(result)
                    existing_run_ids.add(run_id)
                    new_count += 1
                else:
                    failed_runs.append(result)
                
                pbar.set_postfix({
                    "new": new_count,
                    "consec_exist": consecutive_existing,
                    "saved": saved_this_attempt
                })
                
                # Save batch to CSV
                if save_csv and len(records) >= batch_size:
                    existing_columns = _save_batch_to_csv(
                        records, csv_filename, existing_columns
                    )
                    saved_this_attempt += len(records)
                    records = []
            
            pbar.close()
            
            # Save remaining records
            if save_csv and records:
                existing_columns = _save_batch_to_csv(records, csv_filename, existing_columns)
                saved_this_attempt += len(records)
            
            # Success! 
            if early_stopped:
                print(f"▶ Complete (early stopped)!")
            else:
                print(f"\n▶ Complete!")
            print(f"   Iterated: {iter_count} runs")
            print(f"   New runs saved this session: {saved_this_attempt}")
            print(f"   Total in CSV: {len(existing_run_ids)}")
            if failed_runs:
                print(f"   Failed: {len(failed_runs)}")
            
            df = pd.read_csv(csv_filename, low_memory=False)
            print(f"▶ Final DataFrame shape: {df.shape}")
            return df
            
        except Exception as e:
            pbar.close()
            
            # Save what we have before retrying
            if save_csv and records:
                existing_columns = _save_batch_to_csv(records, csv_filename, existing_columns)
                saved_this_attempt += len(records)
                records = []
            
            if _is_retryable_error(e) and attempt < max_retries - 1:
                wait_time = retry_delay * (1.5 ** attempt) + np.random.random() * 3
                print(f"\n⚠ Error at iteration {iter_count}: {str(e)[:80]}")
                print(f"   Saved {saved_this_attempt} runs this attempt")
                print(f"   Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"\n✗ Fatal error after {attempt + 1} attempts: {e}")
                break
    
    # Return whatever we have
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename, low_memory=False)
        print(f"▶ Partial results - DataFrame shape: {df.shape}")
        return df
    return pd.DataFrame()


def _save_batch_to_csv(records, csv_filename, existing_columns=None):
    """Save a batch of records to CSV, handling column alignment."""
    if not records:
        return existing_columns
    
    df_new = pd.DataFrame(records)
    
    if os.path.exists(csv_filename):
        # Read existing to align columns
        if existing_columns is None:
            df_existing = pd.read_csv(csv_filename, nrows=0)
            existing_columns = df_existing.columns.tolist()
        
        # Find new columns
        new_cols = [c for c in df_new.columns if c not in existing_columns]
        
        if new_cols:
            # Add new columns to existing file
            df_existing = pd.read_csv(csv_filename, low_memory=False)
            for col in new_cols:
                df_existing[col] = None
            df_existing.to_csv(csv_filename, index=False)
            existing_columns = df_existing.columns.tolist()
        
        # Align new data columns and append
        for col in existing_columns:
            if col not in df_new.columns:
                df_new[col] = None
        df_new = df_new[existing_columns]
        df_new.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        # Create new file
        df_new.to_csv(csv_filename, index=False)
        existing_columns = df_new.columns.tolist()
    
    return existing_columns


# ---------------------------------------------------------------------
# Helper: serialize values into a *string* key for grouping
# ---------------------------------------------------------------------
def _serialize_for_grouping(val):
    """Convert any Python object (including lists/dicts/arrays) into a stable string representation
    so pandas can group on it safely.

    Equal configs -> equal strings.
    """
    # Preserve missingness
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "__NaN__"

    # Numpy arrays -> list
    if isinstance(val, np.ndarray):
        return "__ndarray__:" + json.dumps(val.tolist(), sort_keys=False)

    # Dicts -> sorted by key for stability
    if isinstance(val, dict):
        return "__dict__:" + json.dumps(val, sort_keys=True, default=str)

    # Lists/tuples/sets -> serialized list
    if isinstance(val, (list, tuple, set)):
        if isinstance(val, set):
            seq = sorted(list(val))
        else:
            seq = list(val)
        return "__seq__:" + json.dumps(seq, sort_keys=False, default=str)

    # Fallback: stringify scalars/other objects
    return f"__val__:{repr(val)}"


# ---------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------
def aggregate_across_seeds(
    df,
    seed_col="seed",
    metric_prefix="summary.",
    output_filename=None,
):
    """Aggregate W&B runs across seeds, computing mean/std/count for each metric.

    Grouping columns:
      - All columns EXCEPT:
        * seed columns
        * run_id, run_name, state
        * columns starting with metric_prefix (e.g. "summary.")
      - And we explicitly drop columns that vary within the same core config
        (like the sklearn pipelines).
    """

    print(f"▶ Using seed column: {seed_col}")

    # 3) Identify metric columns -----------------------------------------------------
    metric_cols = [c for c in df.columns if c.startswith(metric_prefix)]
    if not metric_cols:
        raise ValueError(f"No metric columns found with prefix '{metric_prefix}'")
    print(f"▶ Found {len(metric_cols)} metric columns with prefix '{metric_prefix}'")

    # 4) Determine initial candidate grouping columns --------------------------------
    exclude_cols = set([seed_col] + ["run_id", "run_name", "state"])
    exclude_cols.update(metric_cols)

    candidate_grouping = [c for c in df.columns if c not in exclude_cols]

    # Explicitly drop columns that are per-run rather than per-config
    bad_group_cols = {
        "paths.output_dir",
        "logger.wandb.save_dir",
        "trainer.default_root_dir",
        "callbacks.model_checkpoint.dirpath",
    }
    candidate_grouping = [
        c for c in candidate_grouping
        if c not in bad_group_cols
        and not (c.startswith("dataset.baselines.") and c.endswith(".pipeline"))
    ]

    print(f"▶ Initial grouping candidates (after dropping bad cols): {len(candidate_grouping)}")

    # 5) (Optional) drop columns that are unique per run -----------------------------
    # This keeps things like logger.wandb.id, ckpt_path, etc. out of grouping.
    # Serialize candidate columns first to handle unhashable types (lists, dicts, etc.)
    df_temp = df.copy()
    for col in candidate_grouping:
        df_temp[col] = df_temp[col].apply(_serialize_for_grouping)

    n_rows = len(df)
    grouping_cols = []
    for col in candidate_grouping:
        nunique = df_temp[col].nunique(dropna=False)
        if nunique < 0.9 * n_rows:
            grouping_cols.append(col)

    print(f"▶ Final grouping columns (after nunique filter): {len(grouping_cols)}")

    # 6) Make a copy, serialize grouping cols, and coerce metrics to numeric --------
    df_group = df.copy()

    # Serialize grouping columns so lists/dicts/arrays are safe keys
    for col in grouping_cols:
        df_group[col] = df_group[col].apply(_serialize_for_grouping)

    # Coerce metrics to numeric; non-numeric become NaN
    for col in metric_cols:
        df_group[col] = pd.to_numeric(df_group[col], errors="coerce")

    # Drop metric columns that are entirely NaN after coercion
    numeric_metric_cols = [c for c in metric_cols if df_group[c].notna().any()]
    dropped = sorted(set(metric_cols) - set(numeric_metric_cols))
    print(f"▶ Numeric metric columns kept: {len(numeric_metric_cols)}")
    if dropped:
        print(f"▶ Dropped {len(dropped)} all-NaN / non-numeric metric columns (e.g.): {dropped[:5]}")

    if not numeric_metric_cols:
        raise ValueError("After filtering, no numeric metric columns remain to aggregate.")

    # 7) Build aggregation dict and group -------------------------------------------
    agg_dict = {col: ["mean", "std", "count"] for col in numeric_metric_cols}

    grouped = df_group.groupby(grouping_cols, dropna=False, sort=False)

    # 7a) Enforce exactly 3 seeds per group ----------------------------------------
    group_sizes = grouped.size()
    groups_with_3_seeds = group_sizes[group_sizes == 3].index

    if len(groups_with_3_seeds) == 0:
        raise ValueError("No groups found with exactly 3 seeds. Cannot aggregate.")

    if len(groups_with_3_seeds) < len(group_sizes):
        n_filtered = len(group_sizes) - len(groups_with_3_seeds)
        print(f"▶ Filtering: {n_filtered} groups removed (did not have exactly 3 seeds)")
        print(f"▶ Keeping: {len(groups_with_3_seeds)} groups with exactly 3 seeds")

        # Filter df_group to only include rows from groups with exactly 3 seeds
        df_group_filtered = df_group.set_index(grouping_cols).loc[groups_with_3_seeds].reset_index()
        grouped = df_group_filtered.groupby(grouping_cols, dropna=False, sort=False)
    else:
        print(f"▶ All {len(group_sizes)} groups have exactly 3 seeds")

    aggregated = grouped[numeric_metric_cols].agg(agg_dict)

    # 8) Flatten MultiIndex columns --------------------------------------------------
    new_cols = []
    for metric, stat in aggregated.columns:
        if stat == "mean":
            new_cols.append(metric)
        else:
            new_cols.append(f"{metric}_{stat}")

    aggregated.columns = new_cols
    aggregated = aggregated.reset_index()

    print(f"▶ Aggregated shape: {aggregated.shape}")
    print(f"▶ Number of unique experiment configurations: {len(aggregated)}")

    if output_filename:
        aggregated.to_csv(output_filename, index=False)
        print(f"▶ Saved aggregated results to: {output_filename}")

    return aggregated


def diagnose_grouping_conflicts(
    df,
    metric_prefix="summary.",
    seed_cols=None,
):
    """Diagnose which non-metric columns vary across runs that share the same 'core config'
    (model/dataset/optimizer), and thus would break grouping across seeds if you include them in
    groupby.

    Prints:
      - basic stats about how many configs have multiple runs
      - columns that vary inside those configs (sorted by how often they vary)
    """

    # 1) Detect seed columns ---------------------------------------------------------
    if seed_cols is None:
        seed_cols = []
        possible_seed_cols = [
            "seed",
            "dataset.split_params.data_seed",
            "dataset.loader.parameters.data_seed",
        ]
        for col in possible_seed_cols:
            if col in df.columns:
                seed_cols.append(col)
        if not seed_cols:
            seed_cols = [c for c in df.columns if "seed" in c.lower()]

    print(f"▶ Using seed columns (for diagnostics only): {seed_cols}")

    # 2) Metric columns --------------------------------------------------------------
    metric_cols = [c for c in df.columns if c.startswith(metric_prefix)]
    exclude_cols = set(metric_cols) | set(seed_cols) | {"run_id", "run_name", "state"}

    # 3) Candidate non-metric config-ish columns ------------------------------------
    candidate_cols = [c for c in df.columns if c not in exclude_cols]

    # 4) Serialize candidate columns so groupby / nunique won't choke ----------------
    df_ser = df.copy()
    for col in candidate_cols:
        df_ser[col] = df_ser[col].apply(_serialize_for_grouping)

    # 5) Define a "core config" using sensible prefixes ------------------------------
    base_prefixes = [
        "model.backbone.",
        "model.readout.",
        "model.feature_encoder.",
        "model.model_name",
        "model.model_domain",
        "dataset.loader.parameters.",
        "dataset.parameters.",
        "optimizer.parameters.",
    ]
    base_group_cols = []

    for col in candidate_cols:
        if any(col.startswith(p) for p in base_prefixes):
            base_group_cols.append(col)

    # Add a couple of standalone columns if present
    for col in ["task_name", "evaluator.task"]:
        if col in candidate_cols:
            base_group_cols.append(col)

    # Deduplicate while preserving order
    seen = set()
    base_group_cols = [c for c in base_group_cols if not (c in seen or seen.add(c))]

    if not base_group_cols:
        print("⚠ No base_group_cols found with the chosen prefixes. "
              "You may need to adjust base_prefixes.")
        return

    print(f"▶ Core config grouping columns ({len(base_group_cols)}):")
    for c in base_group_cols:
        print("   ", c)

    # 6) Group by core config and look at group sizes --------------------------------
    grouped_core = df_ser.groupby(base_group_cols, dropna=False)
    group_sizes = grouped_core.size()

    print(f"\n▶ Number of unique core configs: {len(group_sizes)}")
    print(f"▶ Runs per core config: min={group_sizes.min()}, "
          f"max={group_sizes.max()}, mean={group_sizes.mean():.2f}")
    multi_core = group_sizes[group_sizes > 1]

    if multi_core.empty:
        print("⚠ No core config has more than one run. "
              "Either you truly have no repeated configs, or the core "
              "grouping is too fine; try removing some base_prefixes.")
        return

    print(f"▶ Core configs with >1 run (where seeds *should* aggregate): {len(multi_core)}")

    # Restrict to those multi-run core configs
    idx_multi = multi_core.index
    df_multi = df_ser.set_index(base_group_cols).loc[idx_multi].reset_index()

    # 7) For all other candidate columns, see if they vary *within* a core config ----
    other_cols = sorted(set(candidate_cols) - set(base_group_cols))

    varying_info = []  # (col, frac_groups_vary, max_nunique)

    grouped_multi = df_multi.groupby(base_group_cols, dropna=False)
    n_groups = len(multi_core)

    for col in other_cols:
        nunique_per_group = grouped_multi[col].nunique(dropna=False)
        max_nunique = nunique_per_group.max()
        if max_nunique > 1:
            frac_vary = (nunique_per_group > 1).sum() / n_groups
            varying_info.append((col, frac_vary, int(max_nunique)))

    if not varying_info:
        print("\n▶ No additional columns vary within core configs.")
        print("  That means the issue is likely that the core config itself "
              "is too detailed for seeds to line up.")
        return

    varying_info.sort(key=lambda x: x[1], reverse=True)

    print("\n▶ Columns that vary within core configs (and would break seed aggregation):")
    print("   (col, fraction_of_core_configs_where_it_varies, max_nunique_within_a_config)")
    for col, frac, max_n in varying_info[:50]:  # show top 50
        print(f"   {col:60s}  frac={frac:6.3f}, max_nunique={max_n}")

    # Optionally, also print some "safe" columns that never vary
    stable_cols = [
        col for col in other_cols
        if col not in {v[0] for v in varying_info}
    ]
    if stable_cols:
        print("\n▶ Example columns that are stable within core configs (safe to include if desired):")
        for c in stable_cols[:30]:
            print("   ", c)


# ---------------------------------------------------------------------
# Helper: decode serialized values
# ---------------------------------------------------------------------
def _decode_val(x):
    """Decode serialized values like "__val__:'sagn'" -> "sagn"."""
    if isinstance(x, str):
        if x == "__NaN__":
            return np.nan
        if x.startswith("__val__:"):
            inner = x[len("__val__:"):]
            # Strip simple quotes if present
            if inner.startswith("'") and inner.endswith("'"):
                return inner[1:-1]
            # Try to literal-eval numbers / booleans
            try:
                return ast.literal_eval(inner)
            except Exception:
                return inner
    return x


# ---------------------------------------------------------------------
# Find best runs by val/f1_macro and create tables per dataset
# ---------------------------------------------------------------------
def get_best_runs_by_val_f1_macro(
    aggregated_df,
    metric_col="summary.val/f1_macro",
    data_name_col="dataset.loader.parameters.data_name",
    method_col="dataset.loader.parameters.method",
    node_sample_ratio_col="dataset.loader.parameters.node_sample_ratio",
    datasets=None,
):
    # Make a copy to avoid modifying the original
    df = aggregated_df.copy()

    # Check required columns exist
    required_cols = [metric_col, data_name_col, method_col, node_sample_ratio_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Decode serialized values for the grouping columns
    for col in [data_name_col, method_col, node_sample_ratio_col]:
        if col in df.columns:
            df[col] = df[col].apply(_decode_val)

    # Ensure metric is numeric
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    # Filter out rows with NaN in required columns
    df = df.dropna(subset=[data_name_col, method_col, node_sample_ratio_col, metric_col])

    if len(df) == 0:
        raise ValueError("No valid rows found after filtering")

    # Get datasets to process
    if datasets is None:
        datasets = sorted(df[data_name_col].dropna().unique())
    else:
        # Filter to only requested datasets
        df = df[df[data_name_col].isin(datasets)]
        if len(df) == 0:
            raise ValueError(f"No data found for datasets: {datasets}")

    # Find the std column for the metric
    metric_std_col = f"{metric_col}_std"
    df[metric_std_col] = pd.to_numeric(df[metric_std_col], errors="coerce")

    # Group by dataset, method, and node_sample_ratio, and find the best run
    # (highest val/f1_macro) for each combination
    grouped = df.groupby([data_name_col, method_col, node_sample_ratio_col], dropna=False)

    # For each group, get the row with the maximum metric value
    best_runs = []
    for (data_name, method, ratio), group in grouped:
        # Find the row with the maximum metric value
        best_idx = group[metric_col].idxmax()
        best_run = group.loc[best_idx].copy()
        best_runs.append(best_run)

    best_df = pd.DataFrame(best_runs)

    # Create pivot tables for each dataset (combined mean ± std)
    result_tables = {}

    for dataset in datasets:
        dataset_data = best_df[best_df[data_name_col] == dataset].copy()

        if len(dataset_data) == 0:
            print(f"⚠ No data found for dataset: {dataset}")
            continue

        # Create pivot tables for mean and std
        pivot_table_mean = dataset_data.pivot_table(
            values=metric_col,
            index=method_col,
            columns=node_sample_ratio_col,
            aggfunc='first',
        )

        pivot_table_std = dataset_data.pivot_table(
            values=metric_std_col,
            index=method_col,
            columns=node_sample_ratio_col,
            aggfunc='first',
        )

        # Sort methods and ratios for better readability
        if len(pivot_table_mean) > 0:
            pivot_table_mean = pivot_table_mean.sort_index()
            pivot_table_mean = pivot_table_mean.sort_index(axis=1)
            pivot_table_std = pivot_table_std.sort_index()
            pivot_table_std = pivot_table_std.sort_index(axis=1)

        # Combine mean and std into formatted strings: "mean ± std"
        combined_table = pivot_table_mean.copy()
        for idx in pivot_table_mean.index:
            for col in pivot_table_mean.columns:
                mean_val = pivot_table_mean.loc[idx, col]
                std_val = pivot_table_std.loc[idx, col]

                if pd.notna(mean_val) and pd.notna(std_val):
                    combined_table.loc[idx, col] = f"{mean_val:.4f} ± {std_val:.4f}"
                elif pd.notna(mean_val):
                    combined_table.loc[idx, col] = f"{mean_val:.4f}"
                else:
                    combined_table.loc[idx, col] = "NaN"

        result_tables[dataset] = combined_table

    return result_tables

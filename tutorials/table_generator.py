import os
import json
import wandb
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Helper to flatten nested config dicts
# ---------------------------------------------------------------------
def flatten_config(config, parent_key='', sep='.'):
    """
    Flatten a nested dictionary by joining keys with a separator.
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------
# Load runs from W&B or from a cached CSV
# ---------------------------------------------------------------------
def load_results_dataframe(
    wandb_username,
    wandb_project,
    csv_filename="wandb_results.csv",
    force_load=False,
    save_csv=True,
    filters=None,
):
    """
    Load results from W&B and return a DataFrame with all available metrics and configs.
    """
    if filters is None:
        filters = {}

    if os.path.exists(csv_filename) and not force_load:
        df = pd.read_csv(csv_filename)
        print(f"▶ Loaded existing CSV file: {csv_filename}")
    else:
        api = wandb.Api()
        runs = api.runs(f"{wandb_username}/{wandb_project}", filters=filters)
        print(f"▶ Number of runs fetched from W&B: {len(runs)}")

        records = []
        for run in runs:
            cfg = run.config.copy() or {}
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
            records.append(row)

        df = pd.DataFrame(records)

        if save_csv:
            df.to_csv(csv_filename, index=False)
            print(f"▶ Saved results to: {csv_filename}")

    print(f"▶ DataFrame shape: {df.shape}")
    print(f"\n▶ Column names ({len(df.columns)} total):")
    print("-" * 80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")
    print("-" * 80)

    return df


# ---------------------------------------------------------------------
# Helper: serialize values into a *string* key for grouping
# ---------------------------------------------------------------------
def _serialize_for_grouping(val):
    """
    Convert any Python object (including lists/dicts/arrays) into a stable
    string representation so pandas can group on it safely.

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
    seed_cols=None,
    fold_cols=None,
    metric_prefix="summary.",
    output_filename=None,
):
    """
    Aggregate W&B runs across seeds, computing mean/std/count for each metric.

    Grouping columns:
      - All columns EXCEPT:
        * seed columns
        * run_id, run_name, state
        * columns starting with metric_prefix (e.g. "summary.")
      - And we explicitly drop columns that vary within the same core config
        (like the sklearn pipelines).
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
            # Fallback: any column name containing "seed"
            seed_cols = [c for c in df.columns if "seed" in c.lower()]

    print(f"▶ Using seed columns: {seed_cols}")

    # 2) Detect fold columns (optional) ----------------------------------------------
    if fold_cols is None:
        fold_cols = []
        possible_fold_cols = [
            "fold",
            "dataset.split_params.fold",
            "dataset.loader.parameters.fold",
        ]
        for col in possible_fold_cols:
            if col in df.columns:
                fold_cols.append(col)

    print(f"▶ Using fold columns: {fold_cols}")

    # 3) Identify metric columns -----------------------------------------------------
    metric_cols = [c for c in df.columns if c.startswith(metric_prefix)]
    if not metric_cols:
        raise ValueError(f"No metric columns found with prefix '{metric_prefix}'")
    print(f"▶ Found {len(metric_cols)} metric columns with prefix '{metric_prefix}'")

    # 4) Determine initial candidate grouping columns --------------------------------
    exclude_cols = set(seed_cols + ["run_id", "run_name", "state"])
    exclude_cols.update(metric_cols)

    candidate_grouping = [c for c in df.columns if c not in exclude_cols]

    # Explicitly drop the problematic baseline pipeline columns
    bad_group_cols = [
        "dataset.baselines.svm.pipeline",
        "dataset.baselines.elastic_net.pipeline",
    ]
    candidate_grouping = [c for c in candidate_grouping if c not in bad_group_cols]

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
        if nunique < n_rows:
            grouping_cols.append(col)

    # Make sure fold columns remain in grouping (if you want per-fold stats)
    for fc in fold_cols:
        if fc in df.columns and fc not in grouping_cols:
            grouping_cols.append(fc)

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
    """
    Diagnose which non-metric columns vary across runs that share the same
    'core config' (model/dataset/optimizer), and thus would break grouping
    across seeds if you include them in groupby.

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
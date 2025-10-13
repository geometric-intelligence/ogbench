"""MotrPac dataset processor."""

import os

import numpy as np
import pandas as pd

from scripts.utils import create_dataset_metadata, download_file, upload_to_huggingface


def process_motrpac(output_dir: str = "temp_data") -> None:
    """Download and process MotrPac dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Download URLs
    urls = {
        "proteomics": "https://d1yw74buhe0ts0.cloudfront.net/static/motrpac-data-hub/publications/data/related-studies/heritage-proteomics/HERITAGE_proteomics_somalogic.xlsx",
        "analytes": "https://d1yw74buhe0ts0.cloudfront.net/static/motrpac-data-hub/publications/data/related-studies/heritage-proteomics/HERITAGE_somalogic_analytes.xlsx",
    }

    # Download files
    for name, url in urls.items():
        file_path = os.path.join(output_dir, f"motrpac_{name}.xlsx")
        if not os.path.exists(file_path):
            print(f"Downloading {name}...")
            download_file(url, file_path)

    # Load data
    proteomics_df = pd.read_excel(os.path.join(output_dir, "motrpac_proteomics.xlsx"), header=3)
    analytes_df = pd.read_excel(os.path.join(output_dir, "motrpac_analytes.xlsx"))

    # 1) Use analytes.xlsx to select & name protein columns
    # The analytes file has Baseline/Post-training/Response columns with IDs
    if "Baseline" in analytes_df.columns:
        analyte_ids = analytes_df["Baseline"].astype(str).tolist()
    else:
        # Fallback to original logic
        id_col = next(
            c
            for c in analytes_df.columns
            if c.lower() in {"somamer", "seqid", "target", "analyte"}
        )
        analyte_ids = analytes_df[id_col].astype(str).tolist()

    # Keep only analyte columns that exist
    present = [c for c in proteomics_df.columns if str(c) in analyte_ids]
    raw_data = proteomics_df[present].copy()
    raw_data.columns = [str(c) for c in raw_data.columns]  # stable names

    # 2) Targets: compute relative ΔVO₂max and responder labels
    # Extract baselines and deltas robustly by column names
    col_base = next(c for c in proteomics_df.columns if "baseline vo2" in c.lower())
    col_delta = next(c for c in proteomics_df.columns if "delta vo2" in c.lower())

    vo2_base = pd.to_numeric(proteomics_df[col_base], errors="coerce")
    vo2_delta = pd.to_numeric(proteomics_df[col_delta], errors="coerce")
    vo2_post = vo2_base + vo2_delta

    mask = (~vo2_base.isna()) & (~vo2_post.isna())  # keep only valid pairs
    raw_data = raw_data.loc[mask]
    vo2_base = vo2_base.loc[mask]
    vo2_post = vo2_post.loc[mask]

    delta_abs = vo2_post - vo2_base
    delta_rel = (vo2_post - vo2_base) / vo2_base  # relative improvement

    # Responder tasks
    responder15 = (delta_rel > 0.15).astype(int)  # binary

    # 3) Extract baseline covariates
    cov_cols = {
        "age": [c for c in proteomics_df.columns if c.lower() == "age"],
        "sex": [c for c in proteomics_df.columns if c.lower() == "sex"],
        "bmi": [c for c in proteomics_df.columns if "bmi" in c.lower()],
        "race": [c for c in proteomics_df.columns if "race" in c.lower()],
        "body_fat_pct": [c for c in proteomics_df.columns if "body fat" in c.lower()],
        "ffm": [c for c in proteomics_df.columns if "fat-free" in c.lower() or "ffm" in c.lower()],
    }

    cov = (
        pd.DataFrame(
            {
                "age": pd.to_numeric(proteomics_df[cov_cols["age"][0]], errors="coerce")
                if cov_cols["age"]
                else np.nan,
                "sex": proteomics_df[cov_cols["sex"][0]] if cov_cols["sex"] else np.nan,
                "bmi": pd.to_numeric(proteomics_df[cov_cols["bmi"][0]], errors="coerce")
                if cov_cols["bmi"]
                else np.nan,
                "race": proteomics_df[cov_cols["race"][0]] if cov_cols["race"] else np.nan,
                "body_fat_pct": pd.to_numeric(
                    proteomics_df[cov_cols["body_fat_pct"][0]], errors="coerce"
                )
                if cov_cols["body_fat_pct"]
                else np.nan,
                "ffm": pd.to_numeric(proteomics_df[cov_cols["ffm"][0]], errors="coerce")
                if cov_cols["ffm"]
                else np.nan,
                "vo2_baseline": vo2_base,
                "vo2_post": vo2_post,
                "delta_vo2_abs": delta_abs,
                "delta_vo2_rel": delta_rel,
            }
        )
        .loc[mask]
        .reset_index(drop=True)
    )

    # 4) Batch / plate effects handling
    plate_col = next(
        (c for c in proteomics_df.columns if c.lower() in {"plate", "batch", "run"}), None
    )
    if plate_col:
        batch = proteomics_df.loc[mask, plate_col].astype(str).reset_index(drop=True)
        cov["batch"] = batch

    # 5) Light QC at ingest time (no learning from labels)
    col_na = raw_data.isna().mean()
    raw_data = raw_data.loc[:, col_na <= 0.1]  # drop analytes with >10% NA
    # Leave remaining NaNs as-is; downstream will impute on train only

    # 6) log transform
    raw_data = np.log1p(raw_data)  # log1p for stability
    raw_data = pd.DataFrame(raw_data, columns=raw_data.columns).reset_index(drop=True)

    # 7) Emit multiple targets (regression + classification)
    out = os.path.join(output_dir, "motrpac")
    os.makedirs(out, exist_ok=True)

    raw_data.reset_index(drop=True).to_parquet(os.path.join(out, "motrpac_data.parquet"))

    pd.DataFrame({"target": delta_rel.reset_index(drop=True)}).to_parquet(
        os.path.join(out, "motrpac_targets_vo2_rel.parquet")
    )
    pd.DataFrame({"target": responder15.reset_index(drop=True).astype(np.int64)}).to_parquet(
        os.path.join(out, "motrpac_targets_responder15.parquet")
    )

    cov.reset_index(drop=True).to_parquet(os.path.join(out, "motrpac_covariates.parquet"))

    # 8) Create metadata
    target_stats = {
        "delta_rel": {
            "mean": float(delta_rel.mean()),
            "std": float(delta_rel.std()),
            "min": float(delta_rel.min()),
            "max": float(delta_rel.max()),
        },
        "responder15": {
            "pos": int((responder15 == 1).sum()),
            "neg": int((responder15 == 0).sum()),
        },
    }

    metadata = create_dataset_metadata(
        dataset_name="motrpac",
        download_urls=urls,
        num_samples=len(raw_data),
        num_features=raw_data.shape[1],
        target_stats=target_stats,
    )

    # Upload to HuggingFace
    data_files = {
        "data": os.path.join(out, "motrpac_data.parquet"),
        "targets_vo2_rel": os.path.join(out, "motrpac_targets_vo2_rel.parquet"),
    }

    upload_to_huggingface("motrpac", data_files, metadata)

    print("Successfully processed and uploaded MotrPac dataset")
    print(f"  Samples: {len(raw_data)}")
    print(f"  Features: {raw_data.shape[1]}")
    print(f"  Analytes used: {len(analyte_ids)}")
    print(f"  Target stats: {target_stats}")

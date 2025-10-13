"""Utility functions for dataset processing."""

import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict

import huggingface_hub
import requests


def download_file(url: str, file_path: str) -> None:
    """Download a file from URL to local path.

    Args:
        url: URL to download from
        file_path: Local path to save the file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def create_dataset_metadata(
    dataset_name: str,
    download_urls: Dict[str, str],
    num_samples: int,
    num_features: int,
    target_stats: Dict[str, float],
) -> Dict[str, Any]:
    """Create metadata dictionary for a dataset.

    Args:
        dataset_name: Name of the dataset
        download_urls: Dictionary mapping dataset keys to download URLs
        num_samples: Number of samples in the dataset
        num_features: Number of features in the dataset
        target_stats: Statistics about the target variable

    Returns:
        Metadata dictionary
    """
    return {
        "dataset_name": dataset_name,
        "download_urls": download_urls,
        "download_timestamp": datetime.now().isoformat(),
        "statistics": {
            "num_samples": num_samples,
            "num_features": num_features,
            "target_stats": target_stats,
        },
    }


def upload_to_huggingface(
    dataset_name: str,
    data_files: Dict[str, str],
    metadata: Dict[str, Any],
) -> None:
    """Upload dataset files and metadata to HuggingFace Hub.

    Args:
        dataset_name: Name of the dataset (used for subdirectory naming)
        data_files: Dictionary mapping file names to local file paths
        metadata: Metadata dictionary to save
    """
    try:
        api = huggingface_hub.HfApi()
        repo_id = "geometric-intelligence/bgbench"

        # Create repository if it doesn't exist
        try:
            huggingface_hub.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"Did not create repository {repo_id}: {e}")

        # Upload files to root of repository with dataset-specific naming
        for file_name, file_path in data_files.items():
            if os.path.exists(file_path):
                # Rename files to include dataset name prefix for hf_omics compatibility
                hf_file_name = f"{dataset_name}_{file_name}.parquet"
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=hf_file_name,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"Uploaded {hf_file_name}")
            else:
                print(f"Warning: File {file_path} does not exist")

        # Save metadata as JSON
        metadata_file = f"{dataset_name}_metadata.json"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            api.upload_file(
                path_or_fileobj=f.name,
                path_in_repo=metadata_file,
                repo_id=repo_id,
                repo_type="dataset",
            )
        print(f"Uploaded {metadata_file}")
    except (
        huggingface_hub.utils.RepositoryNotFoundError,
        huggingface_hub.utils.HfHubHTTPError,
    ) as e:
        print(f"⚠️  HuggingFace upload failed (authentication required): {e}")
        print(f"Data files saved locally in temp_data/{dataset_name}/")
        print("To upload later, set up authentication with: huggingface-cli login")
    except Exception as e:
        print(f"⚠️  HuggingFace upload failed: {e}")
        print(f"Data files saved locally in temp_data/{dataset_name}/")

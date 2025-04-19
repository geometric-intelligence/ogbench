#!/usr/bin/env python3

import os
import requests
import pandas as pd
from tqdm import tqdm

def download_file(url, output_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_addneuromed():
    """
    Download AddNeuroMed datasets GSE63060 and GSE63061 from GEO.
    """
    # Create output directory if it doesn't exist
    output_dir = "addneuromed_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # GEO dataset URLs
    datasets = {
        'GSE63060': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63060&format=file',
        'GSE63061': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63061&format=file'
    }
    
    for dataset, url in datasets.items():
        print(f"Downloading {dataset}...")
        try:
            # Download the dataset
            output_file = os.path.join(output_dir, f"{dataset}.txt.gz")
            download_file(url, output_file)
            print(f"Successfully downloaded {dataset}")
            
        except Exception as e:
            print(f"Error downloading {dataset}: {str(e)}")

if __name__ == "__main__":
    download_addneuromed()
#!/usr/bin/env python3

import os
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def download_addneuromed():
    """
    Download AddNeuroMed datasets GSE63060 and GSE63061 from GEO.
    """
    # Import required R packages
    try:
        importr('GEOquery')
    except:
        print("Installing required R packages...")
        utils = importr('utils')
        utils.install_packages('GEOquery')
        importr('GEOquery')
    
    # Create output directory if it doesn't exist
    output_dir = "addneuromed_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Download datasets
    datasets = ['GSE63060', 'GSE63061']
    
    for dataset in datasets:
        print(f"Downloading {dataset}...")
        try:
            # Get the dataset
            gse = robjects.r(f'getGEO("{dataset}", GSEMatrix=TRUE, getGPL=FALSE)')
            
            # Save the expression data
            exprs = robjects.r('exprs')(gse[0])
            output_file = os.path.join(output_dir, f"{dataset}_expression.csv")
            robjects.r(f'write.csv(exprs, "{output_file}")')
            
            # Save the phenotype data
            pdata = robjects.r('pData')(gse[0])
            output_file = os.path.join(output_dir, f"{dataset}_phenotype.csv")
            robjects.r(f'write.csv(pdata, "{output_file}")')
            
            print(f"Successfully downloaded and saved {dataset}")
            
        except Exception as e:
            print(f"Error downloading {dataset}: {str(e)}")

if __name__ == "__main__":
    download_addneuromed() 
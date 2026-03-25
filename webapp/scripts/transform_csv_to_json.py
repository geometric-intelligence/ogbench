#!/usr/bin/env python3
"""Transform best_configs_summary.csv to results.json for the webapp leaderboard."""

import csv
import json
from pathlib import Path

# Model name mapping: CSV name -> Display name
MODEL_NAME_MAP = {
    'GATv4': 'MLA-GNN',
    'gatv2': 'GATv2',
    'gcn': 'GCN',
    'gin': 'GIN',
    'sage': 'GraphSAGE',
    'sagn': 'SAGN',
    'mlp': 'MLP',
    'chebnet': 'ChebNet',
}

# ML Baselines with fixed test F1 macro scores (don't depend on graph config)
BASELINES = {
    'parkinsons': {
        'ElasticNet': {'test_f1_macro': 0.64518},
        'SVM': {'test_f1_macro': 0.66422}
    },
    'motrpac': {
        'ElasticNet': {'test_f1_macro': 0.57419},
        'SVM': {'test_f1_macro': 0.54287}
    },
    'addneuromed': {
        'ElasticNet': {'test_f1_macro': 0.55762},
        'SVM': {'test_f1_macro': 0.46252}
    }
}


def transform_csv_to_json(csv_path: Path, output_path: Path) -> dict:
    """Transform CSV to JSON format for the webapp."""
    results = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Get and map model name
            model_csv = row['model']
            model = MODEL_NAME_MAP.get(model_csv, model_csv)
            
            dataset = row['dataset']
            method = row['method']
            node_sample_ratio = row['node_sample_ratio']
            readout = row['readout']
            
            # Create the graph_config and key
            graph_config = f"{dataset}|{node_sample_ratio}|{method}|{readout}"
            key = f"{graph_config}|{model}"
            
            # Extract metrics with safe float conversion
            def safe_float(val, default=0.0):
                try:
                    return float(val) if val else default
                except (ValueError, TypeError):
                    return default
            
            entry = {
                'graph_config': graph_config,
                'model': model,
                'dataset': dataset,
                'readout': readout,
                'node_sample_ratio': safe_float(node_sample_ratio),
                'method': method,
                # Validation metrics (for ranking)
                'val_accuracy': safe_float(row.get('summary.best_val/accuracy')),
                'val_accuracy_std': safe_float(row.get('summary.best_val/accuracy_std')),
                'val_f1_macro': safe_float(row.get('val_f1_macro')),
                'val_f1_macro_std': safe_float(row.get('summary.best_val/f1_macro_std')),
                # Test metrics (for display)
                'test_accuracy': safe_float(row.get('summary.best_test/accuracy')),
                'test_accuracy_std': safe_float(row.get('summary.best_test/accuracy_std')),
                'test_f1_macro': safe_float(row.get('summary.best_test/f1_macro')),
                'test_f1_macro_std': safe_float(row.get('summary.best_test/f1_macro_std')),
                'test_f1_weighted': safe_float(row.get('summary.best_test/f1_weighted')),
                'test_f1_weighted_std': safe_float(row.get('summary.best_test/f1_weighted_std')),
                'auroc': safe_float(row.get('summary.best_test/auroc')),
                'auroc_std': safe_float(row.get('summary.best_test/auroc_std')),
            }
            
            results[key] = entry
    
    # Add baseline entries for each dataset
    for dataset, models in BASELINES.items():
        for model, metrics in models.items():
            graph_config = f"{dataset}|baseline"
            key = f"{graph_config}|{model}"
            
            entry = {
                'graph_config': graph_config,
                'model': model,
                'dataset': dataset,
                'readout': 'baseline',
                'node_sample_ratio': 0.0,  # N/A for baselines
                'method': 'baseline',
                # Baselines don't have validation metrics in the same way
                'val_accuracy': 0.0,
                'val_accuracy_std': 0.0,
                'val_f1_macro': 0.0,
                'val_f1_macro_std': 0.0,
                # Test metrics
                'test_accuracy': 0.0,  # Not provided
                'test_accuracy_std': 0.0,
                'test_f1_macro': metrics['test_f1_macro'],
                'test_f1_macro_std': 0.0,
                'test_f1_weighted': 0.0,
                'test_f1_weighted_std': 0.0,
                'auroc': 0.0,
                'auroc_std': 0.0,
            }
            
            results[key] = entry
    
    # Write to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Transformed {len(results)} entries to {output_path}")
    print(f"  - CSV entries: {len(results) - sum(len(m) for m in BASELINES.values())}")
    print(f"  - Baseline entries: {sum(len(m) for m in BASELINES.values())}")
    
    # Print unique models
    models = set(entry['model'] for entry in results.values())
    print(f"  - Models: {sorted(models)}")
    
    # Print unique datasets
    datasets = set(entry['dataset'] for entry in results.values())
    print(f"  - Datasets: {sorted(datasets)}")
    
    return results


def main():
    # Paths relative to script location
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    
    csv_path = repo_root / 'tutorials' / 'stats' / 'best_configs_summary.csv'
    output_path = script_dir.parent / 'public' / 'data' / 'results.json'
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return 1
    
    transform_csv_to_json(csv_path, output_path)
    return 0


if __name__ == '__main__':
    exit(main())

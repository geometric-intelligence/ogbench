#!/bin/bash
# Sklearn baselines on exactly the feature sets the GNNs see:
# same k-fold splits, same train-only corrections, same node selection.
# One hyperparameter configuration is selected by mean validation score across
# all five folds and then reused for every fold, matching the GNN Optuna trials.
# No adjacency matrix is built; baselines never use the graph.

set -uo pipefail

DATASETS="parkinsons,addneuromed,motrpac,brca,tuberculosis,smoking"
FOLDS="0,1,2,3,4"
NODE_RATIOS="1.0,0.8,0.5,0.3"
METHODS="variance,random,correlation,distance_correlation"

echo "========================================="
echo "Standard baselines (SVM, Elastic Net)"
echo "datasets=${DATASETS} folds=${FOLDS}"
echo "ratios=${NODE_RATIOS} methods=${METHODS}"
echo "hparam_selection=global_kfold_mean_validation"
echo "========================================="

if python ogbench/baseline.py --multirun \
    "dataset=${DATASETS}" \
    dataset.split_params.split_type=k-fold \
    dataset.split_params.k=5 \
    "dataset.split_params.data_seed=${FOLDS}" \
    "dataset.loader.parameters.node_sample_ratio=${NODE_RATIOS}" \
    "dataset.loader.parameters.method=${METHODS}" \
    seed=42 \
    baseline_filter=standard; then
    echo "  -> Success: standard baselines"
else
    echo "  -> Failed: standard baselines"
fi

echo ""
echo "Done. Check WandB for per-fold results (run names include r{ratio}_m{method}_k-fold_fold{N})."

#!/bin/bash

# Node sample ratios and selection methods matching the GNN experiment grid
NODE_RATIOS="0.3,0.5,1.0,full"
METHODS="variance,correlation,distance_correlation,random"

# All datasets
ALL_DATASETS=(
    "addneuromed"
    "motrpac"
    "parkinsons"
    "brca"
)

echo "========================================="
echo "Phase 1: Standard baselines (single run per dataset)"
echo "========================================="

for dataset in "${ALL_DATASETS[@]}"; do
    echo "[standard] Running dataset=${dataset}"

    if python ogbench/baseline.py \
        dataset="${dataset}" \
        baseline_filter=standard; then
        echo "  -> Success: standard baselines for ${dataset}"
    else
        echo "  -> Failed: standard baselines for ${dataset}"
    fi
done

echo ""
echo "========================================="
echo "Phase 2: GNN-features baselines (sweep node_ratio x method)"
echo "========================================="

for dataset in "${ALL_DATASETS[@]}"; do
    echo "[gnn_features] Running dataset=${dataset} | ratios=${NODE_RATIOS} | methods=${METHODS}"

    if python ogbench/baseline.py \
        dataset="${dataset}" \
        baseline_filter=gnn_features \
        "dataset.loader.parameters.node_sample_ratio=${NODE_RATIOS}" \
        "dataset.loader.parameters.method=${METHODS}" \
        --multirun; then
        echo "  -> Success: gnn_features baselines for ${dataset}"
    else
        echo "  -> Failed: gnn_features baselines for ${dataset}"
    fi
done

echo ""
echo "All baseline experiments completed!"
echo ""
echo "Results Summary:"
echo "- Standard baselines: ${#ALL_DATASETS[@]} datasets x 2 models (SVM, Elastic Net)"
echo "- GNN-features baselines: ${#ALL_DATASETS[@]} datasets x 2 models x $(echo ${NODE_RATIOS} | tr ',' '\n' | wc -l) ratios x $(echo ${METHODS} | tr ',' '\n' | wc -l) methods"
echo "- Check WandB for detailed results"

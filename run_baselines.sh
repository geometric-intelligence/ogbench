#!/bin/bash

# Function to run baseline experiment
run_baseline_experiment() {
    local dataset=$1
    local experiment_num=$2
    local total_experiments=$3

    echo "[${experiment_num}/${total_experiments}] Running baseline: dataset=${dataset}"

    # Run the baseline experiment
    if python ogbench/baseline.py dataset="${dataset}"; then
        echo "✅ Success: baseline dataset=${dataset}"
    else
        echo "❌ Failed: baseline dataset=${dataset}"
    fi
}

# Datasets to test (only the 3 main datasets)
datasets=(
    "addneuromed"
    "motrpac"
    "parkinsons"
)

# Counter for tracking progress
total_experiments=${#datasets[@]}
current_experiment=0

echo "Starting baseline experiments: ${total_experiments} total datasets"
echo "Datasets: ${datasets[*]}"
echo "----------------------------------------"

# Run baseline experiments for each dataset
for dataset in "${datasets[@]}"; do
    current_experiment=$((current_experiment + 1))
    run_baseline_experiment "$dataset" "${current_experiment}" "${total_experiments}"
    echo "----------------------------------------"
done

echo "All baseline experiments completed!"
echo ""
echo "Results Summary:"
echo "- Check WandB dashboard for detailed results and visualizations"
echo "- Baseline results are logged with tags: ['baseline', 'sklearn', dataset_name]"
echo "- Each dataset runs multiple baseline models (SVM, Elastic Net, etc.)"
echo "- Results include comprehensive plots: confusion matrices, ROC curves, feature importance"
echo ""
echo "To view results:"
echo "1. WandB: https://wandb.ai/your-username/bgbench-baselines"
echo "2. Local logs: Check logs/ directory for detailed output"
echo "3. Generated plots: Saved in output directories for each experiment"

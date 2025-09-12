#!/bin/bash

# Models to test
models=(
    "chebnet"
    "gatv4"
    "gat"
    "gatv2"
    "gcn"
    "mlp"
    "graph_sage"
    "sagn"
)

# Datasets to test
datasets=(
    "addneuromed"
    "covidaki"
    "motrpac"
    "parkinsons"
)

# Sampling methods to test
sampling_methods=(
    "variance"
    "random"
    "correlation"
)

# Node ratios to test
node_ratios=(
    "1"
    "0.5"
    "0.25"
)

# Counter for tracking progress
total_experiments=$((${#models[@]} * ${#datasets[@]} * ${#sampling_methods[@]} * ${#node_ratios[@]}))
current_experiment=0

echo "Starting experiments: ${total_experiments} total combinations"
echo "Models: ${models[*]}"
echo "Datasets: ${datasets[*]}"
echo "Sampling methods: ${sampling_methods[*]}"
echo "Node ratios: ${node_ratios[*]}"
echo "----------------------------------------"

# Run cartesian product of sampling methods, models, datasets, and node ratios
for sampling_method in "${sampling_methods[@]}"; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for node_ratio in "${node_ratios[@]}"; do
                current_experiment=$((current_experiment + 1))
                echo "[${current_experiment}/${total_experiments}] Running: sampling_method=${sampling_method} model=${model} dataset=${dataset} node_ratio=${node_ratio}"

                # Run the experiment
                if python ogbench/run.py dataset="${dataset}" model="${model}" dataset.loader.parameters.method="${sampling_method}" dataset.loader.parameters.node_sample_ratio="${node_ratio}"; then
                    echo "✅ Success: sampling_method=${sampling_method} model=${model} dataset=${dataset} node_ratio=${node_ratio}"
                else
                    echo "❌ Failed: sampling_method=${sampling_method} model=${model} dataset=${dataset} node_ratio=${node_ratio}"
                fi

                echo "----------------------------------------"
            done
        done
    done
done

echo "All experiments completed!"

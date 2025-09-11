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

# Counter for tracking progress
total_experiments=$((${#models[@]} * ${#datasets[@]} * ${#sampling_methods[@]}))
current_experiment=0

echo "Starting experiments: ${total_experiments} total combinations"
echo "Models: ${models[*]}"
echo "Datasets: ${datasets[*]}"
echo "Sampling methods: ${sampling_methods[*]}"
echo "----------------------------------------"

# Run cartesian product of sampling methods, models, and datasets
for sampling_method in "${sampling_methods[@]}"; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo "[${current_experiment}/${total_experiments}] Running: sampling_method=${sampling_method} model=${model} dataset=${dataset}"

            # Run the experiment
            if python ogbench/run.py dataset="${dataset}" model="${model}" dataset.loader.parameters.method="${sampling_method}"; then
                echo "✅ Success: sampling_method=${sampling_method} model=${model} dataset=${dataset}"
            else
                echo "❌ Failed: sampling_method=${sampling_method} model=${model} dataset=${dataset}"
            fi

            echo "----------------------------------------"
        done
    done
done

echo "All experiments completed!"

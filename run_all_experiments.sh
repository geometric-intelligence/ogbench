#!/bin/bash

# Parse command line arguments
PARALLEL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--parallel]"
            exit 1
            ;;
    esac
done

# Function to detect available GPUs
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# Function to check if GPUs are available (not running other processes)
check_gpu_availability() {
    local gpu_count=$1
    if [ "$gpu_count" -eq 0 ]; then
        echo "No GPUs detected. Running in CPU mode."
        return 0
    fi

    echo "Checking GPU availability..."
    for ((i=0; i<gpu_count; i++)); do
        local gpu_util
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $i 2>/dev/null)
        if [ "$gpu_util" -gt 10 ]; then
            echo "Warning: GPU $i is currently in use (${gpu_util}% utilization)"
            echo "Consider waiting or using --parallel with fewer processes"
        fi
    done
}

# Function to run experiment with specific GPU
run_experiment() {
    local gpu_id=$1
    local sampling_method=$2
    local model=$3
    local dataset=$4
    local experiment_num=$5
    local total_experiments=$6

    echo "[${experiment_num}/${total_experiments}] Running on GPU ${gpu_id}: sampling_method=${sampling_method} model=${model} dataset=${dataset}"

    # Set CUDA_VISIBLE_DEVICES for this process
    export CUDA_VISIBLE_DEVICES=$gpu_id

    # Run the experiment
    if python ogbench/run.py dataset="${dataset}" model="${model}" dataset.loader.parameters.method="${sampling_method}"; then
        echo "✅ Success on GPU ${gpu_id}: sampling_method=${sampling_method} model=${model} dataset=${dataset}"
    else
        echo "❌ Failed on GPU ${gpu_id}: sampling_method=${sampling_method} model=${model} dataset=${dataset}"
    fi
}

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

# Detect available GPUs
gpu_count=$(detect_gpus)
echo "Detected ${gpu_count} GPUs"

# Check GPU availability if parallel mode is enabled
if [ "$PARALLEL" = true ]; then
    check_gpu_availability "$gpu_count"
fi

# Counter for tracking progress
total_experiments=$((${#models[@]} * ${#datasets[@]} * ${#sampling_methods[@]}))
current_experiment=0

echo "Starting experiments: ${total_experiments} total combinations"
echo "Models: ${models[*]}"
echo "Datasets: ${datasets[*]}"
echo "Sampling methods: ${sampling_methods[*]}"
echo "Parallel mode: ${PARALLEL}"
if [ "$PARALLEL" = true ] && [ "$gpu_count" -gt 0 ]; then
    echo "Using ${gpu_count} GPUs for parallel execution"
fi
echo "----------------------------------------"

# Run experiments
if [ "$PARALLEL" = true ] && [ "$gpu_count" -gt 0 ]; then
    # Parallel execution using background processes
    echo "Starting parallel execution with ${gpu_count} GPUs..."

    # Array to store background process IDs
    declare -a pids=()

    # Run cartesian product of sampling methods, models, and datasets
    for sampling_method in "${sampling_methods[@]}"; do
        for model in "${models[@]}"; do
            for dataset in "${datasets[@]}"; do
                current_experiment=$((current_experiment + 1))

                # Calculate which GPU to use (round-robin)
                gpu_id=$(((current_experiment - 1) % gpu_count))

                # Run experiment in background
                run_experiment $gpu_id "$sampling_method" "$model" "$dataset" $current_experiment $total_experiments &
                pids+=($!)

                # Limit concurrent processes to number of GPUs
                if [ "${#pids[@]}" -ge "$gpu_count" ]; then
                    # Wait for any process to complete
                    wait -n
                    # Remove completed processes from array
                    for i in "${!pids[@]}"; do
                        if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                            unset "pids[$i]"
                        fi
                    done
                    # Reindex array
                    pids=("${pids[@]}")
                fi
            done
        done
    done

    # Wait for all remaining background processes to complete
    echo "Waiting for all experiments to complete..."
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

else
    # Sequential execution (original behavior)
    echo "Starting sequential execution..."

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
fi

echo "All experiments completed!"

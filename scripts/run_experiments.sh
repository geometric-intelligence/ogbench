#!/bin/bash

# Common hyperparameters to explore
LR_VALUES=(1e-2 1e-3 1e-1)
WEIGHT_DECAY_VALUES=(1e-4 2e-4 4e-4)
DROPOUT_VALUES=(0.1 0.2 0.5 0.7)
NUM_PROPAGATIONS_VALUES=(2 20 50)
AGGREGATION_RATIO_VALUES=(0.5 0.75 0.9 0.99)
NUM_MLP_LAYERS_VALUES=(1 2 3 4)
HIDDEN_CHANNELS_VALUES=(64 128 256)

# Models to explore
MODELS=("gcn" "gatv2" "graphsage" "nrgnn" "rtgnn" "engcn" "sagn" "mla_gnn")

# Datasets to explore
DATASETS=("addneuromed" "abide" "hcp")

# Create a function to run a single experiment
run_experiment() {
    local model=$1
    local dataset=$2
    local lr=$3
    local weight_decay=$4
    local dropout=$5
    local num_propagations=$6
    local aggregation_ratio=$7
    local num_mlp_layers=$8
    local hidden_channels=$9

    echo "Running experiment: model=$model, dataset=$dataset, lr=$lr, weight_decay=$weight_decay, dropout=$dropout, num_propagations=$num_propagations, aggregation_ratio=$aggregation_ratio, num_mlp_layers=$num_mlp_layers, hidden_channels=$hidden_channels"

    python train.py \
        experiment=$model \
        data=$dataset \
        model.optimizer.lr=$lr \
        model.optimizer.weight_decay=$weight_decay \
        model.net.dropout=$dropout \
        model.net.num_propagations=$num_propagations \
        model.net.aggregation_ratio=$aggregation_ratio \
        model.net.num_mlp_layers=$num_mlp_layers \
        model.net.hidden_channels=$hidden_channels
}

# Create a function to run experiments for a specific model and dataset
run_model_dataset_experiments() {
    local model=$1
    local dataset=$2

    for lr in "${LR_VALUES[@]}"; do
        for weight_decay in "${WEIGHT_DECAY_VALUES[@]}"; do
            for dropout in "${DROPOUT_VALUES[@]}"; do
                for num_propagations in "${NUM_PROPAGATIONS_VALUES[@]}"; do
                    for aggregation_ratio in "${AGGREGATION_RATIO_VALUES[@]}"; do
                        for num_mlp_layers in "${NUM_MLP_LAYERS_VALUES[@]}"; do
                            for hidden_channels in "${HIDDEN_CHANNELS_VALUES[@]}"; do
                                run_experiment "$model" "$dataset" "$lr" "$weight_decay" "$dropout" "$num_propagations" "$aggregation_ratio" "$num_mlp_layers" "$hidden_channels"
                            done
                        done
                    done
                done
            done
        done
    done
}

# Main loop to run all experiments
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        run_model_dataset_experiments "$model" "$dataset"
    done
done 
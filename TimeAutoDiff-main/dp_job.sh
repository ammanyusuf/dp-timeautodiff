#!/bin/bash

# Create output directories for different epsilon values
mkdir -p experiments/single_sequence/dp_0.1
mkdir -p experiments/single_sequence/dp_0.5
mkdir -p experiments/single_sequence/dp_1.0
mkdir -p experiments/single_sequence/dp_2.0
mkdir -p experiments/single_sequence/dp_5.0

mkdir -p experiments/multi_sequence/dp_0.1
mkdir -p experiments/multi_sequence/dp_0.5
mkdir -p experiments/multi_sequence/dp_1.0
mkdir -p experiments/multi_sequence/dp_2.0
mkdir -p experiments/multi_sequence/dp_5.0

# Function to run experiment with cleanup
run_dp_experiment() {
    echo "Running DP experiment with arguments: $@"
    python Model_Code/timeautodiff_run.py "$@"
    echo "DP experiment completed."
    sleep 5  # Give time for GPU memory to fully clear
}

# Single-sequence datasets with different epsilon values
echo "Running single-sequence DP experiments..."

# AirQuality
for eps in 0.1 0.5 1.0 2.0 5.0; do
    echo "Running AirQuality with epsilon = $eps"
    run_dp_experiment \
        --dataset "Dataset/Single-Sequence/AirQuality.csv" \
        --save_path "experiments/single_sequence/dp_${eps}/airquality_model.pt" \
        --output_dir "experiments/single_sequence/dp_${eps}/airquality_samples" \
        --use_dp \
        --epsilon $eps
done

# Hurricane
for eps in 0.1 0.5 1.0 2.0 5.0; do
    echo "Running Hurricane with epsilon = $eps"
    run_dp_experiment \
        --dataset "Dataset/Single-Sequence/Hurricane.csv" \
        --save_path "experiments/single_sequence/dp_${eps}/hurricane_model.pt" \
        --output_dir "experiments/single_sequence/dp_${eps}/hurricane_samples" \
        --use_dp \
        --epsilon $eps
done

# Metro Traffic
for eps in 0.1 0.5 1.0 2.0 5.0; do
    echo "Running Metro Traffic with epsilon = $eps"
    run_dp_experiment \
        --dataset "Dataset/Single-Sequence/Metro_Traffic.csv" \
        --save_path "experiments/single_sequence/dp_${eps}/metro_traffic_model.pt" \
        --output_dir "experiments/single_sequence/dp_${eps}/metro_traffic_samples" \
        --use_dp \
        --epsilon $eps
done

# Pollution Data
for eps in 0.1 0.5 1.0 2.0 5.0; do
    echo "Running Pollution Data with epsilon = $eps"
    run_dp_experiment \
        --dataset "Dataset/Single-Sequence/Pollution Data.csv" \
        --save_path "experiments/single_sequence/dp_${eps}/pollution_model.pt" \
        --output_dir "experiments/single_sequence/dp_${eps}/pollution_samples" \
        --use_dp \
        --epsilon $eps
done

echo "Single-sequence DP experiments completed."

# Multi-sequence datasets
echo "Running multi-sequence DP experiments..."

# Card Fraud
for eps in 0.1 0.5 1.0 2.0 5.0; do
    echo "Running Card Fraud with epsilon = $eps"
    run_dp_experiment \
        --dataset "Dataset/Multi-Sequence Data/card_fraud.csv" \
        --multi_sequence \
        --column_to_partition "User" \
        --save_path "experiments/multi_sequence/dp_${eps}/card_fraud_model.pt" \
        --output_dir "experiments/multi_sequence/dp_${eps}/card_fraud_samples" \
        --use_dp \
        --epsilon $eps
done

# NASDAQ100
for eps in 0.1 0.5 1.0 2.0 5.0; do
    echo "Running NASDAQ100 with epsilon = $eps"
    run_dp_experiment \
        --dataset "Dataset/Multi-Sequence Data/nasdaq100_2019.csv" \
        --multi_sequence \
        --column_to_partition "Symbol" \
        --save_path "experiments/multi_sequence/dp_${eps}/nasdaq_model.pt" \
        --output_dir "experiments/multi_sequence/dp_${eps}/nasdaq_samples" \
        --use_dp \
        --epsilon $eps
done

echo "Multi-sequence DP experiments completed."
echo "All DP experiments completed successfully!"
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

echo "Running AirQuality"
run_dp_experiment \
    --dataset "Dataset/Single-Sequence/AirQuality.csv" \
    --save_path "experiments/single_sequence/dp/airquality_model.pt" \
    --output_dir "experiments/single_sequence/dp/airquality_samples" \
    --use_dp 

# Hurricane
echo "Running Hurricane"
run_dp_experiment \
    --dataset "Dataset/Single-Sequence/Hurricane.csv" \
    --save_path "experiments/single_sequence/dp/hurricane_model.pt" \
    --output_dir "experiments/single_sequence/dp/hurricane_samples" \
    --use_dp

# Metro Traffic
echo "Running Metro Traffic"
run_dp_experiment \
    --dataset "Dataset/Single-Sequence/Metro_Traffic.csv" \
    --save_path "experiments/single_sequence/dp/metro_traffic_model.pt" \
    --output_dir "experiments/single_sequence/dp/metro_traffic_samples" \
    --use_dp

# Pollution Data
echo "Running Pollution Data"
run_dp_experiment \
    --dataset "Dataset/Single-Sequence/Pollution Data.csv" \
    --save_path "experiments/single_sequence/dp/pollution_model.pt" \
    --output_dir "experiments/single_sequence/dp/pollution_samples" \
    --use_dp

echo "Single-sequence DP experiments completed."

# Multi-sequence datasets
echo "Running multi-sequence DP experiments..."

# Card Fraud
echo "Running Card Fraud"
run_dp_experiment \
    --dataset "Dataset/Multi-Sequence Data/card_fraud.csv" \
    --multi_sequence \
    --column_to_partition "User" \
    --save_path "experiments/multi_sequence/dp/card_fraud_model.pt" \
    --output_dir "experiments/multi_sequence/dp/card_fraud_samples" \
    --use_dp \

# NASDAQ100
echo "Running NASDAQ100"
run_dp_experiment \
    --dataset "Dataset/Multi-Sequence Data/nasdaq100_2019.csv" \
    --multi_sequence \
    --column_to_partition "Symbol" \
    --save_path "experiments/multi_sequence/dp/nasdaq_model.pt" \
    --output_dir "experiments/multi_sequence/dp/nasdaq_samples" \
    --use_dp \

echo "Multi-sequence DP experiments completed."
echo "All DP experiments completed successfully!"
#!/bin/bash

# Create output directories
mkdir -p experiments/single_sequence/non_dp
mkdir -p experiments/multi_sequence/non_dp

# Function to run experiment with cleanup
run_experiment() {
    echo "Running experiment with arguments: $@"
    python Model_Code/timeautodiff_run.py "$@"
    echo "Experiment completed."
    sleep 5  # Give time for GPU memory to fully clear
}

# Single-sequence datasets
echo "Running single-sequence experiments..."

# AirQuality
run_experiment \
    --dataset "Dataset/Single-Sequence/AirQuality.csv" \
    --save_path "experiments/single_sequence/non_dp/airquality_model.pt" \
    --output_dir "experiments/single_sequence/non_dp/airquality_samples"

# # Hurricane
# run_experiment \
#     --dataset "Dataset/Single-Sequence/Hurricane.csv" \
#     --save_path "experiments/single_sequence/non_dp/hurricane_model.pt" \
#     --output_dir "experiments/single_sequence/non_dp/hurricane_samples"

# # Metro Traffic
# run_experiment \
#     --dataset "Dataset/Single-Sequence/Metro_Traffic.csv" \
#     --save_path "experiments/single_sequence/non_dp/metro_traffic_model.pt" \
#     --output_dir "experiments/single_sequence/non_dp/metro_traffic_samples"

# # Pollution Data
# run_experiment \
#     --dataset "Dataset/Single-Sequence/Pollution Data.csv" \
#     --save_path "experiments/single_sequence/non_dp/pollution_model.pt" \
#     --output_dir "experiments/single_sequence/non_dp/pollution_samples"

echo "Single-sequence experiments completed."

# # Multi-sequence datasets
# echo "Running multi-sequence experiments..."

# # Card Fraud
# run_experiment \
#     --dataset "Dataset/Multi-Sequence Data/card_fraud.csv" \
#     --multi_sequence \
#     --column_to_partition "User" \
#     --save_path "experiments/multi_sequence/non_dp/card_fraud_model.pt" \
#     --output_dir "experiments/multi_sequence/non_dp/card_fraud_samples"

# # NASDAQ100
# run_experiment \
#     --dataset "Dataset/Multi-Sequence Data/nasdaq100_2019.csv" \
#     --multi_sequence \
#     --column_to_partition "Symbol" \
#     --save_path "experiments/multi_sequence/non_dp/nasdaq_model.pt" \
#     --output_dir "experiments/multi_sequence/non_dp/nasdaq_samples"

# echo "Multi-sequence experiments completed."
echo "All experiments completed successfully!"
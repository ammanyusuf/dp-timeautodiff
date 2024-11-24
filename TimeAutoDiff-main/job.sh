#!/bin/bash

# Create output directories
mkdir -p experiments/single_sequence
mkdir -p experiments/multi_sequence

# Single-sequence datasets
echo "Running single-sequence experiments..."

# AirQuality
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/AirQuality.csv" \
    --save_path "experiments/single_sequence/airquality_model.pt" \
    --output_dir "experiments/single_sequence/airquality_samples"

# Hurricane
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/Hurricane.csv" \
    --pretrained_path "experiments/single_sequence/airquality_model.pt" \
    --save_path "experiments/single_sequence/hurricane_model.pt" \
    --output_dir "experiments/single_sequence/hurricane_samples"

# Metro Traffic
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/Metro_Traffic.csv" \
    --pretrained_path "experiments/single_sequence/hurricane_model.pt" \
    --save_path "experiments/single_sequence/metro_traffic_model.pt" \
    --output_dir "experiments/single_sequence/metro_traffic_samples"

# Pollution Data
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/Pollution Data.csv" \
    --pretrained_path "experiments/single_sequence/metro_traffic_model.pt" \
    --save_path "experiments/single_sequence/pollution_model.pt" \
    --output_dir "experiments/single_sequence/pollution_samples"

echo "Single-sequence experiments completed."

# Multi-sequence datasets
echo "Running multi-sequence experiments..."

# Card Fraud
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Multi-Sequence Data/card_fraud.csv" \
    --multi_sequence \
    --column_to_partition "User" \
    --save_path "experiments/multi_sequence/card_fraud_model.pt" \
    --output_dir "experiments/multi_sequence/card_fraud_samples"

# NASDAQ100
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Multi-Sequence Data/nasdaq100_2019.csv" \
    --multi_sequence \
    --column_to_partition "Symbol" \
    --pretrained_path "experiments/multi_sequence/card_fraud_model.pt" \
    --save_path "experiments/multi_sequence/nasdaq_model.pt" \
    --output_dir "experiments/multi_sequence/nasdaq_samples"

echo "Multi-sequence experiments completed."
echo "All experiments completed successfully!"
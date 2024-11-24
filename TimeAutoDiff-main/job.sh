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

# ETTh
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/ETTh.csv" \
    --save_path "experiments/single_sequence/etth_model.pt" \
    --output_dir "experiments/single_sequence/etth_samples"

# Hurricane
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/Hurricane.csv" \
    --save_path "experiments/single_sequence/hurricane_model.pt" \
    --output_dir "experiments/single_sequence/hurricane_samples"

# Metro Traffic
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/Metro_Traffic.csv" \
    --save_path "experiments/single_sequence/metro_traffic_model.pt" \
    --output_dir "experiments/single_sequence/metro_traffic_samples"

# Pollution Data
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/Pollution Data.csv" \
    --save_path "experiments/single_sequence/pollution_model.pt" \
    --output_dir "experiments/single_sequence/pollution_samples"

# AI4I 2020
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/ai4i2020.csv" \
    --save_path "experiments/single_sequence/ai4i_model.pt" \
    --output_dir "experiments/single_sequence/ai4i_samples"

# COVID-19
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/covid-19.csv" \
    --save_path "experiments/single_sequence/covid19_model.pt" \
    --output_dir "experiments/single_sequence/covid19_samples"

# Energy Data
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/energy_data.csv" \
    --save_path "experiments/single_sequence/energy_model.pt" \
    --output_dir "experiments/single_sequence/energy_samples"

# Stock Data
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Single-Sequence/stock_data.csv" \
    --save_path "experiments/single_sequence/stock_model.pt" \
    --output_dir "experiments/single_sequence/stock_samples"

echo "Single-sequence experiments completed."

# Multi-sequence datasets
echo "Running multi-sequence experiments..."

# Ad Data
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Multi-Sequence Data/Ad_Data.csv" \
    --multi_sequence \
    --column_to_partition "Campaign" \
    --save_path "experiments/multi_sequence/ad_data_model.pt" \
    --output_dir "experiments/multi_sequence/ad_data_samples"

# Card Fraud
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Multi-Sequence Data/card_fraud.csv" \
    --multi_sequence \
    --column_to_partition "card_id" \
    --save_path "experiments/multi_sequence/card_fraud_model.pt" \
    --output_dir "experiments/multi_sequence/card_fraud_samples"

# NASDAQ100
python Model_Code/timeautodiff_run.py \
    --dataset "Dataset/Multi-Sequence Data/nasdaq100_2019.csv" \
    --multi_sequence \
    --column_to_partition "Symbol" \
    --save_path "experiments/multi_sequence/nasdaq_model.pt" \
    --output_dir "experiments/multi_sequence/nasdaq_samples"

echo "Multi-sequence experiments completed."
echo "All experiments completed successfully!"
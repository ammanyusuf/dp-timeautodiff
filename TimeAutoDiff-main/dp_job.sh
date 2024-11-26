#!/bin/bash

# Define arrays for datasets and epsilon values
declare -a SINGLE_SEQUENCE_DATASETS=(
    "AirQuality"
    "Hurricane"
    "Metro_Traffic"
    "Pollution Data"
)

declare -a MULTI_SEQUENCE_DATASETS=(
    "card_fraud:User"
    "nasdaq100_2019:Symbol"
)

declare -a EPSILON_VALUES=(
    "0.1"
    "0.5"
    "1.0"
    "2.0"
    "5.0"
)

# Create output directories
for epsilon in "${EPSILON_VALUES[@]}"; do
    mkdir -p "experiments/single_sequence/dp_${epsilon}"
    mkdir -p "experiments/multi_sequence/dp_${epsilon}"
done

# Function to run experiment with cleanup
run_dp_experiment() {
    echo "Running DP experiment with arguments: $@"
    python Model_Code/timeautodiff_run.py "$@"
    echo "DP experiment completed."
    sleep 5  # Give time for GPU memory to fully clear
}

# Function to format dataset name for file paths
format_dataset_name() {
    echo "$1" | tr ' ' '_' | tr '[:upper:]' '[:lower:]'
}

echo "Running single-sequence DP experiments..."

# Process single-sequence datasets
for dataset in "${SINGLE_SEQUENCE_DATASETS[@]}"; do
    echo "Running ${dataset}"
    formatted_name=$(format_dataset_name "$dataset")
    
    for epsilon in "${EPSILON_VALUES[@]}"; do
        run_dp_experiment \
            --dataset "Dataset/Single-Sequence/${dataset}.csv" \
            --save_path "experiments/single_sequence/dp_${epsilon}/${formatted_name}_model.pt" \
            --output_dir "experiments/single_sequence/dp_${epsilon}" \
            --use_dp
    done
done

echo "Single-sequence DP experiments completed."
echo "Running multi-sequence DP experiments..."

# Process multi-sequence datasets
for dataset_info in "${MULTI_SEQUENCE_DATASETS[@]}"; do
    # Split dataset info into name and partition column
    IFS=':' read -r dataset partition_col <<< "$dataset_info"
    echo "Running ${dataset}"
    formatted_name=$(format_dataset_name "$dataset")
    
    for epsilon in "${EPSILON_VALUES[@]}"; do
        run_dp_experiment \
            --dataset "Dataset/Multi-Sequence Data/${dataset}.csv" \
            --multi_sequence \
            --column_to_partition "${partition_col}" \
            --save_path "experiments/multi_sequence/dp_${epsilon}/${formatted_name}_model.pt" \
            --output_dir "experiments/multi_sequence/dp_${epsilon}" \
            --use_dp
    done
done

echo "Multi-sequence DP experiments completed."
echo "All DP experiments completed successfully!"
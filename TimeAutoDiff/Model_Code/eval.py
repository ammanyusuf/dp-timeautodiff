import numpy as np
import pandas as pd
import torch
import os
import time
import random
import logging
import argparse
import matplotlib.pyplot as plt
import timeautoencoder as tae
import timediffusion as tdf
import timediffusion_cond_label as ctdf
import DP as dp
import process_edited as pce

# from Evaluation_Metrics import mt, pdm, correl
from Evaluation_Metrics import Metrics as mt
from Evaluation_Metrics import predictive_metrics as pdm
from Evaluation_Metrics import correl


def compute_utility_metrics(real_data, synth_data, real_df, synth_df, iterations=2000, n_samples=10):
    """Compute utility metrics for synthetic data evaluation.
    
    Args:
        real_data: Original data tensor
        synth_data: Synthetic data tensor
        real_df: Original dataframe
        synth_df: Synthetic dataframe
        iterations: Number of iterations for discriminative metrics
        n_samples: Number of times to compute each metric
        
    Returns:
        Dictionary containing mean and std of each metric
    """
    result_disc = []
    result_pred = []
    result_tmp = []
    result_corr = []
    

    for _ in range(n_samples):
        random_integers = [random.randint(0, len(real_df)-1) for _ in range(2000)]
        
        # Discriminative score
        disc_score = mt.discriminative_score_metrics(real_data, synth_data, iterations)
        
        # Predictive score
        pred_score = pdm.predictive_score_metrics(real_data, synth_data, 5)
        
        # Temporal score
        temp_score = mt.temp_disc_score(real_data, synth_data, iterations)
        
        # Correlation score
        corr_score = correl.final_correlation(
            real_df.iloc[random_integers,:],
            synth_df.iloc[random_integers,:]
        )
        
        result_disc.append(disc_score)
        result_pred.append(pred_score)
        result_tmp.append(temp_score)
        result_corr.append(corr_score)
    
    metrics = {
        'discriminative': {
            'mean': float(np.mean(result_disc)),
            'std': float(np.std(result_disc))
        },
        'predictive': {
            'mean': float(np.mean(result_pred)),
            'std': float(np.std(result_pred))
        },
        'temporal': {
            'mean': float(np.mean(result_tmp)),
            'std': float(np.std(result_tmp))
        },
        'correlation': {
            'mean': float(np.mean(result_corr)),
            'std': float(np.std(result_corr))
        }
    }
    
    return metrics

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute utility metrics for synthetic data.")
    parser.add_argument('--real_data_path_csv', type=str, required=True, help="Path to the real data CSV file.")
    parser.add_argument('--synth_data_path_csv', type=str, required=True, help="Path to the synthetic data CSV file.")
    parser.add_argument('--real_data_path', type=str, required=True, help="Path to the real data pt file.")
    parser.add_argument('--synth_data_path', type=str, required=True, help="Path to the synthetic data pt file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the metrics results.")
    parser.add_argument('--iterations', type=int, default=2000, help="Number of iterations for discriminative metrics.")
    parser.add_argument('--n_samples', type=int, default=10, help="Number of samples to compute each metric.")
    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load the real and synthetic data
    print("Loading data...")
    real_data = torch.load(args.real_data_path)
    synth_data = torch.load(args.synth_data_path)
    real_df = pd.read_csv(args.real_data_path_csv)
    synth_df = pd.read_csv(args.synth_data_path_csv)

    # Compute utility metrics
    print("Computing utility metrics...")
    metrics = compute_utility_metrics(
        real_data=real_data,
        synth_data=synth_data,
        real_df=real_df,
        synth_df=synth_df,
        iterations=args.iterations,
        n_samples=args.n_samples
    )

    # Save metrics to a JSON file
    metrics_file_path = os.path.join(args.output_dir, "utility_metrics.json")
    print(f"Saving metrics to {metrics_file_path}...")
    with open(metrics_file_path, 'w') as f:
        import json
        json.dump(metrics, f, indent=4)

    print("Utility metrics computation completed.")

if __name__ == "__main__":
    main()
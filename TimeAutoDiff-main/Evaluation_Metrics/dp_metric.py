import torch
import numpy as np
from sklearn.metrics import accuracy_score
from .Metrics import discriminative_score_metrics, temp_disc_score

def compute_privacy_metrics(real_data, synth_data, epsilon, iterations=1000):
    """
    Compute privacy metrics for the generated synthetic data.
    
    Args:
        real_data: Original data tensor
        synth_data: Synthetic data tensor generated with DP
        epsilon: Privacy budget used
        iterations: Number of iterations for discriminative metrics
    
    Returns:
        Dictionary containing privacy metrics:
        - discriminative_score: How well a discriminator can distinguish real vs synthetic
        - temporal_privacy: Privacy score for temporal correlations
        - epsilon: Privacy budget used
        - relative_error: Average relative error between real and synthetic
    """
    
    # Basic discriminative score (real vs synthetic)
    disc_score = discriminative_score_metrics(real_data, synth_data, iterations)
    
    # Temporal privacy score
    temp_score = temp_disc_score(real_data, synth_data, iterations)
    
    # Compute relative error between distributions
    rel_error = torch.mean(torch.abs(real_data - synth_data) / (torch.abs(real_data) + 1e-6))
    
    metrics = {
        'discriminative_score': disc_score,
        'temporal_privacy': temp_score,
        'epsilon': epsilon,
        'relative_error': rel_error.item()
    }
    
    return metrics

def evaluate_privacy_utility_tradeoff(real_data, synth_data_dict, utility_scores):
    """
    Evaluate privacy-utility tradeoff across different epsilon values.
    
    Args:
        real_data: Original data tensor
        synth_data_dict: Dictionary mapping epsilon values to synthetic data
        utility_scores: Dictionary mapping epsilon values to utility scores
    
    Returns:
        Dictionary containing tradeoff metrics for each epsilon
    """
    tradeoff_results = {}
    
    for epsilon, synth_data in synth_data_dict.items():
        # Get privacy metrics
        privacy_metrics = compute_privacy_metrics(real_data, synth_data, epsilon)
        
        # Get corresponding utility score
        utility = utility_scores[epsilon]
        
        # Compute privacy-utility tradeoff score
        # Higher score means better tradeoff
        tradeoff_score = utility / (1 + privacy_metrics['discriminative_score'])
        
        tradeoff_results[epsilon] = {
            'privacy_metrics': privacy_metrics,
            'utility': utility,
            'tradeoff_score': tradeoff_score
        }
    
    return tradeoff_results
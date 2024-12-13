import argparse
import os
import torch
import pandas as pd
import numpy as np
import timeautoencoder as tae
import timediffusion_cond_label as ctdf
import timediffusion as tdf
import DP as dp
import process_edited as pce
import random
import matplotlib.pyplot as plt
import gc
# from Evaluation_Metrics import mt, pdm, correl
from Evaluation_Metrics import Metrics as mt
from Evaluation_Metrics import predictive_metrics as pdm
from Evaluation_Metrics import correl


def load_dataset(dataset_path, is_multi_sequence=False, column_to_partition=None):
    """Load and preprocess dataset."""
    print(f"Loading dataset from: {dataset_path}")
    real_df = pd.read_csv(dataset_path)

    # Handle date column if present
    if "date" in real_df.columns:
        real_df1 = real_df.drop("date", axis=1)
        real_df2 = real_df.copy()
    else:
        real_df1 = real_df.copy()
        real_df2 = real_df.copy()

    threshold = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_multi_sequence:
        if not column_to_partition:
            raise ValueError("Must specify column_to_partition for multi-sequence data")
        processed_data, time_info = dp.partition_multi_seq(
            real_df, threshold, column_to_partition
        )
    else:
        # Process single-sequence data
        processed_data = dp.splitData(real_df1, 24, threshold)
        time_info = dp.splitTimeData(real_df2, processed_data.shape[1]).to(device)

        # Convert to float32 and move to device
        # processed_data = processed_data.float().to(device)
        # time_info = time_info.float().to(device)

    return real_df, real_df1, processed_data, time_info

def start_vae_pretraining(real_df,
    processed_data,
    time_info,
    config,
    column_to_partition=None,
    save_path=None,
    load_path=None,):
   if column_to_partition:
        real_df = real_df.drop(column_to_partition, axis=1)
    tae.pre_train_vae(
    public_df=real_df,
    processed_data=processed_data,
    channels=64,
    hidden_size=200,
    num_layers=1,
    lr=1e-3,
    weight_decay=1e-6,
    n_epochs=10000,
    batch_size=64,
    threshold=threshold,
    emb_dim=128,
    time_dim=8,
    lat_dim=7,
    save_dir=save_dir,
    device='cuda')
    print('finished pre-training')



def train_ae(
    real_df,
    processed_data,
    time_info,
    config,
    column_to_partition=None,
    save_path=None,
    load_path=None,
):
    """Train or load autoencoder model."""
    # # Ensure data is on correct device
    if column_to_partition:
        real_df = real_df.drop(column_to_partition, axis=1)
    # Train autoencoder
    processed_data.to(config["device"])
    ae_results = tae.train_autoencoder(
        real_df,
        processed_data,
        channels=config["channels"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        threshold=config["threshold"],
        min_beta=config["min_beta"],
        max_beta=config["max_beta"],
        emb_dim=config["emb_dim"],
        time_dim=config["time_dim"],
        lat_dim=config["lat_dim"],
        device=config["device"],
        save_path=save_path,
        load_path=load_path,
    )

    # # Auto-encoder Training
    # n_epochs = 20000; eps = 1e-5
    # weight_decay = 1e-6 ; lr = 2e-4; hidden_size = 200; num_layers = 1; batch_size = 50
    # channels = 64; min_beta = 1e-5; max_beta = 0.1; emb_dim = 128; time_dim = 8
    # lat_dim = 7; seq_col = 'Symbol'; threshold = 1; device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # column_to_partition = 'Symbol'
    # real_df = real_df.drop(column_to_partition, axis=1)

    # ae_results = tae.train_autoencoder(real_df, processed_data, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, \
    #                         batch_size, threshold,  min_beta, max_beta, emb_dim, time_dim, lat_dim, device)

    return ae_results


def train_diffusion(latent_features, time_info, config, is_multi_sequence=False):
    """Train diffusion model."""
    if is_multi_sequence:
        # Use conditional diffusion for multi-sequence data
        num_classes = len(latent_features)
        diff_model = ctdf.train_diffusion(
            latent_features,
            time_info.to(config["device"]),
            hidden_dim=config["hidden_size"],
            num_layers=config["num_layers"],
            diffusion_steps=config["n_steps"],
            n_epochs=config["n_epochs"],
            num_classes=num_classes,
        )
    else:
        # Use standard diffusion for single-sequence data
        # time = time_info.to(config["device"])
        diff_model = tdf.train_diffusion(
            latent_features=latent_features,
            time_info=time_info,
            hidden_dim=config["hidden_size"],
            num_layers=config["num_layers"],
            diffusion_steps=config["n_steps"],
            n_epochs=config["n_epochs"],
        )

    return diff_model


def generate_samples(
    real_df,
    processed_data,
    ae_model,
    latent_features,
    time_info,
    diff_model,
    ae_config,
    diff_config,
    is_multi_sequence=False,
    label=None,
):
    """Generate samples using trained models."""

    if is_multi_sequence:
        # Generate samples with conditional diffusion
        _, T, _ = processed_data.shape
        lbl = [label]
        latent_features = latent_features[lbl, :, :]
        time_info = time_info[lbl, :, :]

        N, _, _ = latent_features.shape
        t_grid = torch.linspace(0, 1, T).view(1, -1, 1).to(diff_config["device"])

        samples = ctdf.sample(
            t_grid.repeat(N, 1, 1),
            latent_features.detach().to(diff_config["device"]),
            diff_model,
            time_info,
            label,
            cfg_scale=3,
        )
    else:
        # Generate samples with standard diffusion
        N, T, _ = latent_features.shape
        t_grid = torch.linspace(0, 1, T).view(1, -1, 1).to(diff_config["device"])
        samples = tdf.sample(
            t_grid.repeat(N, 1, 1),
            latent_features.detach().to(diff_config["device"]),
            diff_model,
            time_info,
        )

    # Decode samples using autoencoder
    with torch.no_grad():
        gen_output = ae_model.decoder(samples.to(ae_config["device"]))

    # Post-process generated data
    data_size, seq_len, _ = latent_features.shape
    synth_data = pce.convert_to_tensor(
        real_df, gen_output, ae_config["threshold"], data_size, seq_len
    )
    _synth_data = pce.convert_to_table(real_df, synth_data, ae_config["threshold"])

    return _synth_data, samples


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


def run_dp_experiments(real_df, processed_data, time_info, ae_config, epsilons=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """Run experiments with different privacy budgets (epsilon) to analyze privacy-utility tradeoff.
    
    Args:
        real_df: Original dataframe
        processed_data: Processed tensor data
        time_info: Time information tensor
        ae_config: Autoencoder configuration dictionary
        epsilons: List of privacy budgets to test
    
    Returns:
        Dictionary containing results for each epsilon value
    """
    from Evaluation_Metrics.dp_metric import compute_privacy_metrics, evaluate_privacy_utility_tradeoff
    
    results = {}
    synth_data_dict = {}
    utility_scores = {}
    
    # Diffusion configuration
    diff_config = {
        "hidden_size": 250,
        "num_layers": 2,
        "n_epochs": 20000,
        "n_steps": 100,
        "device": ae_config["device"]
    }
    
    # Run experiments for each epsilon
    for epsilon in epsilons:
        print(f"\nRunning experiment with epsilon = {epsilon}")
        
        # Train autoencoder with current epsilon
        ae_results = tae.train_dp_autoencoder(
            real_df,
            processed_data,
            channels=ae_config["channels"],
            hidden_size=ae_config["hidden_size"],
            num_layers=ae_config["num_layers"],
            lr=ae_config["lr"],
            weight_decay=ae_config["weight_decay"],
            n_epochs=ae_config["n_epochs"],
            batch_size=ae_config["batch_size"],
            threshold=ae_config["threshold"],
            min_beta=ae_config["min_beta"],
            max_beta=ae_config["max_beta"],
            emb_dim=ae_config["emb_dim"],
            time_dim=ae_config["time_dim"],
            lat_dim=ae_config["lat_dim"],
            device=ae_config["device"],
            epsilon=epsilon
        )
        
        ae_model, latent_features = ae_results[0], ae_results[1]
        final_epsilon = ae_results[7]  # Get the actual epsilon spent
        
        # Train diffusion model
        print("Training diffusion model...")
        diff_model = train_diffusion(
            latent_features, 
            time_info, 
            diff_config,
            is_multi_sequence=False
        )
        
        # Generate synthetic data
        print("Generating synthetic data...")
        synth_data, latent_samples = generate_samples(
            real_df,
            processed_data,
            ae_model,
            latent_features,
            time_info,
            diff_model,
            ae_config,
            diff_config,
            is_multi_sequence=False
        )
        
        # Convert synthetic data to dataframe for utility metrics
        synth_df = pce.convert_to_table(real_df, synth_data, ae_config["threshold"])
        
        # Calculate utility metrics
        print("Computing utility metrics...")
        utility_metrics = compute_utility_metrics(
            processed_data,
            synth_data,
            real_df,
            synth_df
        )
        
        # Calculate privacy metrics
        print("Computing privacy metrics...")
        privacy_metrics = compute_privacy_metrics(
            processed_data,
            synth_data,
            final_epsilon
        )
        
        # Store results
        results[final_epsilon] = {
            'ae_model': ae_model,
            'diff_model': diff_model,
            'latent_features': latent_features,
            'synthetic_data': synth_data,
            'synthetic_df': synth_df,
            'utility_metrics': utility_metrics,
            'privacy_metrics': privacy_metrics,
            'target_epsilon': epsilon,
            'actual_epsilon': final_epsilon
        }
        
        # Store for tradeoff analysis
        synth_data_dict[final_epsilon] = synth_data
        utility_scores[final_epsilon] = np.mean([
            utility_metrics['discriminative']['mean'],
            utility_metrics['predictive']['mean'],
            utility_metrics['temporal']['mean'],
            utility_metrics['correlation']['mean']
        ])
    
    # Evaluate privacy-utility tradeoff
    tradeoff_results = evaluate_privacy_utility_tradeoff(
        processed_data,
        synth_data_dict,
        utility_scores
    )
    
    return results, tradeoff_results


def clear_gpu_memory():
    """Clear GPU memory between runs"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def run_original_experiment(args):
    """Original experiment implementation moved from main()"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Clear GPU memory at start
        clear_gpu_memory()

        # Autoencoder configuration
        ae_config = {
            "channels": 64,
            "hidden_size": 200,
            "num_layers": 1,
            "lr": 2e-4,
            "weight_decay": 1e-6,
            "n_epochs": 20000,
            "batch_size": 50,
            "threshold": 1,
            "min_beta": 1e-5,
            "max_beta": 0.1,
            "emb_dim": 128,
            "time_dim": 8,
            "lat_dim": 7,
            "device": device,
        }

        # Diffusion configuration
        diff_config = {
            "hidden_size": 250,
            "num_layers": 2,
            "n_epochs": 20000,
            "n_steps": 100,
            "device": device,
        }

        # Load dataset
        real_df, real_df1, processed_data, time_info = load_dataset(
            args.dataset,
            is_multi_sequence=args.multi_sequence,
            column_to_partition=args.column_to_partition,
        )

        # Train autoencoder
        print("Training/loading autoencoder...")

        ae_results = train_ae(
            real_df1,
            processed_data,
            time_info,
            ae_config,
            column_to_partition=args.column_to_partition,
            save_path=args.save_path,
            load_path=args.pretrained_path,
        )

        # Train diffusion model
        print("Training diffusion model...")
        ae_model, latent_features = ae_results[0], ae_results[1]
        diff_model = train_diffusion(
            latent_features, time_info, diff_config, is_multi_sequence=args.multi_sequence
        )

        # Generate samples
        print("Generating samples...")
        if args.multi_sequence and args.label is not None:
            # For conditional generation
            label = [args.label]
            latent_features = latent_features[label, :, :]
            time_info = time_info[label, :, :]

        synth_data, latent_samples = generate_samples(
            real_df1,
            processed_data,
            ae_model,
            latent_features,
            time_info,
            diff_model,
            ae_config,
            diff_config,
            is_multi_sequence=args.multi_sequence,
            label=args.label if args.multi_sequence else None,
        )

        # Save results
        output_dir = os.path.join(args.output_dir, "non_dp")
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
        torch.save(synth_data, os.path.join(output_dir, f"{dataset_name}_synth.pt"))
        torch.save(processed_data, os.path.join(output_dir, f"{dataset_name}_real.pt"))
        torch.save(latent_samples, os.path.join(output_dir, f"{dataset_name}_latent.pt"))

        # Plot results
        if args.multi_sequence:
            if args.label is not None:
                plot_multi_sequence_results(
                    real_df1, 
                    synth_data, 
                    processed_data, 
                    ae_config["threshold"],
                    args.label,
                    output_dir
                )
        else:
            plot_single_sequence_results(
                real_df1,
                synth_data,
                processed_data,
                ae_config["threshold"],
                output_dir
            )

        print(f"Generated samples saved to {output_dir}")
        
        # After training is complete, clear memory
        if device == "cuda":
            del ae_model
            del diff_model
            clear_gpu_memory()

        return synth_data, processed_data, latent_samples

    except Exception as e:
        print(f"Error in experiment: {str(e)}")
        clear_gpu_memory()
        raise e


def run_dp_experiment(args):
    """Run experiment with differential privacy"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Clear GPU memory at start
        clear_gpu_memory()

        # Autoencoder configuration
        ae_config = {
            "channels": 64,
            "hidden_size": 200,
            "num_layers": 1,
            "lr": 2e-4,
            "weight_decay": 1e-6,
            "n_epochs": 20000,
            "batch_size": 50,
            "threshold": 1,
            "min_beta": 1e-5,
            "max_beta": 0.1,
            "emb_dim": 128,
            "time_dim": 8,
            "lat_dim": 7,
            "device": device,
        }

        # Load dataset
        real_df, real_df1, processed_data, time_info = load_dataset(
            args.dataset,
            is_multi_sequence=args.multi_sequence,
            column_to_partition=args.column_to_partition,
        )

        # Run DP experiments with different epsilon values
        results, tradeoff_results = run_dp_experiments(
            real_df1, 
            processed_data, 
            time_info, 
            ae_config,
            epsilons=[0.1, 0.5, 1.0, 2.0, 5.0]
        )

        # Save results for each epsilon
        output_dir = os.path.join(args.output_dir, "dp")
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

        for epsilon, result in results.items():
            epsilon_dir = os.path.join(output_dir, f"epsilon_{epsilon}")
            os.makedirs(epsilon_dir, exist_ok=True)
            
            torch.save(result['synthetic_data'], os.path.join(epsilon_dir, f"{dataset_name}_synth.pt"))
            torch.save(result['latent_features'], os.path.join(epsilon_dir, f"{dataset_name}_latent.pt"))
            
            # Save metrics
            metrics = {
                'utility_metrics': result['utility_metrics'],
                'target_epsilon': result['target_epsilon'],
                'actual_epsilon': result['actual_epsilon']
            }
            torch.save(metrics, os.path.join(epsilon_dir, f"{dataset_name}_metrics.pt"))
        
        # Save tradeoff results
        torch.save(tradeoff_results, os.path.join(output_dir, f"{dataset_name}_tradeoff.pt"))
        torch.save(processed_data, os.path.join(output_dir, f"{dataset_name}_real.pt"))

        print(f"Generated samples and metrics saved to {output_dir}")
        
        # After all experiments, clear memory
        if device == "cuda":
            clear_gpu_memory()

        return results, tradeoff_results

    except Exception as e:
        print(f"Error in DP experiment: {str(e)}")
        clear_gpu_memory()
        raise e


def plot_single_sequence_results(real_df, synth_data, real_data, threshold, output_dir):
    """Plot histograms comparing real and synthetic data for single sequence case.
    
    Args:
        real_df: Original dataframe
        synth_data: Generated synthetic data tensor
        real_data: Original data tensor
        threshold: Threshold for parser
        output_dir: Directory to save plots
    """
    B, L, K = real_data.shape

    # Reshape data
    sd_reshaped = synth_data.reshape(B * L, K)
    pd_reshaped = real_data.reshape(B * L, K)

    # Get column names
    parser = pce.DataFrameParser().fit(real_df, threshold)
    col_name = parser.column_name()

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=K, figsize=(33.1, 23.4/5))

    for k in range(K):
        axes[k].hist(pd_reshaped[:, k].cpu().detach(), bins=50, color='blue', alpha=0.5, label='Real')
        axes[k].hist(sd_reshaped[:, k].cpu().detach(), bins=50, color='red', alpha=0.5, label='Synthetic')

        # Adding legends
        axes[k].legend()
        axes[k].set_title(col_name[k], fontsize=15)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'feature_comparison.png')
    plt.savefig(plot_path, dpi=500)
    plt.close()
    
    print(f"Feature comparison plot saved to {plot_path}")


def plot_multi_sequence_results(real_df, synth_data, real_data, threshold, label, output_dir):
    """Plot histograms comparing real and synthetic data for multi-sequence case.
    
    Args:
        real_df: Original dataframe
        synth_data: Generated synthetic data tensor
        real_data: Original data tensor
        threshold: Threshold for parser
        label: Label for the sequence
        output_dir: Directory to save plots
    """
    # Convert real data to table format
    _real_data = pce.convert_to_table(real_df, real_data[label,:,:], threshold)

    B, L, K = synth_data.shape

    # Reshape data
    sd_reshaped = synth_data.reshape(B * L, K)
    pd_reshaped = _real_data.reshape(B * L, K)

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=K, figsize=(20, 3))
    column_names = real_df.columns.tolist()

    for k in range(K):
        title = column_names[k]
        
        axes[k].hist(pd_reshaped[:,k].cpu().detach(), bins=50, color='blue', alpha=0.5, label='Real')
        axes[k].hist(sd_reshaped[:,k].cpu().detach(), bins=50, color='red', alpha=0.5, label='Synthetic')
        
        axes[k].legend()
        axes[k].set_title(title)

    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'feature_comparison_label_{label}.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Feature comparison plot for label {label} saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="TimeAutoDiff Training and Generation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--multi_sequence", action="store_true", help="Use multi-sequence mode"
    )
    parser.add_argument(
        "--column_to_partition",
        type=str,
        help="Column to partition for multi-sequence data",
    )
    parser.add_argument(
        "--pretrained_path", type=str, help="Path to pretrained autoencoder"
    )
    parser.add_argument("--save_path", type=str, help="Path to save trained model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_samples",
        help="Directory to save generated samples",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None, help="Number of samples to generate"
    )
    parser.add_argument(
        "--label",
        type=int,
        default=None,
        help="Label for conditional generation (multi-sequence only)",
    )
    parser.add_argument(
        "--use_dp",
        action="store_true",
        help="Use differential privacy training"
    )

    args = parser.parse_args()

    try:
        if args.use_dp:
            results, tradeoff_results = run_dp_experiment(args)
        else:
            synth_data, processed_data, latent_samples = run_original_experiment(args)
    finally:
        # Always clear GPU memory at the end
        clear_gpu_memory()

if __name__ == "__main__":
    main()

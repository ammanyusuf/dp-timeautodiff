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


def start_vae_pretraining(processed_data, time_info, real_df, real_df1, threshold, save_dir):
    tae.pre_train_vae(
    public_df=real_df1,
    time_info=time_info,
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

def start_training(processed_data, time_info, real_df, real_df1, threshold, output_directory, epsilon, n_epochs):
    device = 'cuda'; 
    weight_decay = 1e-6 ; lr = 2e-4; hidden_size = 200; num_layers = 1; batch_size = 50
    channels = 64; min_beta = 1e-5; max_beta = 0.1; emb_dim = 128; time_dim = 8
    lat_dim = 7; seq_col = 'Symbol'
    #real_df1 = real_df1.drop(column_to_partition, axis=1)

    # Call fine-tune autoencoder with pre-trained weights
    # ae, latent_features, losses

    # ds = tae.fine_tune_autoencoder(real_df1, processed_data, pretrained_weights_dir, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, \
    #                         batch_size, threshold,  min_beta, max_beta, emb_dim, time_dim, lat_dim, device)

    # ds = tae.fine_tune_autoencoder_dp(real_df1, processed_data, pretrained_weights_dir, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, \
    #                         batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device, epsilon)
    ds = tae.train_autoencoder_with_dp(real_df1, processed_data, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, \
                             batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device, epsilon)

    # ds = tae.train_autoencoder(real_df1, processed_data, channels, hidden_size, num_layers, lr, weight_decay,
    #                        n_epochs, batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device)
    ae = ds[0]
    latent_features = ds[1]
    losses = ds[2]

    # Diffusion Training
    time = time_info.to(device)
    hidden_dim = 200; num_layers = 2; diffusion_steps = 100; num_classes = len(latent_features)
    diff = tdf.train_diffusion(latent_features, time, hidden_dim, num_layers, diffusion_steps, n_epochs)
    #diff = tdf.train_diffusion(latent_features, time_info.to(device), hidden_dim, num_layers, diffusion_steps, n_epochs, num_classes)

    latent_features = ds[1]; T = latent_features.shape[1]; time_duration = []
    N, _, _ = latent_features.shape
    t_grid = torch.linspace(0, 1, T).view(1, -1, 1).to(device) 

    samples = tdf.sample(t_grid.repeat(N, 1, 1), latent_features.detach().to(device), diff, time)  

    # Post-process the generated data 
    # gen_output = ds[0].decoder(samples.to(device))  # Apply decoder to generated latent vector
    gen_output = ds[0].decoder(samples.to(device))
    data_size, seq_len, _ = latent_features.shape
    synth_data = pce.convert_to_tensor(real_df1, gen_output, threshold, data_size, seq_len)
    _synth_data = pce.convert_to_table(real_df1, synth_data, threshold)
    torch.save(_synth_data, os.path.join(output_directory, f"synthetic_data.pt"))
    # Draw the plots for marginal of featueres : Real v.s. Synthetic
    _real_data = pce.convert_to_table(real_df1, processed_data, threshold)
    torch.save(_real_data, os.path.join(output_directory, f"real_data.pt"))
    # To see if you want to check latent vectors recovered well.
    B, L, K = latent_features.shape

    pd_reshaped = latent_features.reshape(B * L, K)
    sd_reshaped = samples.reshape(B * L, K)

    # To see if you want to check real-data are recovered well.
    B, L, K = _synth_data.shape

    sd_reshaped = _synth_data.reshape(B * L, K)
    pd_reshaped = _real_data.reshape(B * L, K)

    real_df = pd.DataFrame(pd_reshaped.numpy())
    synth_df = pd.DataFrame(sd_reshaped.numpy())
    # Save the real and synthetic dataframes
    real_df_path = os.path.join(output_directory, 'real_data.csv')
    synth_df_path = os.path.join(output_directory, 'synthetic_data.csv')

    real_df.to_csv(real_df_path, index=False)
    synth_df.to_csv(synth_df_path, index=False)

    print(f"Real data saved to {real_df_path}")
    print(f"Synthetic data saved to {synth_df_path}")

    import matplotlib.pyplot as plt
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
    plot_path = os.path.join(output_directory, 'feature_comparison.png')
    plt.savefig(plot_path, dpi=500)
    plt.close()
    
    print(f"Feature comparison plot saved to {plot_path}")





def compute_utility_metrics(real_df, synth_df, iterations=2000, n_samples=10):
    """Compute utility metrics for synthetic data evaluation.
    
    Args:
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
    
    real_data = torch.tensor(real_df.values).float()
    synth_data = torch.tensor(synth_df.values).float()

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




def main():
    print("Main function started")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, help="hurricane, air")
    parser.add_argument('--data', type=str, help="hurricane, air, others")
    parser.add_argument('--eps', type=float, required=True, help="Privacy epsilon value for differential privacy training")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training (default: 50)")
    args = parser.parse_args()
    args.pretrain = args.pretrain.lower() == 'true'
    print(f"Received data argument: {args.data}")
    print(f"Using epsilon value for privacy: {args.eps}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    main_start_time = time.time()  # Start time tracking
    train_dir = '/arc/project/st-mijungp-1/wangzn/'
    output_directory_base = '/scratch/st-mijungp-1/wangzn/TAD_output'
    output_directory = os.path.join(output_directory_base, f"eps_{args.eps}_epochs_{args.epochs}")
    save_dir = 'TimeAutoDiff/Models/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.pretrain == True:
        print('start pretraining...')
        if args.data == 'hurricane':
            file_path = os.path.join(train_dir,'dp-timeautodiff/TimeAutoDiff-main/Dataset/Single-Sequence/Hurricane.csv')
            print(file_path)
            real_df = pd.read_csv(file_path)
            real_df1 = real_df

            # Pre-processing Data
            threshold = 1
            processed_data = dp.splitData(real_df1, 24, threshold);
            time_info = dp.splitTimeData(real_df1, processed_data.shape[1]).to(device).float()
            save_dir = os.path.join(save_dir, 'hurricane')
            start_vae_pretraining(processed_data, time_info, real_df, real_df1, threshold, save_dir)
    else: 
        # save_dir = os.path.join(save_dir, 'hurricane')
        # print("The pre-trained model is saved in: " + save_dir)
        print('start training...')
        if args.data == 'hurricane':
            print(f"*****Dataset: hurricane") 
            file_path = os.path.join(train_dir,'dp-timeautodiff/TimeAutoDiff-main/Dataset/Single-Sequence/Hurricane.csv')
            # Read dataframe
            print(file_path)
            real_df = pd.read_csv(file_path)
            real_df1 = real_df

            # Pre-processing Data
            threshold = 1
            print(f"******Pre-processing Data: hurricane") 
            processed_data = dp.splitData(real_df1, 12, threshold)
            time_info = dp.splitTimeData(real_df1, processed_data.shape[1]).to(device)
            print(f"******Start training: hurricane") 
            output_dir = os.path.join(output_directory, 'hurricane')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            start_training(processed_data, time_info, real_df, real_df1, threshold, output_dir, args.eps, args.epochs)
            
        if args.data == 'air':
            print(f"*****Dataset: air") 
            file_path = os.path.join(train_dir,'dp-timeautodiff/TimeAutoDiff-main/Dataset/Single-Sequence/AirQuality.csv')
            # Read dataframe
            print(file_path)
            real_df = pd.read_csv(file_path)
            real_df1 = real_df

            # Pre-processing Data
            threshold = 1; device = 'cuda'
            print(f"******Pre-processing Data: air") 
            processed_data = dp.splitData(real_df1, 24, threshold)
            time_info = dp.splitTimeData(real_df1, processed_data.shape[1]).to(device)
            print(f"******Start training: air") 
            output_dir = os.path.join(output_directory, 'air')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            start_training(processed_data, time_info, real_df, real_df1, threshold, output_dir, args.eps, args.epochs)
            

        if args.data == 'traffic':
            #2013
            print(f"*****Dataset: traffic") 
            file_path = os.path.join(train_dir,'dp-timeautodiff/TimeAutoDiff-main/Dataset/Single-Sequence/Metro_Traffic.csv')
            # Read dataframe
            print(file_path)
            real_df = pd.read_csv(file_path)
            real_df1 = real_df.iloc[0:8575, :]

            # Pre-processing Data
            threshold = 1; device = 'cuda'
            print(f"******Pre-processing Data: traffic") 
            processed_data = dp.splitData(real_df1, 24, threshold)
            time_info = dp.splitTimeData(real_df1, processed_data.shape[1]).to(device)
            print(f"******Start training: traffic") 
            output_dir = os.path.join(output_directory, 'traffic')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            start_training(processed_data, time_info, real_df, real_df1, threshold, output_dir, args.eps, args.epochs)

        if args.data == 'covid':
            print(f"*****Dataset: covid") 
            file_path = os.path.join(train_dir,'TimeAutoDiff/Dataset/Single-Sequence/covid-19.csv')
            # Read dataframe
            print(file_path)
            real_df = pd.read_csv(file_path)
            real_df1 = real_df

            # Pre-processing Data
            threshold = 1; device = 'cuda'
            print(f"******Pre-processing Data: covid") 
            processed_data = dp.splitData(real_df1, 24, threshold)
            time_info = dp.splitTimeData(real_df1, processed_data.shape[1]).to(device)
            print(f"******Start training: covid") 
            output_dir = os.path.join(output_directory, 'covid')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            start_training(processed_data, time_info, real_df, real_df1, threshold, output_dir, args.eps, args.epochs)


        if args.data == 'pollution':
            #2010
            print(f"*****Dataset: pollution") 
            file_path = os.path.join(train_dir,'dp-timeautodiff/TimeAutoDiff-main/Dataset/Single-Sequence/Pollution.csv')
            # Read dataframe
            print(file_path)
            real_df = pd.read_csv(file_path)
            real_df1 = real_df.iloc[0:8762, :]

            # Pre-processing Data
            threshold = 1; device = 'cuda'
            print(f"******Pre-processing Data: pollution") 
            processed_data = dp.splitData(real_df1, 24, threshold)
            time_info = dp.splitTimeData(real_df1, processed_data.shape[1]).to(device)
            print(f"******Start training: pollution") 
            output_dir = os.path.join(output_directory, 'pollution')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            start_training(processed_data, time_info, real_df, real_df1, threshold, output_dir, args.eps, args.epochs)

    # End time tracking
    elapsed_time = time.time() - main_start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")  # Print the total runtime



if __name__ == '__main__':
    main()



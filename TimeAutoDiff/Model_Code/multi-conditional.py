import logging
import os
import matplotlib.pyplot as plt
import argparse
import time
import numpy as np
import timeautoencoder as tae
import timediffusion_cond_label as ctdf
import DP as dp
import pandas as pd
import torch
import os
import time
import process_edited as pce
import random

def start_vae_pretraining(processed_data, time_info, real_df, real_df1, column_to_partition, threshold, save_dir):
    real_df1 = real_df1.drop(column_to_partition, axis=1)
    tae.pre_train_vae(
    public_df=real_df1,
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


def start_training(processed_data, time_info, real_df, real_df1, column_to_partition, threshold, output_directory, save_file_name):
    device = 'cuda'; 
    # # Auto-encoder Training
    # n_epochs = 20000; eps = 1e-5
    # weight_decay = 1e-6 ; lr = 2e-4; hidden_size = 200; num_layers = 1; batch_size = 50
    # channels = 64; min_beta = 1e-5; max_beta = 0.1; emb_dim = 128; time_dim = 8
    # lat_dim = 7
    # real_df1 = real_df1.drop(column_to_partition, axis=1)

    # ds = tae.train_autoencoder(real_df1, processed_data, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, \
    #                         batch_size, threshold,  min_beta, max_beta, emb_dim, time_dim, lat_dim, device)


    pretrained_weights = {
    "encoder": "path/to/save_pretrained_models/pretrained_encoder.pth",
    "decoder": "path/to/save_pretrained_models/pretrained_decoder.pth"
    }

    # Pass these weights to the fine-tune function
    ae, latent_features, losses = tae.fine_tune_autoencoder(
        real_df1, processed_data, pretrained_weights, channels=64, hidden_size=200, num_layers=1, lr=1e-4, 
        weight_decay=1e-6, n_epochs=5000, batch_size=50, threshold=threshold, min_beta=1e-5, 
        max_beta=0.1, emb_dim=128, time_dim=8, lat_dim=7, device='cuda'
    )


    # Diffusion Training
    latent_features = ds[1];
    hidden_dim = 250; num_layers = 2; diffusion_steps = 100; n_epochs = 20000; num_classes = len(latent_features)
    diff = ctdf.train_diffusion(latent_features, time_info.to(device), hidden_dim, num_layers, diffusion_steps, n_epochs, num_classes)

    # Sampling time-series tabular data of 8th entity 
    N, T, _ = processed_data.shape; 

    _, time_info = dp.partition_multi_seq(real_df, threshold, column_to_partition);

    lbl = torch.arange(0, len(latent_features));  # Generate the entire Sequence  
    label = torch.repeat_interleave(lbl, T, dim=0).reshape(len(lbl),T)

    label=[8]; # 8th 
    latent_features = latent_features[label,:,:]
    time_info = time_info[label,:,:]

    N, _, _ = latent_features.shape
    t_grid = torch.linspace(0, 1, T).view(1, -1, 1).to(device) 

    samples = ctdf.sample(t_grid.repeat(N, 1, 1), latent_features.detach().to(device), diff, time_info, label, cfg_scale = 3)

    # Post-process the generated data 
    gen_output = ds[0].decoder(samples.to(device))  # Apply decoder to generated latent vector

    data_size, seq_len, _ = latent_features.shape
    synth_data = pce.convert_to_tensor(real_df1, gen_output, threshold, data_size, seq_len)
    _synth_data = pce.convert_to_table(real_df1, synth_data, threshold)



    # Draw the plots for marginal of featueres : Real v.s. Synthetic
    _real_data = pce.convert_to_table(real_df1, processed_data[label,:,:], threshold)

    # To see if you want to check real-data are recovered well.
    B, L, K = _synth_data.shape

    sd_reshaped = _synth_data.reshape(B * L, K)
    pd_reshaped = _real_data.reshape(B * L, K)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=K, figsize=(20, 3))
    column_names = real_df1.columns.tolist()

    for k in range(0,K):
        title = column_names[k]
        
        axes[k].hist(pd_reshaped[:,k].cpu().detach(), bins=50, color='blue', alpha=0.5, label='Real')
        axes[k].hist(sd_reshaped[:,k].cpu().detach(), bins=50, color='red', alpha=0.5, label='Synthetic')
        
        # Move the legend line inside the with statement
        axes[k].legend()
        axes[k].set_title(title)

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, save_file_name))


def main():
    print("Main function started")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=bool, help="epileptic, others")
    parser.add_argument('--data', type=str, help="epileptic, others")
    args = parser.parse_args()
    print(f"Received data argument: {args.data}")

    main_start_time = time.time()  # Start time tracking
    train_dir = '/arc/project/st-mijungp-1/wangzn/'
    output_directory = '/scratch/st-mijungp-1/wangzn/TAD_output'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.pretrain == True:
        print('start pretraining...')
        if args.data == 'nasdaq':
            file_path = os.path.join(train_dir,'TimeAutoDiff/Dataset/Multi-Sequence/nasdaq100_2019.csv')
            print(file_path)
            real_df = pd.read_csv(file_path)
            real_df1 = real_df.drop('date', axis=1)

            # Pre-processing Data
            threshold = 1; column_to_partition = 'Symbol'
            processed_data, time_info = dp.partition_multi_seq(real_df, threshold, column_to_partition);
            save_dir = os.path.join(train_dir,'TimeAutoDiff/Models/nasdaq_pretrained')
            start_vae_pretraining(processed_data, time_info, real_df, real_df1, column_to_partition, threshold, save_dir)


    else: 
        if args.data == 'epileptic':
            print(f"*****Dataset: epi") 
            file_path = os.path.join(train_dir,'Epileptic/data_unpivoted.csv')
            # Read dataframe
            print(file_path)
            real_df = pd.read_csv(file_path)
            real_df1 = real_df.drop(['timestamp','time_index', 'second_within_window', 'time'], axis=1)

            # Pre-processing Data
            threshold = 1; column_to_partition = 'id'
            print(f"******Pre-processing Data: epi") 
            processed_data, time_info = dp.partition_multi_seq_sec(real_df, threshold, column_to_partition)
            print(f"******Start training: epi") 
            save_file_name = 'epi.png'
            start_training(processed_data, time_info, real_df, real_df1, column_to_partition, threshold, output_directory, save_file_name)
            
        if args.data == 'nasdaq':
            file_path = os.path.join(train_dir,'TimeAutoDiff/Dataset/Multi-Sequence/nasdaq100_2019.csv')

            print(file_path)
            real_df = pd.read_csv(file_path)
            real_df1 = real_df.drop('date', axis=1)

            # Pre-processing Data
            threshold = 1; column_to_partition = 'Symbol'
            processed_data, time_info = dp.partition_multi_seq(real_df, threshold, column_to_partition);

            save_file_name = 'nasdaq.png'
            start_training(processed_data, time_info, real_df, real_df1, column_to_partition, threshold, output_directory, save_file_name)

    # End time tracking
    elapsed_time = time.time() - main_start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")  # Print the total runtime



if __name__ == '__main__':
    main()
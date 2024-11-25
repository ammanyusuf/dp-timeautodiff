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

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    torch.save(synth_data, os.path.join(args.output_dir, f"{dataset_name}_synth.pt"))
    torch.save(processed_data, os.path.join(args.output_dir, f"{dataset_name}_real.pt"))
    torch.save(
        latent_samples, os.path.join(args.output_dir, f"{dataset_name}_latent.pt")
    )

    print(f"Generated samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()

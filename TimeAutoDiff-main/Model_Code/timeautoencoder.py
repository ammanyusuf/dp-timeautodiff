import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tqdm.notebook
import gc
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import process_edited as pce
from torch.optim import Adam
import DP as dp
import math
from rich.progress import Progress
import torch.nn.utils.clip_grad as clip_grad
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
import queue
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


device = "cuda" if torch.cuda.is_available() else "cpu"


################################################################################################################
def compute_sine_cosine(v, num_terms):
    num_terms = torch.tensor(num_terms).to(device)
    v = v.to(device)

    # Compute the angles for all terms
    angles = (
        2 ** torch.arange(num_terms).float().to(device)
        * torch.tensor(math.pi).to(device)
        * v.unsqueeze(-1)
    )

    # Compute sine and cosine values for all angles
    sine_values = torch.sin(angles)
    cosine_values = torch.cos(angles)

    # Reshape sine and cosine values for concatenation
    sine_values = sine_values.view(*sine_values.shape[:-2], -1)
    cosine_values = cosine_values.view(*cosine_values.shape[:-2], -1)

    # Concatenate sine and cosine values along the last dimension
    result = torch.cat((sine_values, cosine_values), dim=-1)

    return result


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.update_gate = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.new_gate = nn.Linear(hidden_size + hidden_size, hidden_size)
        
    def forward(self, x, h=None):
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)
            
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            
        x = self.input_layer(x)  # [batch_size, seq_len, hidden_size]
        
        output = torch.zeros(batch_size, seq_len, self.hidden_size, device=x.device)
        h_t = h
        
        for t in range(seq_len):
            x_t = x[:, t]  # [batch_size, hidden_size]
            
            combined = torch.cat([x_t, h_t], dim=1)  # [batch_size, 2*hidden_size]
            z_t = torch.sigmoid(self.update_gate(combined))
            r_t = torch.sigmoid(self.reset_gate(combined))
            combined_reset = torch.cat([x_t, r_t * h_t], dim=1)
            n_t = torch.tanh(self.new_gate(combined_reset))
            
            h_t = (1 - z_t) * n_t + z_t * h_t
            output[:, t] = h_t
        
        return output, h_t.unsqueeze(0)


################################################################################################################
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.RNN = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, d_last_states = self.RNN(x)
        y_hat_logit = self.fc(d_last_states[-1])
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat


################################################################################################################
class Embedding_data(nn.Module):
    def __init__(self, input_size, emb_dim, n_bins, n_cats, n_nums, cards):
        super().__init__()

        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards

        self.n_disc = self.n_bins + self.n_cats
        self.num_categorical_list = [2] * self.n_bins + self.cards

        if self.n_disc != 0:
            # Create a list to store individual embeddings
            self.embeddings_list = nn.ModuleList()

            # Create individual embeddings for each variable
            for num_categories in self.num_categorical_list:
                embedding = nn.Embedding(num_categories, emb_dim)
                self.embeddings_list.append(embedding)

        if self.n_nums != 0:
            self.mlp_nums = nn.Sequential(
                nn.Linear(
                    16 * n_nums, 16 * n_nums
                ),  # this should be 16 * n_nums, 16 * n_nums
                nn.SiLU(),
                nn.Linear(16 * n_nums, 16 * n_nums),
            )

        self.mlp_output = nn.Sequential(
            nn.Linear(
                emb_dim * self.n_disc + 16 * n_nums, emb_dim
            ),  # this should be 16 * n_nums, 16 * n_nums
            nn.ReLU(),
            nn.Linear(emb_dim, input_size),
        )

    def forward(self, x):
        x_disc = x[:, :, 0 : self.n_disc].long().to(device)
        x_nums = x[:, :, self.n_disc : self.n_disc + self.n_nums].to(device)

        x_emb = torch.Tensor().to(device)

        # Binary + Discrete Variables
        if self.n_disc != 0:
            variable_embeddings = [
                embedding(x_disc[:, :, i])
                for i, embedding in enumerate(self.embeddings_list)
            ]
            x_disc_emb = torch.cat(variable_embeddings, dim=2)
            x_emb = x_disc_emb

        # Numerical Variables
        if self.n_nums != 0:
            x_nums = compute_sine_cosine(x_nums, num_terms=8)
            x_nums_emb = self.mlp_nums(x_nums)
            x_emb = torch.cat([x_emb, x_nums_emb], dim=2)

        final_emb = self.mlp_output(x_emb)

        return final_emb


################################################################################################################
# def get_torch_trans(heads = 8, layers = 1, channels = 64):
#    encoder_layer = nn.TransformerEncoderLayer(d_model = channels, nhead = heads, dim_feedforward=64, activation = "gelu")
#    return nn.TransformerEncoder(encoder_layer, num_layers = layers)

# class Transformer_Block(nn.Module):
#    def __init__(self, channels):
#        super().__init__()
#        self.channels = channels

#        self.conv_layer1 = nn.Conv1d(1, self.channels, 1)
#        self.feature_layer = get_torch_trans(heads = 8, layers = 1, channels = self.channels)
#        self.conv_layer2 = nn.Conv1d(self.channels, 1, 1)

#    def forward_feature(self, y, base_shape):
#        B, channels, L, K = base_shape
#        if K == 1:
#            return y.squeeze(1)
#        y = y.reshape(B, channels, L, K).permute(0, 2, 1, 3).reshape(B*L, channels, K)
#        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
#        y = y.reshape(B, L, channels, K).permute(0, 2, 1, 3)
#        return y

#    def forward(self, x):
#        x = x.unsqueeze(1)
#        B, input_channel, K, L = x.shape
#        base_shape = x.shape

#        x = x.reshape(B, input_channel, K*L)

#        conv_x = self.conv_layer1(x).reshape(B, self.channels, K, L)
#        x = self.forward_feature(conv_x, conv_x.shape)
#        x = self.conv_layer2(x.reshape(B, self.channels, K*L)).squeeze(1).reshape(B, K, L)

#        return x


################################################################################################################
class DeapStack(nn.Module):
    def __init__(
        self,
        channels,
        batch_size,
        seq_len,
        n_bins,
        n_cats,
        n_nums,
        cards,
        input_size,
        hidden_size,
        num_layers,
        cat_emb_dim,
        time_dim,
        lat_dim,
    ):
        super().__init__()
        self.Emb = Embedding_data(
            input_size, cat_emb_dim, n_bins, n_cats, n_nums, cards
        )
        self.time_encode = nn.Sequential(
            nn.Linear(time_dim, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
        )

        self.encoder_mu = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_logvar = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fc_mu = nn.Linear(hidden_size, lat_dim)
        self.fc_logvar = nn.Linear(hidden_size, lat_dim)

        # self.cont_normed = nn.LayerNorm((seq_len, n_nums))
        # self.decoder_Transformer = Transformer_Block(channels)
        # self.decoder_rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(lat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.channels = channels
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.disc = self.n_bins + self.n_cats
        self.sigmoid = torch.nn.Sigmoid()

        self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins else None
        self.cats_linears = (
            nn.ModuleList([nn.Linear(hidden_size, card) for card in cards])
            if n_cats
            else None
        )
        self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums else None

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, x):
        x = self.Emb(x)
        # x = self.encoder_Transformer(x)
        # x = x + self.time_encode(time_info)

        mu_z, _ = self.encoder_mu(x)
        logvar_z, _ = self.encoder_logvar(x)

        mu_z = self.fc_mu(mu_z)
        logvar_z = self.fc_logvar(logvar_z)
        emb = self.reparametrize(mu_z, logvar_z)

        return emb, mu_z, logvar_z

    def decoder(self, latent_feature):
        decoded_outputs = dict()
        latent_feature = self.decoder_mlp(latent_feature)

        B, L, K = latent_feature.shape

        if self.bins_linear:
            decoded_outputs["bins"] = self.bins_linear(latent_feature)

        if self.cats_linears:
            decoded_outputs["cats"] = [
                linear(latent_feature) for linear in self.cats_linears
            ]

        if self.nums_linear:
            decoded_outputs["nums"] = self.sigmoid(self.nums_linear(latent_feature))

        return decoded_outputs

    def forward(self, x):
        emb, mu_z, logvar_z = self.encoder(x)
        outputs = self.decoder(emb)
        return outputs, emb, mu_z, logvar_z

class DeapStack_DP(nn.Module):
    def __init__(
        self,
        channels,
        batch_size,
        seq_len,
        n_bins,
        n_cats,
        n_nums,
        cards,
        input_size,
        hidden_size,
        num_layers,
        cat_emb_dim,
        time_dim,
        lat_dim,
    ):
        super().__init__()
        self.Emb = Embedding_data(
            input_size, cat_emb_dim, n_bins, n_cats, n_nums, cards
        )
        # self.time_encode = nn.Sequential(
        #     nn.Linear(time_dim, input_size),
        #     nn.ReLU(),
        #     nn.Linear(input_size, input_size),
        # )

        self.encoder_mu = CustomGRU(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_logvar = CustomGRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fc_mu = nn.Linear(hidden_size, lat_dim)
        self.fc_logvar = nn.Linear(hidden_size, lat_dim)

        # self.cont_normed = nn.LayerNorm((seq_len, n_nums))
        # self.decoder_Transformer = Transformer_Block(channels)
        # self.decoder_rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(lat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.channels = channels
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.disc = self.n_bins + self.n_cats
        self.sigmoid = torch.nn.Sigmoid()

        self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins else None
        self.cats_linears = (
            nn.ModuleList([nn.Linear(hidden_size, card) for card in cards])
            if n_cats
            else None
        )
        self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums else None

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, x):
        x = self.Emb(x)
        # x = self.encoder_Transformer(x)
        # x = x + self.time_encode(time_info)

        mu_z, _ = self.encoder_mu(x)
        logvar_z, _ = self.encoder_logvar(x)

        mu_z = self.fc_mu(mu_z)
        logvar_z = self.fc_logvar(logvar_z)
        emb = self.reparametrize(mu_z, logvar_z)

        return emb, mu_z, logvar_z

    def decoder(self, latent_feature):
        decoded_outputs = dict()
        latent_feature = self.decoder_mlp(latent_feature)

        B, L, K = latent_feature.shape

        if self.bins_linear:
            decoded_outputs["bins"] = self.bins_linear(latent_feature)

        if self.cats_linears:
            decoded_outputs["cats"] = [
                linear(latent_feature) for linear in self.cats_linears
            ]

        if self.nums_linear:
            decoded_outputs["nums"] = self.sigmoid(self.nums_linear(latent_feature))

        return decoded_outputs

    def forward(self, x):
        emb, mu_z, logvar_z = self.encoder(x)
        outputs = self.decoder(emb)
        return outputs, emb, mu_z, logvar_z


def auto_loss(inputs, reconstruction, n_bins, n_nums, n_cats, beta, cards):
    """Calculating the loss for DAE network.
    BCE for masks and reconstruction of binary inputs.
    CE for categoricals.
    MSE for numericals.
    reconstruction loss is weighted average of mean reduction of loss per datatype.
    mask loss is mean reduced.
    final loss is weighted sum of reconstruction loss and mask loss.
    """
    B, L, K = inputs.shape

    bins = inputs[:, :, 0:n_bins]
    cats = inputs[:, :, n_bins : n_bins + n_cats].long()
    nums = inputs[:, :, n_bins + n_cats : n_bins + n_cats + n_nums]

    # reconstruction_losses = dict()
    disc_loss = 0
    num_loss = 0

    if "bins" in reconstruction:
        disc_loss += F.binary_cross_entropy_with_logits(reconstruction["bins"], bins)

    if "cats" in reconstruction:
        cats_losses = []
        for i in range(len(reconstruction["cats"])):
            cats_losses.append(
                F.cross_entropy(
                    reconstruction["cats"][i].reshape(B * L, cards[i]),
                    cats[:, :, i].unsqueeze(2).reshape(B * L, 1).squeeze(1),
                )
            )
        disc_loss += torch.stack(cats_losses).mean()

    if "nums" in reconstruction:
        num_loss = F.mse_loss(reconstruction["nums"], nums)

    # reconstruction_loss = torch.stack(list(reconstruction_losses.values())).mean()

    return disc_loss, num_loss


def save_autoencoder(model, optimizer, epoch, loss, save_path):
    """Save the autoencoder model and training state."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        save_path,
    )


def load_autoencoder(model, optimizer, load_path, device):
    """Load a pretrained autoencoder model."""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def train_autoencoder(
    real_df,
    processed_data,
    channels,
    hidden_size,
    num_layers,
    lr,
    weight_decay,
    n_epochs,
    batch_size,
    threshold,
    min_beta,
    max_beta,
    emb_dim,
    time_dim,
    lat_dim,
    device,
    save_path=None,
    load_path=None,
):
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()
    data = torch.tensor(data.astype("float32")).unsqueeze(0)

    datatype_info = parser.datatype_info()
    n_bins = datatype_info["n_bins"]
    n_cats = datatype_info["n_cats"]
    n_nums = datatype_info["n_nums"]
    cards = datatype_info["cards"]

    N, seq_len, input_size = processed_data.shape
    ae = DeapStack(
        channels,
        batch_size,
        seq_len,
        n_bins,
        n_cats,
        n_nums,
        cards,
        input_size,
        hidden_size,
        num_layers,
        emb_dim,
        time_dim,
        lat_dim,
    ).to(device)

    optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 0
    best_loss = float("inf")

    # Load pretrained model if specified
    if load_path is not None:
        start_epoch, best_loss = load_autoencoder(ae, optimizer_ae, load_path, device)
        print(f"Loaded pretrained model from epoch {start_epoch} with loss {best_loss}")

    inputs = processed_data.to(device)

    losses = []
    recons_loss = []
    KL_loss = []
    beta = max_beta

    lambd = 0.7
    best_train_loss = float("inf")
    all_indices = list(range(N))

    with Progress() as progress:
        training_task = progress.add_task("[red]Training...", total=n_epochs)

        for epoch in range(n_epochs):
            ######################### Train Auto-Encoder #########################
            batch_indices = random.sample(all_indices, batch_size)

            optimizer_ae.zero_grad()
            outputs, _, mu_z, logvar_z = ae(inputs[batch_indices, :, :])

            disc_loss, num_loss = auto_loss(
                inputs[batch_indices, :, :],
                outputs,
                n_bins,
                n_nums,
                n_cats,
                beta,
                cards,
            )
            temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
            loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

            loss_Auto = num_loss + disc_loss + beta * loss_kld
            loss_Auto.backward()
            optimizer_ae.step()
            progress.update(
                training_task,
                advance=1,
                description=f"Epoch {epoch}/{n_epochs} - Loss: {loss_Auto.item():.4f}",
            )

            if loss_Auto < best_train_loss:
                best_train_loss = loss_Auto
                patience = 0
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd

            # recons_loss.append(num_loss.item() + disc_loss.item())
            # KL_loss.append(loss_kld.item())

    output, latent_features, _, _ = ae(processed_data)

    # Save model if specified
    if save_path is not None:
        save_autoencoder(ae, optimizer_ae, epoch, loss_Auto, save_path)
        print(f"Saved model to {save_path}")

    return (ae, latent_features.detach(), output, losses, recons_loss, mu_z, logvar_z)


def train_dp_autoencoder(
    real_df,
    processed_data,
    channels,
    hidden_size,
    num_layers,
    lr,
    weight_decay,
    n_epochs,
    batch_size,
    threshold,
    min_beta,
    max_beta,
    emb_dim,
    time_dim,
    lat_dim,
    device,
    epsilon,
    delta=1e-5,
    max_grad_norm=1.0,
    save_path=None,
    load_path=None,
):
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()
    data = torch.tensor(data.astype("float32")).unsqueeze(0)

    datatype_info = parser.datatype_info()
    n_bins = datatype_info["n_bins"]
    n_cats = datatype_info["n_cats"]
    n_nums = datatype_info["n_nums"]
    cards = datatype_info["cards"]

    N, seq_len, input_size = processed_data.shape
    ae = DeapStack_DP(
        channels,
        batch_size,
        seq_len,
        n_bins,
        n_cats,
        n_nums,
        cards,
        input_size,
        hidden_size,
        num_layers,
        emb_dim,
        time_dim,
        lat_dim,
    ).to(device)

    ae = ModuleValidator.fix(ae)
    if not ModuleValidator.is_valid(ae):
        print(ModuleValidator.validate(ae, strict=False))
        raise ValueError("Autoencoder model is not compatible with DP training")

    optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    if load_path is not None:
        start_epoch, best_loss = load_autoencoder(ae, optimizer_ae, load_path, device)
        print(f"Loaded pretrained model from epoch {start_epoch} with loss {best_loss}")

    train_dataset = torch.utils.data.TensorDataset(processed_data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    privacy_engine = PrivacyEngine()
    
    ae, optimizer_ae, train_loader = privacy_engine.make_private_with_epsilon(
        module=ae,
        optimizer=optimizer_ae,
        data_loader=train_loader,
        epochs=n_epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    best_loss = float("inf")

    # inputs = processed_data.to(device)

    losses = []
    recons_loss = []
    KL_loss = []
    beta = max_beta

    # lambd = 0.7
    best_train_loss = float("inf")
    # all_indices = list(range(N))
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
    ) as progress:
        training_task = progress.add_task("Training Epochs...", total=n_epochs)
        batch_task = None
        for epoch in range(n_epochs):
            ae.train()
            epoch_loss = 0.0
            num_batches = 0
            # TODO: kinda hackyDynamically determine the total number of batches
            batch_count = 0
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=128,
                optimizer=optimizer_ae
            ) as memory_safe_data_loader:
                for _ in memory_safe_data_loader:
                    batch_count += 1

            # TODO: this is kinda hacky, and is not updating the correct batch 
            if batch_task is not None:
                progress.remove_task(batch_task)
            batch_task = progress.add_task(
                f"Epoch {epoch + 1}/{n_epochs}: Training Batches",
                total=batch_count
            )
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=128,
                optimizer=optimizer_ae
            ) as memory_safe_data_loader:
                for batch_idx, batch_data in enumerate(memory_safe_data_loader):                    
                    optimizer_ae.zero_grad()
                    batch_input = batch_data[0].to(device)
                    outputs, _, mu_z, logvar_z = ae(batch_input)

                    disc_loss, num_loss = auto_loss(
                        batch_input,
                        outputs,
                        n_bins,
                        n_nums,
                        n_cats,
                        beta,
                        cards,
                    )
                    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
                    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

                    loss_Auto = num_loss + disc_loss + beta * loss_kld
                    loss_Auto.backward()

                    optimizer_ae.step()
                    epoch_loss += loss_Auto.item()
                    num_batches += 1
                    progress.update(
                        batch_task,
                        advance=1,
                        description=f"Batch {batch_idx + 1}/{batch_count} - Loss: {loss_Auto.item():.4f}"
                    )
            epsilon = privacy_engine.get_epsilon(delta)
            progress.update(
                training_task,
                advance=1,
                description=f"Epoch {epoch + 1}/{n_epochs} - Avg Loss: {epoch_loss / num_batches:.4f} - ε: {epsilon:.2f}",
            )

            if loss_Auto < best_train_loss:
                best_train_loss = loss_Auto
                patience = 0
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd

            # losses.append(loss_Auto.item())
            # recons_loss.append(num_loss.item() + disc_loss.item())
            # KL_loss.append(loss_kld.item())

    output, latent_features, _, _ = ae(processed_data)

    # Save model if specified
    if save_path is not None:
        save_autoencoder(ae, optimizer_ae, epoch, loss_Auto, save_path)
        print(f"Saved model to {save_path}")

    return (ae, latent_features.detach(), output, losses, recons_loss, mu_z, logvar_z, epsilon)




def pre_train_vae(public_df, processed_data, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold, emb_dim, time_dim, lat_dim, save_dir, device):
    """
    Pre-trains a Variational Autoencoder (VAE) on a public dataset and saves the encoder/decoder weights.

    Args:
        public_df: Public dataset for pre-training.
        processed_data: Tensor representation of the public dataset.
        channels: Number of channels in the VAE architecture.
        hidden_size: Hidden size for GRU layers.
        num_layers: Number of GRU layers.
        lr: Learning rate.
        weight_decay: Weight decay for regularization.
        n_epochs: Number of training epochs.
        batch_size: Batch size.
        threshold: Threshold for preprocessing.
        emb_dim: Embedding dimension.
        time_dim: Time embedding dimension.
        lat_dim: Latent space dimension.
        save_dir: Directory to save the pre-trained models.
        device: Device to run training on ('cuda' or 'cpu').
    Returns:
        None
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Parse and preprocess the public dataset
    parser = pce.DataFrameParser().fit(public_df, threshold)
    data = parser.transform()
    data = torch.tensor(data.astype('float32')).unsqueeze(0)
        
    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']
    n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']
    cards = datatype_info['cards']
    
    N, seq_len, input_size = processed_data.shape
    ae = DeapStack(channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, input_size, hidden_size, num_layers, emb_dim, time_dim, lat_dim).to(device)
    
    optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    inputs = processed_data.to(device)
    beta = 0.1  # You can adjust this or keep it fixed for pre-training
    best_train_loss = float('inf')
    all_indices = list(range(N))
    
    with Progress() as progress:
        training_task = progress.add_task("[blue]Pre-Training VAE...", total=n_epochs)

        for epoch in range(n_epochs):
            # Sample a batch of data
            batch_indices = random.sample(all_indices, batch_size)
            batch_data = inputs[batch_indices, :, :]
            
            optimizer_ae.zero_grad()
            outputs, _, mu_z, logvar_z = ae(batch_data)
            
            # Compute losses
            disc_loss, num_loss = auto_loss(batch_data, outputs, n_bins, n_nums, n_cats, beta, cards)
            temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
            loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
            loss_Auto = num_loss + disc_loss + beta * loss_kld

            # Backpropagation and optimizer step
            loss_Auto.backward()
            optimizer_ae.step()

            progress.update(training_task, advance=1, description=f"Epoch {epoch}/{n_epochs} - Loss: {loss_Auto.item():.4f}")

            # Update best loss
            if loss_Auto < best_train_loss:
                best_train_loss = loss_Auto

    # Save pre-trained weights
    torch.save({
        'encoder_mu': ae.encoder_mu.state_dict(),
        'encoder_logvar': ae.encoder_logvar.state_dict(),
        'fc_mu': ae.fc_mu.state_dict(),
        'fc_logvar': ae.fc_logvar.state_dict(),
    }, os.path.join(save_dir, "pretrained_encoder.pth"))

    torch.save(ae.decoder_mlp.state_dict(), os.path.join(save_dir, "pretrained_decoder.pth"))
    print(f"Pre-trained VAE saved to {save_dir}")



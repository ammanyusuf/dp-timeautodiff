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
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################################################################################################
def compute_sine_cosine(v, num_terms):
    num_terms = torch.tensor(num_terms).to(device)
    v = v.to(device)

    # Compute the angles for all terms
    angles = 2**torch.arange(num_terms).float().to(device) * torch.tensor(math.pi).to(device) * v.unsqueeze(-1)

    # Compute sine and cosine values for all angles
    sine_values = torch.sin(angles)
    cosine_values = torch.cos(angles)

    # Reshape sine and cosine values for concatenation
    sine_values = sine_values.view(*sine_values.shape[:-2], -1)
    cosine_values = cosine_values.view(*cosine_values.shape[:-2], -1)

    # Concatenate sine and cosine values along the last dimension
    result = torch.cat((sine_values, cosine_values), dim=-1)

    return result
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################################################
class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        """
        A custom implementation of GRU suitable for the current setup.
        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units in the GRU.
            num_layers: Number of GRU layers.
            batch_first: If True, the input/output tensors are of shape (batch, seq, feature).
        """
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Input transformation layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # GRU gates
        self.update_gate = nn.ModuleList(
            [nn.Linear(hidden_size + hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.reset_gate = nn.ModuleList(
            [nn.Linear(hidden_size + hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.new_gate = nn.ModuleList(
            [nn.Linear(hidden_size + hidden_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x, h=None):
        """
        Forward pass for the custom GRU.
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) if batch_first=True.
            h: Initial hidden state of shape (num_layers, batch, hidden_size).
        Returns:
            output: Output features of the GRU for all timesteps.
            h_t: Final hidden state for each layer (num_layers, batch, hidden_size).
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)

        if h is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        x = self.input_layer(x)  # Transform input features
        h_t = h

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]  # Current timestep input
            h_next = []

            for layer in range(self.num_layers):
                h_prev = h_t[layer]
                combined = torch.cat([x_t, h_prev], dim=1)  # Combine input and previous hidden state

                z_t = torch.sigmoid(self.update_gate[layer](combined))  # Update gate
                r_t = torch.sigmoid(self.reset_gate[layer](combined))  # Reset gate
                n_t = torch.tanh(self.new_gate[layer](torch.cat([x_t, r_t * h_prev], dim=1)))  # New gate

                h_next_layer = (1 - z_t) * n_t + z_t * h_prev  # Compute new hidden state
                h_next.append(h_next_layer)
                x_t = h_next_layer  # Pass to the next layer

            h_t = torch.stack(h_next)  # Stack hidden states from all layers
            outputs.append(h_t[-1])  # Store the output from the last layer

        outputs = torch.stack(outputs, dim=1)  # Convert list to tensor
        return outputs, h_t
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
        self.num_categorical_list = [2]*self.n_bins + self.cards
        
        if self.n_disc != 0:
            # Create a list to store individual embeddings
            self.embeddings_list = nn.ModuleList()
            
            # Create individual embeddings for each variable
            for num_categories in self.num_categorical_list:
                embedding = nn.Embedding(num_categories, emb_dim)
                self.embeddings_list.append(embedding)
        
        if self.n_nums != 0:
            self.mlp_nums = nn.Sequential(nn.Linear(16 * n_nums, 16 * n_nums),  # this should be 16 * n_nums, 16 * n_nums
                                          nn.SiLU(),
                                          nn.Linear(16 * n_nums, 16 * n_nums))
            
        self.mlp_output = nn.Sequential(nn.Linear(emb_dim * self.n_disc + 16 * n_nums, emb_dim), # this should be 16 * n_nums, 16 * n_nums
                                       nn.ReLU(),
                                       nn.Linear(emb_dim, input_size))
        
    def forward(self, x):
        
        x_disc = x[:,:,0:self.n_disc].long().to(device)
        x_nums = x[:,:,self.n_disc:self.n_disc+self.n_nums].to(device)
        
        x_emb = torch.Tensor().to(device)
        
        # Binary + Discrete Variables
        if self.n_disc != 0:
            variable_embeddings = [embedding(x_disc[:,:,i]) for i, embedding in enumerate(self.embeddings_list)]
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
def get_torch_trans(heads = 8, layers = 1, channels = 64):
   encoder_layer = nn.TransformerEncoderLayer(d_model = channels, nhead = heads, dim_feedforward=64, activation = "gelu")
   return nn.TransformerEncoder(encoder_layer, num_layers = layers)

class Transformer_Block(nn.Module):
   def __init__(self, channels):
       super().__init__()
       self.channels = channels
        
       self.conv_layer1 = nn.Conv1d(1, self.channels, 1)
       self.feature_layer = get_torch_trans(heads = 8, layers = 1, channels = self.channels)
       self.conv_layer2 = nn.Conv1d(self.channels, 1, 1)
    
   def forward_feature(self, y, base_shape):
       B, channels, L, K = base_shape
       if K == 1:
           return y.squeeze(1)
       y = y.reshape(B, channels, L, K).permute(0, 2, 1, 3).reshape(B*L, channels, K)
       y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
       y = y.reshape(B, L, channels, K).permute(0, 2, 1, 3)
       return y
    
   def forward(self, x):
       x = x.unsqueeze(1)
       B, input_channel, K, L = x.shape
       base_shape = x.shape

       x = x.reshape(B, input_channel, K*L)       
        
       conv_x = self.conv_layer1(x).reshape(B, self.channels, K, L)
       x = self.forward_feature(conv_x, conv_x.shape)
       x = self.conv_layer2(x.reshape(B, self.channels, K*L)).squeeze(1).reshape(B, K, L)
        
       return x

################################################################################################################
# class DeapStack(nn.Module):
#     def __init__(self, channels, batch_size, seq_len, time_info, n_bins, n_cats, n_nums, cards, input_size, hidden_size, num_layers, cat_emb_dim, time_dim, lat_dim):
#         super().__init__()
#         self.time_info = time_info.to(device)  # Store time_info as a class attribute
#         print(f"Initializing DeapStack with input_size: {input_size}")
#         self.Emb = Embedding_data(input_size, cat_emb_dim, n_bins, n_cats, n_nums, cards)
#         self.time_encode = nn.Sequential(nn.Linear(time_dim, input_size),
#                                          nn.ReLU(),
#                                          nn.Linear(input_size, input_size))

# ############################### Transformer ########################################    
#         self.encoder_Transformer = Transformer_Block(channels)
# ############################### Transformer ########################################
#         self.encoder_mu = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.encoder_logvar = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
#         self.fc_mu = nn.Linear(hidden_size, lat_dim)
#         self.fc_logvar = nn.Linear(hidden_size, lat_dim)
# ############################### Transformer ########################################
#         self.cont_normed = nn.LayerNorm((seq_len, n_nums))
#         self.decoder_Transformer = Transformer_Block(channels)
#         self.decoder_rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
# ############################### Transformer ########################################
#         self.decoder_mlp = nn.Sequential(nn.Linear(lat_dim, hidden_size),
#                                          nn.ReLU(),
#                                          nn.Linear(hidden_size, hidden_size))
        
#         self.channels = channels
#         self.n_bins = n_bins
#         self.n_cats = n_cats
#         self.n_nums = n_nums
#         self.disc = self.n_bins + self.n_cats
#         self.sigmoid = torch.nn.Sigmoid ()
        
#         self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins else None
#         self.cats_linears = nn.ModuleList([nn.Linear(hidden_size, card) for card in cards]) if n_cats else None 
#         self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums else None

#     def reparametrize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def encoder(self, x):
#         print(f"Encoding input with shape: {x.shape}")
#         x = self.Emb(x)
# ############################### Transformer ########################################
#         # encoded_time_info = self.time_encode(self.time_info)
#         #     if encoded_time_info.shape != x.shape:
#         #         raise ValueError(f"Shape mismatch: x {x.shape}, encoded_time_info {encoded_time_info.shape}")
            
#         #     x = x + encoded_time_info
# ############################### Transformer ########################################
#         mu_z, _ = self.encoder_mu(x)
#         logvar_z, _ = self.encoder_logvar(x)
        
#         mu_z = self.fc_mu(mu_z); logvar_z = self.fc_logvar(logvar_z)
#         emb = self.reparametrize(mu_z, logvar_z)
        
#         return emb, mu_z, logvar_z

#     def decoder(self, latent_feature):
#         decoded_outputs = dict()
#         latent_feature = self.decoder_mlp(latent_feature)
        
#         B, L, K = latent_feature.shape
        
#         if self.bins_linear:
#             decoded_outputs['bins'] = self.bins_linear(latent_feature)

#         if self.cats_linears:
#             decoded_outputs['cats'] = [linear(latent_feature) for linear in self.cats_linears]

#         if self.nums_linear:
#             decoded_outputs['nums'] = self.sigmoid(self.nums_linear(latent_feature))

#         return decoded_outputs

#     def forward(self, x):
#         print(f"Forward pass with input shape: {x.shape}")
#         emb, mu_z, logvar_z = self.encoder(x)
#         outputs = self.decoder(emb)
#         return outputs, emb, mu_z, logvar_z
    


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
        # self.time_encode = nn.Sequential(
        #     nn.Linear(time_dim, input_size),
        #     nn.ReLU(),
        #     nn.Linear(input_size, input_size),
        # )
        # self.encoder_Transformer = Transformer_Block(channels)

        self.encoder_mu = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_logvar = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True
        )

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
    """ Calculating the loss for DAE network.
        BCE for masks and reconstruction of binary inputs.
        CE for categoricals.
        MSE for numericals.
        reconstruction loss is weighted average of mean reduction of loss per datatype.
        mask loss is mean reduced.
        final loss is weighted sum of reconstruction loss and mask loss.
    """
    B, L, K = inputs.shape

    bins = inputs[:,:,0:n_bins]
    cats = inputs[:,:,n_bins:n_bins+n_cats].long()
    nums = inputs[:,:,n_bins+n_cats:n_bins+n_cats+n_nums]

    #reconstruction_losses = dict()
    disc_loss = 0; num_loss = 0;
    
    if 'bins' in reconstruction:
        disc_loss += F.binary_cross_entropy_with_logits(reconstruction['bins'], bins)

    if 'cats' in reconstruction:
        cats_losses = []
        for i in range(len(reconstruction['cats'])):
            cats_losses.append(F.cross_entropy(reconstruction['cats'][i].reshape(B*L, cards[i]), \
                                               cats[:,:,i].unsqueeze(2).reshape(B*L, 1).squeeze(1)))
        disc_loss += torch.stack(cats_losses).mean()

    if 'nums' in reconstruction:
        num_loss = F.mse_loss(reconstruction['nums'], nums)

    #reconstruction_loss = torch.stack(list(reconstruction_losses.values())).mean()

    return disc_loss, num_loss



# def train_autoencoder(real_df, processed_data, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device):

#     parser = pce.DataFrameParser().fit(real_df, threshold)
#     data = parser.transform()
#     data = torch.tensor(data.astype('float32')).unsqueeze(0)
        
#     datatype_info = parser.datatype_info()
#     n_bins = datatype_info['n_bins']; n_cats = datatype_info['n_cats']
#     n_nums = datatype_info['n_nums']; cards = datatype_info['cards']
    
#     N, seq_len, input_size = processed_data.shape
#     ae = DeapStack(channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, input_size, hidden_size, num_layers, emb_dim, time_dim, lat_dim).to(device)
    
#     optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

#     inputs = processed_data.to(device)
        
#     losses = []
#     recons_loss = []
#     KL_loss = []
#     beta = max_beta
    
#     lambd = 0.7
#     best_train_loss = float('inf')
#     all_indices = list(range(N))
    
#     with Progress() as progress:
#         training_task = progress.add_task("[red]Training...", total=n_epochs)

#         for epoch in range(n_epochs):
#             ######################### Train Auto-Encoder #########################
#             batch_indices = random.sample(all_indices, batch_size)
    
#             optimizer_ae.zero_grad()
#             outputs, _, mu_z, logvar_z = ae(inputs[batch_indices,:,:])
            
#             disc_loss, num_loss = auto_loss(inputs[batch_indices,:,:], outputs, n_bins, n_nums, n_cats, beta, cards)
#             temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
#             loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
            
#             loss_Auto = num_loss + disc_loss + beta * loss_kld
#             loss_Auto.backward()
#             optimizer_ae.step() 
#             progress.update(training_task, advance=1, description=f"Epoch {epoch}/{n_epochs} - Loss: {loss_Auto.item():.4f}")
            
#             if loss_Auto < best_train_loss:
#                 best_train_loss = loss_Auto
#                 patience = 0
#             else:
#                 patience += 1
#                 if patience == 10:
#                     if beta > min_beta:
#                         beta = beta * lambd
            
#             #recons_loss.append(num_loss.item() + disc_loss.item())
#             #KL_loss.append(loss_kld.item())
    
#     output, latent_features, _, _ = ae(processed_data)
        
#     #return (ae, latent_features.detach(), output, losses, recons_loss, KL_loss)
#     return (ae, latent_features.detach(), output, losses, recons_loss, mu_z, logvar_z)

def train_autoencoder(
    real_df, processed_data, channels, hidden_size, num_layers, lr, weight_decay, n_epochs,
    batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device
):
    import time
    from torch.optim import Adam
    from rich.progress import Progress

    # Parse and preprocess data
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()
    data = torch.tensor(data.astype('float32')).unsqueeze(0).to(device)

    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']
    n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']
    cards = datatype_info['cards']

    N, seq_len, input_size = processed_data.shape
    processed_data = processed_data.to(device)  # Ensure processed_data is on device
    ae = DeapStack(
        channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards,
        input_size, hidden_size, num_layers, emb_dim, time_dim, lat_dim
    ).to(device)

    # Optimizer
    optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(processed_data),
        batch_size=batch_size,
        shuffle=True,
    )

    beta = max_beta
    losses = []

    # Start timing
    start_time = time.time()
    epoch_times = []

    with Progress() as progress:
        training_task = progress.add_task("[red]Training Autoencoder...", total=n_epochs)

        for epoch in range(n_epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch in train_loader:
                batch_data = batch[0].to(device)

                optimizer_ae.zero_grad()

                # Forward pass
                outputs, _, mu_z, logvar_z = ae(batch_data)

                # Compute losses
                disc_loss, num_loss = auto_loss(batch_data, outputs, n_bins, n_nums, n_cats, beta, cards)
                temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
                loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
                loss_Auto = num_loss + disc_loss + beta * loss_kld

                # Backward pass
                loss_Auto.backward()
                optimizer_ae.step()

                epoch_losses.append(loss_Auto.item())

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_epoch_loss)
            # Calculate elapsed time and remaining time
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            epoch_times.append(epoch_duration)
            avg_time_per_epoch = sum(epoch_times) / len(epoch_times)
            remaining_time = avg_time_per_epoch * (n_epochs - (epoch + 1))

            # Format the time
            formatted_remaining_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            formatted_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(epoch_end - start_time))

            # Print training status
            print(
                f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_epoch_loss:.4f} - "
                f"Elapsed: {formatted_elapsed_time} - Remaining: {formatted_remaining_time}",
                flush=True,
            )

            progress.update(training_task, advance=1, description=f"Epoch {epoch}/{n_epochs} - Loss: {avg_epoch_loss:.4f}")

    # Extract latent features
    output, latent_features, _, _ = ae(processed_data)
    print("Training completed.")

    return ae, latent_features.detach(), losses



from opacus import PrivacyEngine

import time

def train_autoencoder_with_dp(
    real_df, processed_data, channels, hidden_size, num_layers, lr, weight_decay, n_epochs,
    batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device, epsilon, delta=1e-5, max_grad_norm=1.0
):
    import os
    from torch.optim import Adam
    from opacus import PrivacyEngine
    from rich.progress import Progress
    import time

    # Parse and preprocess data
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()
    data = torch.tensor(data.astype('float32')).unsqueeze(0).to(device)

    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']
    n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']
    cards = datatype_info['cards']

    N, seq_len, input_size = processed_data.shape
    processed_data = processed_data.to(device)  # Ensure processed_data is on device
    ae = DeapStack(
        channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards,
        input_size, hidden_size, num_layers, emb_dim, time_dim, lat_dim
    ).to(device)

    # Replace GRU with CustomGRU
    ae.encoder_mu = CustomGRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
    ae.encoder_logvar = CustomGRU(input_size, hidden_size, num_layers, batch_first=True).to(device)

    optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    # Set up Privacy Engine
    privacy_engine = PrivacyEngine()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(processed_data),
        batch_size=batch_size,
        shuffle=True,
    )

    ae, optimizer_ae, train_loader = privacy_engine.make_private_with_epsilon(
        module=ae,
        optimizer=optimizer_ae,
        data_loader=train_loader,
        epochs=n_epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    beta = max_beta
    losses = []

    # Start timing
    start_time = time.time()
    epoch_times = []

    with Progress() as progress:
        training_task = progress.add_task("[red]Training with DP...", total=n_epochs)

        for epoch in range(n_epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch in train_loader:
                batch_data = batch[0].to(device)
                # print(f"Batch data device: {batch_data.device}")  # Debug

                optimizer_ae.zero_grad()

                # Debug model and input device
                # print(f"Model device: {next(ae.parameters()).device}")
                outputs, _, mu_z, logvar_z = ae(batch_data)

                disc_loss, num_loss = auto_loss(batch_data, outputs, n_bins, n_nums, n_cats, beta, cards)
                temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
                loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
                loss_Auto = num_loss + disc_loss + beta * loss_kld

                loss_Auto.backward()
                optimizer_ae.step()

                epoch_losses.append(loss_Auto.item())

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_epoch_loss)
            # Calculate elapsed time and remaining time
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            epoch_times.append(epoch_duration)
            avg_time_per_epoch = sum(epoch_times) / len(epoch_times)
            remaining_time = avg_time_per_epoch * (n_epochs - (epoch + 1))

            # Format the time
            formatted_remaining_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            formatted_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(epoch_end - start_time))

            # Print training status
            print(
                f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_epoch_loss:.4f} - "
                f"Elapsed: {formatted_elapsed_time} - Remaining: {formatted_remaining_time}",
                flush=True,
            )

            progress.update(training_task, advance=1, description=f"Epoch {epoch}/{n_epochs} - Loss: {avg_epoch_loss:.4f}")

    # Extract latent features
    output, latent_features, _, _ = ae(processed_data)
    print(f"Training completed with ε={privacy_engine.get_epsilon(delta):.2f}, δ={delta}")

    return ae, latent_features.detach(), losses


def pre_train_vae(public_df, time_info, processed_data, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold, emb_dim, time_dim, lat_dim, save_dir, device):
    """
    Pre-trains a Variational Autoencoder (VAE) on a public dataset and saves the encoder/decoder weights.

    Args:
        public_df: Public dataset for pre-training.
        processed_data: Tensor representation of the public dataset.
        channels: Number of channels in the VAE architecture.
        hidden_size: Hiddenx + self.time_encode size for GRU layers.
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
    #ae = DeapStack(channels, batch_size, seq_len, time_info, n_bins, n_cats, n_nums, cards, input_size, hidden_size, num_layers, emb_dim, time_dim, lat_dim).to(device)
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



# def fine_tune_autoencoder(
#     real_df, processed_data, pretrained_weights_dir, channels, hidden_size, num_layers, lr,
#     weight_decay, n_epochs, batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device
# ):
#     """
#     Fine-tunes a pre-trained Variational Autoencoder (VAE) on a private dataset.

#     Args:
#         real_df: Private dataset as a Pandas DataFrame.
#         processed_data: Tensor representation of the private dataset.
#         pretrained_weights_dir: Directory containing pre-trained weights.
#         channels: Number of channels in the VAE architecture.
#         hidden_size: Hidden size for GRU layers.
#         num_layers: Number of GRU layers.
#         lr: Learning rate.
#         weight_decay: Weight decay for regularization.
#         n_epochs: Number of training epochs.
#         batch_size: Batch size for training.
#         threshold: Threshold for preprocessing.
#         min_beta: Minimum beta value for KL divergence weighting.
#         max_beta: Maximum beta value for KL divergence weighting.
#         emb_dim: Embedding dimension.
#         time_dim: Time encoding dimension.
#         lat_dim: Latent space dimension.
#         device: Device to run training on ('cuda' or 'cpu').

#     Returns:
#         ae: Fine-tuned autoencoder model.
#         latent_features: Latent features extracted by the fine-tuned model.
#         losses: List of training losses for each epoch.
#     """
#     import os
#     from torch.optim import Adam
#     import random
#     from rich.progress import Progress

#     # Initialize the autoencoder model
#     datatype_info = pce.DataFrameParser().fit(real_df, threshold).datatype_info()
#     n_bins = datatype_info['n_bins']
#     n_cats = datatype_info['n_cats']
#     n_nums = datatype_info['n_nums']
#     cards = datatype_info['cards']

#     seq_len = processed_data.shape[1]
#     input_size = processed_data.shape[2]
#     print(f"Input size during fine-tuning: {input_size}")

#     # Initialize the autoencoder model
#     ae = DeapStack(
#         channels=channels, batch_size=batch_size, seq_len=seq_len, n_bins=n_bins, n_cats=n_cats,
#         n_nums=n_nums, cards=cards, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
#         cat_emb_dim=emb_dim, time_dim=time_dim, lat_dim=lat_dim
#     ).to(device)

#     # Load pre-trained weights
#     print("Loading pre-trained weights...")
#     encoder_weights_path = os.path.join(pretrained_weights_dir, "pretrained_encoder.pth")
#     decoder_weights_path = os.path.join(pretrained_weights_dir, "pretrained_decoder.pth")
    
#     encoder_weights = torch.load(encoder_weights_path)
    
#     # Reinitialize GRU layers to match the new input size
#     print("Reinitializing GRU layers...")
#     ae.encoder_mu = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#     ae.encoder_logvar = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#     ae.encoder_mu.flatten_parameters()
#     ae.encoder_logvar.flatten_parameters()

#     # Load compatible weights
#     ae.fc_mu.load_state_dict(encoder_weights['fc_mu'])
#     ae.fc_logvar.load_state_dict(encoder_weights['fc_logvar'])
#     ae.decoder_mlp.load_state_dict(torch.load(decoder_weights_path))

#     print("Pre-trained weights loaded successfully.")

#     # Prepare optimizer and inputs
#     optimizer = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
#     inputs = processed_data.to(device)
#     beta = max_beta
#     lambd = 0.7  # Beta reduction factor
#     best_train_loss = float('inf')
#     all_indices = list(range(inputs.shape[0]))

#     losses = []

#     # Fine-tuning loop
#     print("Starting fine-tuning...")
#     with Progress() as progress:
#         training_task = progress.add_task("[green]Fine-Tuning VAE...", total=n_epochs)

#         for epoch in range(n_epochs):
#             batch_indices = random.sample(all_indices, batch_size)
#             batch_data = inputs[batch_indices, :, :]

#             optimizer.zero_grad()
#             outputs, _, mu_z, logvar_z = ae(batch_data)

#             # Compute losses
#             disc_loss, num_loss = auto_loss(batch_data, outputs, n_bins, n_nums, n_cats, beta, cards)
#             temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
#             loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
#             loss = num_loss + disc_loss + beta * loss_kld

#             # Backpropagation
#             loss.backward()
#             optimizer.step()

#             # Update progress
#             losses.append(loss.item())
#             progress.update(training_task, advance=1, description=f"Epoch {epoch}/{n_epochs} - Loss: {loss.item():.4f}")

#             # Beta annealing
#             if loss.item() < best_train_loss:
#                 best_train_loss = loss.item()
#             else:
#                 if beta > min_beta:
#                     beta *= lambd

#     # Extract latent features
#     print("Extracting latent features...")
#     _, latent_features, _, _ = ae(inputs)

#     print("Fine-tuning completed.")
#     return ae, latent_features.detach(), losses

class SafeGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(SafeGRU, self).__init__()
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

    def forward(self, x, h=None):
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)

        if h is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # Extract timestep t
            h_new = []
            for layer, cell in enumerate(self.gru_cells):
                h_t = cell(x_t, h[layer])
                h_new.append(h_t)
                x_t = h_t  # Pass output to the next layer
            h = h_new
            outputs.append(h[-1])  # Collect the final output of the last layer

        outputs = torch.stack(outputs, dim=1)  # Convert list to tensor
        return outputs, torch.stack(h, dim=0)

def fine_tune_autoencoder_dp(
    real_df, processed_data, pretrained_weights_dir, channels, hidden_size, num_layers, lr,
    weight_decay, n_epochs, batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device, 
    epsilon, delta=1e-5, max_grad_norm=1.0
):
    """
    Fine-tunes a pre-trained Variational Autoencoder (VAE) with Differential Privacy (DP) on a private dataset.

    Args:
        real_df: Private dataset as a Pandas DataFrame.
        processed_data: Tensor representation of the private dataset.
        pretrained_weights_dir: Directory containing pre-trained weights.
        channels: Number of channels in the VAE architecture.
        hidden_size: Hidden size for GRU layers.
        num_layers: Number of GRU layers.
        lr: Learning rate.
        weight_decay: Weight decay for regularization.
        n_epochs: Number of training epochs.
        batch_size: Batch size for training.
        threshold: Threshold for preprocessing.
        min_beta: Minimum beta value for KL divergence weighting.
        max_beta: Maximum beta value for KL divergence weighting.
        emb_dim: Embedding dimension.
        time_dim: Time encoding dimension.
        lat_dim: Latent space dimension.
        device: Device to run training on ('cuda' or 'cpu').
        epsilon: Differential privacy budget (epsilon).
        delta: Differential privacy budget (delta).
        max_grad_norm: Maximum gradient norm for clipping.
    Returns:
        ae: Fine-tuned autoencoder model.
        latent_features: Latent features extracted by the fine-tuned model.
        losses: List of training losses for each epoch.
    """
    import os
    from torch.optim import Adam
    from opacus import PrivacyEngine
    from rich.progress import Progress
    from opacus.validators import ModuleValidator
    from opacus.accountants import RDPAccountant

    # Initialize the autoencoder model
    datatype_info = pce.DataFrameParser().fit(real_df, threshold).datatype_info()
    n_bins = datatype_info['n_bins']
    n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']
    cards = datatype_info['cards']

    seq_len = processed_data.shape[1]
    input_size = processed_data.shape[2]
    print(f"Input size during fine-tuning: {input_size}")

    # Initialize the autoencoder model
    ae = DeapStack(
        channels=channels, batch_size=batch_size, seq_len=seq_len, n_bins=n_bins, n_cats=n_cats,
        n_nums=n_nums, cards=cards, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        cat_emb_dim=emb_dim, time_dim=time_dim, lat_dim=lat_dim
    ).to(device)

    #  Load pre-trained weights
    print("Loading pre-trained weights...")
    encoder_weights_path = os.path.join(pretrained_weights_dir, "pretrained_encoder.pth")
    decoder_weights_path = os.path.join(pretrained_weights_dir, "pretrained_decoder.pth")
    encoder_weights = torch.load(encoder_weights_path)
    # Replace GRU layers with SafeGRU
    ae.encoder_mu = SafeGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    ae.encoder_logvar = SafeGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    # Load compatible weights
    ae.fc_mu.load_state_dict(encoder_weights['fc_mu'])
    ae.fc_logvar.load_state_dict(encoder_weights['fc_logvar'])
    ae.decoder_mlp.load_state_dict(torch.load(decoder_weights_path))
    print("Pre-trained weights loaded successfully.")
    # Avoid invoking flatten_parameters explicitly
    for gru_layer in [ae.encoder_mu, ae.encoder_logvar]:
        gru_layer._flat_weights = []
        
    # Ensure DP compatibility
    ae = ModuleValidator.fix(ae)
    if not ModuleValidator.is_valid(ae):
        raise ValueError("Autoencoder model is not compatible with DP training")
    optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    # Prepare privacy engine
    privacy_engine = PrivacyEngine()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(processed_data),
        batch_size=batch_size,
        shuffle=True
    )
    ae, optimizer_ae, train_loader = privacy_engine.make_private_with_epsilon(
        module=ae,
        optimizer=optimizer_ae,
        data_loader=train_loader,
        epochs=n_epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    beta = max_beta
    lambd = 0.7
    best_train_loss = float('inf')
    losses = []

    # Fine-tuning loop
    print("Starting fine-tuning with Differential Privacy...")
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
        ) as progress:
        training_task = progress.add_task("[green]Fine-Tuning VAE...", total=n_epochs)

        for epoch in range(n_epochs):
            epoch_losses = []
            for batch in train_loader:
                batch_data = batch[0].to(device)

                optimizer_ae.zero_grad()
                outputs, _, mu_z, logvar_z = ae(batch_data)

                # Compute losses
                disc_loss, num_loss = auto_loss(batch_data, outputs, n_bins, n_nums, n_cats, beta, cards)
                temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
                loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
                loss = num_loss + disc_loss + beta * loss_kld

                # Backpropagation with DP-enabled optimizer
                loss.backward()
                optimizer_ae.step()

                epoch_losses.append(loss.item())

            # Update progress
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_epoch_loss)
            progress.update(
                training_task,
                advance=1,
                description=f"Epoch {epoch}/{n_epochs} - Loss: {avg_epoch_loss:.4f}"
            )

            # Beta annealing
            if avg_epoch_loss < best_train_loss:
                best_train_loss = avg_epoch_loss
            else:
                if beta > min_beta:
                    beta *= lambd

    # Extract latent features
    print("Extracting latent features...")
    _, latent_features, _, _ = ae(processed_data.to(device))

    print("Fine-tuning completed.")
    return ae, latent_features.detach(), losses


# def fine_tune_autoencoder_dp(
#     real_df, processed_data, pretrained_weights_dir, channels, hidden_size, num_layers, lr,
#     weight_decay, n_epochs, batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device, 
#     epsilon, delta=1e-5, max_grad_norm=1.0
# ):
#     """
#     Fine-tunes a pre-trained Variational Autoencoder (VAE) with Differential Privacy (DP) on a private dataset.

#     Args:
#         real_df: Private dataset as a Pandas DataFrame.
#         processed_data: Tensor representation of the private dataset.
#         pretrained_weights_dir: Directory containing pre-trained weights.
#         channels: Number of channels in the VAE architecture.
#         hidden_size: Hidden size for GRU layers.
#         num_layers: Number of GRU layers.
#         lr: Learning rate.
#         weight_decay: Weight decay for regularization.
#         n_epochs: Number of training epochs.
#         batch_size: Batch size for training.
#         threshold: Threshold for preprocessing.
#         min_beta: Minimum beta value for KL divergence weighting.
#         max_beta: Maximum beta value for KL divergence weighting.
#         emb_dim: Embedding dimension.
#         time_dim: Time encoding dimension.
#         lat_dim: Latent space dimension.
#         device: Device to run training on ('cuda' or 'cpu').
#         epsilon: Differential privacy budget (epsilon).
#         delta: Differential privacy budget (delta).
#         max_grad_norm: Maximum gradient norm for clipping.
#     Returns:
#         ae: Fine-tuned autoencoder model.
#         latent_features: Latent features extracted by the fine-tuned model.
#         losses: List of training losses for each epoch.
#     """
#     import os
#     from torch.optim import Adam
#     from opacus import PrivacyEngine  # Import from Opacus library
#     import random
#     from rich.progress import Progress

#     # Initialize the autoencoder model
#     datatype_info = pce.DataFrameParser().fit(real_df, threshold).datatype_info()
#     n_bins = datatype_info['n_bins']
#     n_cats = datatype_info['n_cats']
#     n_nums = datatype_info['n_nums']
#     cards = datatype_info['cards']

#     seq_len = processed_data.shape[1]
#     input_size = processed_data.shape[2]
#     print(f"Input size during fine-tuning: {input_size}")

#     # Initialize the autoencoder model
#     ae = DeapStack(
#         channels=channels, batch_size=batch_size, seq_len=seq_len, n_bins=n_bins, n_cats=n_cats,
#         n_nums=n_nums, cards=cards, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
#         cat_emb_dim=emb_dim, time_dim=time_dim, lat_dim=lat_dim
#     ).to(device)

#     # Load pre-trained weights
#     print("Loading pre-trained weights...")
#     encoder_weights_path = os.path.join(pretrained_weights_dir, "pretrained_encoder.pth")
#     decoder_weights_path = os.path.join(pretrained_weights_dir, "pretrained_decoder.pth")
    
#     encoder_weights = torch.load(encoder_weights_path)
    
#     # Reinitialize GRU layers to match the new input size
#     print("Reinitializing GRU layers...")
#     ae.encoder_mu = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#     ae.encoder_logvar = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

#     # Load compatible weights
#     ae.fc_mu.load_state_dict(encoder_weights['fc_mu'])
#     ae.fc_logvar.load_state_dict(encoder_weights['fc_logvar'])
#     ae.decoder_mlp.load_state_dict(torch.load(decoder_weights_path))

#     print("Pre-trained weights loaded successfully.")

#     # # Prepare optimizer and DP engine
#     # optimizer = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

#     # # Attach Opacus Privacy Engine
#     # privacy_engine = PrivacyEngine()
#     # ae, optimizer, data_loader = privacy_engine.make_private(
#     #     module=ae,
#     #     optimizer=optimizer,
#     #     data_loader=None,  # Placeholder, will manually handle batching
#     #     noise_multiplier = compute_noise_multiplier(epsilon=epsilon,delta=delta,sample_rate=batch_size / len(processed_data),num_epochs=n_epochs),
#     #     max_grad_norm=max_grad_norm
#     # )

#     # Make autoencoder DP-compatible
#     ae = ModuleValidator.fix(ae)
#     if not ModuleValidator.is_valid(ae):
#         raise ValueError("Autoencoder model is not compatible with DP training")

#     optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

#     # Initialize privacy engine
#     privacy_engine = PrivacyEngine()
    
#     # Create data loader
#     train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(processed_data),
#         batch_size=batch_size,
#         shuffle=True
#     )
    
#     # Make the model private
#     ae, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
#         module=ae,
#         optimizer=optimizer_ae,
#         data_loader=train_loader,
#         epochs=n_epochs,
#         target_epsilon=epsilon,
#         target_delta=delta,
#         max_grad_norm=max_grad_norm,
#     )






#     inputs = processed_data.to(device)
#     beta = max_beta
#     lambd = 0.7  # Beta reduction factor
#     best_train_loss = float('inf')
#     all_indices = list(range(inputs.shape[0]))

#     losses = []

#     # Fine-tuning loop
#     print("Starting fine-tuning with Differential Privacy...")
#     with Progress() as progress:
#         training_task = progress.add_task("[green]Fine-Tuning VAE...", total=n_epochs)

#         for epoch in range(n_epochs):
#             # Sample a batch of data
#             batch_indices = random.sample(all_indices, batch_size)
#             batch_data = inputs[batch_indices, :, :]

#             optimizer.zero_grad()
#             outputs, _, mu_z, logvar_z = ae(batch_data)

#             # Compute losses
#             disc_loss, num_loss = auto_loss(batch_data, outputs, n_bins, n_nums, n_cats, beta, cards)
#             temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
#             loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
#             loss = num_loss + disc_loss + beta * loss_kld

#             # Backpropagation with DP-enabled optimizer
#             loss.backward()
#             optimizer.step()

#             epsilon = privacy_engine.get_epsilon(delta)
#             # Update progress
#             losses.append(loss.item())
#             progress.update(training_task, advance=1, description=f"Epoch {epoch}/{n_epochs} - Loss: {loss.item():.4f}")

#             # Beta annealing
#             if loss.item() < best_train_loss:
#                 best_train_loss = loss.item()
#             else:
#                 if beta > min_beta:
#                     beta *= lambd

#     # Extract latent features
#     print("Extracting latent features...")
#     _, latent_features, _, _ = ae(inputs)

#     print("Fine-tuning completed.")
#     return ae, latent_features.detach(), losses



def fine_tune_autoencoder_dp2(
    real_df, processed_data, pretrained_weights_dir, channels, hidden_size, num_layers, lr,
    weight_decay, n_epochs, batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device, 
    epsilon, delta=1e-5, max_grad_norm=1.0
):
    """
    Fine-tunes a pre-trained Variational Autoencoder (VAE) with Differential Privacy (DP) on a private dataset.

    Args:
        real_df: Private dataset as a Pandas DataFrame.
        processed_data: Tensor representation of the private dataset.
        pretrained_weights_dir: Directory containing pre-trained weights.
        channels: Number of channels in the VAE architecture.
        hidden_size: Hidden size for GRU layers.
        num_layers: Number of GRU layers.
        lr: Learning rate.
        weight_decay: Weight decay for regularization.
        n_epochs: Number of training epochs.
        batch_size: Batch size for training.
        threshold: Threshold for preprocessing.
        min_beta: Minimum beta value for KL divergence weighting.
        max_beta: Maximum beta value for KL divergence weighting.
        emb_dim: Embedding dimension.
        time_dim: Time encoding dimension.
        lat_dim: Latent space dimension.
        device: Device to run training on ('cuda' or 'cpu').
        epsilon: Differential privacy budget (epsilon).
        delta: Differential privacy budget (delta).
        max_grad_norm: Maximum gradient norm for clipping.
    Returns:
        ae: Fine-tuned autoencoder model.
        latent_features: Latent features extracted by the fine-tuned model.
        losses: List of training losses for each epoch.
    """
    import os
    from torch.optim import Adam
    from opacus import PrivacyEngine
    from rich.progress import Progress
    from opacus.validators import ModuleValidator

    # Initialize the autoencoder model
    datatype_info = pce.DataFrameParser().fit(real_df, threshold).datatype_info()
    n_bins = datatype_info['n_bins']
    n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']
    cards = datatype_info['cards']

    seq_len = processed_data.shape[1]
    input_size = processed_data.shape[2]
    print(f"Input size during fine-tuning: {input_size}")

    # Initialize the autoencoder model
    ae = DeapStack(
        channels=channels, batch_size=batch_size, seq_len=seq_len, n_bins=n_bins, n_cats=n_cats,
        n_nums=n_nums, cards=cards, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        cat_emb_dim=emb_dim, time_dim=time_dim, lat_dim=lat_dim
    ).to(device)

    # Replace GRU layers with CustomGRU
    ae.encoder_mu = CustomGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).to(device)
    ae.encoder_logvar = CustomGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).to(device)

    # Load pre-trained weights
    print("Loading pre-trained weights...")
    encoder_weights_path = os.path.join(pretrained_weights_dir, "pretrained_encoder.pth")
    decoder_weights_path = os.path.join(pretrained_weights_dir, "pretrained_decoder.pth")
    encoder_weights = torch.load(encoder_weights_path)
    ae.fc_mu.load_state_dict(encoder_weights['fc_mu'])
    ae.fc_logvar.load_state_dict(encoder_weights['fc_logvar'])
    ae.decoder_mlp.load_state_dict(torch.load(decoder_weights_path))
    print("Pre-trained weights loaded successfully.")

    # Ensure DP compatibility
    ae = ModuleValidator.fix(ae)
    if not ModuleValidator.is_valid(ae):
        raise ValueError("Autoencoder model is not compatible with DP training")
    optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    # Prepare privacy engine
    privacy_engine = PrivacyEngine()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(processed_data),
        batch_size=batch_size,
        shuffle=True
    )
    ae, optimizer_ae, train_loader = privacy_engine.make_private_with_epsilon(
        module=ae,
        optimizer=optimizer_ae,
        data_loader=train_loader,
        epochs=n_epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    beta = max_beta
    lambd = 0.7
    best_train_loss = float('inf')
    losses = []

    # Fine-tuning loop
    print("Starting fine-tuning with Differential Privacy...")
    with Progress() as progress:
        training_task = progress.add_task("[green]Fine-Tuning VAE...", total=n_epochs)

        for epoch in range(n_epochs):
            epoch_losses = []
            for batch in train_loader:
                batch_data = batch[0].to(device)

                optimizer_ae.zero_grad()
                outputs, _, mu_z, logvar_z = ae(batch_data)

                # Compute losses
                disc_loss, num_loss = auto_loss(batch_data, outputs, n_bins, n_nums, n_cats, beta, cards)
                temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
                loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
                loss = num_loss + disc_loss + beta * loss_kld

                # Backpropagation with DP-enabled optimizer
                loss.backward()
                optimizer_ae.step()

                epoch_losses.append(loss.item())

            # Update progress
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_epoch_loss)
            progress.update(
                training_task,
                advance=1,
                description=f"Epoch {epoch}/{n_epochs} - Loss: {avg_epoch_loss:.4f}"
            )

            # Beta annealing
            if avg_epoch_loss < best_train_loss:
                best_train_loss = avg_epoch_loss
            else:
                if beta > min_beta:
                    beta *= lambd

    # Extract latent features
    print("Extracting latent features...")
    _, latent_features, _, _ = ae(processed_data.to(device))

    print("Fine-tuning completed.")
    return ae, latent_features.detach(), losses
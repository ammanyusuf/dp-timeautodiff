import logging
import os
import matplotlib.pyplot as plt
import argparse
import time
import numpy as np
import timeautoencoder as tae
import timediffusion as tdf
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


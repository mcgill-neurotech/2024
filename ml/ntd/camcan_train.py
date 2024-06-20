
import os
import numpy as np
import torch
import wandb

from icecream import ic
from torch.utils.data import DataLoader

from datasets import CamCAN_MEG
from networks import LongConv, BaseConv
from diffusion_model import Diffusion
from utils.kernels_and_diffusion_utils import WhiteNoiseProcess, OUProcess

# Device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
DATA_PATH = "/meg/meg1/users/mlapatrie/data/Scouted_MEG_CamCAN_300/train_data/" #"D:/Scouted_MEG_CamCAN_300/train_data/"
SAVE_PATH = "/meg/meg1/users/mlapatrie/data/CamCAN200/CamCAN_tests/diffusion_camcan_pca.pt"

NUM_TIMESTEPS = 5000
SIGNAL_LENGTH = 600
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 150

PCA = True
PCA_COMPONENTS = 200 # Also the number of channels. If PCA false, is still used as the number of channels.

# Load data

sub_ccids = [i for i in os.listdir(DATA_PATH) if i.startswith("CC")]

dataset = CamCAN_MEG(
    signal_length=SIGNAL_LENGTH,
    rois="all",
    sub_ids=sub_ccids,
    pca=PCA,
    pca_sub_ids=sub_ccids,
    n_components=PCA_COMPONENTS,
    with_class_cond=False,
    folder_path=DATA_PATH,
    pca_folder_path=DATA_PATH,
)

train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)

ic("Loaded the data")

# Initialize model

TIME_DIM = 10
COND_CHANNELS = 0
HIDDEN_CHANNELS = 32
IN_KERNEL_SIZE = 17
OUT_KERNEL_SIZE = 17
SLCONV_KERNEL_SIZE = 32
NUM_SCALES = 5
DECAY_MIN = 2
DECAY_MAX = 2
HEADS = 3
USE_FFT_CONV = True

KERNEL_SIZE = 29
DILATION = 1

network = LongConv(
    signal_length=SIGNAL_LENGTH,
    signal_channel=PCA_COMPONENTS,
    time_dim=TIME_DIM,
    cond_channel=COND_CHANNELS,
    hidden_channel=HIDDEN_CHANNELS,
    in_kernel_size=IN_KERNEL_SIZE,
    out_kernel_size=OUT_KERNEL_SIZE,
    slconv_kernel_size=SLCONV_KERNEL_SIZE,
    num_scales=NUM_SCALES,
    decay_min=DECAY_MIN,
    decay_max=DECAY_MAX,
    heads=HEADS,
    use_fft_conv=USE_FFT_CONV,
)

network = BaseConv(
        
    signal_length=SIGNAL_LENGTH,
    signal_channel=PCA_COMPONENTS,
    time_dim=TIME_DIM,
    cond_channel=COND_CHANNELS,
    hidden_channel=HIDDEN_CHANNELS,
    kernel_size=KERNEL_SIZE,
    dilation=DILATION,
    activation_type="tanh",

)

noise_sampler = WhiteNoiseProcess(1.0, SIGNAL_LENGTH)

diffusion_model = Diffusion(
    network=network,
    diffusion_time_steps=NUM_TIMESTEPS,
    noise_sampler=noise_sampler,
    mal_dist_computer=noise_sampler,
    schedule="linear",
    start_beta=0.0001,
    end_beta=0.02,
).to(device)

ic("Initialized the model")

# wandb
wandb.init(project="diffusion-camcan", mode="online",
           config={
                   "Number of timesteps": NUM_TIMESTEPS,
                   "Signal length": SIGNAL_LENGTH,
                   "Number of channels": PCA_COMPONENTS,
                   "Number of epochs": NUM_EPOCHS,
                   "Batch size": BATCH_SIZE,
                   "Learning rate": LEARNING_RATE,
                   "Time dim": TIME_DIM,
                   "Cond channels": COND_CHANNELS,
                   "Hidden channels": HIDDEN_CHANNELS,
                   "In kernel size": IN_KERNEL_SIZE,
                   "Out kernel size": OUT_KERNEL_SIZE,
                   "SlConv kernel size": SLCONV_KERNEL_SIZE,
                   "Number of scales": NUM_SCALES,
                   "Decay min": DECAY_MIN,
                   "Decay max": DECAY_MAX,
                   "Heads": HEADS,
                   "FFT Conv": USE_FFT_CONV,
                   })

# Train model

optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=LEARNING_RATE)

for i in range(NUM_EPOCHS):
    
    loss_per_epoch = []
    
    for batch in train_loader:
        
        optimizer.zero_grad()
        
        signal = batch["signal"][:, :PCA_COMPONENTS, :]
        
        loss = torch.mean(diffusion_model.train_batch(signal.to(device)))
        
        loss.backward()
        optimizer.step()
        
        loss_per_epoch.append(loss.item())
        wandb.log({"loss": loss})
        
    epoch_loss = np.mean(loss_per_epoch)
    ic(f"Epoch {i}: Loss {epoch_loss}")
    wandb.log({"epoch_loss": epoch_loss})
    
# Save model
torch.save(diffusion_model.state_dict(), SAVE_PATH)
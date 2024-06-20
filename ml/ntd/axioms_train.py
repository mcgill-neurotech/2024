
import os
import torch
import numpy as np
import wandb

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import AdamW
from icecream import ic

from networks import LongConv
from diffusion_model import Diffusion
from utils.kernels_and_diffusion_utils import WhiteNoiseProcess, OUProcess

# wandb
wandb.init(project="axioms", mode="online")

# Device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
NUM_TIMESTEPS = 1000
BATCH_SIZE = 32
FS = 300
LENGTH = 2
signal_length = FS * LENGTH

SAVE_PATH = "/meg/meg1/users/mlapatrie/data/axioms/"

# Data
# Generate sinusoidal segments with different frequencies and amplitudes with phase 0 and pi/2

low_freqency = 1
high_frequency = 11

low_amplitude = 1
high_amplitude = 100

low_phase = 0
high_phase = 0


# Generate data
data = []
cond = []
signals = []
signals_with_phase = []

for f in range(low_freqency, high_frequency + 1):
    
    for a in range(low_amplitude, high_amplitude + 1):
        
        for p in range(low_phase, high_phase + 1):
        
            # Generate sinusoidal signal with phase 0
            t = torch.linspace(0, LENGTH, signal_length)
            signal = (a * torch.sin(2 * torch.pi * f * t + p * torch.pi / 180)).unsqueeze(0)
            
            data.append(signal)
            cond.append([f, a, p])
        
np.save(os.path.join(SAVE_PATH, "signals/signals_no_phase.npy"), signals)
np.save(os.path.join(SAVE_PATH, "signals/signals_with_phase.npy"), signals_with_phase)

dataset = [{"signal": signal, "cond": torch.tensor(cond)} for signal, cond in zip(data, cond)]

train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

# Initialize the model

network = LongConv(
    signal_length=signal_length,
    signal_channel=1,
    time_dim=10,
    cond_channel=0,
    hidden_channel=20,
    in_kernel_size=17,
    out_kernel_size=17,
    slconv_kernel_size=17,
    num_scales=5,
    decay_min=2,
    decay_max=2,
    heads=3,
    use_fft_conv=True,
    activation_type="tanh",
)

noise_sampler = WhiteNoiseProcess(1.0, signal_length)

diffusion_model = Diffusion(
    network=network,
    diffusion_time_steps=NUM_TIMESTEPS,
    noise_sampler=noise_sampler,
    mal_dist_computer=noise_sampler,
    schedule="quad",
    start_beta=0.0001,
    end_beta=0.02,
).to(device)


# Train the model

NUM_EPOCHS = 500
LEARNING_RATE = 1e-3

optimizer = AdamW(diffusion_model.parameters(), lr=LEARNING_RATE)

for i in range(NUM_EPOCHS):
    
    epoch_loss = []
    
    for batch in train_loader:
        
        signal = batch["signal"].to(device)
        #cond = batch["cond"].to(device)
        #cond = cond.unsqueeze(2).repeat(1, 1, signal_length)
        
        optimizer.zero_grad()
        
        loss = torch.mean(diffusion_model.train_batch(signal, cond=None))
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss.append(loss.item())
        
        wandb.log({"loss": loss.item()})
        #print(f"Epoch {i}, loss: {loss.item()}")
        
    wandb.log({"loss": np.mean(epoch_loss)})
    
torch.save(network.state_dict(), os.path.join(SAVE_PATH, "saved_models/phase_network.pt"))
torch.save(diffusion_model.state_dict(), os.path.join(SAVE_PATH, "saved_models/phase.pt"))

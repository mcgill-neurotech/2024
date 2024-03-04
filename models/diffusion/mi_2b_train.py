

# Imports
import os
import wandb
import torch
import numpy as np
import scipy.io as io

import yaml # pip install pyyaml

from torch import optim
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from classification.classifiers import load_data, CSPClassifier, Classifier
from ntd.networks import LongConv
from ntd.diffusion_model import Diffusion
from ntd.utils.kernels_and_diffusion_utils import WhiteNoiseProcess

# Device (uses the gpu if you have one, otherwise uses the cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# WandB (logs the training process. You can keep the mode="disabled" if you don't want it or use online to log it to wandb.ai)
wandb.init(project="diffusion-mi", mode="disabled")

# This code trains a model on the CSP components. To train on the eeg data, use the Classifier class and create your own preprocessing pipeline.

# CONSTANTS
DATA_PATH = "C:/Repos/Python/motor-imagery-classification-2024/data/2b_iv/"
SAVE_PATH = "/meg/meg1/users/mlapatrie/data/MI/saved_models/mi_01.pt"
CONF_PATH = "C:/Repos/Python/motor-imagery-classification-2024/models/diffusion/conf/"

# Load parameters from config files

# Load training parameters (There must be a better way, but this works just fine)
with open(os.path.join(CONF_PATH, "train.yaml"), "r") as f:
    train_yaml = yaml.safe_load(f)
    
with open(os.path.join(CONF_PATH, "classifier.yaml"), "r") as f:
    classifier_yaml = yaml.safe_load(f)
    
with open(os.path.join(CONF_PATH, "network.yaml"), "r") as f:
    network_yaml = yaml.safe_load(f)
    
with open(os.path.join(CONF_PATH, "diffusion.yaml"), "r") as f:
    diffusion_yaml = yaml.safe_load(f)

# Load data
dataset_mat = {}
for i in range(1,10):
	mat_train,mat_test = load_data(DATA_PATH,i)
	dataset_mat[f"subject_{i}"] = {"train":mat_train,"test":mat_test}

classifier = CSPClassifier(dataset_mat, t_baseline=classifier_yaml["t_baseline"], t_epoch=classifier_yaml["t_epoch"])

# The classifier has the function __len__ and __getitem__ making it a valid dataset for the DataLoader
train_loader = DataLoader(
    classifier,
    train_yaml["batch_size"],
)

# Initialize model

network = LongConv(
    signal_length=network_yaml["signal_length"],
    signal_channel=network_yaml["signal_channel"], # The CSP classifier components
    time_dim=network_yaml["time_dim"],
    cond_channel=network_yaml["cond_channel"], # The cond channel will contain the cue (0 or 1)
    hidden_channel=network_yaml["hidden_channel"],
    in_kernel_size=network_yaml["in_kernel_size"],
    out_kernel_size=network_yaml["out_kernel_size"],
    slconv_kernel_size=network_yaml["slconv_kernel_size"],
    num_scales=network_yaml["num_scales"],
    decay_min=network_yaml["decay_min"],
    decay_max=network_yaml["decay_max"],
    heads=network_yaml["heads"],
    use_fft_conv=network_yaml["use_fft_conv"],
).to(device)

noise_sampler = WhiteNoiseProcess(1.0, network_yaml["signal_length"]).to(device)

diffusion_model = Diffusion(
    network=network,
    diffusion_time_steps=diffusion_yaml["num_timesteps"],
    noise_sampler=noise_sampler,
    mal_dist_computer=noise_sampler,
    schedule=diffusion_yaml["schedule"],
    start_beta=diffusion_yaml["start_beta"],
    end_beta=diffusion_yaml["end_beta"],
).to(device)


# Train model

optimizer = optim.AdamW(
    network.parameters(),
    lr=train_yaml["lr"],
)

loss_per_epoch = []


for i in range(train_yaml["num_epochs"]):
    
    epoch_loss = []
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Repeat the cue signal to match the signal length
        cond = batch["cue"].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, network_yaml["signal_length"]).to(device)
        
        print(batch["signal"].shape)
        
        loss = diffusion_model.train_batch(batch["signal"].to(device), cond=cond)
        loss = torch.mean(loss)
        
        epoch_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
    epoch_loss = np.mean(epoch_loss)
    loss_per_epoch.append(epoch_loss)
    
    wandb.log({"loss": epoch_loss})
    print(f"Epoch {i} loss: {epoch_loss}")
    
    
# Save model
torch.save(diffusion_model.state_dict(), SAVE_PATH)

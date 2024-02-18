

# Imports
import os
import wandb
import torch
import numpy as np
import scipy.io as io

from torch import optim
from torch.utils.data import DataLoader

from ntd.networks import LongConv
from ntd.diffusion_model import Diffusion
from ntd.utils.kernels_and_diffusion_utils import WhiteNoiseProcess
from ntd.mi_datasets import subject_dataset, Classifier, SubjectClassifier, CSPClassifier


wandb.init(project="diffusion-mi", mode="disabled")

# CONSTANTS
DATA_PATH = "C:/Repos/Python/motor-imagery-classification/data/2b_iv/"#"/meg/meg1/users/mlapatrie/data/MI/2b_iv"
SAVE_PATH = "/meg/meg1/users/mlapatrie/data/MI/saved_models/mi_01.pt"

t_baseline = 3.75
t_epoch = 5.75

NUM_TIMESTEPS = 1000
SIGNAL_LENGTH = 500

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 500


# Load data
def load_data(folder,idx):
	path_train = os.path.join(folder,f"B0{idx}T.mat")
	path_test = os.path.join(folder,f"B0{idx}E.mat")
	mat_train = io.loadmat(path_train)["data"]
	mat_test = io.loadmat(path_test)["data"]
	return mat_train,mat_test


dataset_mat = {}
for i in range(1,10):
	mat_train,mat_test = load_data(DATA_PATH,i)
	dataset_mat[f"subject_{i}"] = {"train":mat_train,"test":mat_test}

classifier = CSPClassifier(dataset_mat, t_baseline=t_baseline, t_epoch=t_epoch)

train_loader = DataLoader(
    classifier,
    BATCH_SIZE,
)

print(train_loader)

# Initialize model

network = LongConv(
    signal_length=SIGNAL_LENGTH,
    signal_channel=2,
    time_dim=10,
    cond_channel=1, # The cond channel will contain the cue (0 or 1)
    hidden_channel=5,
    in_kernel_size=17,
    out_kernel_size=17,
    slconv_kernel_size=17,
    num_scales=5,
    decay_min=2,
    decay_max=2,
    heads=3,
    use_fft_conv=False,
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
)


# Train model

optimizer = optim.AdamW(
    network.parameters(),
    lr=LEARNING_RATE,
)

loss_per_epoch = []


for i in range(NUM_EPOCHS):
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Repeat the cue signal to match the signal length
        cond = batch["cue"].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, SIGNAL_LENGTH)
        
        print(batch["signal"].shape)
        
        loss = diffusion_model.train_batch(batch["signal"], cond=cond)
        loss = torch.mean(loss)
        
        loss.backward()
        optimizer.step()
        
    loss_per_epoch.append(loss.item())
    
    wandb.log({"loss": loss.item()})
    print(f"Epoch {i} loss: {loss_per_epoch[-1]}")
    
    
# Save model
torch.save(diffusion_model.state_dict(), SAVE_PATH)

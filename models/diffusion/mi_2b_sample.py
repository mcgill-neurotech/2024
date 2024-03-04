
import os
import yaml
import torch
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ntd.networks import LongConv
from ntd.diffusion_model import Diffusion
from ntd.utils.kernels_and_diffusion_utils import WhiteNoiseProcess

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
MODEL_PATH = "D:/DDPM_data/MI/saved_models/mi_all.pt"
SAVE_PATH = "/meg/meg1/users/mlapatrie/data/MI/generated_samples/"

NUM_SAMPLES = 30

# Load parameters from config files
CONF_PATH = "C:/Repos/Python/motor-imagery-classification-2024/models/diffusion/conf/"

with open(os.path.join(CONF_PATH, "network.yaml"), "r") as f:
    network_yaml = yaml.safe_load(f)
    
with open(os.path.join(CONF_PATH, "diffusion.yaml"), "r") as f:
    diffusion_yaml = yaml.safe_load(f)


# Load trained model
print("Loading model")
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

diffusion_model.load_state_dict(torch.load(MODEL_PATH))

# Generate samples
cond_zeros = torch.zeros(NUM_SAMPLES, 1, network_yaml["signal_length"])
cond_ones = torch.ones(NUM_SAMPLES, 1, network_yaml["signal_length"])

# Generate and save samples with cue 0
print("Generating samples: cue 0")
samples_zeros, _ = diffusion_model.sample(NUM_SAMPLES, cond=cond_zeros)
samples_zeros = samples_zeros.numpy()
np.save(os.path.join(SAVE_PATH, "samples_zero.npy"), samples_zeros)

# Generate and save samples with cue 1
print("Generating samples: cue 1")
samples_ones, _ = diffusion_model.sample(NUM_SAMPLES, cond=cond_ones)
samples_ones = samples_ones.numpy()
np.save(os.path.join(SAVE_PATH, "samples_one.npy"), samples_ones)


# Plot samples
#print("Plotting the segments")
#num_channels = samples_zeros.shape[1]
#offset = -10
#
## Plot cue zero samples
#fig, ax = plt.subplots()
#plot_overlapping_signal(
#    fig,
#    ax,
#    sig=samples_zeros[0] + offset * np.arange(num_channels)[:, np.newaxis],
#    colors=["blue", "green", "red"],
#)
#
#plt.title("Generated samples, Cue 0")
#plt.show()
#
## Plot cue one samples
#fig, ax = plt.subplots()
#plot_overlapping_signal(
#    fig,
#    ax,
#    sig=samples_ones[0] + offset * np.arange(num_channels)[:, np.newaxis],
#    colors=["blue", "green", "red"],
#)
#
#plt.title("Generated samples, Cue 1")
#plt.show()
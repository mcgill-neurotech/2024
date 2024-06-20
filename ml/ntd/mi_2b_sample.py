
import os
import torch
import numpy as np

from diffusion_model import Diffusion

import matplotlib.pyplot as plt

from networks import LongConv
from utils.kernels_and_diffusion_utils import WhiteNoiseProcess
from utils.plotting_utils import plot_overlapping_signal


# CONSTANTS
MODEL_PATH = "/meg/meg1/users/mlapatrie/data/MI/saved_models/mi_01.pt"
SAVE_PATH = "/meg/meg1/users/mlapatrie/data/MI/generated_samples/"

NUM_SAMPLES = 30
SIGNAL_LENGTH = 500
NUM_TIMESTEPS = 1000


# Load trained model
print("Loading model")
network = LongConv(
    signal_length=SIGNAL_LENGTH,
    signal_channel=3,
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

diffusion_model.load_state_dict(torch.load(MODEL_PATH))

# Generate samples
cond_zeros = torch.zeros(NUM_SAMPLES, 1, SIGNAL_LENGTH)
cond_ones = torch.ones(NUM_SAMPLES, 1, SIGNAL_LENGTH)

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
print("Plotting the segments")
num_channels = samples_zeros.shape[1]
offset = -10

# Plot cue zero samples
fig, ax = plt.subplots()
plot_overlapping_signal(
    fig,
    ax,
    sig=samples_zeros[0] + offset * np.arange(num_channels)[:, np.newaxis],
    colors=["blue", "green", "red"],
)

plt.title("Generated samples, Cue 0")
plt.show()

# Plot cue one samples
fig, ax = plt.subplots()
plot_overlapping_signal(
    fig,
    ax,
    sig=samples_ones[0] + offset * np.arange(num_channels)[:, np.newaxis],
    colors=["blue", "green", "red"],
)

plt.title("Generated samples, Cue 1")
plt.show()
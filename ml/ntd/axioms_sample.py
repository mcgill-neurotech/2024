
import os
import torch
import numpy as np

from icecream import ic

from diffusion_model import Diffusion

import matplotlib.pyplot as plt

from networks import LongConv
from utils.kernels_and_diffusion_utils import WhiteNoiseProcess
from utils.plotting_utils import plot_overlapping_signal

# Device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
MODEL_PATH = "/meg/meg1/users/mlapatrie/data/axioms/saved_models/phase.pt"
NETWORK_PATH = "/meg/meg1/users/mlapatrie/data/axioms/saved_models/phase_network.pt"
SAVE_PATH = "/meg/meg1/users/mlapatrie/data/axioms/generated_samples/"

NUM_SAMPLES = 30
FS = 300
LENGTH = 2
signal_length = FS * LENGTH
NUM_TIMESTEPS = 1000


# Load trained model
print("Loading model")

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
network.load_state_dict(torch.load(NETWORK_PATH))

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

diffusion_model.load_state_dict(torch.load(MODEL_PATH))

low_freqency = 5
high_frequency = 5

low_amplitude = 1
high_amplitude = 5

low_phase = 0
high_phase = 0

# Generate random cond vectors of shape (NUM_SAMPLES, 3, signal_length)
cond_f = torch.randint(low=low_freqency, high=high_frequency + 1, size=(NUM_SAMPLES,))
cond_a = torch.randint(low=low_amplitude, high=high_amplitude + 1, size=(NUM_SAMPLES,))
cond_p = torch.randint(low=low_phase, high=high_phase + 1, size=(NUM_SAMPLES,))

cond = torch.cat([cond_f.unsqueeze(1), cond_a.unsqueeze(1), cond_p.unsqueeze(1)], dim=1)
cond = cond.unsqueeze(2).repeat(1, 1, signal_length).to(device)

samples, _ = diffusion_model.sample(NUM_SAMPLES, cond=None)
samples = samples.cpu().numpy()
np.save(os.path.join(SAVE_PATH, "samples.npy"), samples)

for s in range(len(samples)):
    # Plot samples
    print("Plotting the segments")
    num_channels = samples.shape[1]
    offset = -10
    
    # Plot cue zero samples
    fig, ax = plt.subplots()
    plot_overlapping_signal(
        fig,
        ax,
        sig=samples[s] + offset * np.arange(num_channels)[:, np.newaxis],
        colors=["blue"],
    )
    
    plt.title(f"Frequency: {cond_f[s]}, Amplitude: {cond_a[s]}, Phase: {cond_p[s]}")
    plt.show()
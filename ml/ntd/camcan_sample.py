
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
MODEL_PATH = "/meg/meg1/users/mlapatrie/data/CamCAN200/CamCAN_tests/diffusion_camcan.pt"
SAVE_PATH = "/meg/meg1/users/mlapatrie/data/CamCAN200/CamCAN_tests/generated_segments"

NUM_SAMPLES = 1

NUM_TIMESTEPS = 50000
SIGNAL_LENGTH = 600
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 150

PCA = True
PCA_COMPONENTS = 5 # Also the number of channels. If PCA false, is still used as the number of channels.


# Load trained model
print("Loading model")

# Initialize model

TIME_DIM = 10
COND_CHANNELS = 0
HIDDEN_CHANNELS = 16
IN_KERNEL_SIZE = 17
OUT_KERNEL_SIZE = 17
SLCONV_KERNEL_SIZE = 17
NUM_SCALES = 5
DECAY_MIN = 2
DECAY_MAX = 2
HEADS = 3
USE_FFT_CONV = True

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

diffusion_model.load_state_dict(torch.load(MODEL_PATH))

ic("Initialized the model")


samples, _ = diffusion_model.sample(NUM_SAMPLES, cond=None)
samples = samples.cpu().numpy()
#np.save(os.path.join(SAVE_PATH, "samples.npy"), samples)

print(samples[0])
print(samples[0].shape)
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
        colors=["blue", "green", "red", "orange", "purple"],
    )
    
    #plt.title(f"Frequency: {cond_f[s]}, Amplitude: {cond_a[s]}, Phase: {cond_p[s]}")
    plt.show()
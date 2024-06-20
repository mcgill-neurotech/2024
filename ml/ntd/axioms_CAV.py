# Implementation of CAVs on pre-trained diffusion models
# CAVs were introduced in the paper "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)" by Kim et al. (2018).

# Here, we implement CAVs to differentiate between individuals. To do so, each CAV will represent the seperation between an individual and the rest of the cohort.
# Furthermore, we will also implement CAVs to understand group differences. In this case, each CAV will represent the seperation between one group and the rest of the cohort.

# We will then compute the sensitivity of the model to each CAV, giving us a measure of how much the model relies on each CAV to make predictions.

# Here are the steps we will follow:
# 1. Load the pre-trained model
# 2. Load the data
# 3. Create data groups
# 4. Feed the data to the model and get the activations
# 5. Train classifier between a group and the rest of the cohort
# 6. Compute CAVs
# 7. Compute sensitivity of the model to each CAV
# 8. Visualize CAVs


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from torch.utils.data import DataLoader

from networks import LongConv
from utils.kernels_and_diffusion_utils import WhiteNoiseProcess
from diffusion_model import Diffusion

# Device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Step 0: Define the functions
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook


# Step 1: Load the pre-trained model
model_name = "ap.pt"
model_path = "/meg/meg1/users/mlapatrie/data/axioms/saved_models/"

signal_length = 600
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

diffusion_model.load_state_dict(torch.load(os.path.join(model_path, model_name)))


# Step 2: Load the data

fs = 300
segment_length = 2 # in seconds

low_frequency = 1
high_frequency = 1
frequency_step = 0.1

low_amplitude = 1
high_amplitude = 100
amplitude_step = 1

low_phase = 0
high_phase = 180

# Generate data
data = []
cond = []
signals = []
signals_with_phase = []

# Group 1
frequencies = np.arange(low_frequency, high_frequency + 1, frequency_step)
amplitudes = np.arange(low_amplitude, high_amplitude + 1, amplitude_step)
phases = np.arange(low_phase, high_phase + 1, 1)
for f in frequencies:
    
    for a in amplitudes:
        
        for p in phases:
        
            # Generate sinusoidal signal with phase 0
            t = torch.linspace(0, segment_length, signal_length)
            signal = (a * torch.sin(2 * torch.pi * f * t + p * torch.pi / 180)).unsqueeze(0)
            
            data.append(signal)
            cond.append([f, a, p])

dataset_1 = [{"signal": signal, "cond": torch.tensor(cond)} for signal, cond in zip(data, cond)]


# Group 2

data = []
cond = []
signals = []
signals_with_phase = []

low_frequency = 10
high_frequency = 10
frequency_step = 0.1

low_amplitude = 1
high_amplitude = 100
amplitude_step = 1

low_phase = 0
high_phase = 180

frequencies = np.arange(low_frequency, high_frequency + 1, frequency_step)
amplitudes = np.arange(low_amplitude, high_amplitude + 1, amplitude_step)
phases = np.arange(low_phase, high_phase + 1, 1)
for f in frequencies:
    
    for a in amplitudes:
        
        for p in phases:
        
            # Generate sinusoidal signal with phase 0
            t = torch.linspace(0, segment_length, signal_length)
            signal = (a * torch.sin(2 * torch.pi * f * t + p * torch.pi / 180)).unsqueeze(0)
            
            data.append(signal)
            cond.append([f, a, p])

dataset_2 = [{"signal": signal, "cond": torch.tensor(cond)} for signal, cond in zip(data, cond)]


dataset_1_loader = DataLoader(dataset_1, 64)
dataset_2_loader = DataLoader(dataset_2, 64)

# Step 3: Create data groups
# We will create two groups: one for the first subject and one for the rest of the cohort.
#g1_sig = torch.from_numpy(np.array([segment["signal"] for segment in dataset_1])).to(device)
#g1_cond = torch.from_numpy(np.array([segment["cond"] for segment in dataset_1])).to(device)

#g2_sig = torch.from_numpy(np.array([segment["signal"] for segment in dataset_2])).to(device)
#g2_cond = torch.from_numpy(np.array([segment["cond"] for segment in dataset_2])).to(device)

#print(f"g1 data shape: {g1_sig.shape}")
#print(f"g2 data shape: {g2_sig.shape}")

#print(f"g1 cond shape: {g1_cond.shape}")
#print(f"g2 cond shape: {g2_cond.shape}")

g1_cond = None
g2_cond = None

# Step 4: Feed the data to the model and get the activations


# Register forward hook
target_layer = diffusion_model.network.conv_pool[15]
target_layer.register_forward_hook(get_activation("SLConv"))

# Forward pass individual data
g1_activations = []
for batch in dataset_1_loader:
    time_index = torch.full((len(batch["signal"]),), 999, dtype=torch.float).to(device)
    
    diffusion_model.network.forward(batch["signal"].to(device), t=time_index, cond=g1_cond)
    
    g1_batch_activations = activations["SLConv"].cpu().detach().numpy()
    g1_activations.append(g1_batch_activations.mean(axis=-1))

g1_activations = np.concatenate(g1_activations)
print(g1_activations.shape)

# Forward pass group data
g2_activations = []
for batch in dataset_2_loader:
    time_index = torch.full((len(batch["signal"]),), 999, dtype=torch.float).to(device)
    diffusion_model.network.forward(batch["signal"].to(device), t=time_index, cond=g2_cond)
    
    g2_batch_activations = activations["SLConv"].cpu().detach().numpy()
    g2_activations.append(g2_batch_activations.mean(axis=-1))

g2_activations = np.concatenate(g2_activations)
print(g2_activations.shape)

# Step 5: Train classifier between a group and the rest of the cohort

train_test_split = 0.8
g1_train_segments = int(train_test_split * len(g1_activations))

g2_train_segments = int(train_test_split * len(g2_activations))


#svc = SVC(kernel="linear")
regression_model = LogisticRegression(max_iter=1000)
X = np.concatenate([g1_activations[:g1_train_segments], g2_activations[:g2_train_segments]])
Y = np.concatenate([np.ones(g1_train_segments), np.zeros(g2_train_segments)])

X_test = np.concatenate([g1_activations[-g1_train_segments:], g2_activations[-g2_train_segments:]])
Y_test = np.concatenate([np.ones(g1_train_segments), np.zeros(g2_train_segments)])

print(X_test.shape)
print(Y_test.shape)

regression_model.fit(X, Y)

print(f"Accuracy: {regression_model.score(X_test, Y_test)}")

cav = regression_model.coef_[0]
print(cav)


normed_cav = cav / np.linalg.norm(cav)

scale_factor = 0.1

X_perturbed_positive = X_test + scale_factor * normed_cav
X_perturbed_negative = X_test - scale_factor * normed_cav

for i in range(10):
    
    X_perturbed_positive += scale_factor * normed_cav
    X_perturbed_negative -= scale_factor * normed_cav
    
    plt.plot(X_perturbed_positive[0])
    plt.show()
    
    plt.plot(X_perturbed_negative[0])
    plt.show()


print(f"Pos perturbed: {regression_model.score(X_perturbed_positive, Y_test)}")
print(f"Neg perturbed: {regression_model.score(X_perturbed_negative, Y_test)}")

print(cav)

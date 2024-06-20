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
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from networks import LongConv, BaseConv
from utils.kernels_and_diffusion_utils import WhiteNoiseProcess
from diffusion_model import Diffusion

from datasets import CamCAN_MEG

# Device
device = ("cuda" if torch.cuda.is_available else "cpu")


# Step 0: Define the functions
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook


# Step 1: Load the pre-trained model
model_name = "diffusion_camcan_pca.pt"
model_path = "/meg/meg1/users/mlapatrie/data/CamCAN200/CamCAN_tests/"

NUM_TIMESTEPS = 5000
SIGNAL_LENGTH = 600
PCA = False
PCA_COMPONENTS = 200


# Load trained model
print("Loading model")

TIME_DIM = 10
COND_CHANNELS = 0
HIDDEN_CHANNELS = 32
IN_KERNEL_SIZE = 17
OUT_KERNEL_SIZE = 17
SLCONV_KERNEL_SIZE = 17
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

model = Diffusion(
    network=network,
    diffusion_time_steps=NUM_TIMESTEPS,
    noise_sampler=noise_sampler,
    mal_dist_computer=noise_sampler,
    schedule="linear",
    start_beta=0.0001,
    end_beta=0.02,
).to(device)


model.load_state_dict(torch.load(os.path.join(model_path, model_name)))

# Step 2: Load the data

fs = 300
segment_length = 2 # in seconds

# Get file paths
data_path = "/meg/meg1/users/mlapatrie/data/Scouted_MEG_CamCAN_300/test_data/"
ccids = [i for i in os.listdir(data_path) if i.endswith(".mat")]
print(f"Number of subjects: {len(ccids)}")

# Get subjects used for training to find the pca components
train_path = "/meg/meg1/users/mlapatrie/data/Scouted_MEG_CamCAN_300/train_data/"
ccids_train = [i for i in os.listdir(train_path) if i.endswith(".mat")]
print(f"Number of subjects used for training: {len(ccids_train)}")

# Load the data
data = [] # data will contain a dataset for each subject in the cohort. Each dataset will contain the signal and the class condition for the entire recording.

for ccid in ccids:
    data.append(
        CamCAN_MEG(
            signal_length=fs*segment_length,
            sub_ids=[ccid],
            rois="all",
            folder_path=data_path,
            start_index=0,
            end_index=-1,
            pca=PCA,
            n_components=PCA_COMPONENTS,
            pca_sub_ids=ccids_train,
            pca_folder_path=train_path,#os.path.join(model_path, "pca_components_5.npy"),
            with_class_cond=False,
        )
    )
        
#np.save(os.path.join(model_path, "pca_components_200.npy"), data[0].pca_components)
#print("saved")

# Step 3: Create data groups
# We will create two groups: one for the first subject and one for the rest of the cohort.

cavs1 = []
cavs2 = []

for i in range(len(data)):
    # Group 1
    sub_ind = i
    ind_data = [segment for segment in data[sub_ind]]
    ind_sig = torch.from_numpy(np.array([segment["signal"] for segment in ind_data]))
    #ind_cond = torch.from_numpy(np.array([segment["cond"] for segment in ind_data]))

    #sub_ind_test = 14
    #ind_data_test = [segment for segment in data[sub_ind_test]]
    #ind_sig_test = torch.from_numpy(np.array([segment["signal"] for segment in ind_data_test]))
    #ind_cond_test = torch.from_numpy(np.array([segment["cond"] for segment in ind_data_test]))


    # Group 2
    segments_per_ind = 5 # We don't take the entire recording to have a balanced dataset
    group_subjects = [subject_data for i, subject_data in enumerate(data) if i != sub_ind]
    group_data = [segment for sub_data in group_subjects for i, segment in enumerate(sub_data) if i < segments_per_ind]
    group_sig = torch.from_numpy(np.array([segment["signal"] for segment in group_data]))
    #group_cond = torch.from_numpy(np.array([segment["cond"] for segment in group_data]))

    print(f"Individual data shape: {ind_sig.shape}")
    print(f"Group data shape: {group_sig.shape}")

    #print(f"Individual cond shape: {ind_cond.shape}")
    #print(f"Group cond shape: {group_cond.shape}")


    # Step 4: Feed the data to the model and get the activations

    # Register forward hook
    target_layer = model.network.conv_pool[5]
    print(target_layer)
    target_layer.register_forward_hook(get_activation("SLConv"))

    # Forward pass individual data
    time_index = torch.full((len(ind_sig),), 0, dtype=torch.float).to(device)
    model.network.forward(ind_sig.to(device), t=time_index, cond=None)

    ind_activations = activations["SLConv"].cpu().detach().numpy()
    ind_activations = ind_activations.mean(axis=-1)

    # Test data
    #time_index = model.unormalized_probs.multinomial(num_samples=len(ind_sig_test), replacement=True).to(torch.float)
    #model.network.forward(ind_sig_test, t=time_index, cond=ind_cond_test)
    #
    #ind_test_activations = activations["SLConv"].detach().numpy()
    #ind_test_activations = ind_test_activations.mean(axis=-1)

    # Forward pass group data
    
    time_index = torch.full((len(group_sig),), 0, dtype=torch.float).to(device)
    model.network.forward(group_sig.to(device), t=time_index, cond=None)

    group_activations = activations["SLConv"].cpu().detach().numpy()
    group_activations = group_activations.mean(axis=-1)

    # Step 5: Train classifier between a group and the rest of the cohort

    ind_train_length = 60 # In seconds
    ind_train_segments = int(ind_train_length / segment_length)

    group_train_segments = 30

    # Shuffle the group data
    group_indices = np.arange(len(group_activations))
    np.random.shuffle(group_indices)
    group_activations = group_activations[group_indices]

    regression_model = SVC(kernel="linear")
    #regression_model = LogisticRegression(max_iter=1000)
    X = np.concatenate([ind_activations[:ind_train_segments], group_activations[:group_train_segments]])
    Y = np.concatenate([np.ones(ind_train_segments), np.zeros(group_train_segments)])
    print(X.shape)

    X_test = np.concatenate([ind_activations[-ind_train_segments:], group_activations[-group_train_segments:]])
    Y_test = np.concatenate([np.ones(ind_train_segments), np.zeros(group_train_segments)])

    print(X_test.shape)
    print(Y_test.shape)

    regression_model.fit(X, Y)

    print(f"Accuracy: {regression_model.score(X_test, Y_test)}")

    cav = regression_model.coef_[0]
    cavs1.append(cav)
    
    regression_model.fit(X_test, Y)
    
    cav = regression_model.coef_[0]
    cavs2.append(cav)
    
    


    #normed_cav = cav / np.linalg.norm(cav)
    #
    #scale_factor = 0.1
    #
    #X_perturbed_positive = X_test + scale_factor * normed_cav
    #X_perturbed_negative = X_test - scale_factor * normed_cav
    #
    #print(f"Pos perturbed: {regression_model.score(X_perturbed_positive, Y_test)}")
    #print(f"Neg perturbed: {regression_model.score(X_perturbed_negative, Y_test)}")
    #
    #print(cav)

# Compute the cosine similarity between the CAVs
#cav_similarities = np.zeros((len(cavs1), len(cavs2)))

#for i in range(len(cavs1)):
#    for j in range(len(cavs2)):
#        cav_similarities[i, j] = np.dot(cavs1[i], cavs2[j]) / (np.linalg.norm(cavs1[i]) * np.linalg.norm(cavs2[j]))
#        
# Show heatmap
#np.save("/meg/meg1/users/mlapatrie/data/CamCAN200/cavs1.npy", cavs1)
#np.save("/meg/meg1/users/mlapatrie/data/CamCAN200/cavs3.npy", cavs2)
#np.save("/meg/meg1/users/mlapatrie/data/CamCAN200/cav_fingerprinting.npy", cav_similarities)
#plt.imshow(cav_similarities, cmap="viridis")
#plt.show()

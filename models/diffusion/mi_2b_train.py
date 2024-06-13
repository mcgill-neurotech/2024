# Imports
import os
import wandb
import torch
import numpy as np
import scipy.io as io

import yaml # pip install pyyaml

from torch import optim
from torch.utils.data import DataLoader
import json
from datetime import datetime
import pickle

import optuna # pip install optuna (this is for the Bayesian optimization and subsequent analysis)
from lightning.fabric import Fabric
from einops import repeat

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append("../../")

from classification.classifiers import load_data, SimpleCSP
from classification.loaders import EEGDataset, CSP_subject_dataset, subject_dataset
from ntd.networks import LongConv
from ntd.diffusion_model import Diffusion
from ntd.utils.kernels_and_diffusion_utils import WhiteNoiseProcess

# Device (uses the gpu if you have one, otherwise uses the cpu)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

torch.set_float32_matmul_precision("medium")

# WandB (logs the training process. You can keep the mode="disabled" if you don't want it or use online to log it to wandb.ai)
wandb.init(project="diffusion-mi", mode="online")

# This code trains a model on the CSP components. To train on the eeg data, use the Classifier class and create your own preprocessing pipeline.

# CONSTANTS
DATA_PATH = "../../data/2b_iv"
SAVE_PATH = "../../saved_models/diff_1.pt"
CONF_PATH = "../diffusion/conf"

torch.manual_seed(0)

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

dataset = {}
for i in range(1,10):
    mat_train,mat_test = load_data("../../data/2b_iv",i)
    dataset[f"subject_{i}"] = {"train":mat_train,"test":mat_test}

REAL_DATA = "../../data/2b_iv/raw"

TRAIN_SPLIT = 9*[["train"]]
TEST_SPLIT = 9*[["test"]]

CHANNELS = [0,1,2]

train_dataset = EEGDataset(subject_splits=TRAIN_SPLIT,
                    dataset=None,
                    save_paths=[REAL_DATA],
                    subject_dataset_type=subject_dataset,
                    channels=CHANNELS,
                    sanity_check=False,
                    length=2.05)

test_dataset = EEGDataset(subject_splits=TEST_SPLIT,
                    dataset=None,
                    save_paths=[REAL_DATA],
                    channels=CHANNELS,
                    sanity_check=False,
                    length=2.05)

print(train_dataset.data[0].shape)
network_yaml["signal_length"] = train_dataset.data[0].shape[-1]
network_yaml["signal_channel"] = train_dataset.data[0].shape[1]
print(network_yaml["signal_length"])

# float-16 can have some stability problems outside of FFT
fabric = Fabric(accelerator="cuda",precision="bf16-mixed")
fabric.launch()

# EVALUATION CODE
def evaluate_generated_signals(classifier_model, generated_signals, labels):
    # Get predictions from classifier model
    classifier_model.eval()
    with torch.no_grad():
        predictions = classifier_model.predict(generated_signals)

    # Calculate accuracy
    predicted_labels = predictions.argmax(dim=1)
    accuracy = (predicted_labels == labels).float().mean().item()

    return accuracy

# GENERATE SIGNALS
def generate_samples(diffusion_model, 
                     condition,
                     k=5):
    # it's a bit hard to predict memory consumption so splitting in mini-batches to be safe
    num_samples = 250
    cond = 0
    if (condition == 0):
        cond = torch.zeros(num_samples, 1, network_yaml["signal_length"]).to(DEVICE)
    elif (condition == 1):
        cond = torch.ones(num_samples, 1, network_yaml["signal_length"]).to(DEVICE)
    
    diffusion_model.eval()

    print(f"Generating samples: cue {condition}")
    complete_samples = []
    with fabric.autocast():
        with torch.no_grad():
            for i in range(k):
                samples, _ = diffusion_model.sample(num_samples, cond=cond)
                samples = samples.cpu().numpy()
                print(samples.shape)
                complete_samples.append(samples)
    complete_samples = np.float32(np.concatenate(complete_samples))
    print(complete_samples.shape)
    return complete_samples

# Objective function for Bayesian Optimization
previous_optim_val = 0 # Keep the last value in case of an assertion error
def objective(trial):

    print(f"Starting trial {trial.number}")

    lr = 6E-4

    num_epochs = 500

    time_dim = 12
    hidden_channel = trial.suggest_int('hidden_channel', 16, 64, step=16)

    kernel_size = trial.suggest_int('kernel_size', 35, 65, step=10) 
    num_scales = trial.suggest_int('num_scales', 2, 4, step=1)

    decay_min = 2
    decay_max = 2
    
    activation_type = "leaky_relu"

    # https://github.com/fkodom/fft-conv-pytorch
    use_fft_conv = kernel_size * (2 ** (num_scales - 1)) >= 100

    num_timesteps = 1024
    schedule = "linear"
    start_beta = 0.0001
    end_beta = 0.08

    train_loader = DataLoader(
        train_dataset,
        train_yaml["batch_size"]
    )

    network = LongConv(
        signal_length=network_yaml["signal_length"],
        signal_channel=network_yaml["signal_channel"], # The CSP classifier components
        time_dim=time_dim,
        cond_channel=network_yaml["cond_channel"], # The cond channel will contain the cue (0 or 1)
        hidden_channel=hidden_channel,
        in_kernel_size=kernel_size,
        out_kernel_size=kernel_size,
        slconv_kernel_size=kernel_size,
        num_scales=num_scales,
        decay_min=decay_min,
        decay_max=decay_max,
        heads=network_yaml["heads"],
        activation_type=activation_type,
        use_fft_conv=use_fft_conv,
    ).to(DEVICE)

    noise_sampler = WhiteNoiseProcess(1.0, network_yaml["signal_length"]).to(DEVICE)

    diffusion_model = Diffusion(
        network=network,
        diffusion_time_steps=num_timesteps,
        noise_sampler=noise_sampler,
        mal_dist_computer=noise_sampler,
        schedule=schedule,
        start_beta=start_beta,
        end_beta=end_beta,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        network.parameters(),
        lr=lr,
    )

    diffusion_model,optimizer = fabric.setup(diffusion_model,optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    loss_per_epoch = []

    stop_counter = 0
    min_delta = 0.05
    tolerance = 30

    print(f"training with kernel size {kernel_size}, scale: {num_scales}, hidden_dim: {hidden_channel}")
    for i in range(num_epochs):
        
        epoch_loss = []
        for batch in train_loader:
            
            with fabric.autocast():

                signal,cue = batch
                cond = cue.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, network_yaml["signal_length"]).to(DEVICE)
                
                loss = diffusion_model.train_batch(signal.to(DEVICE), cond=cond)
            loss = torch.mean(loss)
            
            epoch_loss.append(loss.item())
            
            fabric.backward(loss)
            # loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss = np.mean(epoch_loss)
        loss_per_epoch.append(epoch_loss)
        
        wandb.log({f"loss_{trial.number}": epoch_loss,
                f"epoch":i})
        print(f"Epoch {i} loss: {epoch_loss}")

        print(f"diff: {epoch_loss - min(loss_per_epoch)}")

        trial.report(epoch_loss, i)
    
    return min(loss_per_epoch)

if __name__ == "__main__":

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15)
    sampler = optuna.samplers.TPESampler(seed=10)
    study = optuna.create_study(direction="minimize", pruner=pruner,sampler=sampler)
    study.optimize(objective, n_trials=50)

    # Analyze results
    print(f"Best trial: {study.best_trial.params}")

    timestamp = f"params_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"

    with open(f"{timestamp}.json","w") as f:
        json.dump(study.best_trial.params,f)

    # Hyperparameter importance
    importance = optuna.importance.get_param_importances(study)
    print("Hyperparameter importance:")
    for param, value in importance.items():
        print(f"{param}: {value}")

    # Save model
    # torch.save(diffusion_model.state_dict(), SAVE_PATH)


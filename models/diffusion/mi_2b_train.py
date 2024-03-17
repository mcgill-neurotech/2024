# Imports
import os
import wandb
import torch
import numpy as np
import scipy.io as io

import yaml # pip install pyyaml

from torch import optim
from torch.utils.data import DataLoader

import optuna # pip install optuna (this is for the Bayesian optimization) and subsequent analysis)

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
DATA_PATH = "D:/Neurotech/motor-imagery-classification-2024/data/2b_iv"
SAVE_PATH = "D:/Neurotech/motor-imagery-classification-2024/data/saved_models/diff_1.pt"
CONF_PATH = "D:/Neurotech/motor-imagery-classification-2024/models/diffusion/conf"
PRE_TRAINED_CLASSIFIER_PATH = "" ### FILL IN ###

# REAL DATA ACCURACY
REAL_DATA_ACCURACY_ZERO = 0.72 ### FILL IN ###
REAL_DATA_ACCURACY_ONE = 0.72 ### FILL IN ###

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

# Objective function for Bayesian Optimization
def objective(trial):
    # Hyperparameters to be optimized

    # Training parameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    num_epochs = trial.suggest_int('num_epochs', 10, 150)

    # Network parameters
    # Note that kernel sizes are same
    time_dim = trial.suggest_int('time_dim', 10, 18, step=2)
    hidden_channel = trial.suggest_int('hidden_channel', 16, 64, step=16)
    kernel_size = trial.suggest_int('kernel_size', 15, 65, step=10)
    num_scales = trial.suggest_int('num_scales', 1, 5, step=1)
    decay_min = trial.suggest_int('decay_min', 1, 4, step=1)
    decay_max = trial.suggest_int('decay_min', decay_min, 4, step=1)
    activation_type = trial.suggest_categorical('activation_type', ["gelu", "leaky_relu"])
    use_fft_conv = trial.suggest_categorical('use_fft_conv', [True, False])

    # Diffusion parameters
    # Note that start_beta seems to always be 0.001 so we don't need to test that
    num_timesteps = trial.suggest_int('num_timesteps', 100, 1000, step=100)
    schedule = trial.suggest_categorical('schedule', ["linear", "quadratic", "cosine"])
    # If the schedule is not cosine, we need to test the end_beta
    start_beta = -1
    end_beta = -1
    if schedule != "cosine":
        start_beta = network_yaml["start_beta"]
        end_beta = trial.suggest_float('end_beta', 0.01, 0.08, step=0.01)
        
    train_loader = DataLoader(
        classifier,
        train_yaml["batch_size"]
    )

    # Initialize model
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
    ).to(device)

    noise_sampler = WhiteNoiseProcess(1.0, network_yaml["signal_length"]).to(device)

    diffusion_model = Diffusion(
        network=network,
        diffusion_time_steps=num_timesteps,
        noise_sampler=noise_sampler,
        mal_dist_computer=noise_sampler,
        schedule=schedule,
        start_beta=start_beta,
        end_beta=end_beta,
    ).to(device)

    # Optimizer (also testing learning rate here)
    optimizer = optim.AdamW(
        network.parameters(),
        lr=lr,
    )

    loss_per_epoch = []

    # Train model
    for i in range(num_epochs):
        
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
    
    # Evaluate synthetic data performance
    generated_signals_zero = generate_samples(diffusion_model, condition=0)
    generated_signals_one = generate_samples(diffusion_model, condition=1)

    synthetic_accuracy_zero = evaluate_generated_signals(classifier, generated_signals_zero, labels=0)
    synthetic_accuracy_one = evaluate_generated_signals(classifier, generated_signals_one, labels=1)

    synthetic_total_accuracy = (synthetic_accuracy_zero + synthetic_accuracy_one) / 2

    # Evaluate real data performance
    real_total_accuracy = (REAL_DATA_ACCURACY_ZERO + REAL_DATA_ACCURACY_ONE) / 2

    # Compute the difference in accuracy
    accuracy_difference = abs(real_total_accuracy - synthetic_total_accuracy)

    # Log results
    wandb.log({"synthetic_accuracy": synthetic_total_accuracy})
    wandb.log({"real_accuracy": real_total_accuracy})
    wandb.log({"accuracy_difference": accuracy_difference})

    # Determine if trial should be pruned or not
    trial.report(accuracy_difference, i)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return accuracy_difference

# BAYESIAN OPTIMIZATION
# Can modify pruner as necessary (n_startup_trials refers to the number of trials
# before pruning starts. n_warmup_steps refers to the number of epochs before pruning)
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=50)

# Analyze results
print(f"Best trial: {study.best_trial.params}")

# Hyperparameter importance
importance = optuna.importance.get_param_importances(study)
print("Hyperparameter importance:")
for param, value in importance.items():
    print(f"{param}: {value}")

# Save model
# torch.save(diffusion_model.state_dict(), SAVE_PATH)

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
def generate_samples(diffusion_model, condition):
    num_samples = 30
    cond = 0
    if (condition == 0):
        cond = torch.zeros(num_samples, 1, network_yaml["signal_length"])
    elif (condition == 1):
        cond = torch.ones(num_samples, 1, network_yaml["signal_length"])
    
    diffusion_model.eval()

    print(f"Generating samples: cue {condition}")
    with torch.no_grad():
        samples, _ = diffusion_model.sample(num_samples, cond=cond)
        samples = samples.numpy()

    return samples
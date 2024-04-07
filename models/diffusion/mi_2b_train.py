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

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from classification.classifiers import load_data, CSPClassifier, Classifier
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

# Load data
### PREPROCESSES THE DATA USING THE CSP CLASSIFIER ###
#idataset_mat_diffusion = {}
#for i in range(1,8):
#	mat_train,mat_test = load_data(DATA_PATH,i)
#	dataset_mat_diffusion[f"subject_{i}"] = {"train":mat_train,"test":mat_test}
#  
#dataset_mat_classifier = {}
#for i in range(8,10):
#	mat_train,mat_test = load_data(DATA_PATH,i)
#	dataset_mat_classifier[f"subject_{i}"] = {"train":mat_train,"test":mat_test}
#
#diffusion_classifier = CSPClassifier(dataset_mat_diffusion, t_baseline=classifier_yaml["t_baseline"], t_epoch=classifier_yaml["t_epoch"])
#diffusion_classifier.set_epoch(3.5,2)
#signal_shape = diffusion_classifier.get_shape()
#print(f"signal shape: {signal_shape}")
#network_yaml["signal_length"] = signal_shape[-1]
#network_yaml["signal_channel"] = signal_shape[1]
#print(network_yaml["signal_length"])

### LOADS THE PREPROCESSED DATA ###
# Load preprocessed pickled data
preprocessed_data_path = "../../data/preprocessed_fake.pt"
preprocessed_data = torch.load(preprocessed_data_path)
print(preprocessed_data.shape)
network_yaml["signal_length"] = preprocessed_data.shape[-1]
network_yaml["signal_channel"] = preprocessed_data.shape[1]
print(network_yaml["signal_length"])


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
    # it's a bit hard to predict memory consumption so splitting in mini-batches to be safe
    num_samples = 105
    cond = 0
    if (condition == 0):
        cond = torch.zeros(num_samples, 1, network_yaml["signal_length"]).to(DEVICE)
    elif (condition == 1):
        cond = torch.ones(num_samples, 1, network_yaml["signal_length"]).to(DEVICE)
    
    diffusion_model.eval()

    print(f"Generating samples: cue {condition}")
    complete_samples = []
    with torch.no_grad():
        for i in range(6):
            samples, _ = diffusion_model.sample(num_samples, cond=cond)
            samples = samples.cpu().numpy()
            print(samples.shape)
            complete_samples.append(samples)
    complete_samples = np.concatenate(complete_samples)
    print(complete_samples.shape)
    return complete_samples

# Objective function for Bayesian Optimization
previous_optim_val = 0 # Keep the last value in case of an assertion error
def objective(trial):
    # THE FOLLOWING HYPERPARAMETERS ARE THE ONES WE ARE TRYING TO TEST AND OPTIMIZE
    # Training hyperparameters

    print(f"Starting trial {trial.number}")
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    # num_epochs = trial.suggest_int('num_epochs', 10, 150)
    num_epochs = 150
    # we don't need to optimize for number of epochs
    # we can just check for convergence

    # Network hyperparameters
    # Note that kernel sizes are same
    time_dim = trial.suggest_int('time_dim', 10, 18, step=2)
    hidden_channel = trial.suggest_int('hidden_channel', 16, 64, step=16)

    kernel_size = trial.suggest_int('kernel_size', 15, 65, step=10) 
    num_scales = trial.suggest_int('num_scales', 1, 5, step=1)

    decay_min = trial.suggest_int('decay_min', 1, 4, step=1)
    decay_max = trial.suggest_int('decay_max', decay_min, 4, step=1)

    # we can probably keep the activation type constant, it shouldn't interplay with other parameters that much at this scale
    
    # activation_type = trial.suggest_categorical('activation_type', ["gelu", "leaky_relu"])
    activation_type = "leaky_relu"
    # use_fft_conv = trial.suggest_categorical('use_fft_conv', [True, False]) 
    # FFT Conv isn't a parameter it's just the algorithm used to compute the convolution
    # https://github.com/fkodom/fft-conv-pytorch
    use_fft_conv = kernel_size * (2 ** (num_scales - 1)) >= 100

    # Diffusion hyperparameters
    # Note that start_beta seems to always be 0.001 so we don't need to test that
    # we should use a single scheduler that makes sense for prototyping and then optimize the scheduler for the best model
    # num_timesteps = trial.suggest_int('num_timesteps', 100, 1000, step=100)
    # schedule = trial.suggest_categorical('schedule', ["linear", "quad", "cosine"])
    num_timesteps = 500
    schedule = "cosine"
    # If the schedule is not cosine, we need to test the end_beta
    start_beta = -1
    end_beta = -1
    if schedule != "cosine":
        start_beta = diffusion_yaml["start_beta"]
        end_beta = trial.suggest_float('end_beta', 0.01, 0.08, step=0.01)
        
    # Load data
    train_loader = DataLoader(
        preprocessed_data,
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

    # Optimizer (also testing learning rate here)
    optimizer = optim.AdamW(
        network.parameters(),
        lr=lr,
    )

    loss_per_epoch = []

    stop_counter = 0
    min_delta = 0.05
    tolerance = 20

    try:
        # Train model
        for i in range(num_epochs):
            
            epoch_loss = []
            for batch in train_loader:
                
                # Repeat the cue signal to match the signal length
                cond = batch["cue"].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, network_yaml["signal_length"]).to(DEVICE)
                
                loss = diffusion_model.train_batch(batch["signal"].to(DEVICE), cond=cond)
                loss = torch.mean(loss)
                
                epoch_loss.append(loss.item())
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            epoch_loss = np.mean(epoch_loss)
            loss_per_epoch.append(epoch_loss)
            
            wandb.log({f"loss_{trial.number}": epoch_loss,
                    f"epoch":i})
            print(f"Epoch {i} loss: {epoch_loss}")

            print(f"diff: {epoch_loss - min(loss_per_epoch)}")

            if epoch_loss - min(loss_per_epoch) >= min_delta*min(loss_per_epoch):
                stop_counter += 1
            if stop_counter > tolerance:
                break
    except Exception as e:
        print("Error during training.\nSkipping to next trial.")
        print(e)
        return previous_optim_val
        
    
    # Evaluate synthetic data performance
    generated_signals_zero = generate_samples(diffusion_model, condition=0)
    generated_signals_one = generate_samples(diffusion_model, condition=1)
    
    accuracies = []
    kappas = []

    test_classifier = CSPClassifier(dataset_mat_classifier,
                                        t_baseline=classifier_yaml["t_baseline"],
                                        t_epoch=classifier_yaml["t_epoch"],
                                        start=classifier_yaml["start"],
                                        length=classifier_yaml["length"],)
    
    full_x,full_y = test_classifier.get_train(cut=True)
    
    # already checked at 0 with accuracy of 79%
    for real_fake_split in range(10, 90, 10):
        
        # Train new classifier with a mix of generated and real data
        
        # Change real_fake_split percent of the test_classifier data to generated signals
        n = int(len(full_x) * real_fake_split / 100)

        shuffling = np.random.permutation(full_x.shape[0])

        split_x = full_x[shuffling]
        split_y = full_y[shuffling]
        split_x[0:n//2] = generated_signals_one[0:n//2]
        split_y[0:n//2] = 1

        split_x[n//2:n] = generated_signals_zero[n//2:n]
        split_y[n//2:n] = 0

        test_classifier.fit((split_x,split_y))
        # test_classifier.eval()
        
        results = test_classifier.test(verbose=False)
        accuracies.append(results["test"][-2])
        kappas.append(results["test"][-1])
    
    best_split = np.argmax(accuracies)
    best_accuracy = accuracies[best_split]

    # Log results
    wandb.log({"best_accuracy": best_accuracy,
               "best_split":best_split,
               "trial":trial.number},)

    # Determine if trial should be pruned or not
    trial.report(best_accuracy, i)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    previous_optim_val = best_accuracy
    
    return best_accuracy 

if __name__ == "__main__":

    # BAYESIAN OPTIMIZATION
    # Can modify pruner as necessary (n_startup_trials refers to the number of trials
    # before pruning starts. n_warmup_steps refers to the number of epochs before pruning)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    sampler = optuna.samplers.TPESampler(seed=10)
    study = optuna.create_study(direction="maximize", pruner=pruner,sampler=sampler)
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


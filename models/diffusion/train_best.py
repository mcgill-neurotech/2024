from lightning.fabric import Fabric
import os
import sys
sys.path.append("../..")

from classification.classifiers import load_data, CSPClassifier, Classifier
from ntd.networks import LongConv
from ntd.diffusion_model import Diffusion
from ntd.utils.kernels_and_diffusion_utils import WhiteNoiseProcess
from torch import optim
from torch.utils.data import DataLoader
import json
from datetime import datetime
import torch
import yaml
import json
import numpy as np
import wandb

DATA_PATH = "../../data/2b_iv"
SAVE_PATH = "../../saved_models/"
if not os.path.isdir(SAVE_PATH):
	os.makedirs(SAVE_PATH)
CONF_PATH = "../diffusion/conf"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

torch.set_float32_matmul_precision("medium")

with open(os.path.join(CONF_PATH, "train.yaml"), "r") as f:
    train_yaml = yaml.safe_load(f)
    
with open(os.path.join(CONF_PATH, "classifier.yaml"), "r") as f:
    classifier_yaml = yaml.safe_load(f)
    
with open(os.path.join(CONF_PATH, "network.yaml"), "r") as f:
    network_yaml = yaml.safe_load(f)
    
with open(os.path.join(CONF_PATH, "diffusion.yaml"), "r") as f:
    diffusion_yaml = yaml.safe_load(f)


# Load data
dataset_mat_diffusion = {}
for i in range(1,8):
	mat_train,mat_test = load_data(DATA_PATH,i)
	dataset_mat_diffusion[f"subject_{i}"] = {"train":mat_train,"test":mat_test}
  
dataset_mat_classifier = {}
for i in range(8,10):
	mat_train,mat_test = load_data(DATA_PATH,i)
	dataset_mat_classifier[f"subject_{i}"] = {"train":mat_train,"test":mat_test}

diffusion_classifier = CSPClassifier(dataset_mat_diffusion, t_baseline=classifier_yaml["t_baseline"], t_epoch=classifier_yaml["t_epoch"])
diffusion_classifier.set_epoch(3.5,2.05)
signal_shape = diffusion_classifier.get_shape()
print(f"signal shape: {signal_shape}")
network_yaml["signal_length"] = signal_shape[-1]
network_yaml["signal_channel"] = signal_shape[1]
print(network_yaml["signal_length"])

with open(r"params_2024_03_28_22_46.json","r") as f:
	best_params = json.load(f)

def generate_samples(fabric,diffusion_model, condition, n_iter=20):
    # it's a bit hard to predict memory consumption so splitting in mini-batches to be safe
    num_samples = 200
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
            for i in range(n_iter):
                samples, _ = diffusion_model.sample(num_samples, cond=cond)
                samples = samples.cpu().numpy()
                print(samples.shape)
                complete_samples.append(samples)
    complete_samples = np.float32(np.concatenate(complete_samples))
    print(complete_samples.shape)
    return complete_samples

def train(fabric):
		lr = best_params["lr"]
		num_epochs = 150
		time_dim = best_params["time_dim"]
		hidden_channel = best_params["hidden_channel"]
		kernel_size = best_params["kernel_size"]
		num_scales = best_params["num_scales"]
		decay_min = 2
		decay_max = 2
		activation_type = "leaky_relu"
		use_fft_conv = kernel_size * (2 ** (num_scales - 1)) >= 100
		num_timesteps = 250
		schedule = "linear"
		# If the schedule is not cosine, we need to test the end_beta
		start_beta = 0.0001
		end_beta = 0.08
              
		train_loader = DataLoader(
			diffusion_classifier,
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
		)
              
		noise_sampler = WhiteNoiseProcess(1.0, network_yaml["signal_length"])

		diffusion_model = Diffusion(
			network=network,
			diffusion_time_steps=num_timesteps,
			noise_sampler=noise_sampler,
			mal_dist_computer=noise_sampler,
			schedule=schedule,
			start_beta=start_beta,
			end_beta=end_beta,
		)

		# Optimizer (also testing learning rate here)
		optimizer = optim.AdamW(
			network.parameters(),
			lr=lr,
		)
        
		wandb.watch(diffusion_model)
		diffusion_model,optimizer = fabric.setup(diffusion_model,optimizer)
		train_loader = fabric.setup_dataloaders(train_loader)

		loss_per_epoch = []

		stop_counter = 0
		min_delta = 0.05
		tolerance = 20
			# Train model
		for i in range(num_epochs):
			
			epoch_loss = []
			for batch in train_loader:
				
				with fabric.autocast():
				# Repeat the cue signal to match the signal length
					# print(batch["signal"].shape)
					cond = batch["cue"].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, network_yaml["signal_length"]).to(DEVICE)
					
					loss = diffusion_model.train_batch(batch["signal"].to(DEVICE), cond=cond)
				loss = torch.mean(loss)
				
				epoch_loss.append(loss.item())
				
				fabric.backward(loss)
				# loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				
			epoch_loss = np.mean(epoch_loss)
			loss_per_epoch.append(epoch_loss)
			
			wandb.log({f"loss": epoch_loss,
					f"epoch":i})
			print(f"Epoch {i} loss: {epoch_loss}")

			print(f"diff: {epoch_loss - min(loss_per_epoch)}")

			if epoch_loss - min(loss_per_epoch) >= min_delta*min(loss_per_epoch):
				stop_counter += 1
			if stop_counter > tolerance:
				break

		generated_signals_zero = generate_samples(fabric,diffusion_model, condition=0,n_iter=20)
		generated_signals_one = generate_samples(fabric,diffusion_model, condition=1,n_iter=20)
		np.save(os.path.join(SAVE_PATH,"generated_zeros.npy"),generated_signals_zero)
		np.save(os.path.join(SAVE_PATH,"generated_ones.npy"),generated_signals_one)
		torch.save(diffusion_model.state_dict(),os.path.join(SAVE_PATH,"best_model.pt"))


if __name__ == "__main__":
		wandb.init(project="diffusion-mi", mode="online",name="best_model")
		fabric = Fabric(accelerator="cuda",precision="bf16-mixed")
		fabric.launch()
		train(fabric)
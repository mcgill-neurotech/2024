from lightning.fabric import Fabric
import os
import sys
sys.path.append("../..")

from classification.classifiers import load_data, CSPClassifier, Classifier, SimpleCSP
from classification.loaders import EEGDataset, CSP_subject_dataset
from ntd.networks import LongConv, DoubleLongConv
from ntd.diffusion_model import Diffusion
from ntd.utils.kernels_and_diffusion_utils import WhiteNoiseProcess
from torch import optim
from torch.utils.data import DataLoader
import json
from datetime import datetime
import torch
from torch import nn
import yaml
import json
import numpy as np
import wandb
from einops import repeat
from pytorch_lightning.utilities.model_summary import ModelSummary
from lightning_fabric.utilities.seed import seed_everything
from classification.classifiers import DeepClassifier , SimpleCSP, k_fold_splits
from classification.loaders import subject_dataset
from models.unet.eeg_unets import Unet,UnetConfig, BottleNeckClassifier, Unet1D
import pickle
import argparse

DATA_PATH = "../../data/2b_iv"
SAVE_PATH = "../../saved_models/CSP_slc"
if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)
CONF_PATH = "../diffusion/conf"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

torch.set_float32_matmul_precision('medium')
seed_everything(0)

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

DEBUG = False

if DEBUG:
	print("---\n---\nCurrently in debug mode\n---\n---")
     
NUM_TIMESTEPS = 1000
DIFFUSION_LR = 6E-4
SCHEDULE = "linear"
START_BETA = 1E-4
END_BETA = 8E-2
DIFFUSION_NUM_EPOCHS = 180 if not DEBUG else 1
DIFFUSION_BATCH_SIZE = 64
CLASSIFICATION_MAX_EPOCHS = 100 if not DEBUG else 1
# c = np.split(np.arange(54),6)
# c = np.concatenate([c[0],c[2]])
# CHANNELS = c

train_dataset = EEGDataset(subject_splits=TRAIN_SPLIT,
                    dataset=None,
                    save_paths=[REAL_DATA],
                    dataset_type=subject_dataset,
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
with open("best_tiny.json","r") as f:
    best_params = json.load(f)

def generate_samples(fabric,
                     diffusion_model, 
					 condition,
                     batch_size=200,
                     n_iter=20,
                     w=0):
    # it's a bit hard to predict memory consumption so splitting in mini-batches to be safe
    num_samples = batch_size
    cond = 0
    if (condition == 0):
        cond = torch.zeros(num_samples, 1, network_yaml["signal_length"]).to(DEVICE)
    elif (condition == 1):
        cond = torch.ones(num_samples, 1, network_yaml["signal_length"]).to(DEVICE)
    
    diffusion_model.eval()

    print(f"Generating samples: cue {condition}")
    k = 1 if DEBUG else n_iter
    complete_samples = []
    with fabric.autocast():
        with torch.no_grad():
            for i in range(k):
                samples, _ = diffusion_model.sample(num_samples, cond=cond,w=w)
                samples = samples.cpu().numpy()
                print(samples.shape)
                complete_samples.append(samples)
    complete_samples = np.float32(np.concatenate(complete_samples))
    if DEBUG:
        complete_samples = repeat(complete_samples,"n ... -> (n k) ...",k=n_iter)
    print(complete_samples.shape)
    return complete_samples

def check(train_split,
		  test_split,
		  fake_paths,
		  channels=CHANNELS):

	accuracies = []
		
	test_classifier = SimpleCSP(train_split=train_split,
								test_split=test_split,
								dataset=None,
								save_paths=[REAL_DATA],
								channels=channels,
								length=2.05)

	full_x,full_y = test_classifier.get_train()

	print(f"full x shape: {full_x.shape}")


	real_acc = test_classifier.fit(preprocess=True)

	print(f"reaching an accuracy of {real_acc} without fake data")

	for real_fake_split in range(25, 101, 25):
		
		test_classifier = SimpleCSP(train_split=train_split,
								test_split=test_split,
								dataset=None,
								fake_paths=fake_paths,
								fake_percentage=real_fake_split/100,
								save_paths=[REAL_DATA],
								channels=channels,
								length=2.05)

		acc = test_classifier.fit(preprocess=True)

		accuracies.append(acc)
					
		print(f"Reaching an accuracy of {acc} using {real_fake_split}% fake data")

	return real_acc,accuracies

def diffusion(fabric,
          path,
          train_split,
          test_split,
          subject_id,
          train=True,
          generate=True,
          train_real=None):
        lr = 6E-4
        num_epochs = DIFFUSION_NUM_EPOCHS
        time_dim = 12
        hidden_channel = best_params["hidden_channel"]
        kernel_size = best_params["kernel_size"]
        num_scales = best_params["num_scales"]
        decay_min = 2
        decay_max = 2
        activation_type = "leaky_relu"
        use_fft_conv = kernel_size * (2 ** (num_scales - 1)) >= 100
        num_timesteps = NUM_TIMESTEPS
        schedule = "linear"
        # If the schedule is not cosine, we need to test the end_beta
        start_beta = 0.0001
        end_beta = 0.08
              
        train_set = EEGDataset(subject_splits=train_split,
                    dataset=None,
                    save_paths=[REAL_DATA],
                    dataset_type=subject_dataset,
                    channels=CHANNELS,
                    sanity_check=False,
                    length=2.05)

        test_set = EEGDataset(subject_splits=test_split,
                        dataset=None,
                        save_paths=[REAL_DATA],
                        channels=CHANNELS,
                        sanity_check=False,
                        length=2.05)
              
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
        )
              
        print(ModelSummary(network))
              
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
        tolerance = 30

        save_path = os.path.join(path,f"subject_{subject_id}")
              
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
              
            # Train model

        if train:
             
            for i in range(num_epochs):
                
                epoch_loss = []
                for batch in train_loader:
                    
                    with fabric.autocast():
                    # Repeat the cue signal to match the signal length
                        # print(batch["signal"].shape)
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
                
                wandb.log({f"loss": epoch_loss,
                        f"epoch":i})
                print(f"Epoch {i} loss: {epoch_loss}")

                print(f"diff: {epoch_loss - min(loss_per_epoch)}")

                if epoch_loss - min(loss_per_epoch) >= min_delta*min(loss_per_epoch):
                    stop_counter += 1
                if stop_counter > tolerance:
                    break

            torch.save(diffusion_model.state_dict(),os.path.join(save_path,"best_model.pt"))

        diffusion_model.load_state_dict(torch.load(os.path.join(save_path,"best_model.pt")))
        one_path = os.path.join(save_path,"generated_ones.npy")
        zero_path = os.path.join(save_path,"generated_zeros.npy")
        fake_paths = [zero_path,one_path]

        if generate:
            generated_signals_zero = generate_samples(fabric,diffusion_model, condition=0,n_iter=6,
											batch_size=275,w=0)
            generated_signals_one = generate_samples(fabric,diffusion_model, condition=1,n_iter=6,
                                            batch_size=275,w=0)
            np.save(zero_path,generated_signals_zero)
            np.save(one_path,generated_signals_one)
              
        csp_real,accuracies = check(train_split=train_split,
						test_split=test_split,
						fake_paths=fake_paths,
						channels=CHANNELS)
	
        max_acc = np.argmax(accuracies)
        print(f"Reaching a maximal accuracy of {accuracies[max_acc]} for CSP using {(max_acc+1)*15}% fake vs {csp_real}")	

        results = {"accuracies_csp":accuracies}
        results["csp_real"] = csp_real

        cnn_results = {}

        if train_real is not None:
            print(f"Training without fake data: {train_real}")

        Unet1D = UnetConfig(
            input_shape=(512),
            input_channels=3,
            conv_op=nn.Conv1d,
            norm_op=nn.InstanceNorm1d,
            non_lin=nn.ReLU,
            pool_op=nn.AvgPool1d,
            up_op=nn.ConvTranspose1d,
            starting_channels=32,
            max_channels=256,
            conv_group=1,
            conv_padding=(1),
            conv_kernel=(3),
            pool_fact=2,
            deconv_group=1,
            deconv_padding=(0),
            deconv_kernel=(2),
            deconv_stride=(2),
            residual=True
        )

        fine_tune = False

        for idx,p in enumerate([0.5,1]):

            train_real = train_real if train_real is not None else (idx==0)

            classifier = BottleNeckClassifier((2048,1024),)
            unet = Unet(Unet1D,classifier)
            if fine_tune:
                unet.load_state_dict(torch.load(os.path.join(save_path,f"unet_state_dict_{subject_id}.pt")))

            if train_real:
                fake,real = train_classification(fabric=fabric,
                                        unet=unet,
                                        fake_percentage=p,
                                        fake_paths=fake_paths,
                                        train_split=train_split,
                                        test_split=test_split,
                                        train_real=train_real,
                                        fine_tune=False,
                                        subject_id=subject_id,
                                        w=0)
                cnn_results[f"{subject_id}_real"] = real
                print(f"Reaching an accuracy of {real} without fake data")
                wandb.log({f"accuracy_{subject_id}_percentage_{0}":real})
            else:
                fake = train_classification(fabric=fabric,
                                        unet=unet,
                                        fake_percentage=p,
                                        fake_paths=fake_paths,
                                        train_split=train_split,
                                        test_split=test_split,
                                        train_real=train_real,
                                        fine_tune=False,
                                        subject_id=subject_id,
                                        w=0)
            print(f"Reaching an accuracy of {fake} with {p} fake")
            wandb.log({f"accuracy_{subject_id}_percentage_{p}":fake})
            cnn_results[f"{subject_id}_{p}_{1}"] = fake
        
        results["cnn"] = cnn_results
        return results
        


def train_classification(fabric,
						 unet,
						 fake_percentage,
						 fake_paths,
						 train_split,
						 test_split,
						 train_real,
						 fine_tune,
						 subject_id,
						 w):
	
	deep_clf = DeepClassifier(
		model=unet,
		save_paths=[REAL_DATA],
		fake_data=fake_paths,
		train_split=train_split,
		test_split=test_split,
		fake_percentage=fake_percentage,
		dataset=None,
		dataset_type=subject_dataset,
		length=2.05,
		index_cutoff=512,
        channels=CHANNELS
	)

	with_fake = deep_clf.fit(fabric=fabric,
			 num_epochs=CLASSIFICATION_MAX_EPOCHS,
			 lr=1E-4,
			 weight_decay=1E-4,
			 verbose=True,
			 optimizer=None,
			 stop_threshold=10,
			 log=True,
			 id=f"subject_{subject_id}_percentage_{fake_percentage}_weight_{w}")
	
	if fine_tune:
		print("\n---\nFine-tuning model\n---\n")
		to_fine_tune = [unet.encoder,
			unet.decoder,
			unet.middle_conv,
			unet.class_embed,]

		to_optimize = [{"params":i.parameters(),
			"lr":2E-5,
			"weight_decay":1E-4} for i in to_fine_tune]

		to_optimize.append({"params":unet.auxiliary_clf.parameters(),
			"lr":1E-4,
			"weight_decay":1E-4})

		optimizer = optim.AdamW(to_optimize)
	else:
		optimizer = None
	
	deep_clf.setup_dataloaders(use_fake=False)
	if train_real:
		without_fake = deep_clf.fit(fabric=fabric,
				num_epochs=CLASSIFICATION_MAX_EPOCHS,
				lr=1E-4,
				weight_decay=1E-4,
				verbose=True,
				optimizer=optimizer,
				stop_threshold=10,
				log=True,
				id=f"subject_{subject_id}_percentage_{0}")
		
		return with_fake,without_fake
	else:
		return with_fake


def train_clf(fabric,
              subject_id,
              save_path,
              fake_paths,
              fine_tune=False,
              w=0):
    cnn_results = {}

    if train_real is not None:
        print(f"Training without fake data: {train_real}")

    for idx,p in enumerate([0.5,1]):

        train_real = train_real if train_real is not None else (idx==0)

        classifier = BottleNeckClassifier((2048,1024),)
        unet = DiffusionUnet(UnetDiff1D,classifier)
        if fine_tune:
            unet.load_state_dict(torch.load(os.path.join(save_path,f"unet_state_dict_{subject_id}.pt")))

        if train_real:
            fake,real = train_classification(fabric=fabric,
                                    unet=unet,
                                    fake_percentage=p,
                                    fake_paths=fake_paths,
                                    train_split=train_split,
                                    test_split=test_split,
                                    train_real=train_real,
                                    fine_tune=fine_tune,
                                    subject_id=subject_id,
                                    w=w)
            cnn_results[f"{subject_id}_real"] = real
            print(f"Reaching an accuracy of {real} without fake data")
            wandb.log({f"accuracy_{subject_id}_percentage_{0}":real})
        else:
            fake = train_classification(fabric=fabric,
                                    unet=unet,
                                    fake_percentage=p,
                                    fake_paths=fake_paths,
                                    train_split=train_split,
                                    test_split=test_split,
                                    train_real=train_real,
                                    fine_tune=fine_tune,
                                    subject_id=subject_id,
                                    w=w)
        print(f"Reaching an accuracy of {fake} with {p} fake")
        wandb.log({f"accuracy_{subject_id}_percentage_{p}":fake})
        cnn_results[f"{subject_id}_{p}_{w}"] = fake

    return cnn_results

def k_fold(experiment_name,
		   k=9,
		   n=9,
		   w=0,
		   train=True,
		   fine_tune=False,
		   train_real=None,
		   generate=True):
	
    fabric = Fabric(accelerator="cuda",precision="bf16-mixed")

    splits = k_fold_splits(k,n,leave_out=False)

    for split in splits:
        print(f"train: {split[0]}")
        print(f"test: {split[1]}")

    results = {}

    full_folder = os.path.join(SAVE_PATH,experiment_name)

    for idx,split in enumerate(splits):
            
        results_k = diffusion(fabric=fabric,
                              path=full_folder,
                              train_split=split[0],
                              test_split=split[1],
                              subject_id=idx,
                              train=train,
                              generate=generate)

        results[f"split_{idx}"] = results_k

    with open(os.path.join(full_folder,f"results_{w}.p"),"wb") as f:
        pickle.dump(results,f)



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("-n","--name",help="experiment name",type=str)
        parser.add_argument("-k","--k_fold",help="number of folds",type=int)

        args = parser.parse_args() 

        wandb.init(project="slc-diffusion-mi", mode="disabled",
                name=args.name)
        
        k_fold(args.name,args.k_fold,9,w=0,train=False,train_real=None,generate=True)
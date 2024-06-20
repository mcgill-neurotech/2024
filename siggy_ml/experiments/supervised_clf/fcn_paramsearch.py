import sys
sys.path.append("../../../motor-imagery-classification-2024/")

import torch
from torch import nn
import optuna
from datetime import datetime
import json
from classification.classifiers import DeepClassifier
from classification.loaders import load_data
from models.unet.eeg_unets import UnetConfig,Unet,BottleNeckClassifier
import lightning as L
from lightning import Fabric
import wandb

torch.set_float32_matmul_precision('medium')

FABRIC = Fabric(accelerator="cuda",precision="bf16-mixed")

def objective(trial):
	conv_kernel = (trial.suggest_int("conv_kernel",3,23,step=4))
	bottleneck_dim = trial.suggest_int("max_channels",128,1024,step=128)
	pool = trial.suggest_categorical("pool_op",choices=["avg","max"])

	pool_op = nn.AvgPool1d if pool == "avg" else nn.MaxPool1d

	config = UnetConfig(
		input_shape=(256),
		input_channels=3,
		conv_op=nn.Conv1d,
		norm_op=nn.InstanceNorm1d,
		non_lin=nn.ReLU,
		pool_op=pool_op,
		up_op=nn.ConvTranspose1d,
		starting_channels=trial.suggest_int("starting_channels",8,32,step=8),
		max_channels=bottleneck_dim,
		conv_group=1,
		conv_kernel=conv_kernel,
		conv_padding=conv_kernel//2,
		pool_fact=2,
		deconv_group=1,
		deconv_padding=(0),
		deconv_kernel=(2),
		deconv_stride=(2),
		residual=trial.suggest_categorical("residual",choices=[True,False]),
		conv_pdrop=trial.suggest_float("conv_drop",0,0.25,step=0.05)
	)

	unet = Unet(config,nn.Identity)
	classifier = BottleNeckClassifier((unet.out_shape[-1],trial.suggest_int("mlp_dim",128,1024,step=128)),pool="max")
	unet.auxiliary_clf = classifier
	unet.to("cuda")

	train_split = 6*[["train","test"]] + 3*[["train"]]
	test_split = 6*[[]] + 3* [["test"]]

	clf = DeepClassifier(unet,dataset,train_split=train_split,test_split=test_split)

	lr = trial.suggest_float('lr', 1e-5, 1e-3,log=True)
	weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2,log=True)

	max_accuracy = clf.fit(FABRIC,
						num_epochs=50,
						lr=lr,
						weight_decay=weight_decay,
						verbose=False)
	print(f"---\nreached an accuracy of {max_accuracy}\n---")

	wandb.log({"max_accuracy":max_accuracy})

	return max_accuracy


if __name__ == "__main__":

	wandb.init(project="mi-classification",mode="online")

	pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
	sampler = optuna.samplers.TPESampler(seed=0)
	study = optuna.create_study(direction="maximize", pruner=pruner,sampler=sampler)

	dataset = {}
	for i in range(1,10):
		mat_train,mat_test = load_data("../../data/2b_iv/",i)
		dataset[f"subject_{i}"] = {"train":mat_train,"test":mat_test}

	study.optimize(objective,200)

	print(f"Best trial: {study.best_trial.params}")

	timestamp = f"params_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"

	with open(f"{timestamp}.json","w") as f:
		json.dump(study.best_trial.params,f)
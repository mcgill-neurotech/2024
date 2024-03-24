import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,dataloader
from scipy.signal import filtfilt, iirnotch, butter
from classifiers import subject_dataset
from einops import rearrange
import os
import scipy

def load_data(folder,idx):
	path_train = os.path.join(folder,f"B0{idx}T.mat")
	path_test = os.path.join(folder,f"B0{idx}E.mat")
	mat_train = scipy.io.loadmat(path_train)["data"]
	mat_test = scipy.io.loadmat(path_test)["data"]
	return mat_train,mat_test

class EEGDataset(Dataset):
	def __init__(self,
				 dataset, 
				 subject_splits,
				 fs=250, 
				 t_baseline=0, 
				 t_epoch=9,
				 start=3.5,
				 length=2):
		
		self.fs = fs
		self.t_baseline = t_baseline
		self.t_epoch = t_epoch
		self.data = self.load_data(dataset,subject_splits)
		self.set_epoch(start,length)

	def __len__(self):
		return self.data[0].shape[0]

	def __getitem__(self, idx):
		return self.data[0][idx,:,int(self.input_start*250):int(self.input_end*250)], self.data[1][idx]

	def load_data(self,
				  dataset,
				  subject_splits):
		
		"""
		Function for getting the full dataset into a numpy array
		The subject_dataset does the heavy lifting of getting the data into numpy.
		This function mostly helps against leakage for combining subject data without session leakage.
		"""
		
		epochs = []
		cues = []

		
		for (k,v),splits in zip(dataset.items(),subject_splits):
			for split in splits:
				set = subject_dataset(v[split],self.fs,self.t_baseline,self.t_epoch)
				epochs.append(np.float32(set.epochs))
				cues.append(set.cues)

		epochs = rearrange(np.concatenate(epochs,0),"n t d -> n d t")
		cues = np.concatenate(cues,0)
		epochs, cues = self.preprocess(epochs,cues)
		return ((np.float32(epochs).copy()),cues.copy())
	
	def set_epoch(self,start,length):
		self.input_start = start + self.t_baseline
		self.input_end = self.input_start + length
	
	def filter(self,
			x,
			notch_freq=50):
		
		nyquist = self.fs/2
		b,a = iirnotch(notch_freq,30,self.fs)
		x = filtfilt(b,a,x)
		return x

	def preprocess(self,x,y):
		"""
		Apply filters and additional preprocessing to measurements
		"""
		n,t,d = x.shape
		x = rearrange(x,"n d t -> (n d) t")
		x = self.filter(x)
		x = rearrange(x,"(n d) t -> n d t",n=n)
		return x,y
	
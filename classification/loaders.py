import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,dataloader
from scipy.signal import filtfilt, iirnotch, butter
from einops import rearrange
import os
import scipy

def load_data(folder,idx):
	path_train = os.path.join(folder,f"B0{idx}T.mat")
	path_test = os.path.join(folder,f"B0{idx}E.mat")
	mat_train = scipy.io.loadmat(path_train)["data"]
	mat_test = scipy.io.loadmat(path_test)["data"]
	return mat_train,mat_test

class subject_dataset:

	def __init__(self,
			  mat,
			  fs = 250,
			  t_baseline = 0.3,
			  t_epoch = 4):

		self.fs = fs
		self.t_baseline = t_baseline
		self.t_epoch = t_epoch

		self.epochs = []
		self.cues = []
		self.dfs = []
		self.timestamps = []

		for i in range(mat.shape[-1]):
			epochs,cues,df = self.load_data(mat,i)
			self.epochs.append(epochs)
			self.cues.append(cues)
			self.dfs.append(df)

		self.epochs = np.concatenate(self.epochs,axis=0)
		# removing 1 to get 0,1 instead of 1,2
		self.cues = np.concatenate(self.cues,0)	- 1	

	def  load_data(self,mat,index=0):

		"""
		Loading the epochs,cues, and dataframe for one trial
		For now, we are not removing samples with artifacts
		since it can't be done on the fly.

		Args:
			mat: array
			index: trial index

		Returns:
			epochs: array of epoch electrode data
			cues: array of cues (labels) for each epoch
			df: dataframe of trial data
		"""
		
		electrodes = mat[0][index][0][0][0].squeeze()
		timestamps = mat[0][index][0][0][1].squeeze()
		cues = mat[0][index][0][0][2].squeeze()
		artifacts = mat[0][index][0][0][5]

		epochs,cues = self.create_epochs(electrodes,timestamps,artifacts,cues,
							  self.t_baseline,self.t_epoch,self.fs)
		
		
		df = self.load_df(timestamps,electrodes,cues)
		return epochs,cues,df

	def create_epochs(self,
				   electrodes,
				   timestamps,
				   artifacts,
				   cues,
				   t_baseline,
				   t_epoch,
				   fs):
		
		"""
		creating an array with all epochs from a trial

		Args:
			electrode: electrode data
			timestamps: cue indices
			artifacts: artifact presence array
			cues: labels for cues
			t_baseline: additional time for baseline
			t_epoch: time of epoch
		"""
		
		n_samples = int(fs*t_epoch)
		n_samples_baseline = int(fs*t_baseline)
		epochs = []
		cues_left = []
		for i,j,c in zip(timestamps,artifacts,cues):
			if j==0:
				epochs.append(electrodes[i-n_samples_baseline:i+n_samples,:])
				cues_left.append(c)
		epochs = np.stack(epochs)
		cues_left = np.asarray(cues_left)
		return epochs,cues_left

	def load_df(self,
			 timestamps,
			 electrodes,
			 cues):

		"""
		Loading the channel values and adding a timestamp column.
		Useful for visualizing an entire trial

		Args:
			None
		Returns:
			None
		"""

		timeline = np.zeros_like(electrodes[:,0])

		for t,c in zip(timestamps,cues):
			timeline[t] = c

		df = pd.DataFrame(electrodes)

		df["timestamps"] = timeline

		return df
	
	def trial_preprocess(x,*args,**kwargs):
		"""
		Pre-processing step to be applied to entire trial.
		"""
		return x



class EEGDataset(Dataset):
	def __init__(self,
				 dataset, 
				 subject_splits,
				 fs=250, 
				 t_baseline=0, 
				 t_epoch=9,
				 start=3.5,
				 length=2,
				 channels = np.array([0,1,2])):
		
		self.fs = fs
		self.t_baseline = t_baseline
		self.t_epoch = t_epoch
		self.data = self.load_data(dataset,subject_splits,channels)
		self.set_epoch(start,length)

	def __len__(self):
		return self.data[0].shape[0]

	def __getitem__(self, idx):
		return self.data[0][idx,:,int(self.input_start*250):int(self.input_end*250)], self.data[1][idx]

	def load_data(self,
				  dataset,
				  subject_splits,
				  channels):
		
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

		epochs = rearrange(np.concatenate(epochs,0),"n t d -> n d t")[:,channels,:]
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
	
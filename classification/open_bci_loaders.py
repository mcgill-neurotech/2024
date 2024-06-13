import sys

sys.path.append("../../motor-imagery-classification-2024/")

import os
import pandas as pd
import numpy as np
from einops import rearrange
from scipy.signal import filtfilt, iirnotch, butter
from classification.loaders import subject_dataset,EEGDataset
from typing import Optional, Iterable
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def load_files(folder):
	d = {}
	subjects = []
	files = os.listdir(folder)
	for f in files:
		df = pd.read_csv(os.path.join(folder,f))
		subject = str.split(f,"_")[-1]
		if subject not in d.keys():
			d[subject] = [df]
			subjects.append(subject)
		else:
			d[subject].append(df)
	for s in subjects:
		d[s] = {"train":d[s][:-1],"test":[d[s][-1]]}
	return d

def _get_epochs(df,
				indices,
				n_samples_baseline,
				n_samples,
				epochs = [],
				subject_channels=["ch1","ch2","ch3","ch4"]):
		for i in indices:
			# length x n_channels
			epoch = df.loc[i-n_samples_baseline:i+n_samples][subject_channels]
			epochs.append(epoch)
		return epochs

def get_epochs(dfs,
			   fs,
			   t_epoch,
			   t_baseline=0,
			   subject_channels=["ch1","ch2","ch3","ch4"]):

	left_epochs = []
	right_epochs = []

	n_samples = int(fs*t_epoch)
	n_samples_baseline = int(fs*t_baseline)
	
	for df in dfs:
		t = df["timestamp"]
		# print((max(t) - min(t))/60)
		indices = np.arange(len(df))

		left_indices = indices[df["left"] == 1]
		right_indices = indices[df["right"] == 1]

		left_epochs = _get_epochs(df,left_indices,n_samples_baseline,
							n_samples,left_epochs,subject_channels)
		
		right_epochs = _get_epochs(df,right_indices,n_samples_baseline,
							n_samples,right_epochs,subject_channels)
		
	left_epochs = np.stack(left_epochs,0)
	right_epochs = np.stack(right_epochs,0)
	return left_epochs,right_epochs

def sliding_window_view(arr, window_size, step, axis):
	shape = arr.shape[:axis] + ((arr.shape[axis] - window_size) // step + 1, window_size) + arr.shape[axis+1:]
	strides = arr.strides[:axis] + (arr.strides[axis] * step, arr.strides[axis]) + arr.strides[axis+1:]
	strided = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
	return rearrange(strided,"n e ... -> (n e) ...")

def subject_epochs(dfs,
				   subject_channels=["ch2","ch3","ch4","ch5"],
				   stride=25,
				   epoch_length=512):
	left,right = get_epochs(dfs,256,8,0,subject_channels=subject_channels)
	n,l,d = left.shape
	left_epochs = sliding_window_view(left,epoch_length,stride,1)
	right_epochs = sliding_window_view(right,epoch_length,stride,1)
	left_y = np.zeros(len(left_epochs))
	right_y = np.ones(len(right_epochs))
	xs = np.concatenate((left_epochs,right_epochs))
	ys = np.concatenate((left_y,right_y))
	return xs,ys

class OpenBCISubject(subject_dataset):

	def __init__(self, 
			  dfs,
			  subject_channels,
			  fs=256,
			  t_baseline=0,
			  t_epoch=8,
			  stride=25,
			  epoch_length=512):
		
		self.fs = fs
		self.t_baseline = t_baseline
		self.t_epoch = t_epoch

		self.dfs = dfs

		self.epochs,self.cues = subject_epochs(dfs,subject_channels,stride=stride,
										 epoch_length=epoch_length)
		self.epochs,self.cues = self.epoch_preprocess(self.epochs,self.cues)

	def epoch_preprocess(self, x, y, notch_freq=60,low=4,high=50):

		n,t,d = x.shape
		x = rearrange(x,"n t d -> (n d) t")
		b,a = iirnotch(notch_freq,30,self.fs)
		x = filtfilt(b,a,x)
		nyquist = self.fs/2
		b,a = butter(4,[low/nyquist,high/nyquist],"bandpass",analog=False)
		x = filtfilt(b,a,x)
		x = rearrange(x,"(n d) t -> n t d",d=d)
		return x,y

class OpenBCIDataset(EEGDataset):

	def __init__(self,
		subject_splits:list[list[str]],
		dataset:Optional[dict] = None,
		save_paths:Optional[list[str]] = None,
		fake_data=None,
		dataset_type: Optional[subject_dataset] = None,
		fake_percentage:float = 0.5,
		fs:float = 256, 
		t_baseline:float = 0, 
		t_epoch:float = 8,
		channels:Iterable = np.array([0,1,2]),
		sanity_check:bool=False,
		**kwargs,):

		self.fs = fs
		self.t_baseline = t_baseline
		self.t_epoch = t_epoch
		self.fake_percentage = fake_percentage
		self.save_paths = save_paths

		if dataset is None:
			print("Loading saved data")
			self.data = self.load_data(save_paths,subject_splits,channels)
		else:
			print("Saving new data")
			self.save_dataset(dataset,save_paths[0],dataset_type,**kwargs)
			self.data = self.load_data(save_paths,subject_splits,channels)

		if fake_data is not None:
			print("we have fake data")
			self.data = self.load_fake(*fake_data)

		if sanity_check:
			self.sanity_check()

		print(f"final data shape: {self.data[0].shape}")

	def save_dataset(self, 
			dataset: dict, 
			path: str, 
			dataset_type: OpenBCISubject,
			**kwargs,):
		
		if not os.path.isdir(path):
			os.makedirs(path)

		for idx,(k,subject) in enumerate(dataset.items()):
			for split in ["train","test"]:
				set = dataset_type(subject[split],fs=self.fs,t_baseline=self.t_baseline,
								   t_epoch=self.t_epoch,**kwargs)
				epochs = np.float32(set.epochs)
				cues = set.cues
				np.save(os.path.join(path,f"subject_{idx}_{split}_epochs.npy"),epochs)
				np.save(os.path.join(path,f"subject_{idx}_{split}_cues.npy"),cues)

	def load_data(self,
			   paths,
			   subject_splits,
			   channels):
		
		epochs = []
		cues = []

		for path in paths:

			for idx,splits in enumerate(subject_splits):
				for split in splits:
					epochs.append(np.load(os.path.join(path,f"subject_{idx}_{split}_epochs.npy")))
					cues.append(np.load(os.path.join(path,f"subject_{idx}_{split}_cues.npy")))

		epochs = rearrange(np.concatenate(epochs,0),"n t d -> n d t")[:,channels,:]
		cues = np.concatenate(cues,0)

		print(epochs.shape)
		print(cues.shape)

		epochs, cues = self.preprocess(epochs,cues)
		return ((np.float32(epochs).copy()),cues.copy())
	
	def plot_sample(self):

		x = self.data[0][0]
		d,t = x.shape

		plt.figure(figsize=(10,d*2))

		for i in range(d):

			plt.subplot(d,1,i+1)
			plt.plot(x[i])
			plt.title(f"Channel {i+1}")
			plt.xlabel("")
			plt.ylabel("Amplitude ($\mu V$)")

			if i < d - 1:
				plt.xticks([])

		plt.tight_layout()
		plt.show()

class CSPOpenBCISubject(OpenBCISubject):

	def __init__(self, 
			  dfs,
			  subject_channels,
			  fs=256,
			  t_baseline=0,
			  t_epoch=8,
			  **kwargs):
		super().__init__(dfs,subject_channels, fs, t_baseline, t_epoch,**kwargs)

	def epoch_preprocess(self, x, y, notch_freq=60, low=4, high=50):
		x,y = super().epoch_preprocess(x, y, notch_freq, low, high)

		ax = []

		for i in range(1,10):
			ax.append(self.bandpass(x,self.fs,4*i,4*i+4))
		x = np.concatenate(ax,-1)
		mu = np.mean(x,axis=-1)
		sigma = np.std(x,axis=-1)
		x = (x-rearrange(mu,"n d -> n d 1"))/rearrange(sigma,"n d -> n d 1")
		return x,y

	def bandpass(self,
			  x,
			  fs,
			  low,
			  high,):
		
		nyquist = fs/2
		b,a = butter(4,[low/nyquist,high/nyquist],"bandpass",analog=False)
		n,t,d = x.shape
		x = rearrange(x,"n t d -> (n d) t")
		x = filtfilt(b,a,x)
		x = rearrange(x,"(n d) t -> n t d",n=n)
		return x
	
if __name__ == "__main__":
	print(f"path {os.getcwd()}")
	files = load_files("data/collected_data")
	train_split = 2*[["train"]]
	test_split = 2*[["test"]]
	save_path = os.path.join("processed","raw")
	csp_save_path = os.path.join("processed","data/collected_data/csp")

	train_csp_dataset = OpenBCIDataset(
		subject_splits=train_split,
		dataset=files,
		save_paths=[csp_save_path],
		fake_data=None,
		dataset_type=CSPOpenBCISubject,
		channels=np.arange(0,2*9),
		subject_channels=["ch2","ch5"],
		stride=128,
		epoch_length=512
	)

	test_csp_dataset = OpenBCIDataset(
		subject_splits=test_split,
		dataset=files,
		save_paths=[csp_save_path],
		fake_data=None,
		dataset_type=CSPOpenBCISubject,
		channels=np.arange(0,2*9),
		subject_channels=["ch2","ch5"],
		stride=128,
		epoch_length=512
	)

	x_train,y_train = train_csp_dataset.data
	x_test,y_test = test_csp_dataset.data
	x_train,y_train = np.float64(x_train),np.float64(y_train)
	x_test,y_test = np.float64(x_test),np.float64(y_test)

	csp = CSP(n_components=x_train.shape[1],reg=None,log=True,norm_trace=False)
	svm = SVC(C=1)

	clf = Pipeline(steps=[("csp",csp),
						("classification",svm)])

	clf.fit(x_train,y_train)

	y_train_pred = clf.predict(x_train)
	y_test_pred = clf.predict(x_test)
	train_acc = accuracy_score(y_train,y_train_pred)
	test_acc = accuracy_score(y_test,y_test_pred)

	print(f"train accuracy: {train_acc}")
	print(f"test accuracy: {test_acc}")

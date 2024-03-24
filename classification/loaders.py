import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,dataloader
from .classifiers import subject_dataset
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
                 t_baseline=0.3, 
                 t_epoch=4):
        
        self.fs = fs
        self.t_baseline = t_baseline
        self.t_epoch = t_epoch
        self.data = self.load_data(dataset,subject_splits)

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

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
        return (epochs,cues)

    def preprocess(self,x,y):
        """
		Apply filters and additional preprocessing to measurements
		"""
        n,t,d = x.shape
        x = rearrange(x,"n d t -> (n d) t")
        x = self.filter(x)
        x = rearrange(x,"(n d) t -> n d t",n=n)
        return x,y
    
    def filter(self,x):
        """
		Apply a sequence of filters to a set of measurements
		"""
        return x
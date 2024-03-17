import os
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from einops import rearrange
from scipy.signal import filtfilt, iirnotch, butter
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import ConfusionMatrixDisplay
from mne.decoding import CSP


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

	def load_data(self,mat,index=0):

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


class Classifier:

    def __init__(self,
            dataset,
            t_baseline=0,
            t_epoch=4,
            fs=250):
    
        self.fs = fs
        self.t_epoch = t_epoch
        self.t_baseline = t_baseline
        self.train_epochs = []
        self.train_y = []
        self.val_epochs = []
        self.val_y = [] 
        for k,v in dataset.items():
            s_train = subject_dataset(v["train"],fs,t_baseline,t_epoch)
            s_val = subject_dataset(v["test"],fs,t_baseline,t_epoch)
            self.train_epochs.append(s_train.epochs)
            self.train_y.append(s_train.cues)
            self.val_epochs.append(s_val.epochs)
            self.val_y.append(s_val.cues)   
        self.train_epochs = rearrange(np.concatenate(self.train_epochs,0),"n t d -> n d t")
        self.val_epochs = rearrange(np.concatenate(self.val_epochs,0),"n t d -> n d t")
        self.train_y = np.concatenate(self.train_y,0)
        self.val_y = np.concatenate(self.val_y,0)   
        self.x_train,self.y_train = self.preprocess(self.train_epochs,self.train_y)
        self.x_val,self.y_val = self.preprocess(self.val_epochs,self.val_y) 
    def fit(self,*args,**kwargs):
        pass    
    def filter(self,x): 
        """
        Apply a sequence of filters to a set of measurements
        """
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

    def predict(self,x):    
        """
        Returns a continuous and a discrete prediction
        """
        
        y = np.random.random(x.shape[0])
        return y,np.round(y)
    
    def get_train(self):
        return self.x_train,self.y_train
    
    def get_test(self):
        return self.x_val,self.y_val
    
    def test(self,
    	  verbose=True):   
        outs = {}
        for split in ["train","test"]:
            x,y = self.get_test() if split == "test" else self.get_train()
            y_pred,y_discrete = self.predict(x)
            
            acc = accuracy_score(y,y_discrete)
            confusion = confusion_matrix(y,y_discrete,normalize="true") 
            kappa = cohen_kappa_score(y_discrete,y) 
            if verbose: 
                print(f"{split} kappa score: {kappa}")  
                print(f"{split} accuracy: {acc}")
                print(confusion)
                cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels = [False, True])
                cm_display.plot()
                plt.show()
            outs[split] = (y,y_pred,y_discrete,acc,kappa)
        return outs
    
    def plot_first(self,i=0):   
        plt.plot(self.x_train[i,0,:],label="clean")
        plt.plot(self.train_epochs[i,0,:],alpha=0.5,label="raw")
        plt.legend()
        plt.show()
        
    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, index):
        return {"signal": self.x_train[index].astype(np.float32), "cue": self.y_train[index].astype(np.float32)}
  
  
class CSPClassifier(Classifier):

    def __init__(self,
                dataset,
                t_baseline,
                t_epoch,
                start=3.75,
                length = 0.5,
                fs=250):

        super().__init__(dataset,t_baseline,t_epoch,fs)

        self.csp = CSP(n_components=18,reg=None,log=True,norm_trace=False)
        self.svm = SVC(C=1)

        self.clf = Pipeline(steps=[("csp",self.csp),
                            ("selection",SelectKBest(mutual_info_classif)),
                            ("classification",self.svm)])
        self.set_epoch(start,length)

    def set_epoch(self,start,length):
        self.input_start = start + self.t_baseline
        self.input_end = self.input_start + length

    def gridCV(self,
            param_grid,
            n_jobs = 4):
        x,y = self.get_train()
        x = x[:,:,int(self.input_start*250):int(self.input_end*250)]
        search = GridSearchCV(self.clf,param_grid,n_jobs=n_jobs)
        search.fit(x,y)
        print(search.best_params_)
        print(f"accuracy of: {search.best_estimator_.score(x,y)}")
        self.clf = search.best_estimator_
		
    def fit(self):
        x,y = self.get_train()
        x = x[:,:,int(self.input_start*250):int(self.input_end*250)]
        self.clf.fit(x,y)

    def predict(self, x):
        x = x[:,:,int(self.input_start*250):int(self.input_end*250)]
        y_cont = self.clf.decision_function(x)
        y_discrete = self.clf.predict(x)
        return y_cont,y_discrete
	
    def filter(self,
            x,
            notch_freq=50):
		
        nyquist = self.fs/2
        b,a = iirnotch(notch_freq,30,self.fs)
        x = filtfilt(b,a,x)
        return x
	
    def passband(self,
                x,
                low,
                high):
		
        nyquist = self.fs/2
        b,a = butter(4,[low/nyquist,high/nyquist],"bandpass",analog=False)
        n,d,t = x.shape
        x = rearrange(x,"n d t -> (n d) t")
        x = filtfilt(b,a,x)
        x = rearrange(x,"(n d) t -> n d t",n=n)
        return x
	
    def preprocess(self, x, y):
        x,y = super().preprocess(x, y)
        x = x[:,[0,2],:]
        ax = []
        
        for i in range(1,10):
            ax.append(self.passband(x,4*i,4*i+4))

        x = np.concatenate(ax,1)
        mu = np.mean(x,axis=-1)
        sigma = np.std(x,axis=-1)
        x = (x-rearrange(mu,"n d -> n d 1"))/rearrange(sigma,"n d -> n d 1")
        return x,y
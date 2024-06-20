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
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
import copy
from typing import Optional, Iterable
from einops import reduce
import wandb

import sys

sys.path.append("../../motor-imagery-classification-2024/")

from classification.loaders import EEGDataset, subject_dataset, CSP_subject_dataset
from models.unet.eeg_unets import Unet,UnetConfig, BottleNeckClassifier, Unet1D

def load_data(folder,idx):
    path_train = os.path.join(folder,f"B0{idx}T.mat")
    path_test = os.path.join(folder,f"B0{idx}E.mat")
    mat_train = scipy.io.loadmat(path_train)["data"]
    mat_test = scipy.io.loadmat(path_test)["data"]
    return mat_train,mat_test

def k_fold_splits(k=9,
				  n_participants=9,
				  leave_out=False):

	participants = np.arange(n_participants)
	np.random.seed(0)
	np.random.shuffle(participants)
	np.random.shuffle(participants)
	val_folds = np.array_split(participants,k)
	folds = []
	for fold in val_folds:
		train_split = []
		test_split = []
		for i in range(n_participants):
			if i in fold:
				if leave_out:
					train_split.append([])
				else:
					train_split.append(["train"])
				test_split.append(["test"])
			else:
				train_split.append(["train","test"])
				test_split.append([])
		folds.append([train_split,test_split])
	return folds


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
    
    def get_shape(self):
        return self.x_train.shape
    
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
                            ("selection",SelectKBest(mutual_info_classif,k=10)),
                            ("classification",self.svm)])
        self.set_epoch(start,length)
        self.clf = Pipeline(steps=[("csp",self.csp),
                            ("selection",SelectKBest(mutual_info_classif)),
                            ("classification",self.svm)])
        self.set_epoch(start,length)

    def set_epoch(self,start,length):
        self.input_start = start + self.t_baseline
        self.input_end = self.input_start + length

    def get_train(self,
                  cut=False):
        if cut:
            x,y = self.get_train()
            x = x[:,:,int(self.input_start*250):int(self.input_end*250)]
            return x,y
        else:
            return super().get_train()

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
        
    def fit(self,
            dataset=None):
        if dataset == None:
            x,y = self.get_train()
            x = x[:,:,int(self.input_start*250):int(self.input_end*250)]
            self.clf.fit(x,y)
        else:
             self.clf.fit(*dataset)
        

    def get_shape(self):
         return self.x_train[:,:,int(self.input_start*250):int(self.input_end*250)].shape

    def predict(self,
                x,
                cut=True):
        if cut:
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
    
class DeepClassifier:
    def __init__(self,
                 model,
                 save_paths:list[str],
                 train_split:list[list[str]],
                 test_split:list[list[str]],
                 dataset:Optional[dict] = None,
				 subject_dataset_type: Optional[subject_dataset] = None,
                 channels:Iterable = np.array([0,1,2]),
                 fake_data: Optional[list[str]] = None,
                 fake_percentage: float = 0.5,
                 batch_size:int = 32, 
                 fs:float = 250, 
				 t_baseline:float = 0, 
				 t_epoch:float = 9,
				 start:float = 3.5,
				 length:float = 2,
                 index_cutoff:int = 256,
                 sanity_check:bool = False,
                 **dataset_kwargs):
        self.fs = fs
        self.t_epoch = t_epoch
        self.t_baseline = t_baseline
        self.batch_size = batch_size
        self.subject_dataset_type = subject_dataset_type
        self.start = start
        self.length = length
        self.channels = channels
        self.save_paths = save_paths
        self.fake_data = fake_data
        self.dataset = dataset
        self.train_split = train_split
        self.test_split = test_split
        self.save_paths = save_paths

        self.setup_dataloaders(use_fake=True,
                               sanity_check=sanity_check,
                               fake_percentage=fake_percentage,
                               test=True,
                               **dataset_kwargs)
        
        self.model = model
        self.init_weights = copy.deepcopy(self.model.state_dict())
        self.index_cutoff = index_cutoff

    def get_loader(self,
                   save_paths:list[str],
                   batch_size:int,
                   subject_splits:list[list[str]],
                   fake_data: Optional[list[str]] = None,
                   dataset:Optional[dict] = None,
                   subject_dataset_type: Optional[subject_dataset] = None,
                   dataset_type: Optional[EEGDataset] = EEGDataset,
                   shuffle=True,
                   sanity_check=False,
                   fake_percentage=0.5,
                   test=False,
                   **dataset_kwargs):
        
        dset = dataset_type(subject_splits=subject_splits,dataset=dataset,
                          save_paths=save_paths,
                          subject_dataset_type=subject_dataset_type,
                          fake_data=fake_data,
                          fake_percentage=fake_percentage,
                          fs=self.fs,t_baseline=self.t_baseline,
                          t_epoch=self.t_epoch,start=self.start,
                          length=self.length,channels=self.channels,
                          sanity_check=sanity_check,
                          **dataset_kwargs)
        if test:
            n_val = int(len(dset)*0.5)
            n_test = len(dset)-n_val
            val_set,test_set = torch.utils.data.random_split(dset,[n_val,n_test])
            # val_set = torch.utils.data.Subset(dset,val_idx)
            # test_set = torch.utils.data.Subset(dset,test_idx)
            val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=shuffle)
            test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=shuffle)
            return val_loader,test_loader

        return DataLoader(dset,batch_size,shuffle=shuffle,)
    
    def setup_dataloaders(self,
                          splits=None,
                          use_fake=True,
                          sanity_check=False,
                          fake_percentage=0.5,
                          test=False,
                          **dataset_kwargs):
        
        if splits is not None:
            train_split = splits[0]
            test_split = splits[1]
        else:
            train_split = self.train_split
            test_split = self.test_split
        
        fake_data = self.fake_data if use_fake else None

        self.train_loader = self.get_loader(batch_size=self.batch_size,
                                            subject_splits=train_split,
                                            dataset=self.dataset,
                                            fake_data=fake_data,
                                            save_paths=self.save_paths,
                                            subject_dataset_type=self.subject_dataset_type,
                                            shuffle=True,
                                            sanity_check=sanity_check,
                                            fake_percentage=fake_percentage,
                                            **dataset_kwargs)

        if test:
            self.val_loader,self.test_loader = self.get_loader(batch_size=self.batch_size,
                                            subject_splits=test_split,
                                            dataset=self.dataset,
                                            save_paths=self.save_paths,
                                            subject_dataset_type=self.subject_dataset_type,
                                            shuffle=False,
                                            sanity_check=sanity_check,
                                            fake_percentage=fake_percentage,
                                            test=test,
                                            **dataset_kwargs)
    
        else:
            self.val_loader = self.get_loader(batch_size=self.batch_size,
                                                subject_splits=test_split,
                                                dataset=self.dataset,
                                                save_paths=self.save_paths,
                                                subject_dataset_type=self.subject_dataset_type,
                                                shuffle=False,
                                                sanity_check=sanity_check,
                                                fake_percentage=fake_percentage,
                                                **dataset_kwargs)
    
    def sample_batch(self):
        return next(iter(self.train_loader))[0][:, :, :self.index_cutoff]

    def fit(self,
            fabric:Fabric, 
            num_epochs=10,
            lr=1E-4,
            weight_decay=1E-4,
            verbose=True,
            validation_step=1,
            optimizer=None,
            stop_threshold=None,
            log=False,
            id=None,
            forward_fn=None,
            test=False,
            setup_test=True):
        self.model.load_state_dict(self.init_weights)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr,weight_decay=weight_decay)
        else:
            print("using specified optimizer")
        stats = []
        if not isinstance(self.model,_FabricModule):
            self.model = fabric.setup(self.model)
        optimizer = fabric.setup_optimizers(optimizer)
        train_loader,val_loader = fabric.setup_dataloaders(self.train_loader,self.val_loader)

        stop_counter = 0
        stop_threshold = num_epochs if stop_threshold is None else stop_threshold

        checkpoint = copy.deepcopy(self.model.state_dict())
        val_losses = [10]

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                labels = labels.to(torch.long)
                with fabric.autocast():
                    inputs, labels = inputs[:, :, :self.index_cutoff], labels
                    if forward_fn is None:
                        outputs = self.model.classify(inputs)
                    else:
                        outputs = forward_fn(model,inputs)
                    loss = criterion(outputs, labels)
                fabric.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            val_loss = 0.0
            correct = 0
            total = 0
            if epoch%validation_step == 0:
                self.model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        labels = labels.to(torch.long)
                        with fabric.autocast():
                            inputs, labels = inputs[:, :, :self.index_cutoff], labels
                            outputs = self.model.classify(inputs)
                            _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        val_loss += criterion(outputs, labels).item()

                val_accuracy = 100 * correct / total

                avg_val_loss = val_loss/len(self.val_loader)

                if avg_val_loss < min(val_losses):
                    print("checkpointing")
                    checkpoint = copy.deepcopy(self.model.state_dict())
                else:
                    print(f"Min loss: {min(val_losses)} vs {avg_val_loss}")

                val_losses.append(avg_val_loss)

                if log:
                    wandb.log({f"epoch_{id}":epoch,
                               f"training_loss_{id}":running_loss/len(self.train_loader),
                               f"validation_loss_{id}":val_loss/len(self.val_loader),
                               f"training_accuracy_{id}":train_accuracy,
                               f"validation_accuracy_{id}":val_accuracy})

                if verbose:
                    print(f'Epoch [{epoch+1}/{num_epochs}], '
                        f'Training Loss: {running_loss/len(self.train_loader):.3f}, '
                        f'Training Accuracy: {train_accuracy:.2f}%, '
                        f'Validation Loss: {val_loss/len(self.val_loader):.3f}, '
                        f'Validation Accuracy: {val_accuracy:.2f}%')
                stats.append(val_accuracy*1+0*train_accuracy)

                if (running_loss/len(self.train_loader))*1.25<val_loss/len(self.val_loader):
                    stop_counter += 1
                if (running_loss/len(self.train_loader))>val_loss/len(self.val_loader):
                    stop_counter -= 1
                if stop_counter < 0:
                    stop_counter = 0
                
            if stop_counter > stop_threshold:
                break
        
        test_loss = 0.0
        correct = 0
        total = 0
        if test:
            if setup_test:
                self.test_loader = fabric.setup_dataloaders(self.test_loader)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    labels = labels.to(torch.long)
                    with fabric.autocast():
                        inputs, labels = inputs[:, :, :self.index_cutoff], labels
                        outputs = self.model.classify(inputs)
                        _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    test_loss += criterion(outputs, labels).item()
            test_acc = 100 * correct / total
            return test_acc

        print('Finished Training')
        return max(stats)
    
    def k_fold_cv(self,
                  fabric,
                  k=9,
                  n=9,
                  lr=1E-3,
                  use_fake=True,
                  weight_decay=1E-4,
                  verbose=False,
                  leave_out=False,
                  **dataset_kwargs):
        folds = k_fold_splits(k,n,leave_out)
        accuracies = []
        for fold in folds:
            self.setup_dataloaders(fold,use_fake)
            self.train_loader = self.get_loader(self.dataset,fold[0],
                                                self.batch_size,True,**dataset_kwargs)
            self.val_loader = self.get_loader(self.dataset,fold[1],
                                              self.batch_size,True,**dataset_kwargs)
            self.model.load_state_dict(self.init_weights)
            max_accuracy = self.fit(fabric,
                        num_epochs=20,
                        lr=lr,
                        weight_decay=weight_decay,
                        verbose=verbose)
            print(f"---\nreached an accuracy of {max_accuracy}\n---")
            accuracies.append(max_accuracy)
        return accuracies
    
    def predict(self,x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
        return predicted
    
class MLPClassifier(nn.Module):
    def __init__(self, input_channels):
        super(MLPClassifier, self).__init__()
        self.conv = nn.Conv1d(input_channels,32,3)
        self.fc = nn.Linear(32, 2) 

    def forward(self, x):
        x = self.conv(x)
        x = reduce(x,"batch channel ... -> batch channel","mean")
        x = self.fc(x)
        return x
    
    def classify(self,x):
        return self.forward(x)
    
    
class SimpleCSP:

    def __init__(self,
                 save_paths:list[str],
                 train_split:list[list[str]],
                 test_split:list[list[str]],
                 dataset:Optional[dict] = None,
                 fake_paths=None,
                 fake_percentage=0.5,
                 dataset_type: Optional[EEGDataset] = EEGDataset,
				 subject_dataset_type: Optional[subject_dataset] = None,
                 channels:Iterable = np.array([0,1,2]),
                 fs:float = 250, 
				 t_baseline:float = 0, 
				 t_epoch:float = 9,
				 start:float = 3.5,
				 length:float = 2.05,
                 sanity_check:bool = False):
        
        self.fs = fs
        self.t_epoch = t_epoch
        self.t_baseline = t_baseline
        self.subject_dataset_type = subject_dataset_type
        self.start = start
        self.length = length
        self.channels = channels

        self.train_set = dataset_type(subject_splits=train_split,dataset=dataset,
                          save_paths=save_paths,subject_dataset_type=subject_dataset_type,
                          fake_data=fake_paths,
                          fake_percentage=fake_percentage,
                          fs=self.fs,t_baseline=self.t_baseline,
                          t_epoch=self.t_epoch,start=self.start,
                          length=self.length,channels=self.channels,
                          sanity_check=sanity_check)
        
        self.test_set = dataset_type(subject_splits=test_split,dataset=dataset,
                          save_paths=save_paths,subject_dataset_type=subject_dataset_type,
                          fake_data=None,
                          fake_percentage=0,
                          fs=self.fs,t_baseline=self.t_baseline,
                          t_epoch=self.t_epoch,start=self.start,
                          length=self.length,channels=self.channels,
                          sanity_check=sanity_check)
        
        self.set_epoch(start,length)
    
    def set_epoch(self,start,length):
        self.input_start = start + self.t_baseline
        self.input_end = self.input_start + length

    def fit(self,
            data = None,
            preprocess = False):

        if data is None:
            x_train,y_train = self.train_set.data[0], self.train_set.data[1]
        else:
            x_train,y_train = data

        if preprocess:
            x_train,y_train = self.preprocess(x_train,y_train)

        x_test,y_test = self.test_set.data[0], self.test_set.data[1]
        x_train = np.float64(x_train)
        x_test = np.float64(x_test)

        print(f"input shape: {x_train.shape}")

        csp = CSP(n_components=x_train.shape[1],reg=None,log=True,norm_trace=False)
        svm = SVC(C=1)

        clf = Pipeline(steps=[("csp",csp),
                            ("classification",svm)])

        clf.fit(x_train,y_train)

        if preprocess:
            x_test,y_test = self.preprocess(x_test,y_test)

        y_discrete = clf.predict(x_test)
            
        acc = accuracy_score(y_test,y_discrete)
        # confusion = confusion_matrix(y_test,y_discrete,normalize="true") 
        # kappa = cohen_kappa_score(y_discrete,y_test) 

        return acc
    
    def get_train(self):
        return self.train_set.data[0], self.train_set.data[1]
    
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
        x = x[:,[0,2],:]
        ax = []
        
        for i in range(1,10):
            ax.append(self.passband(x,4*i,4*i+4))

        x = np.concatenate(ax,1)
        return x,y

if __name__ == "__main__":

    dataset = {}
    for i in range(1,10):
        mat_train,mat_test = load_data("../data/2b_iv",i)
        dataset[f"subject_{i}"] = {"train":mat_train,"test":mat_test}

    save_path = "../data/2b_iv/csp"

    train_split = 6*[["train","test"]] + 3*[["train"]]
    test_split = 6*[[]] + 3* [["test"]]

    channels = np.split(np.arange(0,6*9),6)
    channels = np.concatenate([channels[0],channels[2]])
    
    model = MLPClassifier(18)

    csp_config = UnetConfig(
        input_shape=(256),
        input_channels=18,
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

    mlp = BottleNeckClassifier((4096,512))
    model = Unet(csp_config,mlp)

    clf = DeepClassifier(model=model,
                         train_split=train_split,
                         test_split=test_split,
                         dataset=None,
                         save_paths=[save_path],
                         subject_dataset_type=CSP_subject_dataset,
                         channels=channels,
                         batch_size=32,
                         fs=250,
                         t_baseline=0,
                         t_epoch=9,
                         start=3.5,
                         length=2.05,
                         index_cutoff=512,
                         sanity_check=False)
    
    torch.set_float32_matmul_precision("medium")
    
    fabric = Fabric(accelerator="cuda",precision="bf16-mixed")
    fabric.launch()

    print(clf.sample_batch().shape)
    
    clf.fit(fabric=fabric,
            num_epochs=50,
            lr=1E-3,
            weight_decay=1E-3,
            verbose=True)
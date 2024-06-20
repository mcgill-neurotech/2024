
import os
import numpy as np
from omegaconf import open_dict

from utils.utils import path_loader

from train_diffusion_model import (
        init_diffusion_model, 
        likelihood_computation_wrapper
        )

from datasets import CanCAM_MEG, HCP_fMRI, Matrix_HCP_fMRI


def cross_likelihood(models, config_name, targets, batch_size, save_folder=""):
    """
    Computes the likelihood for each data target on each model.
    
    Args:
        models (list): Contains the paths (str) of each model used for computation
        config_name (str): Name of the config file under the models' directory
        targets (list): Contains the paths (str) of each target data file (.npy) which contains the class data for the computation
        batch_size (int): Batch size
        
    Saves the likelihoods under the same folder as the models'
    """
    for model in models:
        model_name = os.path.basename(model)
        model_path = "/".join(model.split("/")[:-1])
        model_sub = model.split("/")[-2]
        print(model_sub)
        
        model_cfg = path_loader(config_name, model_path)
    
        with open_dict(model_cfg):
            model_cfg.likelihood_experiment = {"batch_size": batch_size}
        
        diffusion_model, network = init_diffusion_model(model_cfg)
        diffusion_model.load_state_dict(path_loader(model_name, model_path))
        
        if not os.path.isdir(os.path.join(model_path, "likelihoods")):
            os.mkdir(os.path.join(model_path, "likelihoods"))
        
        for target in targets:
            target_data = np.load(target, allow_pickle=True)
            likelihood = likelihood_computation_wrapper(model_cfg, diffusion_model, target_data)
                
            #save_name = os.path.basename(target)
            subject = target.split("/")[-2]
            
            np.save(os.path.join(model_path, "likelihoods", save_folder, subject), likelihood)
            
            print("DONE", subject)
        

def cancam_fingerprinting():
    models_path = "/meg/meg1/users/mlapatrie/data/CanCAM200/"
    data_path = "/meg/meg1/users/mlapatrie/data/CanCAM200/"
    subjects = [i for i in os.listdir(data_path) if len(i) == 11]
    
    config_name = "conditional_config.pkl"
    model_name = "conditional_model.pkl"
    data_name = "test_data.npy"
    
    models = []
    targets = []
    for sub in subjects:
        models.append(os.path.join(models_path, sub, model_name))
        targets.append(os.path.join(models_path, sub, data_name))
    
    
    cross_likelihood(models, config_name, targets, batch_size=60)


def cancam_fingerprinting_300(models_path):
    data_path = "/meg/meg1/users/mlapatrie/data/Scouted_MEG_CanCAM_300"
    subjects = [i for i in os.listdir(models_path) if len(i) == 11]
    
    config_name = "conditional_config.pkl"
    model_name = "conditional_model.pkl"
    data_name = "test_data.npy"
    
    models = []
    targets = []
    print(subjects)
    for sub in subjects:
        
        data_set = CanCAM_MEG(
                signal_length=600,
                rois="all",
                pca=True,
                pca_sub_ids=['230412_0615', '230412_2317', '230227_0955', '230227_1420', '230412_0223', '230412_1523', '230227_1554', '230227_1124', '230412_1123', '230227_1731'],
                n_components=30,
                sub_ids=[sub.replace("_", "")],
                with_class_cond=True,
                folder_path=data_path,
        )
        print(data_set)
        np.save(os.path.join(models_path, sub, "test_data.npy"), data_set)
        
        models.append(os.path.join(models_path, sub, model_name))
        targets.append(os.path.join(models_path, sub, data_name))
    
    print(models)
    print(targets)
    cross_likelihood(models, config_name, targets, batch_size=15)
    

def cancam_characteristic_features():
    models_path = "/meg/meg1/users/mlapatrie/data/CanCAM200/"
    data_path = "/meg/meg1/users/mlapatrie/data/CanCAM200/"
    
    config_name = "conditional_config.pkl"
    model_name = "conditional_model.pkl"
    data_name = "test_data.npy"
    
    pairs = [["230412_1244", "230324_1217"], ["230324_1559", "230412_1642"]]
    
    for pair in pairs:
        cross_likelihood([os.path.join(models_path, pair[0], model_name)], config_name, [os.path.join(data_path, pair[0], data_name)], batch_size=60)
        #cross_likelihood([os.path.join(models_path, pair[1], model_name)], config_name, [os.path.join(data_path, pair[0], data_name)], batch_size=60)


def cancam_train_data():
    models_path = "/meg/meg1/users/mlapatrie/data/CanCAM200/"
    data_path = "/meg/meg1/users/mlapatrie/data/CanCAM200/"
    
    config_name = "conditional_config.pkl"
    model_name = "conditional_model.pkl"
    data_name = "train_data.npy"
    
    subjects = ["230412_1244", "230324_1217", "230324_1559", "230412_1642", "230412_0732", "230413_0433"]
    
    for sub in subjects:
        cross_likelihood([os.path.join(models_path, sub, model_name)], config_name, [os.path.join(data_path, sub, data_name)], 90)
    

def hcp_same_model_classification():
    config_name = "conditional_config.pkl"
    model_name = "conditional_model.pkl"
    
    models_path = "/meg/meg1/users/mlapatrie/data/HCP/Models/"
    data_path = "/meg/meg1/users/mlapatrie/data/HCP/test_data/"
    models = [os.path.join(models_path, i, model_name) for i in os.listdir(models_path)]
    targets = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    targets.append(os.path.join(data_path, "Noise.npy"))
    done_targets = os.listdir(models_path + "likelihoods/Classification/")
    for i in done_targets:
        targets.remove(i)
    print(len(targets))
    models.remove("/meg/meg1/users/mlapatrie/data/HCP/Models/HCP_200_REST_LR/conditional_model.pkl")
    models.remove("/meg/meg1/users/mlapatrie/data/HCP/Models/Noise/conditional_model.pkl")
    
    
    cross_likelihood(models, config_name, targets, batch_size=2, save_folder="Classification/")
    

def cancam_networks():
    model_path = "/meg/meg1/users/mlapatrie/data/CanCAM200/CanCAM_20/"
    data_path = "/meg/meg1/users/mlapatrie/data/test_data/"
    data_save = "/meg/meg1/users/mlapatrie/data/CanCAM200/network_experiment/"
    
    config_name = "conditional_config.pkl"
    model_name = "conditional_model.pkl"
    
    networks = [("Visual", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]),
                ("Somatomotor", [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]),
                ("Dorsal attention", [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138]),
                ("Ventral attention", [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]),
                ("Limbic", [50, 51, 52, 53, 54, 55, 150, 151, 152, 153, 154, 155]),
                ("Frontoparietal", [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173]),
                ("Default", [74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199])]

    test_subjects = ["230401_2124", "230406_1425", "230409_1516"]
    files = ["Visual.npy", "Frontoparietal.npy", "Dorsal attention.npy", "Limbic.npy", "Somatomotor.npy", "Ventral attention.npy"]
    
    for subject in test_subjects:
        models = [os.path.join(model_path, model_name)]
        targets = [os.path.join(data_save, subject, f) for f in files]
        
        cross_likelihood(models, config_name, targets, batch_size=30)
        
    """
    for subject in test_subjects:
        data_set = CanCAM_MEG(
            signal_length=1000,
            rois="half",
            sub_ids=subject.replace("_", ""),
            with_class_cond=True,
            folder_path=data_path
        )
        np.save(data_save + subject + "/full.npy", data_set, allow_pickle=True)
        
        for net in networks:
            data_set = CanCAM_MEG(
                signal_length=1000,
                rois="half",
                sub_ids=subject.replace("_", ""),
                with_class_cond=True,
                folder_path=data_path,
                zeroed_idx=net[1]
            )
            
            np.save(data_save + subject + "/" + net[0] + ".npy", data_set, allow_pickle=True)
    """
    

def hcp_networks():
    model_path = "/meg/meg1/users/mlapatrie/data/HCP/Models/Matrices/"
    data_path = "/meg/meg1/users/mlapatrie/data/HCP/Correlation Matrices/test_matrices/"
    data_save = "/meg/meg1/users/mlapatrie/data/HCP/network_experiment/matrices/"
    
    config_name = "conditional_config.pkl"
    model_name = "conditional_model.pkl"
    
    networks = [("Visual", [0, 1, 2, 3, 4, 5, 6, 7, 8, 50, 51, 52, 53, 54, 55, 56, 57]),
                ("Somatomotor", [9, 10, 11, 12, 13, 14, 58, 59, 60, 61, 62, 63, 64, 65]),
                ("Dorsal attention", [15, 16, 17, 18, 19, 20, 21, 22, 66, 67, 68, 69, 70, 71, 72]),
                ("Ventral attention", [23, 24, 25, 26, 27, 28, 29, 73, 74, 75, 76, 77]),
                ("Limbic", [30, 31, 32, 78, 79]),
                ("Frontoparietal", [33, 34, 35, 36, 80, 81, 82, 83, 84, 85, 86, 87, 88]),
                ("Default", [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])]

    test_subjects = ["130417", "128632"]
    files = ["full.npy", "Default.npy", "Dorsal attention.npy", "Frontoparietal.npy", "Limbic.npy", "Somatomotor.npy", "Ventral attention.npy", "Visual.npy"]
    
    for subject in test_subjects:
        models = [os.path.join(model_path, model_name)]
        targets = [os.path.join(data_save, subject, f) for f in files]
        
        cross_likelihood(models, config_name, targets, batch_size=2, save_folder=subject+"/")
        
    """
    for subject in test_subjects:
        data_set = Matrix_HCP_fMRI(
            sub_ids=[subject],
            with_class_cond=True,
            folder_path=data_path
        )
        np.save(data_save + subject + "/full.npy", data_set, allow_pickle=True)
        
        for net in networks:
            data_set = Matrix_HCP_fMRI(
                sub_ids=[subject],
                with_class_cond=True,
                folder_path=data_path,
                zeroed_idx=net[1]
            )
            
            np.save(data_save + subject + "/" + net[0] + ".npy", data_set, allow_pickle=True)
    """
    

def hcp_age_classifier():
    all_subjects_sessions = os.listdir("/meg/meg1/users/mlapatrie/data/HCP/time_series")
    all_subjects_sessions.sort()
    subjects_sessions = all_subjects_sessions[all_subjects_sessions.index("189450_rfMRI_REST1_LR_timeseries.npy")+1:]
    subjects = []
    
    for sess in subjects_sessions:
        if sess[:6] not in subjects:
            subjects.append(sess[:6])
    
    selected_subjects = []
    for s in subjects:
        try:
            data = np.load(f"/meg/meg1/users/mlapatrie/data/HCP/time_series/{s}_rfMRI_REST1_LR_timeseries.npy")
            data2 = np.load(f"/meg/meg1/users/mlapatrie/data/HCP/time_series/{s}_rfMRI_REST2_LR_timeseries.npy")
            
            if len(data[0]) == 1200 and len(data2[0]) == 1200:
                selected_subjects.append(s)
                
        except Exception as e:
            print(e)
                
    print(selected_subjects)
    print(len(selected_subjects))
    
    config_name = "conditional_config.pkl"
    model_name = "conditional_model.pkl"
    
    models_path = "/meg/meg1/users/mlapatrie/data/HCP/Models/"
    data_path = "/meg/meg1/users/mlapatrie/data/HCP/test_data/age_classifier"
    models_names = ["HCP_22-30", "HCP_31-36+", "Female", "Male"]
    models = [os.path.join(models_path, i, model_name) for i in models_names]
    targets = [os.path.join(data_path, i + "_LR.npy") for i in selected_subjects]
    
    print(models)
    print(targets)
    cross_likelihood(models, config_name, targets, batch_size=2)


if __name__ == "__main__":
    cancam_fingerprinting_300("/meg/meg1/users/mlapatrie/data/CanCAM200/old-young_pca/")
    #hcp_same_model_classification()
    #cancam_characteristic_features()
    #cancam_train_data()
    #cancam_networks()
    #hcp_networks()
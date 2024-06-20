
from hydra import compose, initialize
import os
import logging
import pandas as pd
import numpy as np
import scipy.io as io

logging.basicConfig(level=logging.INFO)

from train_diffusion_model import training_and_eval_pipeline


def multiple_subjects(subjects, overrides):
    for i, sub in enumerate(subjects):
        print("SUB", sub)
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(
                config_name="config",
                overrides=overrides[i]
            )
    
        training_and_eval_pipeline(cfg)
    
    
def cancam_mats():
    subjects_mat = os.listdir("/meg/meg1/users/mlapatrie/data/Scouted_MEG_CanCAM")
    subjects = []

    trained = os.listdir("/meg/meg1/users/mlapatrie/data/CanCAM200")
    for sub_mat in subjects_mat:
        if len(sub_mat) == 28 and sub_mat[13:24] not in trained:
            subjects.append(sub_mat[13:24])

    print(len(subjects))
    
    overrides = []
    for sub in subjects:
        overrides.append([
            "base.experiment="+sub,
            "base.tag=conditional",
            "dataset.sub_ids="+sub,
            "+experiments/likelihood_experiment=likelihood_experiment",
        ])
    
    multiple_subjects(subjects, overrides)


def hcp_classes(model_name, conditions):
    file_path = "D:/DDPM_data/HCP/restricted_data.csv"
    metadata = pd.read_csv(file_path)

    subjects_path = "D:/DDPM_data/HCP/time_series/"
    subjects = [i[:6] for i in os.listdir(subjects_path)]

    selected_subjects = []
    subjects_ids = []

    for index, sub_info in metadata.iterrows():
        
        if str(sub_info["Subject"]) in subjects:
            
            # Go through the conditions
            matches = 0
            max_matches = len(conditions.keys())
            for var in conditions.keys():
                for cond in conditions[var]:
                    if str(sub_info[var]) == cond:
                        matches += 1
                        
            if matches >= max_matches:
                selected_subjects.append(sub_info)
                subjects_ids.append(str(sub_info["Subject"]))

    selected_subjects = pd.DataFrame(selected_subjects, index=np.arange(len(selected_subjects)))
    
    overrides = [
        "base.experiment="+model_name,
        "base.tag=conditional",
        "dataset.sub_ids="+("["+", ".join(subjects_ids)+"]"),
        "+experiments/likelihood_experiment=likelihood_experiment",
    ]
    
    multiple_subjects([model_name], [overrides])


def cancam_conditions(conditions):
    file_path = "/meg/meg1/users/mlapatrie/data/CanCAM200/restricted_data.csv"
    metadata = pd.read_csv(file_path)

    subjects_path = "/meg/meg1/users/mlapatrie/data/Scouted_MEG_CanCAM_300/"
    subject_files = os.listdir(subjects_path)
    
    subjects = []
    for sub_path in subject_files:
        mat_file = io.loadmat(os.path.join(subjects_path, sub_path))
        subject_id = mat_file["Comment"][0][4:12]
        subjects.append("sub_" + subject_id)
        
    selected_subjects = []
    subjects_ids = []

    for index, sub_info in metadata.iterrows():
        if str(sub_info["CCID"]) in subjects:
            
            # Go through the conditions
            matches = 0
            max_matches = len(conditions.keys())
            for var in conditions.keys():
                for cond in conditions[var]:
                    if str(sub_info[var]) == cond:
                        matches += 1
                        
            if matches >= max_matches:
                selected_subjects.append(sub_info)
                subjects_ids.append(str(sub_info["CCID"]))

    selected_sub_files = []
    for sub_id in subjects_ids:
        index = subjects.index(sub_id)
        selected_sub_files.append(subject_files[index])
    
    selected_sub_dates = [i[13:-4] for i in selected_sub_files]
    print(selected_sub_dates)
    
    overrides = []
    for sub in selected_sub_dates:
        overrides.append([
            "base.experiment="+sub,
            "base.tag=conditional",
            "dataset.sub_ids="+f"[{sub}]",
            "+experiments/likelihood_experiment=likelihood_experiment",
        ])
        
    multiple_subjects(selected_sub_dates, overrides)

    
def hcp_ages():
    conditions1 = {
        "Age": ["22-25", "26-30"],
        "Gender": ["M", "F"],
    }
    hcp_classes("22-30", conditions1)
    
    conditions2 = {
        "Age": ["31-35", "36+"],
        "Gender": ["M", "F"],
    }
    hcp_classes("31-36+", conditions2)
    

def hcp_sex():
    conditions1 = {
        "Gender": ["M"],
    }
    hcp_classes("Male", conditions1)
    
    conditions2 = {
        "Gender": ["F"],
    }
    hcp_classes("Female", conditions2)
    

def cancam_young_old():
    conditions = {
        "CCID": ["sub_CC110037", "sub_CC110182", "sub_CC110126", "sub_CC110056", "sub_CC110098",
                 "sub_CC721891", "sub_CC721434", "sub_CC721374", "sub_CC721532", "sub_CC721224"]
    }
    cancam_conditions(conditions)
    
    
def cancam_20(transposed=False):
    # Shuffled os.listdir(path_to_30_subjects) with seed 42
    subs = ['230412_0223', '230412_1123', '230412_1919', '230227_1508', '230227_1124', '230412_2317', '230413_0155', '230412_1642', '230412_1523', '230412_1244', '230412_1801', '230227_1420', '230227_1731', '230227_1043', '230412_0847', '230227_1554', '230324_1559', '230412_1004', '230412_0732', '230324_1445']
    subs = [name.replace("_", "") for name in subs]

    name = "CanCAM_20"
    if transposed: name += "_transposed"
    
    overrides = [
            "base.experiment=CanCAM_20",
            "base.tag=conditional",
            "dataset.sub_ids="+str(subs),
            "dataset.transposed="+str(transposed),
            "+experiments/likelihood_experiment=likelihood_experiment",
        ]
    multiple_subjects(["CanCAM_20"], [overrides])
    

def cancam_100(transposed=False):
    # Shuffled os.listdir(path_to_30_subjects) with seed 42
    subs = ['230324_1217', '230324_1559', '230324_1445', '230324_1332', '230413_0433', '230413_0155', '230412_2317', '230413_0036', '230412_2158', '230412_1801', '230412_1919', '230412_1523', '230412_1642', '230412_1004', '230412_1123', '230412_1244', '230412_0847', '230412_0615', '230412_0732', '230412_0223', '230227_0955', '230227_1043', '230227_1124', '230227_1244', '230227_1329', '230227_1420', '230227_1508', '230227_1554', '230227_1644', '230227_1731', '230408_1843', '230407_1634', '230407_2021', '230407_2137', '230407_2253', '230408_0006', '230408_0126', '230408_0240', '230408_0355', '230408_0508', '230408_0623', '230408_0738', '230408_0851', '230408_1005', '230408_1119', '230408_1233', '230408_1347', '230408_1501', '230408_1615', '230408_1729', '230405_1543', '230405_1703', '230405_1825', '230405_1946', '230405_2106', '230405_2228', '230404_0928', '230404_1336', '230404_1459', '230404_1624', '230404_1748', '230404_1912', '230404_2037', '230404_2202', '230404_2323', '230405_0045', '230405_1019', '230405_1139', '230405_1300', '230405_1421', '230403_1130', '230403_1244', '230403_1401', '230403_1525', '230403_1650', '230402_2246', '230403_0006', '230403_0126', '230403_0241', '230403_0357', '230403_0512', '230403_0628', '230403_0745', '230403_0900', '230403_1014', '230401_2241', '230401_2355', '230402_0113', '230402_0228', '230402_0342', '230402_0456', '230402_0609', '230402_0723', '230402_0838', '230402_1000', '230402_1122', '230402_1243', '230401_1848', '230401_2007', '230401_2124']
    subs = [name.replace("_", "") for name in subs]
    
    name = "CanCAM_100"
    if transposed: name += "_transposed"
    
    overrides = [
            "base.experiment=CanCAM_100",
            "base.tag=conditional",
            "dataset.sub_ids="+str(subs),
            "dataset.transposed="+str(transposed),
            "+experiments/likelihood_experiment=likelihood_experiment",
        ]
    multiple_subjects(["CanCAM_100"], [overrides])


if __name__ == "__main__":
    #hcp_ages()
    #cancam_young_old()
    #cancam_100()
    cancam_20()

import scipy.io as io
import os
import numpy as np


# Rename every file by the subject ID
file_path = "D:/Scouted_MEG_CamCAN_300/test_data/" #"/meg/meg1/users/mlapatrie/data/Scouted_MEG_CamCAN_300/train_data"
files = [i for i in os.listdir(file_path) if i.endswith(".mat") and "CC" not in i]
print(len(files))

for f in files:
    print(f)
    mat_file = io.loadmat(os.path.join(file_path, f))
    ccid = mat_file["Comment"][0].split("/")[0][4:]
    
    # If the new name doesnt already exist change name to it. Else, add a number to the end of the name
    if not os.path.exists(os.path.join(file_path, f"{ccid}.mat")):
        os.rename(os.path.join(file_path, f), os.path.join(file_path, f"{ccid}.mat"))
    else:
        i = 1
        while os.path.exists(os.path.join(file_path, f"{ccid}_{i}.mat")):
            i += 1
        os.rename(os.path.join(file_path, f), os.path.join(file_path, f"{ccid}_{i}.mat"))
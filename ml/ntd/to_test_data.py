
from datasets import HCP_fMRI, Noise

import os
import numpy as np

folder_path = "/meg/meg1/users/mlapatrie/data/HCP/time_series/"
save_path = "/meg/meg1/users/mlapatrie/data/HCP/test_data/"
session_list = os.listdir(folder_path)

sub_ids = []
conditions = ["REST1_LR", "REST2_LR"]
data_identifier = "LR"

for session in session_list:
    sub_id = session.split("_")[0]
    if sub_id not in sub_ids:
        sub_ids.append(sub_id)

sub_ids.sort()
sub_ids = sub_ids[108:]
print(sub_ids)

for sub in sub_ids:
    data_set = HCP_fMRI(
            sub_ids=[sub],
            sessions=conditions,
            with_class_cond=True,
            folder_path=folder_path,
        )
    
    np.save(f"{save_path}{sub}_{data_identifier}.npy", data_set, allow_pickle=True)

"""
data_set = Noise(
            time_windows=2,
            with_class_cond=True,
        )
np.save(f"{save_path}Noise.npy", data_set, allow_pickle=True)
"""

networks = [
    ("No mask", []),
    ("Visual", [0, 1, 2, 3, 4, 5, 6, 7, 8, 50, 51, 52, 53, 54, 55, 56, 57]),
    ("Somatomotor", [9, 10, 11, 12, 13, 14, 58, 59, 60, 61, 62, 63, 64, 65]),
    ("Dorsal attention", [15, 16, 17, 18, 19, 20, 21, 22, 66, 67, 68, 69, 70, 71, 72]),
    ("Ventral attention", [23, 24, 25, 26, 27, 28, 29, 73, 74, 75, 76, 77]),
    ("Limbic", [30, 31, 32, 78, 79]),
    ("Frontoparietal", [33, 34, 35, 36, 80, 81, 82, 83, 84, 85, 86, 87, 88]),
    ("Default", [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
]
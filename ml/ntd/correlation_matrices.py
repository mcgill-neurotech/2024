
import matplotlib.pyplot as plt
import numpy as np
import os


data_path = "D:/DDPM_data/HCP/time_series"

files = os.listdir(data_path)
files.sort()
files = files[401:]

sessions = []

for f in files:
    if "REST" in f and "LR" in f:
        if "128632" in f or "129129" in f or "130417" in f:
            sessions.append(f)
        
print(len(sessions))

for session in sessions:
    
    session_info = session.split("_")
    session_name = session_info[0] + "_" + session_info[2][-1]
    print(session_name)
    time_series = np.load(os.path.join(data_path, session))
    
    correlation_matrix = []
    
    for x in time_series:
        row = []
        for y in time_series:
            corr_coef = np.corrcoef(x, y)
            row.append(corr_coef[0][1])
            
        correlation_matrix.append(row)
        
    np.save(f"D:/DDPM_data/HCP/Correlation Matrices/test_matrices/{session_name}.npy", correlation_matrix)

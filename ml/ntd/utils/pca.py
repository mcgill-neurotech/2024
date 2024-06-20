
import numpy as np
import torch
import os
import scipy.io as io
import matplotlib.pyplot as plt

from nilearn import datasets, plotting, image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_mat_to_nilearn_map():
    # Load the Schaefer 2018 atlas
    atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)

    # Atlas filenames
    atlas_filename = atlas_data.maps
    labels = atlas_data.labels

    time_file = os.path.join("D:/Scouted_MEG_CanCAM_300", f"matrix_scout_{'230412_0615'}.mat")
    mat_file = io.loadmat(time_file)
    mat_labels = mat_file["Description"]

    parsed_mat_labels = []

    for label in mat_labels:
        content = "".join(label[0]).split("@")[0].split(" ")
        
        if content[1] == "L": hemisphere = "LH"
        else: hemisphere = "RH"
        
        parsed_mat_labels.append(f"17Networks_{hemisphere}_{content[0]}")
        
    parsed_labels = []

    for label in labels:
        parsed_labels.append(str(label).split("'")[1])

    label_to_index = {label: index for index, label in enumerate(parsed_labels)}

    sorted_indices = [label_to_index[label] for label in parsed_mat_labels]

    return sorted_indices


def load_camcan_data(folder_path, sub_ids, rois, signal_length, start_index=0, end_index=-1, transposed=False):
    """
    Load scouted data from CanCAM dataset.

    Args:
        folder_path: Path to session folder
        sub_id: Patient id
        rois: Array of indeces of the rois
        signal_length: Length of time windows to split into

    Returns:
        data_array: Array of shape (num_time_windows, num_rois, signal_length) if transposed=False
                                   (num_time_windows, signal_length, num_rois) if transposed=True
    """
    
    data_array = []
    
    for sub_id in sub_ids:
        time_file = os.path.join(folder_path, sub_id)
        mat_file = io.loadmat(time_file)
        time = mat_file["Time"]
    
        meg_all_rois = mat_file["Value"]
        
        start_time = time[0][start_index]
        end_time = time[0][end_index]
    
        meg = []
        for roi in rois:
            meg.append(meg_all_rois[roi-1])
    
        time_window = np.where((time > start_time) & (time <= end_time))[1]
        meg_time_window = time[:, time_window]
    
        splitter = (
            np.arange(1, (meg_time_window.shape[1] // signal_length) + 1, dtype=int)
            * signal_length
        )
    
        previous = 0
        for index in splitter:
            splitted = []
            for roi in range(len(rois)):
                splitted.append(meg[roi][previous:index])
                      
            scaler = StandardScaler()
            scaler.fit(splitted)
            splitted = scaler.transform(splitted)
            
            previous = index
            
            if transposed:
                splitted = np.transpose(splitted)
                
            data_array.append(splitted)
            
    print("Shape of CanCAM data:", np.array(data_array).shape)
    
    return np.array(data_array)


def project_on_pca(pca_components, data_array):
    # data_array of shape (num_windows, num_rois, segment_length)
    
    projected_data_array = []
    
    for window in data_array:
        projected_data_array.append(np.dot(pca_components, window))

    return np.array(projected_data_array)


def project_on_pca_torch(pca_components, data_tensor):
    return torch.matmul(torch.tensor(pca_components).to(torch.float), data_tensor)


def project_from_pca(pca_components, projected_data_array):
    # data_array of shape (num_windows, num_rois, segment_length)
    
    data_array = []
    
    for window in projected_data_array:
        print(pca_components.T.shape, window.shape)
        data_array.append(np.dot(pca_components.T, window))

    return np.array(data_array)
    
    
def pca_camcan_meg(data_path, sub_ids, signal_length, start_index=0, end_index=27001, n_components=30, plot=False, save_components=False, save_path=""):
    
    print(sub_ids)
    meg_array = load_camcan_data(data_path, sub_ids, list(range(200)), signal_length, start_index, end_index)
    
    print("Shape", meg_array.shape)
    num_windows, num_rois, length_segments = meg_array.shape
    
    # Swaping axes to (num_windows, length_segments, num_rois)
    reshaped_meg_array = np.swapaxes(meg_array, 1, 2)

    # Reshaping to (num_windows * length_segments, num_rois)
    long_meg_array = reshaped_meg_array.reshape(num_windows * length_segments, num_rois)
    
    pca = PCA(n_components=n_components)
    pca.fit_transform(long_meg_array)

    if plot:
        # Using Kaiser's rule to find the number of 
        index = 0
        for idx, eigenvalue in enumerate(pca.explained_variance_):
            if eigenvalue < 1:
                index = idx - 1
                break
    
    if save_components:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            
        np.save(f"{save_path}/pca_components.npy", pca.components_)
        
    return pca.components_


def visualize_components():
    pca_components = pca_camcan_meg("D:/Scouted_MEG_CanCAM_300", ['230412_0615'], 90, n_components=5)
    
    sorted_indices = get_mat_to_nilearn_map()
    
    # Load Schaefer atlas
    atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
    atlas_filename = atlas_data.maps
    
    # Load the atlas image
    atlas_img = image.load_img(atlas_filename)
    
    for comp in pca_components:
        sorted_comp = comp[np.argsort(sorted_indices)]
        
        # Normalizing
        mean = sorted_comp.mean()
        std = sorted_comp.std()
        
        sorted_comp = (sorted_comp - mean) / std
        
        # Create a brain map with these values
        brain_map = atlas_img.get_fdata().copy()
        for roi_label, value in enumerate(sorted_comp, start=1):
            brain_map[brain_map == roi_label] = value

        # Convert the brain map to a Nifti image
        brain_map_img = image.new_img_like(atlas_img, brain_map)

        # Plot the brain map
        plotting.plot_stat_map(brain_map_img, title='ROI Values Visualization',
                                    display_mode='ortho', colorbar=True, cmap='viridis')
        print("plotted")
        plotting.show()
        

def uncast_generated_segments(samples_path, save_segments: bool = False, save_path: str = "", save_name: str = ""):
    data = np.load(samples_path, allow_pickle=True)
    
    segments = [data[i]["signal"] for i in range(len(data))]

    projected_segments = project_from_pca(components, segments)
        
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    if save_segments:
        for i in range(len(segments)):
            io.savemat(f"{save_path}/{save_name}_{i+1}.mat", {'data': projected_segments[i]})
            
    return projected_segments
        

if __name__ == "__main__":
    sub_dates = [f[13:-4] for f in os.listdir("D:/Scouted_MEG_CanCAM_300") if len(f) == 28]
    components = pca_camcan_meg("D:/Scouted_MEG_CanCAM_300", sub_dates, 600, n_components=30, plot=False, save_components=True, save_path="D:/DDPM_data/CanCAM200/group_models/CanCAM_100/")
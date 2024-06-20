
import numpy as np
import torch.nn as nn
import pickle
import torch
import os

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull

from utils.utils import path_loader
from datasets import CanCAM_MEG
from train_diffusion_model import init_diffusion_model 


def get_activation(name, activations={}):
    def hook(model, input, output):
        activations[name] = output
    return hook


def get_slconv_features_values(model_path, model_name, cfg_name, dataset):
    
    # Loading model
    cfg = path_loader(cfg_name, model_path)
    
    diffusion_model, network = init_diffusion_model(cfg)
    diffusion_model.load_state_dict(path_loader(model_name, model_path))
    
    # Getting cond vector
    try:
        cond = torch.stack([dic["cond"] for dic in dataset])[:len(dataset)]
        cond = cond.to(diffusion_model.device)
    except KeyError:
        cond = None
        
    batch = torch.stack([dic["signal"] for dic in dataset])
    
    # Adding a forward hook to the last SLConv layer   
    activations = {}
    diffusion_model.network.conv_pool[15].register_forward_hook(get_activation('SLConv', activations=activations))
    
    diffusion_model.eval()
    with torch.no_grad():
        time_vector = torch.tensor([0], device=diffusion_model.device).repeat(
                    batch.shape[0]
                )
            
        diffusion_model.network.forward(sig=batch, t=time_vector, cond=cond)
    
    return activations['SLConv'].detach().numpy()


def visualize_features_per_group(features_values_list, group_names=None, dimension_reduction="t-SNE", convex_hull=False, dimensions=2, title="Reduced SLConv features"):
    
    if group_names is None:
        group_names = ["Group {}".format(i+1) for i in range(len(features_values_list))]
    
    features_values = np.concatenate(features_values_list)
    if features_values.ndim == 3:
        features_values = np.mean(features_values, axis=2)
    
    # z-score features_values
    #features_values = (features_values - np.mean(features_values, axis=0)) / np.std(features_values, axis=0)
    
    # features_values.shape: (num_samples, num_features)
    print(f"features_values.shape: {features_values.shape}")
    
    if dimension_reduction == "MDS":
        # Perform MDS on features_values
        mds = MDS(n_components=dimensions)
        features_nd = mds.fit_transform(features_values.astype(dtype=np.float64))
    
    elif dimension_reduction == "t-SNE":
        # Perform t-SNE on features_values
        tsne = TSNE(n_components=dimensions, random_state=0)
        features_nd = tsne.fit_transform(features_values.astype(dtype=np.float64))
    
    elif dimension_reduction == "PCA":
        # Perform PCA on features_values
        pca = PCA(n_components=dimensions)
        features_nd = pca.fit_transform(features_values.astype(dtype=np.float64))
    
    else:
        raise NotImplementedError("Dimension reduction method not implemented")
    
    # features_nd.shape: (num_samples, n_dimensions)
    print(f"features_nd.shape: {features_nd.shape}")
    
    # Unconcatenate mds_features
    features_list_nd = []
    for i in range(len(features_values_list)):
        start_index = int(np.sum([len(features_values_list[j]) for j in range(i)]))
        end_index = int(np.sum([len(features_values_list[j]) for j in range(i+1)]))
        features_list_nd.append(features_nd[start_index:end_index, :])
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot()
    
    if dimensions == 3:
        # Close the 2D plot
        plt.close(fig)
        
        # Create a new 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
    # Set title and labels for axes
    fig.suptitle(title)
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")
    if dimensions == 3:
        ax.set_zlabel("Coordinate 3")
        
    norm = mcolors.Normalize(vmin=0, vmax=len(features_list_nd) - 1)

    legend_handles = []
    for i, group_mds in enumerate(features_list_nd):
        # Remove 5% furthest from mean
        group_mds = group_mds[np.sum(np.abs(group_mds), axis=1).argsort()[:int(len(group_mds) * 0.95)]]
        
        color = plt.cm.gist_rainbow(norm(i))
        
        if dimensions == 2:
            scatter = ax.scatter(group_mds[:, 0], group_mds[:, 1], label=group_names[i], color=color)
            
        elif dimensions == 3:
            scatter = ax.scatter(group_mds[:, 0], group_mds[:, 1], group_mds[:, 2], label=group_names[i], color=color)
        
        legend_handles.append(scatter)

        if convex_hull:
            # Calculate the convex hull for the group
            hull = ConvexHull(group_mds)

            if dimensions == 2:
                # Get the vertices of the convex hull
                vertices = group_mds[hull.vertices]

                # Create a polygon patch from the smoothed vertices and add it to the plot
                polygon = Polygon(vertices, fill=True, facecolor=color, edgecolor=color, linewidth=1, alpha=0.3)
                
                plt.gca().add_patch(polygon)
            
            elif dimensions == 3:
                for simplex in hull.simplices:
                    simplex = np.concatenate((simplex, [simplex[0]]))  # Close the loop
                    vertices = group_mds[simplex]
                    poly = [[vertices[i][0], vertices[i][1], vertices[i][2]] for i in range(len(vertices))]
                    ax.add_collection3d(Poly3DCollection([poly], facecolors=color, linewidths=1, alpha=0.3))


    plt.legend(handles=legend_handles, labels=group_names, loc="upper right")
    plt.grid(True)
    plt.show()
    

def visualize_features_heatmap(features_values_list, individual_idx: list =[0], sample_idx: list = [0], feature_idx: list = [0], n_channels=30, subplots: bool = True, cmap="gist_gray", true_reference=True):
    
    max_val = np.max(features_values_list)
    min_val = np.min(features_values_list)
    
    # If subplots is True, create a figure with subplots of shape 3xn
    if subplots:
        fig, axs = plt.subplots(len(individual_idx), len(sample_idx) * len(feature_idx), figsize=(len(individual_idx)*5, 15))
        fig.suptitle("SLConv features heatmap")
    
    features_values_list = data_grouping(features_values_list, "individual", 74)
    
    for i_idx, i in enumerate(individual_idx):
        for s_idx, s in enumerate(sample_idx):
            sample_features_values = features_values_list[i][s]

            a, b = sample_features_values.shape
            
            assert a % n_channels == 0
            
            sample_features_values = np.reshape(sample_features_values, (a // n_channels, n_channels, b))
            
            for f_idx, f in enumerate(feature_idx):
                feature_values = sample_features_values[f]
                
                if not true_reference:
                    min_val = np.min(feature_values)
                    max_val = np.max(feature_values)
                
                # Add plot to figure
                if subplots and len(individual_idx) > 1:
                    axs[i_idx, (s_idx + f_idx) + s_idx*(len(feature_idx)-1)].imshow(feature_values, cmap=cmap, interpolation="nearest", aspect="auto", vmin=min_val, vmax=max_val)
                    axs[i_idx, (s_idx + f_idx) + s_idx*(len(feature_idx)-1)].set_title(f"Sample {s}, Feature {f}")
                    axs[i_idx, (s_idx + f_idx) + s_idx*(len(feature_idx)-1)].axis('off')
                
                elif subplots:
                    axs[(s_idx + f_idx) + s_idx*(len(feature_idx)-1)].imshow(feature_values, cmap=cmap, interpolation="nearest", aspect="auto", vmin=min_val, vmax=max_val)
                    axs[(s_idx + f_idx) + s_idx*(len(feature_idx)-1)].set_title(f"Sample {s}, Feature {f}")
                
                else:
                    plt.imshow(feature_values, cmap=cmap, interpolation="nearest", aspect="auto")
                    plt.show()
                    
        axs[i_idx, 0].axis('on')
        axs[i_idx, 0].set_ylabel(f"Individual {i}")
        
        if i_idx != len(individual_idx) - 1:
            axs[i_idx, 0].set_xticks([])
        else:
            for subp in range(len(sample_idx) * len(feature_idx)):
                axs[i_idx, subp].axis('on')
                axs[i_idx, subp].set_xlabel("Time")
    
    plt.show()


def feature_variance(feature_values_list, grouping_type="individual"):
    
    feature_values_list = data_grouping(feature_values_list, grouping_type=grouping_type, individual_length=74)
        
    feature_values_list = feature_values_list.mean(axis=3)
    
    # Change axis 1 and 2
    feature_values_list = np.swapaxes(feature_values_list, 1, 2)
    
    # One variance per feature per individual
    individual_variances = np.std(feature_values_list, axis=2)
    individual_variances = np.swapaxes(individual_variances, 0, 1)
    individual_variances = np.mean(individual_variances, axis=1)
    
    # One variance per feature
    feature_values_list = np.swapaxes(feature_values_list, 0, 1)
    feature_values_list = np.reshape(feature_values_list, (feature_values_list.shape[0], feature_values_list.shape[1] * feature_values_list.shape[2]))
    group_variances = np.std(feature_values_list, axis=1)
    
    variance_ratios = individual_variances / group_variances
    
    # Retrieve the best features
    return np.argsort(variance_ratios), variance_ratios


def feature_continuous_distance(features_values_list, continuous_variable):
      
    features_values_list = features_values_list.swapaxes(1, 0)
    
    delta_v = []
    for i in range(len(continuous_variable)):
        row = []
        for j in range(len(continuous_variable)):
            row.append(np.abs(continuous_variable[i] - continuous_variable[j]))
        delta_v.append(row)
    
    delta_v = np.array(delta_v)
    delta_v = delta_v[np.triu_indices(len(delta_v), k=1)]
    
    features_correlation = []
    
    for feature in features_values_list:
        # 444 points
        delta_x = []
        for i in range(len(feature)):
            row = []
            for j in range(len(feature)):
                row.append(np.abs(feature[i] - feature[j]))
            delta_x.append(row)
        
        # Calculate the correlation between delta_x and delta_v
        delta_x = np.array(delta_x)
        delta_x = delta_x[np.triu_indices(len(delta_x), k=1)]       
        
        correlation = np.corrcoef(delta_x, delta_v)[0, 1]
        features_correlation.append(correlation)
        
        print(correlation)
                
    features_correlation = np.array(features_correlation)
    print(features_correlation.shape)
    
    # Retrieve the best features
    return np.argsort(features_correlation), features_correlation


def random_forest_feature_importance(features_values_list, group_idx=[[0, 1], [2, 3], [4, 5]], individual_length=74):
    
    # Shape (num_segments, num_features)
    X = np.reshape(features_values_list.mean(axis=3), (features_values_list.shape[0] * features_values_list.shape[1], features_values_list.shape[2]))
    print(X.shape)
    
    Y = []
    for g_idx in range(len(group_idx)):
        Y += [[g_idx] * individual_length for i_idx in range(len(group_idx[g_idx]))]
    Y = np.array(Y).flatten()
    print(Y)
    print(Y.shape)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, Y_train)
    
    Y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy: {accuracy}")


class SignificantFeaturesModel(nn.Module):
    def __init__(self, num_features):
        super(SignificantFeaturesModel, self).__init__()
        
        self.weights = nn.Parameter(torch.ones(num_features))
    

def significant_features_loss(X, Y, model, lambda_within=1, lambda_between=1):
    # Shape of X: (num_segments, num_features)
    
    X = X * model.weights
    
    unique_labels = np.unique(Y)
    group_centroids = []
    within_group_variances = []
    
    # Compute centroids and within group variances
    for label in unique_labels:
        
        # Mask for the current group
        group_mask = Y == label
        group_segments = X[group_mask]
        
        # Centroid
        centroid = group_segments.mean(dim=0)
        group_centroids.append(centroid)
        
        # Within group variance
        within_group_variance	= ((group_segments - centroid) ** 2).mean()
        within_group_variances.append(within_group_variance)
        
    # Compute within-group variance component of the loss
    L_within = torch.stack(within_group_variances).mean()
    
    # Compute between-group variance component of the loss
    L_between = 0
    for i, centroid_i in enumerate(group_centroids):
        for j, centroid_j in enumerate(group_centroids):
            if i != j:
                # Maximize distance between centroids
                L_between -= (centroid_i - centroid_j).pow(2).sum()
                
    # Compute total loss
    loss = lambda_within*L_within# + lambda_between*L_between
    
    return loss

    
def find_n_closest_groups(group_features_values, individual_features_values, group_idx, n=1):
    
    a, b, c, d = group_features_values.shape
    group_values = np.reshape(group_features_values, (a, b, d, c))
    group_means = np.mean(np.mean(group_values, axis=1), axis=1)
    
    a, b, c, d = individual_features_values.shape
    individual_values = np.reshape(individual_features_values, (a, b, d, c))
    individual_means = np.mean(np.mean(individual_values, axis=1), axis=1)
    
    seed = individual_means[group_idx]
    
    distances = np.linalg.norm(group_means - seed, axis=1)
    
    closest_groups = np.argsort(distances)[:n]
    
    return closest_groups
    

def create_test_data(subjects_idx, subjects_path, pca_path, signal_length=600, n_components=30):
    subjects = [i for i in os.listdir(subjects_path) if len(i) == 28]

    parsed_subjects = []

    for sub in subjects:
        parsed_subjects.append(sub[-15:-4].replace("_", ""))
        
    parsed_subjects = [parsed_subjects[i] for i in subjects_idx]
        
    cancam100_subjects = []
    if not pca_path.endswith(".npy"):
        cancam100_subjects = os.listdir(pca_path)
        cancam100_subjects = [sub[-15:-4] for sub in cancam100_subjects if len(sub) == 28]

    dataset = CanCAM_MEG(
        signal_length=signal_length,
        rois="all",
        sub_ids=parsed_subjects,
        pca=True,
        pca_sub_ids=cancam100_subjects,
        n_components=n_components,
        transposed=False,
        with_class_cond=True,
        start_index=0,
        end_index=-1,
        folder_path=subjects_path,
        pca_folder_path=pca_path,
    )

    return dataset


def n_significant_features(individual_features_values_list, group_features_values_list, group_idx, features_order):
    
    previous_accuracies = []
    for i in range(0, individual_features_values_list.shape[2], 5):
        # Choosing i best features
        chosen_features_order = features_order[0:i]
        
        range_accuracy = 0
        hits = 0
        for ind_idx in range(len(individual_features_values_list)):
            closest_group = find_n_closest_groups(group_features_values_list[:, :, chosen_features_order, :], individual_features_values_list[:, :, chosen_features_order, :], ind_idx, 1)[0]
            if ind_idx in group_idx[closest_group]:
                hits += 1
        
        range_accuracy = hits / len(individual_features_values_list)
        print(f"Accuracy for {i} features: {range_accuracy}")
        print(f"Previous accuracies: {np.mean(previous_accuracies)}")
           
        previous_accuracies.append(range_accuracy)
        
        if i % 100 == 0 and i > 0:
            if np.mean(previous_accuracies[-5:]) < np.mean(previous_accuracies) or np.mean(previous_accuracies) < 0.5:
                break
    
    # Gets the last index of the maximum accuracy and returns the number of features 
    best_index = len(previous_accuracies) - np.argmax(previous_accuracies[::-1]) - 1
    return best_index*5, previous_accuracies[best_index]


def data_grouping(features_values_list, grouping_type="individual", group_idx=[[0, 1], [2, 3]], individual_length=74, fingerprinting_length=15):
    
    if grouping_type == "individual":
        new_features_values_list = []
        for group in features_values_list:
            for ind_idx in range(0, len(group), individual_length):
                new_features_values_list.append(group[ind_idx:ind_idx + individual_length])
    
    elif grouping_type == "fingerprinting":
        new_features_values_list = []
        for group in features_values_list:
            for ind_idx in range(0, len(group), individual_length):
                ind = group[ind_idx:ind_idx + individual_length]
                new_features_values_list.append(ind[0:fingerprinting_length])
                new_features_values_list.append(ind[individual_length - fingerprinting_length:])
                
    elif grouping_type == "group":
        new_features_values_list = []
        for group in group_idx:
            new_features_values_list.append(np.concatenate([features_values_list[i] for i in group]))
                
    else:
        new_features_values_list = features_values_list
        
    return np.array(new_features_values_list)


def extract_features_wrapper(model_path, model_name, cfg_name, groups_idx=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], save_features_list: bool = False, save_path: str = None):
    
    features_values_list = []
    
    for group in groups_idx:
        dataset = create_test_data(group, "/meg/meg1/users/mlapatrie/data/Scouted_MEG_CanCAM_300/test_data", "/meg/meg1/users/mlapatrie/data/CanCAM200/CanCAM_100/pca_components.npy")
    
        features_values = get_slconv_features_values(model_path, model_name, cfg_name, dataset)
    
        features_values_list.append(features_values)
        
    features_values_list = np.array(features_values_list)
        
    if save_features_list:
        np.save(os.path.join(save_path, "features_values.npy"), features_values_list)
    
    return features_values_list
    
    
if __name__ == "__main__":
    ### Get the features values ###
    
    # Creates 3 groups with 5 subjects each
    #groups_idx = [[(i*5 + j) for j in range(5)] for i in range(3)]
    #groups_idx = [[i] for i in range(18)]
    #features_values_list = extract_features_wrapper("/meg/meg1/users/mlapatrie/data/CanCAM200/CanCAM_100/", "conditional_model.pkl", "conditional_config.pkl", groups_idx, True, "/meg/meg1/users/mlapatrie/data/CanCAM200/")
    
    ### Load precomputed features values ###
    ### Start segments ###
    
    #individual_features_values_list = np.load("/meg/meg1/users/mlapatrie/data/CanCAM200/features_values_start.npy")
    #print(individual_features_values_list.shape)  
    #
    ## Creating random pairs
    #n_individuals = len(individual_features_values_list)
    #individuals_idx = [i for i in range(n_individuals)]
    #np.random.shuffle(individuals_idx)
    ##group_idx = [{individuals_idx[i], individuals_idx[i+1]} for i in range(0, n_individuals, 2)]
    #group_idx = [[i] for i in range(len(individual_features_values_list))]
    #
    #group_features_values_list = individual_features_values_list#data_grouping(individual_features_values_list, grouping_type=None, group_idx=group_idx)
    #print(group_features_values_list.shape)
    #
    #
    #### Calculate feature variance ###
    #
    #features_order, variance_ratios = feature_variance(group_features_values_list, grouping_type=None)
    #print(f"Features with the smallest variance ratio: {features_order[:100]}")
    #print(f"Features with the highest variance ratio: {features_order[-100:]}")
    #
    ## Find number of significant features
    ##n_features = n_significant_features(individual_features_values_list, group_features_values_list, group_idx=group_idx, features_order=features_order)
    ##print(f"Number of significant features: {n_features}")
    #
    #chosen_features = features_order[:500]#n_features]   
    #
    #visualize_features_per_group(individual_features_values_list[:, :, chosen_features, :], dimension_reduction="t-SNE", convex_hull=False, dimensions=2, title="Reduced SLConv features")
    #
    #### Load precomputed features values ###
    #### All segments ###
    #
    #individual_features_values_list = np.load("/meg/meg1/users/mlapatrie/data/CanCAM200/features_values_all.npy")
    #print(individual_features_values_list.shape)
    #
    #grouped_features = []
    #
    #group_idx = [[0, 8], [1, 15], [2, 16], [3, 17], [4, 19], [5, 20], [6, 21], [7, 22], [18, 25], [9, 23], [10, 24], [11, 26], [12, 27], [13, 28], [14, 29]]
    #
    #for group in group_idx:
    #    grouped_features.append(np.concatenate((individual_features_values_list[group[0]], individual_features_values_list[group[1]])))
    #
    #grouped_features = np.array([np.array(grouped_features)[0], np.array(grouped_features)[1], np.array(grouped_features)[7], np.array(grouped_features)[8], np.array(grouped_features)[11], np.array(grouped_features)[12]])
    #print(grouped_features.shape)
    #
    #
    #features_order, variance_ratios = feature_variance(grouped_features, grouping_type=None)
    #print(f"Features with the smallest variance ratio: {features_order[:100]}")
    #print(f"Features with the highest variance ratio: {features_order[-100:]}")
    #
    #
    #visualize_features_per_group(grouped_features[:, :, features_order[:500], :], dimension_reduction="t-SNE", convex_hull=False, dimensions=2, title="Reduced SLConv features")
    
    #features_values_list = np.load("D:/DDPM_data/CanCAM200/subjects_matrices.npy")
    #print(features_values_list.shape)
    #
    #group_indices = np.array([[4, 8, 36, 38, 40, 57, 67, 74, 87, 90], [2, 22, 27, 32, 64, 69, 72, 80, 88, 24], [37, 46, 47, 50, 55, 73, 76, 86, 91, 98], 
    #                 [0, 1, 6, 18, 23, 25, 28, 39, 61, 65], [12, 17, 42, 54, 62, 68, 89, 94, 95, 99], [7, 11, 15, 16, 31, 44, 45, 48, 49, 53],
    #                 [9, 19, 20, 29, 30, 34, 43, 51, 52, 56], [3, 5, 10, 13, 14, 21, 26, 33, 35, 41]])
    #
    #print(group_indices.shape)
#
    #features_values_list = np.reshape(features_values_list, (100, 5, 19900))
   #
    #grouped_features_values_list = []
    #
    #for group in group_indices:
    #    grouped_features_values_list.append([features_values_list[i] for i in group])
    #    
    #grouped_features_values_list = np.array(grouped_features_values_list)
    #print(grouped_features_values_list.shape)
    #
    #grouped_features_values_list = np.reshape(grouped_features_values_list, (4, 100, 19900))
    #print(grouped_features_values_list.shape)
    #
    #feature_order, variance_ratios = feature_variance(grouped_features_values_list, grouping_type=None)
    #print(f"Features with the smallest variance ratio: {feature_order[:100]}")
    #print(f"Variation ratios: {variance_ratios[feature_order[:100]]}")
#
    #print(f"Features with the highest variance ratio: {feature_order[-100:]}")
    #print(f"Variation ratios: {variance_ratios[feature_order[-100:]]}")
    #
    #visualize_features_per_group(features_values_list[:, :, feature_order[:10]], dimension_reduction="t-SNE", convex_hull=True, dimensions=2, title="Reduced SLConv features")
    
    
    features_values_list = np.load("D:/DDPM_data/CanCAM200/features_values_small.npy")
    features_values_list = np.reshape(features_values_list, (1, 74*6, 2880, 600))
    print(features_values_list.shape)
    
    #feature_order, features_correlation = feature_continuous_distance(features_values_list.mean(axis=2), [10] * 74 + [20] * 74 + [30] * 74 + [40] * 74 + [50] * 74 + [60] * 74) 
    #np.save("D:/DDPM_data/CanCAM200/feature_order.npy", feature_order)
    
    feature_order = np.load("D:/DDPM_data/CanCAM200/feature_order.npy")
    print(feature_order.shape)
    
    
    visualize_features_per_group(features_values_list[:, :, feature_order[:50], :], dimension_reduction="t-SNE", convex_hull=True, dimensions=2, title="Reduced SLConv features")

import torch
import numpy as np

import matplotlib.pyplot as plt


from utils.utils import path_loader
from train_diffusion_model import init_diffusion_model


activations = {}
gradients = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook


def get_gradients(name):
    def hook(model, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()
    return hook


def neuron_correlation(data, model, layer, segment_start_idx=0, segment_end_idx=-1):
    
    time_index = model.unormalized_probs.multinomial(num_samples=len(data), replacement=True).to(torch.float)
    
    sig = torch.tensor(np.array([segment["signal"] for segment in data]))
    cond = torch.tensor(np.array([segment["cond"] for segment in data]))
        
    # Forward pass
    model.network.forward(sig, t=time_index, cond=cond)
    
    # Concatenate activations to a shape of (num_neurons, segment_length * num_segments)
    activations_long = np.concatenate(activations[layer].detach().numpy()[segment_start_idx:segment_end_idx], axis=1)
        
    # Compute correlation map
    correlation_map = np.corrcoef(activations_long)
    
    return correlation_map


if __name__ == "__main__":
    
    # Initialize the model
    config_name = "conditional_config.pkl"
    model_name = "conditional_model.pkl"
    model_path = "/meg/meg1/users/mlapatrie/data/CanCAM200/CanCAM_100"
    
    model_cfg = path_loader(config_name, model_path)
    
    model, network = init_diffusion_model(model_cfg)
    model.load_state_dict(path_loader(model_name, model_path))

    model.zero_grad()
    model.eval()
    
    # Load real data
    data = np.load(f"/meg/meg1/users/mlapatrie/data/CanCAM200/benchmark_data/300_2s_pca_30.npy", allow_pickle=True)
    
    ### Neuron correlation map ###
    ## Adding a forward hook to the last SLConv layer
    #model.network.conv_pool[15].register_forward_hook(get_activation('SLConv'))
    #
    #correlation_map = neuron_correlation(data, model, "SLConv")
    #plt.imshow(correlation_map)
    #plt.show()
    
    ### Feature attribution ###
    
    # Register hook on last conv layer to get feature values
    model.network.conv_pool[15].register_forward_hook(get_activation("SLConv"))
    
    # Look at the previous layer for grads
    target_layer = model.network.conv_pool[14]
    target_layer.register_forward_hook(get_activation("features"))
    target_layer.register_full_backward_hook(get_gradients("gradients"))
    
    time_index = model.unormalized_probs.multinomial(num_samples=len(data), replacement=True).to(torch.float)

    sig = torch.tensor(np.array([segment["signal"] for segment in data]))
    cond = torch.tensor(np.array([segment["cond"] for segment in data]))
         
    output = model.network.forward(sig, t=time_index, cond=cond)
    
    print(activations["SLConv"].shape)
    activations["SLConv"][1].mean().backward()
    
    # Access saved activations and gradients
    grads = gradients['gradients']
    grads = torch.reshape(grads, (30, 30, 32, 600))
    print(grads.shape)
    features = activations['features']
    
    # Weight the activations with the gradients
    weights = torch.mean(grads, [2], keepdim=True)
    print(weights.shape)
    grad_cam = torch.sum(weights * features, dim=2, keepdim=True)
    print(grad_cam.shape)
    
    grad_cam = torch.relu(grad_cam)
    
    # Normalize the heatmap
    grad_cam = (grad_cam - grad_cam.min()) / grad_cam.max()
    
    grad_cam = torch.reshape(grad_cam, (30, 30, 600))
    
    print(grad_cam.shape)
    
    # Plot the data
    
    input_idx = 1
    
    input_data = np.array(data[input_idx]["signal"])
    grad_cam_data = grad_cam.detach().numpy()[input_idx]
    
    num_channels = input_data.shape[0]
    time_points = np.arange(0, input_data.shape[1])
    
    offsets = np.linspace(0, num_channels*2, num_channels)
    
    plt.figure(figsize=(15, 10))
    
    for i in range(num_channels):
        
        offset_channel_data = input_data[i, :] + offsets[i]
        
        plt.plot(time_points, offset_channel_data)
        
        # Overlaying Grad_CAM heatmap
        for t in range(len(time_points)):
            plt.fill_between([time_points[t], time_points[t]+1],
                             [offset_channel_data[t], offset_channel_data[t]+1],
                             [offset_channel_data[t] + offsets[1]/2, offset_channel_data[t] + offsets[1]/2],
                             color='red', alpha=grad_cam_data[i, t])
        
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('30 PC with Grad-CAM Overlay')
    plt.show()
    
    
    
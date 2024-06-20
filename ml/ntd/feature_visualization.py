
import torch
from torchviz import make_dot
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import neurodsp.sim as sim

from utils.utils import path_loader
from utils.pca import project_on_pca_torch
from train_diffusion_model import init_diffusion_model


signal_length = 2
sampling_frequency = 300
pca_components = 30

num_channels = 200

pca_matrix = np.load(f"/meg/meg1/users/mlapatrie/data/CanCAM200/CanCAM_100/pca_components.npy")

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook


def objective_function(activation_signal, cond, model, layer_name, time_index, feature_idx=0):
    model.network.forward(activation_signal, t=time_index, cond=cond)
    
    activation = activations[layer_name][0][feature_idx]
    
    return -1 * activation.mean().abs(), activation.mean()


def simulate_signal(variables):
    
    # Simulate a sin wave of frequency variables[0]
    t = torch.linspace(0, signal_length, steps=signal_length * sampling_frequency) 
    signal_channel = torch.sin(2 * np.pi * variables[0] * t)
    
    signal = signal_channel.repeat(num_channels, 1)

    # PCA the data
    signal = project_on_pca_torch(pca_matrix, signal)
    
    a, b = signal.shape
    signal = torch.reshape(signal, (1, a, b))
    
    return signal


def optimize_image(sig, cond, model, layer_name, time_index, feature_idx=0, signal_function=lambda x: x, num_iterations=1000, learning_rate=0.001):
    # Initialize the optimizer
    optimizer = optim.Adam([sig], lr=learning_rate)
    
    loss = 0
    activation = 0
    
    loss_history = []
    activation_history = []
    
    for i in range(num_iterations):
        # Zero the gradients
        optimizer.zero_grad()
        activation_signal = signal_function(sig)
        
        # Compute the loss
        loss, activation = objective_function(activation_signal, cond, model, layer_name, time_index, feature_idx)
        
        # Backpropagate the loss
        loss.backward()
        
        vis_graph = make_dot(loss, params=dict(list(model.named_parameters()) + [('sig', sig)]))
        vis_graph.render('model_graph', format='png')
        
        # Take a step
        optimizer.step()
        
        # Append the loss to the history
        loss_history.append(loss.item())
        
        # Append the activation to the history
        activation_history.append(activation.item())
        
        if i % 10 == 0:
            #print(np.linalg.norm(sig.grad))
            print(sig.grad)
            print(f"Iteration {i} - Loss: {loss.item()} - Activation: {activation.item()}")
        
    return sig, loss_history, activation_history


def base_data_wrapper(template_image, data_type="noise"):
    
    if data_type == "zeros":
        # Initializing the signal with zeros
        sig = torch.tensor(np.zeros_like([template_image["signal"]]), dtype=torch.float, requires_grad=True)
    
    elif data_type == "noise":
        # Initializing the signal with noise
        sig = torch.tensor(np.random.rand(1, *template_image["signal"].shape), dtype=torch.float, requires_grad=True)
    
    cond = np.array([template_image["cond"]])
    
    if len(cond.shape) < 3:
        cond = torch.tensor([cond], dtype=torch.float)
    elif len(cond.shape) == 3:
        cond = torch.tensor(cond, dtype=torch.float)
    else:
        raise f"Too many dimensions in cond vector.\nExpected 2 or 3 dimensions, got {len(cond.shape)}."
    
    return sig, cond


def template_data_wrapper(data_type="real", signal_length=2, sampling_frequency=300, pca_components=30):
    
    if data_type == "real":
        return np.load(f"/meg/meg1/users/mlapatrie/data/CanCAM200/benchmark_data/{sampling_frequency}_{signal_length}s_pca_{pca_components}.npy", allow_pickle=True)[0]

    elif data_type == "simulated":
        # The signal key contains the variable to create a simulated segment
        # [freq]
        return {"signal": np.array(0.0), "cond": np.zeros(signal_length * sampling_frequency)}
                       
                       
if __name__ == "__main__":
    
    # Initializing the model
    config_name = "conditional_config.pkl"
    model_name = "conditional_model.pkl"
    model_path = "/meg/meg1/users/mlapatrie/data/CanCAM200/CanCAM_100/"
    
    model_cfg = path_loader(config_name, model_path)
    
    model, network = init_diffusion_model(model_cfg)
    model.load_state_dict(path_loader(model_name, model_path))
    
    # Adding a forward hook to the last SLConv layer
    model.network.conv_pool[15].register_forward_hook(get_activation('SLConv'))
    
    model.zero_grad()
    
    # Loading a template image
    template_image = template_data_wrapper(data_type="real", signal_length=signal_length, sampling_frequency=sampling_frequency, pca_components=pca_components)
    
    # Getting time index
    time_index = model.unormalized_probs.multinomial(num_samples=1, replacement=True).to(torch.float)
    
    # Initializing the signal
    sig, cond = base_data_wrapper(template_image, data_type="zeros")
    
    print(sig.shape)
    print(cond.shape)
    
    learning_rate = 0.1
    num_iterations = 100
    
    optimized_sig, loss_history, activation_history = optimize_image(sig, cond, model, signal_function=lambda x: x, layer_name="SLConv", time_index=time_index, feature_idx=0, num_iterations=num_iterations, learning_rate=learning_rate)
    
    plt.imshow(optimized_sig.detach()[0], aspect='auto')
    plt.show()
    
    
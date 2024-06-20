
from omegaconf import open_dict
import numpy as np
import time
import os

import torch
from einops import repeat

from train_diffusion_model import (
        init_diffusion_model, 
        likelihood_computation_wrapper,
        generate_samples_wrapper
        )

from utils.utils import path_loader
from datasets import CanCAM_MEG


def noise_resiliency(benchmark_path, model_path, model_name, cfg_name, noise_level, seed, batch_size):
    
    cfg = path_loader(cfg_name, model_path)
    
    with open_dict(cfg):
        cfg.likelihood_experiment = {"batch_size": batch_size}
    
    diffusion_model, network = init_diffusion_model(cfg)
    diffusion_model.load_state_dict(path_loader(model_name, model_path))
    
    test_data = np.load(benchmark_path, allow_pickle=True)
    
    # Add noise to test data
    signals = [np.array(test_data[i]["signal"]) for i in range(len(test_data))]
    std_signal = np.std(signals)
    
    for i in range(len(test_data)):
        test_data[i]["signal"] = test_data[i]["signal"] + np.random.normal(0, std_signal * noise_level, size=np.array(test_data[i]["signal"]).shape)
    
    likelihood = likelihood_computation_wrapper(cfg, diffusion_model, test_data)
    
    np.save(os.path.join(model_path, "benchmark", "noise_resiliency", "likelihoods", str(noise_level*100) + "%"), likelihood)
    
    print("DONE", str(noise_level*100) + "%")
    
    
def noise_resiliency_wrapper(benchmark_path, model_path, model_name, cfg_name, noise_levels, seed=42, batch_size=15):    
    
    # Create folder structure
    if not os.path.isdir(os.path.join(model_path, "benchmark")):
        os.mkdir(os.path.join(model_path, "benchmark"))
        
    if not os.path.isdir(os.path.join(model_path, "benchmark", "noise_resiliency")):
        os.mkdir(os.path.join(model_path, "benchmark", "noise_resiliency"))
        
    if not os.path.isdir(os.path.join(model_path, "benchmark", "noise_resiliency", "likelihoods")):
        os.mkdir(os.path.join(model_path, "benchmark", "noise_resiliency", "likelihoods"))
    
    for noise_level in noise_levels:
        noise_resiliency(benchmark_path, model_path, model_name, cfg_name, noise_level, seed=seed, batch_size=batch_size)
        
    # Loading 0% noise likelihoods
    base_llh = np.load(os.path.join(model_path, "benchmark", "noise_resiliency", "likelihoods", "0.0%.npy"))
    
    errors = []
    
    # Compute the mean squared error between llh and base_llh for all noise levels
    for noise_level in noise_levels[1:]:
        llh = np.load(os.path.join(model_path, "benchmark", "noise_resiliency", "likelihoods", str(noise_level*100) + "%.npy"))
        
        # Compute the mean squared error between llh and base_llh
        mse = np.mean(np.square(llh - base_llh))
        errors.append(mse)
    
    np.save(os.path.join(model_path, "benchmark", "noise_resiliency", "errors"), errors)
    
    # Compute AUC of the errors from 0-50% noise and 50-100% noise
    measures = {
        "low_noise": np.mean(errors[:len(errors)//2]),
        "high_noise": np.mean(errors[len(errors)//2:]),
    }
    
    np.save(os.path.join(model_path, "benchmark", "noise_resiliency", "measures"), measures)
    
    print("DONE NOISE RESILIENCY TEST")


def alternate_diffusion(batch_size_ls, diffusion_model, ground_truth_diffusion_model, sample_length, diffusion_type, cond=None):
    
    samples_ls = []
    history = []
    running_batch_count = 0
    
    for bs in batch_size_ls:
        if cond is None:
            batch_cond = cond
        else:
            batch_cond = cond[running_batch_count : running_batch_count + bs]
            batch_cond = batch_cond.to(diffusion_model.device)
            batch_cond = batch_cond.to(ground_truth_diffusion_model.device)

        # Generate samples
        noise_type = "alpha_beta"
        
        sampler = diffusion_model.noise_sampler
        ground_truth_sampler = ground_truth_diffusion_model.noise_sampler

        if cond is not None:
            cond_batch, cond_channel, cond_length = cond.shape
            assert cond_batch == 1 or cond_batch == bs
            assert cond_length == sample_length
            if cond_batch == 1:
                cond = cond.repeat(bs, 1, 1)
                
        assert diffusion_model.network.signal_channel == ground_truth_diffusion_model.network.signal_channel
        assert diffusion_model.diffusion_time_steps == ground_truth_diffusion_model.diffusion_time_steps

        diffusion_model.eval()
        ground_truth_diffusion_model.eval()
        
        print("Starting sample generation")
        with torch.no_grad():
            state = sampler.sample(
                sample_shape=(
                    bs,
                    diffusion_model.network.signal_channel,
                    sample_length,
                )
            )

            history_ls = [state]

            start = time.time()
            if diffusion_type == "smallest_distance":
                time_steps = diffusion_model.diffusion_time_steps
            elif diffusion_type == "longest_distance":
                time_steps = int(diffusion_model.diffusion_time_steps * 3)
            
            running_time_steps = diffusion_model.diffusion_time_steps - 1
                
            for i in range(time_steps):
                if i % 100 == 0:
                    print(f"Step {i+1}/{time_steps}")
                
                if diffusion_type == "smallest_distance":
                    timestep = diffusion_model.diffusion_time_steps - i - 1
                elif diffusion_type == "longest_distance":
                    if i % 3 == 0 and i != 0:
                        running_time_steps += 1
                        timestep = running_time_steps
                    else:
                        running_time_steps -= 1
                        timestep = running_time_steps
            
                if i % 2 == 0:
                    model_in_use = diffusion_model
                else:
                    model_in_use = ground_truth_diffusion_model
                    
                if diffusion_type == "longest_distance":
                    if i % 3 == 0:
                        model_in_use = ground_truth_diffusion_model
                    else:
                        model_in_use = diffusion_model
                
                time_vector = torch.tensor([timestep], device=model_in_use.device).repeat(
                    bs
                )
                samp = sampler.sample(
                    sample_shape=(
                        bs,
                        model_in_use.network.signal_channel,
                        sample_length,
                    )
                )

                if diffusion_type == "smallest_distance":
                    state = (1 / torch.sqrt(model_in_use.alphas[timestep])) * (
                        state
                        - (
                            (
                                (1.0 - model_in_use.alphas[timestep])
                                / (torch.sqrt(1.0 - model_in_use.alpha_bars[timestep]))
                            )
                            * model_in_use.network.forward(
                                state,
                                time_vector,
                                cond=cond,
                            )[0]
                        )
                    )
                elif diffusion_type == "longest_distance" and i % 3 == 0:
                    # Going away from the groundtruth model
                    state = (1 / torch.sqrt(model_in_use.alphas[timestep])) * (
                        state
                        + (
                            (
                                (1.0 - model_in_use.alphas[timestep])
                                / (torch.sqrt(1.0 - model_in_use.alpha_bars[timestep]))
                            )
                            * model_in_use.network.forward(
                                state,
                                time_vector,
                                cond=cond,
                            )[0]
                        )
                    )
                elif diffusion_type == "longest_distance":
                    # For the two other steps, we go towards the model
                    state = (1 / torch.sqrt(model_in_use.alphas[timestep])) * (
                        state
                        - (
                            (
                                (1.0 - model_in_use.alphas[timestep])
                                / (torch.sqrt(1.0 - model_in_use.alpha_bars[timestep]))
                            )
                            * model_in_use.network.forward(
                                state,
                                time_vector,
                                cond=cond,
                            )[0]
                        )
                    )

                if timestep > 0:
                    if noise_type == "beta":
                        sigma = model_in_use._get_beta(timestep)
                    elif noise_type == "alpha_beta":
                        sigma = model_in_use._get_alpha_beta(timestep)
                    state += sigma * samp

                history_ls.append(state)
            
            print(f"Took {time.time() - start}")
        
        samples_ls.append(state)
        history.append(history_ls)
        running_batch_count += bs
        
    return samples_ls, history


def smallest_distance(benchmark_path, model_path, model_name, cfg_name, ground_truth_model_path, ground_truth_model_name, ground_truth_cfg_name, num_samples, batch_size):
    
    # Loading model
    cfg = path_loader(cfg_name, model_path)
    
    with open_dict(cfg):
        cfg.likelihood_experiment = {"batch_size": batch_size}
    
    diffusion_model, network = init_diffusion_model(cfg)
    diffusion_model.load_state_dict(path_loader(model_name, model_path))
    
    # Loading ground truth model
    ground_truth_cfg = path_loader(ground_truth_cfg_name, ground_truth_model_path)
    
    with open_dict(ground_truth_cfg):
        ground_truth_cfg.likelihood_experiment = {"batch_size": batch_size}
        
    ground_truth_diffusion_model, ground_truth_network = init_diffusion_model(ground_truth_cfg)
    ground_truth_diffusion_model.load_state_dict(path_loader(ground_truth_model_name, ground_truth_model_path))
    
    # Loading test data as a template
    test_data = np.load(benchmark_path, allow_pickle=True)
    
    sample_length = test_data[0]["signal"].shape[1]
    print("Sample length:", sample_length)
    
    try:
        cond = torch.stack([dic["cond"] for dic in test_data])[:num_samples]
        cond = cond.to(diffusion_model.device)
        cond = cond.to(ground_truth_diffusion_model.device)
        
    except KeyError:
        cond = None
        
    print("Loaded the models and the cond vectors")
    
    # Generate samples batch-wise
    batch_size_ls = [batch_size] * (num_samples // batch_size)
    if (remainder := num_samples % batch_size) > 0:
        batch_size_ls += [remainder]

    # Setting cond
    if cond is not None:
        if len(cond.shape) == 2:
            cond = repeat(cond, "c t -> b c t", b=num_samples)
        elif len(cond.shape) == 3:
            assert cond.shape[0] == num_samples
        else:
            raise ValueError("Cond must be a 2D or 3D tensor!")

    samples_ls, history = alternate_diffusion(batch_size_ls, diffusion_model, ground_truth_diffusion_model, sample_length, diffusion_type="smallest_distance", cond=cond)
    print("Finished generating the samples")

    all_samples = torch.cat(samples_ls, dim=0).cpu()
    
    # Creating a data object with the samples
    samples_dataset = []
    
    for i, sample in enumerate(all_samples):
        sample_dic = test_data[0].copy()
        sample_dic["signal"] = sample
        samples_dataset.append(sample_dic)
    
    samples_dataset = np.array(samples_dataset)
    np.save(os.path.join(model_path, "benchmark", "smallest_distance", f"samples.npy"), samples_dataset, allow_pickle=True)
    
    # Compute the likelihoods for each sample and each model
    print("Computing the likelihoods on the samples")
    
    # Model
    likelihood = likelihood_computation_wrapper(cfg, diffusion_model, samples_dataset)
    np.save(os.path.join(model_path, "benchmark", "smallest_distance", "likelihoods", "model"), likelihood)
    
    # Ground truth model
    ground_truth_likelihood = likelihood_computation_wrapper(ground_truth_cfg, ground_truth_diffusion_model, samples_dataset)
    np.save(os.path.join(model_path, "benchmark", "smallest_distance", "likelihoods", "ground_truth_model"), ground_truth_likelihood)
    
    # Compute the mean distance
    mean_distance = np.mean(np.absolute(np.array(likelihood) - np.array(ground_truth_likelihood)))
    np.save(os.path.join(model_path, "benchmark", "smallest_distance", "mean"), mean_distance)
    
    return mean_distance


def mean_distance(benchmark_path, model_path, model_name, cfg_name, ground_truth_model_path, ground_truth_model_name, ground_truth_cfg_name, num_samples, batch_size):
    # Loading model
    cfg = path_loader(cfg_name, model_path)
    
    with open_dict(cfg):
        cfg.generate_samples = {"batch_size": batch_size, "num_samples": num_samples}
    
    diffusion_model, network = init_diffusion_model(cfg)
    diffusion_model.load_state_dict(path_loader(model_name, model_path))
    
    # Loading ground truth model
    ground_truth_cfg = path_loader(ground_truth_cfg_name, ground_truth_model_path)
    
    with open_dict(ground_truth_cfg):
        ground_truth_cfg.likelihood_experiment = {"batch_size": batch_size}
        
    ground_truth_diffusion_model, ground_truth_network = init_diffusion_model(ground_truth_cfg)
    ground_truth_diffusion_model.load_state_dict(path_loader(ground_truth_model_name, ground_truth_model_path))
    
    print("Loaded the models")
    
    # Loading test data as a template
    test_data = np.load(benchmark_path, allow_pickle=True)
    print("Generating samples")
    start = time.time()
    samples = generate_samples_wrapper(cfg, diffusion_model, test_data)
    print(f"Took {time.time() - start}")
    
    # Creating a data object with the samples
    samples_dataset = []
    
    for i, sample in enumerate(samples):
        sample_dic = test_data[0].copy()
        sample_dic["signal"] = sample 
        samples_dataset.append(sample_dic)
    
    samples_dataset = np.array(samples_dataset)
    np.save(os.path.join(model_path, "benchmark", "mean_distance", f"samples.npy"), samples_dataset, allow_pickle=True)
    
    # Compute the likelihoods for each sample and each model
    print("Computing the likelihoods on the samples")
    
    # Model
    likelihood = likelihood_computation_wrapper(cfg, diffusion_model, samples_dataset)
    np.save(os.path.join(model_path, "benchmark", "mean_distance", "likelihoods", "model"), likelihood)
    
    # Ground truth model
    ground_truth_likelihood = likelihood_computation_wrapper(ground_truth_cfg, ground_truth_diffusion_model, samples_dataset)
    np.save(os.path.join(model_path, "benchmark", "mean_distance", "likelihoods", "ground_truth_model"), ground_truth_likelihood)
    
    # Compute the mean distance
    mean_distance = np.mean(np.absolute(np.array(likelihood) - np.array(ground_truth_likelihood)))
    np.save(os.path.join(model_path, "benchmark", "mean_distance", "mean"), mean_distance)
    
    return mean_distance


def longest_distance(benchmark_path, model_path, model_name, cfg_name, ground_truth_model_path, ground_truth_model_name, ground_truth_cfg_name, num_samples, batch_size):
    
    # Loading model
    cfg = path_loader(cfg_name, model_path)
    
    with open_dict(cfg):
        cfg.likelihood_experiment = {"batch_size": batch_size}
    
    diffusion_model, network = init_diffusion_model(cfg)
    diffusion_model.load_state_dict(path_loader(model_name, model_path))
    
    # Loading ground truth model
    ground_truth_cfg = path_loader(ground_truth_cfg_name, ground_truth_model_path)
    
    with open_dict(ground_truth_cfg):
        ground_truth_cfg.likelihood_experiment = {"batch_size": batch_size}
        
    ground_truth_diffusion_model, ground_truth_network = init_diffusion_model(ground_truth_cfg)
    ground_truth_diffusion_model.load_state_dict(path_loader(ground_truth_model_name, ground_truth_model_path))
    
    # Loading test data as a template
    test_data = np.load(benchmark_path, allow_pickle=True)
    
    sample_length = test_data[0]["signal"].shape[1]
    print("Sample length:", sample_length)
    
    try:
        cond = torch.stack([dic["cond"] for dic in test_data])[:num_samples]
        cond = cond.to(diffusion_model.device)
        cond = cond.to(ground_truth_diffusion_model.device)
        
    except KeyError:
        cond = None
        
    print("Loaded the models and the cond vectors")
    
    # Generate samples batch-wise
    batch_size_ls = [batch_size] * (num_samples // batch_size)
    if (remainder := num_samples % batch_size) > 0:
        batch_size_ls += [remainder]

    # Setting cond
    if cond is not None:
        if len(cond.shape) == 2:
            cond = repeat(cond, "c t -> b c t", b=num_samples)
        elif len(cond.shape) == 3:
            assert cond.shape[0] == num_samples
        else:
            raise ValueError("Cond must be a 2D or 3D tensor!")

    samples_ls, history = alternate_diffusion(batch_size_ls, diffusion_model, ground_truth_diffusion_model, sample_length, diffusion_type="longest_distance", cond=cond)
    print("Finished generating the samples")
    
    all_samples = torch.cat(samples_ls, dim=0).cpu()
    
    # Creating a data object with the samples
    samples_dataset = []
    
    for i, sample in enumerate(all_samples):
        sample_dic = test_data[0].copy()
        sample_dic["signal"] = sample
        samples_dataset.append(sample_dic)
    
    samples_dataset = np.array(samples_dataset)
    np.save(os.path.join(model_path, "benchmark", "longest_distance", f"samples.npy"), samples_dataset, allow_pickle=True)
    
    # Compute the likelihoods for each sample and each model
    print("Computing the likelihoods on the samples")
    
    # Model
    start = time.time()
    likelihood = likelihood_computation_wrapper(cfg, diffusion_model, samples_dataset)
    np.save(os.path.join(model_path, "benchmark", "longest_distance", "likelihoods", "model"), likelihood)
    print(f"Model likelihoods took: {time.time() - start}")
    
    # Ground truth model
    start = time.time()
    ground_truth_likelihood = likelihood_computation_wrapper(ground_truth_cfg, ground_truth_diffusion_model, samples_dataset)
    np.save(os.path.join(model_path, "benchmark", "longest_distance", "likelihoods", "ground_truth_model"), ground_truth_likelihood)
    print(f"Groundtruth likelihoods took: {time.time() - start}")
        
    # Compute the mean distance
    mean_distance = np.mean(np.absolute(np.array(likelihood) - np.array(ground_truth_likelihood)))
    np.save(os.path.join(model_path, "benchmark", "longest_distance", "mean"), mean_distance)
    
    return mean_distance


def generalization_test_wrapper(benchmark_path, model_path, model_name, cfg_name, ground_truth_model_path, ground_truth_model_name, ground_truth_cfg_name, num_samples=10, batch_size=10):
    
    # Create folder structure
    if not os.path.isdir(os.path.join(model_path, "benchmark")):
        os.mkdir(os.path.join(model_path, "benchmark"))
        
    if not os.path.isdir(os.path.join(model_path, "benchmark", "smallest_distance")):
        os.mkdir(os.path.join(model_path, "benchmark", "smallest_distance"))
        
    if not os.path.isdir(os.path.join(model_path, "benchmark", "smallest_distance", "likelihoods")):
        os.mkdir(os.path.join(model_path, "benchmark", "smallest_distance", "likelihoods"))
        
    if not os.path.isdir(os.path.join(model_path, "benchmark", "mean_distance")):
        os.mkdir(os.path.join(model_path, "benchmark", "mean_distance"))
        
    if not os.path.isdir(os.path.join(model_path, "benchmark", "mean_distance", "likelihoods")):
        os.mkdir(os.path.join(model_path, "benchmark", "mean_distance", "likelihoods"))
    
    if not os.path.isdir(os.path.join(model_path, "benchmark", "longest_distance")):
        os.mkdir(os.path.join(model_path, "benchmark", "longest_distance"))
        
    if not os.path.isdir(os.path.join(model_path, "benchmark", "longest_distance", "likelihoods")):
        os.mkdir(os.path.join(model_path, "benchmark", "longest_distance", "likelihoods"))
    
    print("Created folder structure")
    
    # Computing smallest distance between model and ground_truth_model
    distance = smallest_distance(benchmark_path, model_path, model_name, cfg_name, ground_truth_model_path, ground_truth_model_name, ground_truth_cfg_name, num_samples, batch_size)
    print("Smallest distance:", distance)
    
    # Computing mean distance between model and ground_truth_model
    distance = mean_distance(benchmark_path, model_path, model_name, cfg_name, ground_truth_model_path, ground_truth_model_name, ground_truth_cfg_name, num_samples, batch_size)
    print("Mean distance:", distance)
    
    # Computing longest distance between model and ground_truth_model
    distance = longest_distance(benchmark_path, model_path, model_name, cfg_name, ground_truth_model_path, ground_truth_model_name, ground_truth_cfg_name, num_samples, batch_size)
    print("Longest distance:", distance)
    
    print("DONE GENERALIZATION TEST")
    

def create_benchmark_data():
    subjects = os.listdir("D:/Scouted_MEG_CanCAM_300")

    parsed_subjects = []

    for sub in subjects:
        parsed_subjects.append(sub[-15:-4].replace("_", ""))
        
    old_young_subjects = os.listdir("D:/DDPM_data/CanCAM200/old-young_models/old-young")

    dataset = CanCAM_MEG(
        signal_length=600,
        rois="all",
        sub_ids=parsed_subjects,
        pca=True,
        pca_sub_ids=old_young_subjects,
        n_components=50,
        transposed=False,
        with_class_cond=True,
        start_index=-601,
        end_index=-1,
        folder_path="D:/Scouted_MEG_CanCAM_300",
    )

    np.save("D:/DDPM_data/CanCAM200/benchmark_data/300_2s_pca_50.npy", dataset, allow_pickle=True)


if __name__ == '__main__':
    generalization_test_wrapper("/meg/meg1/users/mlapatrie/data/CanCAM200/benchmark_data/300_2s_pca_30.npy", 
                                "/meg/meg1/users/mlapatrie/data/CanCAM200/old-young_pca_30/230227_0955", "conditional_model.pkl", "conditional_config.pkl",
                                "/meg/meg1/users/mlapatrie/data/CanCAM200/old-young_pca_30/230412_0223", "conditional_model.pkl", "conditional_config.pkl")
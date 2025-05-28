import os
import time
import argparse
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from tqdm import tqdm


num_threads = 1
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)

profiled_layers = (nn.Conv2d, nn.Linear)


def profile_model(model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True
    ) as prof:
        with torch.profiler.record_function("Model Forward Pass"):
            model(input_tensor)

    print(prof.key_averages().table(sort_by="cpu_time_total"))


def benchmark_model(model: torch.nn.Module, input_tensor: torch.Tensor, num_iterations: int) -> None:
    runtimes = []
    for i in tqdm(range(num_iterations)):
        start_time = time.perf_counter_ns()
        model(input_tensor)
        stop_time = time.perf_counter_ns()
        runtimes.append(stop_time - start_time)
    
    median = np.median(runtimes) / 1000000
    mean = np.mean(runtimes) / 1000000
    std = np.std(runtimes) / 1000000
    print(f"Median:\t{median:.3f}\nMean:\t{mean:.3f}\nStd:\t{std:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath", type=str)
    args = parser.parse_args()

    with open(args.config_filepath, "r") as file:
        config = json.load(file)

    # Load model
    model_path = config["model"]
    if os.path.exists(model_path):
        try:
            model = torch.load(model_path).eval()
        except Exception as e:
            print(f"Error loading file: {e}")
    elif model_path in models.list_models():
        model = getattr(models, model_path)().eval()
    else:
        raise ValueError(f"Model '{model_path}' is not found.")

    input_shape = config["input_shape"]
    input_tensor = torch.randn(input_shape)

    print(model)

    print("Insert")
    # insert_print_statements(model)
    
    print("Run")
    model(input_tensor)

    # benchmark_model(model, input_tensor)
    # profile_model(model, input_tensor)

import argparse
import json
import os
import psutil
import time
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm

# Set number of threads
NUM_THREADS = 1
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(NUM_THREADS)

# Get the current process
pid = os.getpid()
process = psutil.Process(pid)

# Bind process to cores 0 to NUM_THREADS-1
process.cpu_affinity(list(range(NUM_THREADS)))

PROFILED_LAYERS = (nn.Conv2d, nn.Linear)

PROFILING_ITERATIONS = 100

layer_start_timestamps = []
layer_stop_timestamps = []
layer_runtimes = []
model_runtimes = []


def _forward_pre_hook(module, input):
    layer_start_timestamps.append(time.perf_counter_ns())


def _forward_hook(module, input, output):
    layer_stop_timestamps.append(time.perf_counter_ns())


def profile_layers(model: torch.nn.Module) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, PROFILED_LAYERS):
            layer.register_forward_pre_hook(_forward_pre_hook)
            layer.register_forward_hook(_forward_hook)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)

    # Load model
    model_path = config["model"]
    if os.path.exists(model_path):
        try:
            model = torch.jit.load(model_path).eval()
        except Exception as e:
            print(f"Error loading file: {e}")
    elif hasattr(models, model_path):
        model = getattr(models, model_path)().eval()
    else:
        raise ValueError(f"Model '{model_path}' is not found.")

    # Load input tensor
    input_tensor = torch.rand(config["input_shape"])

    # Add hook functions to layers
    profile_layers(model)

    # Profiling
    print("Profiling")
    with torch.no_grad():
        for _ in tqdm(range(PROFILING_ITERATIONS)):
            layer_start_timestamps.clear()
            layer_stop_timestamps.clear()
            model_start_timestamp = time.perf_counter_ns()
            model(input_tensor)
            model_stop_timestamp = time.perf_counter_ns()
            model_runtimes.append(np.subtract(model_stop_timestamp, model_start_timestamp))
            layer_runtimes.append(np.subtract(layer_stop_timestamps, layer_start_timestamps))

    # Save layer runtime stats
    layer_runtime_stats_filepath = config["layer_runtime_stats_filepath"]
    if os.path.exists(layer_runtime_stats_filepath):
        print("Remove ")
        os.remove(layer_runtime_stats_filepath)
    with open(layer_runtime_stats_filepath, "w") as file:
        file.writelines(
            f"{layer_id}, {runtime}\n"
            for iteration in layer_runtimes
            for layer_id, runtime in enumerate(iteration)
        )

    # Compute model runtime stats
    model_runtimes_ms = np.array(model_runtimes) / 1000000
    median = np.median(model_runtimes_ms)
    mean = np.mean(model_runtimes_ms)
    std = np.std(model_runtimes_ms)
    print(f"\nmedian: {median}\nmean: {mean}\nstd: {std}\n")

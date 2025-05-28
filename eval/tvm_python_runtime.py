import os
import argparse
import json
import time
import torch
import torchvision.models as models
import ctypes
import tvm
from tvm.contrib import graph_executor
import numpy as np
from tqdm import tqdm

num_cores = 1
os.environ["TVM_NUM_THREADS"] = f"{num_cores}"

repo_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


def benchmark_model(input_tensor: torch.Tensor, lib_path: str) -> None:
    ctypes.CDLL("/net/heap/laudani/tvmonriscv/build/release/build/libprofiling_functions.so", ctypes.RTLD_GLOBAL)
    lib: tvm.runtime.Module = tvm.runtime.load_module(f"{repo_path}/{lib_path}")
    dev = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input("input0", tvm.nd.array(input_tensor.numpy()))

    print("Evaluate inference time cost.")

    runtimes = []
    for i in tqdm(range(100)):
        start_time = time.perf_counter_ns()
        module.run()
        stop_time = time.perf_counter_ns()
        runtimes.append(stop_time-start_time)
    
    median = np.median(runtimes) / 1000000
    mean = np.mean(runtimes) / 1000000
    std = np.std(runtimes) / 1000000
    print(f"Median:\t{median:.3f}\nMean:\t{mean:.3f}\nStd:\t{std:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath", type=str)
    # parser.add_argument("functions_filepath", type=str)
    args = parser.parse_args()

    with open(args.config_filepath, "r") as file:
        config = json.load(file)

    input_shape = config["input_shape"]
    input_tensor = torch.randn(input_shape)

    lib_path = config["lib_path"]

    benchmark_model(input_tensor, lib_path)

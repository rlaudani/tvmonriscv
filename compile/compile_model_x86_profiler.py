import os
import json
import argparse
import time
import tvm
from tvm.contrib import graph_executor
import torch
import torchvision.models as models
import ctypes

from tvm import relay, auto_scheduler

# TODO: Rename function
import lowering.timing_functions_injection
# from strategy.x86 import conv2d, dense

repo_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

num_cores = 1
os.environ["TVM_NUM_THREADS"] = f"{num_cores}"

def build_lib(model: torch.nn.Module, input_shape_list: list[tuple[str, list[int]]], target: tvm.target.Target, lib_path: str) -> str:
    mod, params = tvm.relay.frontend.pytorch.from_pytorch(model, input_shape_list)

    opt_config = {
        "tir.add_lower_pass": [
            (0, lowering.timing_functions_injection.inject_tracing_conv2d())
        ]
    }

    with tvm.transform.PassContext(opt_level=4, config=opt_config):
        lib = relay.build(mod, target=target, params=params)

    lib.export_library(lib_path)

    return lib_path


def run_lib(lib_path: str, device: tvm.runtime.Device) -> None:
    ctypes.CDLL("/net/heap/laudani/tvmonriscv/build/release/build/libprofiling_functions.so", ctypes.RTLD_GLOBAL)
    lib: tvm.runtime.Module = tvm.runtime.load_module(lib_path)
    m = graph_executor.GraphModule(lib["default"](device))
    m.set_input("input0", tvm.nd.array(input_tensor))
    
    for i in range(100):
        start_time = time.time()
        m.run()
        runtime = time.time() - start_time
        print(f"Runtime: {runtime}")


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
            model = torch.load(model_path).eval()
        except Exception as e:
            print(f"Error loading file: {e}")
    elif model_path in models.list_models():
        model = getattr(models, model_path)().eval()
    else:
        raise ValueError(f"Model '{model_path}' is not found.")

    input_shape = config["input_shape"]
    input_tensor = torch.randn(input_shape)
    input_shape_list = [("input0", input_shape)]
    traced_model = torch.jit.trace(model, input_tensor).eval()

    target = tvm.target.Target(f"llvm -mcpu={config['target']}")
    device = tvm.runtime.device("cpu")

    lib_path = config["lib_path"]

    build_lib(traced_model, input_shape_list, target, lib_path)

    # Python runtime
    # lib_path = build_lib(traced_model, input_shape_list, target, lib_path)
    # run_lib(lib_path, device)

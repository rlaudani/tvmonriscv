import os
import time
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as nn_models
from tqdm import tqdm
import ctypes

# from strategy.x86 import conv2d, dense
import lowering.timing_functions_injection

num_cores = 1
os.environ["TVM_NUM_THREADS"] = f"{num_cores}"

repo_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


def create_lib(mod, params, target, name):

    opt_config = {
        "tir.add_lower_pass": [
            (0, lowering.timing_functions_injection.inject_tracing_conv2d())
        ]
    }

    log_file = "tvm_auto_scheduler_data/vgg16_autoschedule.json"

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=4, config=opt_config):
            lib = relay.build(mod, target=target, params=params)

    output_dir = f"{repo_path}/models"
    lib_name = f"{name}_pytorch_lib.so"
    lib_path = f"{output_dir}/{lib_name}"
    lib.export_library(lib_path)

    return lib_name


def run(lib_name):
    ctypes.CDLL("/net/heap/laudani/tvmonriscv/build/release/build/libprofiling_functions.so", ctypes.RTLD_GLOBAL)
    lib: tvm.runtime.Module = tvm.runtime.load_module(f"{repo_path}/models/{lib_name}")
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input("input0", tvm.nd.array(input_tensor.numpy()))
    print("Evaluate inference time cost.")

    runtimes = []
    for i in tqdm(range(1)):
        start_time = time.perf_counter_ns()
        module.run()
        stop_time = time.perf_counter_ns()
        runtimes.append(stop_time-start_time)

    median = np.median(runtimes) / 1000000
    mean = np.mean(runtimes) / 1000000
    std = np.std(runtimes) / 1000000
    print(f"Median:\t{median:.3f}\nMean:\t{mean:.3f}\nStd:\t{std:.3f}")

if __name__ == "__main__":
    model_name = "vgg16"
    model = nn_models.vgg16().eval()
    input_shape = (1, 3, 224, 224)

    input_tensor = torch.randn(input_shape)

    model = torch.jit.trace(model, input_tensor).eval()

    input_name = "input0"
    shape_list = [(input_name, input_shape)]

    target = tvm.target.Target("llvm -mcpu=znver2")
    dev = tvm.cpu(0)

    mod, params = tvm.relay.frontend.pytorch.from_pytorch(model, shape_list)
    
    lib_name = create_lib(mod, params, target, model_name)

    lib_name = "vgg16_pytorch_lib.so"

    run(lib_name)

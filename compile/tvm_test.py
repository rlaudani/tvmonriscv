import os
import time
import tvm
from tvm.contrib import graph_executor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as nn_models
import ctypes
import tvm.testing

import lowering.timing_functions_injection
from strategy.x86 import conv2d, dense

# False: no printf commands, True: insert printf commands
INJECT_PRINT_COMMANDS = False

# False: Simple CNN, True: AlexNet
LARGE_NETWORK = True

num_cores = 1
os.environ["TVM_NUM_THREADS"] = f"{num_cores}"

repo_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


def create_lib(mod, params, target, name):

    with tvm.transform.PassContext(opt_level=4):
        lib = tvm.relay.build(mod, target=target, params=params)

    output_dir = f"{repo_path}/models"
    lib_name = f"{name}_pytorch_lib.so"
    lib_path = f"{output_dir}/{lib_name}"
    lib.export_library(lib_path)

    return lib_name


def run(lib_name):
    # ctypes.CDLL("/net/heap/laudani/tvmonriscv/build/release/build/libprofiling_functions.so", ctypes.RTLD_GLOBAL)
    lib: tvm.runtime.Module = tvm.runtime.load_module(f"{repo_path}/models/{lib_name}")
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input("input0", tvm.nd.array(input_tensor.numpy()))
    print("Evaluate inference time cost.")
    # timing_results = module.benchmark(
    #     device=dev,
    #     repeat=50,
    #     number=2,
    #     end_to_end=False
    # )
    # print(timing_results)

    for i in range(3) :
        start_time = time.time()
        module.run()
        runtime = time.time() - start_time
        print(f"Runtime: {runtime}")


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

    # target = tvm.target.Target("llvm", host="llvm")
    # dev = tvm.cpu(0)

    #target, dev = tvm.testing.enabled_targets()[0]

    print("Target: ", target)
    print("Device: ", dev)

    mod, params = tvm.relay.frontend.pytorch.from_pytorch(model, shape_list)
    
    lib_name = create_lib(mod, params, target, model_name)

    run(lib_name)

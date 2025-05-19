import os
import time
import tvm
from tvm.contrib import graph_executor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as nn_models
import ctypes

import lowering.lowering_pytorch

# False: no printf commands, True: insert printf commands
INJECT_PRINT_COMMANDS = True

# False: Simple CNN, True: AlexNet
ALEXNET = False

num_cores = 1
os.environ["TVM_NUM_THREADS"] = f"{num_cores}"

repo_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def create_lib(mod, params, target, name):

    if INJECT_PRINT_COMMANDS:
        opt_config = {
            "tir.add_lower_pass": [
                (0, lowering.lowering_pytorch.inject_tracing_conv2d())
            ]
        }

        with tvm.transform.PassContext(config=opt_config, opt_level=4):
            lib = tvm.relay.build(mod, target=target, params=params)
    
    else:
        with tvm.transform.PassContext(opt_level=4):
            lib = tvm.relay.build(mod, target=target, params=params)

    output_dir = f"{repo_path}/models"
    lib_name = f"{name}_pytorch_lib.so"
    lib_path = f"{output_dir}/{lib_name}"
    lib.export_library(lib_path)

    return lib_name


def run(lib_name):
    ctypes.CDLL("/net/heap/laudani/tvmonriscv/build/release/build/libcustom_functions.so", ctypes.RTLD_GLOBAL)
    lib: tvm.runtime.Module = tvm.runtime.load_module(f"{repo_path}/models/{lib_name}")
    dev = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input("input0", tvm.nd.array(input_tensor.numpy()))
    print("Evaluate inference time cost.")
    # timing_results = module.benchmark(
    #     device=dev,
    #     repeat=20,
    #     number=5,
    #     cooldown_interval_ms=10,
    #     end_to_end=False
    # )
    # print(timing_results)
    start_time = time.time()
    module.run()
    runtime = time.time() - start_time
    print(f"Runtime: {runtime}")


if __name__ == "__main__":
    if ALEXNET:
        model_name = "alexnet"
        model = nn_models.AlexNet().eval()
        input_shape = (1, 3, 224, 224)
    else:
        model_name = "simple_cnn"
        model = SimpleCNN().eval()
        input_shape = (1, 1, 28, 28)

    input_tensor = torch.randn(input_shape)

    model = torch.jit.trace(model, input_tensor).eval()
    
    input_name = "input0"
    shape_list = [(input_name, input_shape)]

    target = tvm.target.Target("llvm")
    mod, params = tvm.relay.frontend.pytorch.from_pytorch(model, shape_list)
    
    lib_name = create_lib(mod, params, target, model_name)

    run(lib_name)

import torch
import torchvision.models as models
import tvm
import tvm.relay.testing
import numpy as np

import os
repo_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..'))

import sys
sys.path.append(repo_path)


import compile.strategy.conv2d_pytorch
# import compile.strategy.dense_pytorch
import compile.lowering.lowering_pytorch


def build_lq_lib(nn_model, nn_name: str, batch: int, shape_list, store_path: str) -> str :
    """Build a library for a given nn model and batch size

    Args:
        lq_nn: NN model (PyTorch)
        nn_name (str): Name of NN model
        batch (int): Input batch size
        store_path (str): Store path of .so

    Returns:
        str: Filename of created lib
    """
    layout = 'NCHW'

    def __build_lib(shape_list, target, dev, dtype="float32"):
        mod, params = tvm.relay.frontend.pytorch.from_pytorch(nn_model, shape_list)

        # Add lowering passes
        opt_config = {
            "tir.add_lower_pass": [
                (0, tvm.tir.transform.Simplify()),
                (0, tvm.tir.transform.RemoveNoOp()),
                (0, compile.lowering.lowering_pytorch.inject_tracing_conv2d())
            ]
        }

        with tvm.transform.PassContext(config=opt_config, opt_level=1):
            lib = tvm.relay.build(mod, target=target, params=params)
        
        inp_shape_str = "1x3x224x224"
        lib_name = f'{nn_name}_input_{inp_shape_str}_lib.so'
        store_file = f'{store_path}/{lib_name}'
        lib.export_library(store_file)
        print(mod)
        return lib_name


    target, dev = tvm.testing.enabled_targets()[0]
    lib_name = __build_lib(shape_list, target, dev)
    return lib_name

print("PyTorch Version")

# VGG16
model_name="vgg16"

model = models.vgg16()
model.eval()

input_shape = (1, 3, 224, 224)
input_data = torch.randn(input_shape)

script_module = torch.jit.trace(model, input_data)

# Convert to TVM Relay
input_name = "input"
shape_list = [(input_name, input_shape)]

lib_name = build_lq_lib(nn_model=script_module, nn_name=model_name, shape_list=shape_list, batch=1, store_path=f'{repo_path}/models')
print(f'Created lib {lib_name} in {repo_path}/models/')

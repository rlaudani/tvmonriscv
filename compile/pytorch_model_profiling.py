import os
import tvm
import tvm.testing
from tvm.contrib.graph_executor import GraphModule
from tvm.contrib.debugger.debug_executor import GraphModuleDebug
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
import plotly.express as px

# import lowering.lowering_pytorch
# import strategy.conv2d_pytorch

repo_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

def build_lq_lib(model, input_shape: tuple, name: str, batch: int, output_dir: str) -> str:
    """
    Build a library for the given PyTorch model.
    
    Args:
        model: The PyTorch model to compile.
        input_shape: Shape of the input tensor.
        name: Name prefix for the compiled library.
        batch: Batch size.
        output_dir: Output directory for the compiled library.
    
    Returns:
        The filename of the compiled shared library.
    """
    # Trace the model using a sample input tensor 
    input_tensor = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_tensor).eval()

    # Define input name and shape for Relay conversion
    input_name = "input0"
    shape_list = [(input_name, input_shape)]

    # Convert the PyTorch model to Relay IR
    mod, params = tvm.relay.frontend.from_pytorch(scripted_model, shape_list)

    # Set the compilation target and execution device
    target, dev = tvm.testing.enabled_targets()[0]

    # TODO
    # opt_config = {
    #     "tir.add_lower_pass": [
    #         (0, tvm.tir.transform.Simplify()),
    #         (0, tvm.tir.transform.RemoveNoOp()),
    #         (0, lowering.lowering_pytorch.inject_tracing_conv2d())
    #     ]
    # }
    # # Compile the Relay module into an optimized TVM library
    # with tvm.transform.PassContext(config=opt_config, opt_level=4):
    #     lib = tvm.relay.build(mod, target=target, params=params)

    # Compile the Relay module into an optimized TVM library
    with tvm.transform.PassContext(opt_level=4):
        lib = tvm.relay.build(mod, target=target, params=params)

    # Export the compiled library
    input_shape_str = ''.join([str(i) + 'x' for i in input_shape])[:-1]
    lib_name = f"{name}_input_{input_shape_str}_lib.so"
    lib_path = f"{output_dir}/{lib_name}"
    lib.export_library(lib_path)

    print(f'Created lib {lib_name} in {repo_path}/models/')

    return lib_name


def benchmark_model(lib_name: str, input_shape: tuple, file_name: str) -> None:
    """
    Benchmark the given library.
    
    Args:
        lib_name: TODO
        input_shape: Shape of the input tensor.
        file_name: TODO
    """
     # Select execution target and device
    target, dev = tvm.testing.enabled_targets()[0]

    # Load compiled model    
    lib: tvm.runtime.Module = tvm.runtime.load_module(f"{repo_path}/models/{lib_name}")

    # Create graph executor
    gmod = GraphModule(lib["default"](dev))

    # Set model input
    input_tensor = tvm.nd.array(torch.randn(input_shape))
    gmod.set_input("input0", input_tensor)

    # Measure model inference time
    # TODO: Include cooldown?
    # TODO: End-to-end: True or False?
    print("Evaluate inference time cost.")
    timing_results = gmod.benchmark(
        device=dev,
        repeat=100,
        number=5,
        cooldown_interval_ms=10,
        end_to_end=False
    )
    print(timing_results)

    # Save timing results
    with open(f"{repo_path}/results/{file_name}.txt", "w") as f:
        for i in timing_results.results:
            f.write(f"{str(i)}\n")

    print(f'Created file {file_name} in {repo_path}/compile/')


def plot_timing_results(file_name: str, fig_name: str = None, show_fig: bool = True) -> None:
    """
    Plot timing results.

    Args:
        file_name: TODO
        fig_name: TODO
        show_fig: TODO
    """
    # Load runtime data from the CSV file
    df = pd.read_csv(f"{repo_path}/results/{file_name}.txt", header=None, names=["runtime"])
    
    # Create a box plot of the runtime values
    # TODO: Improve figure style
    fig = px.box(df, y="runtime")

    if show_fig:
        fig.show()

    # Save figure
    if fig_name is not None:
        fig_path = f"{repo_path}/figures/{fig_name}.svg"
        fig.write_image(fig_path)
        print(f"Save plot at {fig_path}")


def benchmark_layers(lib_name: str, input_shape: tuple, file_name: str) -> None:
    """
    Benchmark layers in the given library.
    
    Args:
        lib_name: TODO
        input_shape: Shape of the input tensor.
        file_name: TODO
    """
     # Select execution target and device
    target, dev = tvm.testing.enabled_targets()[0]

    # Load compiled model    
    lib: tvm.runtime.Module = tvm.runtime.load_module(f"{repo_path}/models/{lib_name}")

    # Create graph executor
    gmod = GraphModuleDebug(lib["default"](dev))

    #TODO


if __name__ == "__main__":
    # VGG16
    name = "vgg16"
    model = models.vgg16().eval()
    batch_size = 1
    input_shape = (batch_size, 3, 224, 224)
    res_file_name = "vgg16_timing_results"

    # ResNet50
    # name = "resnet50"
    # model = models.resnet50().eval()
    # batch_size = 1
    # input_shape = (batch_size, 3, 224, 224)
    # res_file_name = "resnet50_timing_results"

    # TODO: MobileNet
    # TODO: YOLO

    lib_name = build_lq_lib(model=model, input_shape=input_shape,
                            name=name, batch=1, output_dir=f"{repo_path}/models")

    # TODO: Is this a valid approach to measure runtime?
    benchmark_model("vgg16_input_1x3x224x224_lib.so", input_shape=input_shape, file_name=res_file_name)
    # benchmark_model(lib_name="resnet50_input_1x3x224x224_lib.so", input_shape=input_shape, file_name=res_file_name)

    plot_timing_results(file_name=res_file_name, fig_name="vgg16")
    # plot_timing_results(file_name=res_file_name, fig_name="resnet_50")

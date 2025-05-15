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
import ctypes
import time
import pandas as pd

from tvm import meta_schedule as ms

from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import onnx

import ctypes

import lowering.lowering_pytorch
# import strategy.conv2d_x86_pytorch
import strategy.conv2d_pytorch

repo_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

num_cores = 1
os.environ["TVM_NUM_THREADS"] = f"{num_cores}"


def load_model(mod, params, target, log_file):
    # 3. AutoScheduler Setup
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    print("Gefundene Tasks:")
    for i, task in enumerate(tasks):
        print(f"[Task {i}] {task.desc}: {task.workload_key}")

    # 4. Suche starten
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
    )

    task_scheduler = auto_scheduler.TaskScheduler(tasks, task_weights)
    task_scheduler.tune(tune_option)


def create_lib(mod, params, target, log_file):
    # 5. Mit besten Schedules kompilieren

    opt_config = {
        "tir.add_lower_pass": [
            (0, lowering.lowering_pytorch.inject_tracing_conv2d())
        ]
    }

    # with auto_scheduler.ApplyHistoryBest(log_file):
    #     with tvm.transform.PassContext(config=opt_config, opt_level=2):
    #         lib = relay.build(mod, target=target, params=params)


    # with auto_scheduler.ApplyHistoryBest(log_file):
    #     with tvm.transform.PassContext(opt_level=0):
    #         lib = relay.build(mod, target=target, params=params)


    with tvm.transform.PassContext(config=opt_config, opt_level=1):
        lib = relay.build(mod, target=target, params=params)

    # with tvm.transform.PassContext(opt_level=4):
    #     lib = relay.build(mod, target=target, params=params)

    # Export the compiled library
    name = "vgg16"
    output_dir = f"{repo_path}/models"
    lib_name = f"{name}_pytorch_lib.so"
    lib_path = f"{output_dir}/{lib_name}"
    lib.export_library(lib_path)
    return lib_name


def run(lib_name):
    # 6. Ausführen
    ctypes.CDLL("/net/heap/laudani/tvmonriscv/build/release/build/libcustom_functions.so", ctypes.RTLD_GLOBAL)
    lib: tvm.runtime.Module = tvm.runtime.load_module(f"{repo_path}/models/{lib_name}")
    dev = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input("input", tvm.nd.array(input_tensor.numpy()))
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


def create_dataframe():
    df = pd.read_csv("compile/timestamps.txt", header=None, names=["layer_id", "runtime"])
    df["runtime"] = df["runtime"] / 1000000
    df["iteration"] = df.groupby("layer_id").cumcount() + 1
    df = df[["iteration", "layer_id", "runtime"]]
    return df


def create_boxplots(df, file_path: str = None):
    fig = px.box(df, x="layer_id", y="runtime", log_y=False)
    fig.show()

    if file_path is not None:
        fig.write_image(file_path)
        print(f"Save plot at {file_path}")


if __name__ == "__main__":
    # 1. PyTorch ResNet18 laden und nach ONNX exportieren
    model_name = "vgg16"
    model = models.vgg16(pretrained=True)
    model.eval()

    input_shape = (1, 3, 224, 224)
    input_tensor = torch.randn(input_shape)
    torch.onnx.export(model, input_tensor, f"{model_name}.onnx", input_names=["input"], output_names=["output"], opset_version=11)

    # 2. ONNX nach Relay konvertieren
    onnx_model = onnx.load(f"{model_name}.onnx")
    mod, params = relay.frontend.from_onnx(onnx_model, shape={"input": input_shape})

    # Target
    target = tvm.target.Target("llvm")

    # Log file
    log_file = f"{model_name}_autoschedule.json"

    # load_model(mod, params, target, log_file)

    lib_name = create_lib(mod, params, target, log_file)
    run(lib_name)

    # data = create_dataframe()
    # create_boxplots(data)


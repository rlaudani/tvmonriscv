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
from tqdm import tqdm

from tvm import meta_schedule as ms

from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import onnx

# import logging
# logging.basicConfig(level=logging.DEBUG)

repo_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

num_cores = 1
os.environ["TVM_NUM_THREADS"] = f"{num_cores}"


def setup():
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


model = models.vgg16()
model.eval()

input_shape = (1, 3, 224, 224)
input_tensor = torch.randn(input_shape)
torch.onnx.export(model, input_tensor, "vgg16.onnx", input_names=["input"], output_names=["output"], opset_version=11)

onnx_model = onnx.load("vgg16.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, shape={"input": input_shape})

target = tvm.target.Target("llvm -mcpu=znver2")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

log_file = "tvm_auto_scheduler_data/vgg16_autoschedule.json"
setup()

with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=4):
        lib = relay.build(mod, target=target, params=params)

name = "vgg16"
output_dir = f"{repo_path}/models"
lib_name = f"{name}_pytorch_lib.so"
lib_path = f"{output_dir}/{lib_name}"
lib.export_library(lib_path)

# 6. Ausführen
dev = tvm.cpu()
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("input", tvm.nd.array(input_tensor.numpy()))

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

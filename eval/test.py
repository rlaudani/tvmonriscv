import time
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm

num_threads = 1
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)

profiled_layers = (nn.Conv2d, nn.Linear)

start_times = []
stop_times = []


def insert_hooks(model):

    def _pre_hook(module, input):
        start_times.append(time.perf_counter_ns())

    def _post_hook(module, input, output):
        stop_times.append(time.perf_counter_ns())

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.register_forward_pre_hook(_pre_hook)
            module.register_forward_hook(_post_hook)  


model = models.vgg16()
insert_hooks(model)
input_tensor = torch.rand(1, 3, 224, 224)

total_runtimes = []

NUM_ITERATIONS = 200
NS_TO_MS_DIVISOR = 1000000
WARMUP = 10

for _ in range(WARMUP):
    start_times.clear()
    stop_times.clear()
    model(input_tensor)


for _ in tqdm(range(NUM_ITERATIONS)):
    start_times.clear()
    stop_times.clear()
    model(input_tensor)
    runtimes = np.subtract(stop_times, start_times) / NS_TO_MS_DIVISOR

    with open("eval/runtimes.txt", "a") as file:
        for i, a in enumerate(runtimes):
            file.write(f"{i}, {a}\n")

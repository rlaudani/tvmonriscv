import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import json


PROFILED_LAYERS = (nn.Conv2d, nn.Linear)


def create_dataframe(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, header=None, names=["layer_id", "runtime"])
    df["runtime"] = df["runtime"] / 1000000
    return df


def create_xticks_labels(model: nn.Module, input_shape: tuple[int]) -> list[str]:
    labels = []

    def _hook_func(module, input, output):
        label = f"{module.__class__.__name__}\n({'x'.join(map(str, input[0].shape[1:]))})"
        labels.append(label) 

    for layer in model.modules():
        if isinstance(layer, PROFILED_LAYERS):
            layer.register_forward_hook(_hook_func)

    input_tensor = torch.randn(input_shape)
    model(input_tensor)

    return labels


def create_plot(df: pd.DataFrame, model: nn.Module, input_shape: tuple[int], path: str = None) -> None:
    df.boxplot(column="runtime", by="layer_id")
    labels = create_xticks_labels(model, input_shape)

    assert len(labels) == len(pd.unique(df["layer_id"]))

    plt.ylabel("Runtime [ms]")
    plt.xlabel("Layer")
    plt.xticks(list(range(1, len(labels) + 1)), labels)
    plt.xticks(rotation=45)
    plt.grid(False)
    plt.suptitle("")
    plt.title("")
    plt.show()

    plt.savefig(path, format="svg", bbox_inches="tight")


def print_stats(df: pd.DataFrame) -> None:
    stats_df = df.groupby("layer_id")["runtime"].agg(["median", "mean", "std"])
    print("layer\t\tmedian [ms]\tmean [ms]\tstd [ms]")
    for idx, row in stats_df.iterrows():
        print(f"{idx}\t\t{row['median']:.2f}\t\t{row['mean']:.2f}\t\t{row['std']:.2f}")


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

    input_shape = tuple(config["input_shape"])

    data = create_dataframe(filepath=config["layer_runtime_stats_filepath"])

    fig_filepath = config["fig_filepath"]
    print_stats(data)
    create_plot(data, model, input_shape, fig_filepath)

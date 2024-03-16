from typing import Any, Dict, List
from collections import OrderedDict
import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, args):
        self.args = args
        self.wandb = None
        if args.wandb:
            wandb.init(project=args.wandb_project, name=args.exp_name, config=args)
            self.wandb = wandb

    def log(self, logs: Dict[str, Any]) -> None:
        if self.wandb:
            self.wandb.log(logs)

def plot_data(data_points:List, name:str):
    fig, ax = plt.subplots()
    ax.set_title(name)
    ax.plot(list(range(len(data_points))), data_points)
    plt.show()

def calculate_density(tensor):
    zero_tensor = torch.zeros(tensor.shape, dtype=torch.float).to(tensor.device)
    cmp_tensor = (tensor != zero_tensor)
    return cmp_tensor.sum()/tensor.numel()

def density_weights(model):
    density = OrderedDict()
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            data = layer.weight.data
            density[layer] = calculate_density(data)
    # print("Density of given model is: {}".format(density))
    return density

def density_masks(masks):
    density = OrderedDict()
    for key, mask in masks.items():
        density[key] = calculate_density(mask)
    return density

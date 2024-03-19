from typing import Any, Dict, List
import argparse
import os
import copy
import torch
import wandb
from collections import OrderedDict
import torch.nn as nn


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


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg

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


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # In your arg_parser() function, add an argument for the results file path.
    parser.add_argument("--results_file", type=str, default="results.txt", help="Path to save training results")

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("--model_name", type=str, default="cnn")

    parser.add_argument("--non_iid", type=int, default=1)  # 0: IID, 1: Non-IID
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--n_shards", type=int, default=200)
    parser.add_argument("--frac", type=float, default=0.1)

    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=1)

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="FedAvg")
    parser.add_argument("--exp_name", type=str, default="exp")

    return parser.parse_args()

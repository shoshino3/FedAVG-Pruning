from typing import Any, Dict, List
import argparse
import os
import copy
import torch
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn
import time

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

def plot_data(data_points:List, pruning_side, non_iid, alpha, name:str, ifacc=True):
    non_iid="Non_IID" if non_iid else "IID"
    fig, ax = plt.subplots()
    if ifacc:
        ax.set_title(f"{name},best:{max(data_points)} at epoch {data_points.index(max(data_points))}")
    else:
        ax.set_title(f"{name},best:{min(data_points)} at epoch {data_points.index(min(data_points))}")
    ax.plot(list(range(len(data_points))), data_points)
    filepath = f"curve/{pruning_side}/{non_iid}"
    if non_iid=="Non_IID":
        filepath += f"/alpha_{alpha}"
    os.makedirs(filepath, exist_ok=True)
    plt.savefig(f"{filepath}/{name}.png")
    plt.close()
    
def count_zero_weights(model, s = ""): #sanity check, does not add to the logic of the code
    print(s)
    zero_channel_count = 0
    
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
    
            zero_weights = layer.weight == 0
            zero_channel_count += torch.sum(zero_weights).item()
            
    print("Number of zero channels: ", zero_channel_count)
    


def print_model_size(model, base_dir="models", delete_after=True):
    # Generate a timestamped folder
    timestamp = int(time.time() * 1000)
    folder_path = os.path.join(base_dir, f"model_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    
    # Define the filepath for the model
    filepath = os.path.join(folder_path, "model.pth")
    
    # Save the model
    torch.save(model.state_dict(), filepath)
    
    # Get the file size in MB
    model_size = os.path.getsize(filepath) / (1024 ** 2)
    print(f"Model file size: {model_size:.4f} MB")
    
    # Optionally delete the folder and file
    if delete_after:
        os.remove(filepath)
        os.rmdir(folder_path)
        

def count_parameters(model, trainable_only=True):
    if trainable_only:
        # only count the trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        # count all parameters
        total_params = sum(p.numel() for p in model.parameters())
    return total_params

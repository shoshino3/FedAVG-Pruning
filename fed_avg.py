from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os, time
import argparse
import shutil

from data import MNISTDataset, CIFAR10Dataset, FederatedSampler
from models import CNN, MLP, vgg
from utils import average_weights, Logger, plot_data, print_model_size
from EarlyBird import EarlyBird, actual_prune


class FedAvg:
    """Implementation of FedAvg
    http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.args.pruning_ratio = 0 if not self.args.pruning else self.args.pruning_ratio
        print(f"Running with args: {self.args}")
        self.scheduler = [int(x) for x in self.args.lr_scheduler.split(",")] if self.args.lr_scheduler else []
        self.device = torch.device(
            f"cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
        )
        print(f"Using device: {self.device}")
        # self.logger = Logger(args)
        self.accuracy, self.loss = [], []
        self.eb_epoch = 0

        # Loading training and testing dataset and is processed for clients to receive their own dataset
        self.train_loader, self.test_loader = self._get_data(
            root=self.args.data_root,
            n_clients=self.args.n_clients,
            n_shards=self.args.n_shards,
            non_iid=self.args.non_iid,
            alpha=self.args.alpha
        )
        
        self.is_best = False
        
        if self.args.model_name == "mlp":
            self.root_model = MLP(input_size=784, hidden_size=128, n_classes=10).to(
                self.device
            )
            self.target_acc = 0.97
            
        elif self.args.model_name == "cnn":
            self.root_model = CNN(n_channels=3, n_classes=10).to(self.device)
            self.target_acc = 0.85
            
        elif self.args.model_name == "vgg":
            self.root_model = vgg(dataset='cifar10', depth=19).to(self.device)
            self.target_acc = 0.85
            
        else:
            raise ValueError(f"Invalid model name, {self.args.model_name}")

        print_model_size(self.root_model)

        self.reached_target_at = None  # type: int

    def _get_data(
        self, root: str, n_clients: int, n_shards: int, non_iid: int, alpha: float
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            root (str): path to the dataset.
            n_clients (int): number of clients.
          
            non_iid (int): 0: IID, 1: Non-IID

        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, test_loader
        """
        if self.args.dataset == "mnist":
            train_set = MNISTDataset(root=root, train=True)
            test_set = MNISTDataset(root=root, train=False)
        elif self.args.dataset == "cifar10":
            train_set = CIFAR10Dataset(root=root, train=True)
            test_set = CIFAR10Dataset(root=root, train=False)

        sampler = FederatedSampler(
            train_set, non_iid=non_iid, n_clients=n_clients, n_shards=n_shards, alpha=alpha
        )

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, sampler=sampler)
        test_loader = DataLoader(test_set, batch_size=128)

        return train_loader, test_loader

    def _train_client(
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Tuple[nn.Module, float]:
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """
        client_acc, client_loss = [], []
        model = copy.deepcopy(root_model)
        model.to(self.device)
        model.train()
        
        ###weight_decay is the new addition since early bird uses it. 
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay = 1e-4
        )


        for epoch in range(self.args.n_client_epochs):
            """
            this is from Early Bird implementation where learning rate decreases at epoch = 80 and 120
            """
                  
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                logits = model(data)
                loss = F.nll_loss(logits, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_correct += (logits.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)

            # Calculate average accuracy and loss
            epoch_loss /= (idx+1)
            epoch_acc = epoch_correct / epoch_samples
            client_loss.append(epoch_loss)
            client_acc.append(epoch_acc)

        print(
            f"Client #{client_idx} | Avg Loss: {sum(client_loss)/len(client_loss):.4f} | Avg Acc: {sum(client_acc)/len(client_acc):.4f}",
            # end="\r",
        )

        return model, epoch_loss / self.args.n_client_epochs
        


    def train(self, pre_eb = True) -> None:
        """Train a server model."""
        train_losses = []
        best_prec1 = 0.0

      
        early_bird = EarlyBird(self.args.pruning_ratio) #to change the pruning %
    
        
        epoch_acc_tracker = np.zeros((self.args.n_epochs, 2))
      
        
        for epoch in range(self.args.n_epochs):
            
            self.root_model.train() ## training session mode
            
            
            clients_models = []
            clients_losses = []

            # Randomly select clients
            m = max(int(self.args.frac * self.args.n_clients), 1)
            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)

            ##after 4 epochs and after the emergence of early bird, this 'if' clause gets executed and runs only once.
            if args.pruning and pre_eb and early_bird.early_bird_emerge(self.root_model):  
                print(f"[early_bird with pruning ratio {self.args.pruning_ratio} Found EB at epoch: "+str(epoch+1))
       
                self.eb_epoch = epoch + 1

                print_model_size(self.root_model)
                self.root_model=actual_prune(self.root_model, self.args.pruning_ratio)
                print_model_size(self.root_model)
                pre_eb = False
                
            if epoch in self.scheduler: 
                print(f"Changing learning rate: from {self.args.learning_rate} to {self.args.learning_rate*0.1}")
                self.args.learning_rate *= 0.1

            
            for client_idx in idx_clients:
                # Set client in the sampler
                self.train_loader.sampler.set_client(client_idx)

                # Train client
                client_model, client_loss = self._train_client(
                    root_model=self.root_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                )
                
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)
            
            # Update server model based on clients models
            updated_weights = average_weights(clients_models)
            self.root_model.load_state_dict(updated_weights)
            
            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)

            if (epoch + 1) % self.args.log_every == 0:
                # Test server model
                total_loss, total_acc = self.test()
                avg_train_loss = sum(train_losses) / len(train_losses)

                # Log results
                logs = {
                    "train/loss": avg_train_loss,
                    "test/loss": total_loss,
                    "test/acc": total_acc,
                    "round": epoch,
                }
                if total_acc >= self.target_acc and self.reached_target_at is None:
                    self.reached_target_at = epoch
                    logs["reached_target_at"] = self.reached_target_at
                    print(
                        f"\n -----> Target accuracy {self.target_acc} reached at round {epoch}! <----- \n"
                    )

                # self.logger.log(logs)
                self.accuracy.append(total_acc)
                self.loss.append(total_loss)

                # Print results to CLI
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss:.4f}")
                print(
                    f"---> Avg Test Loss: {total_loss:.4f} | Avg Test Accuracy: {total_acc:.4f}\n"
                )

                # Early stopping
                if self.args.early_stopping and self.reached_target_at is not None:
                    print(f"\nEarly stopping at round #{epoch}...")
                    break
                
        print_model_size(self.root_model)
        
        if self.args.draw_curve:
            plot_data(self.accuracy, self.args.pruning_side, self.args.non_iid, self.args.alpha, f"acc_pr_{self.args.pruning_ratio}", ifacc=True)
            plot_data(self.loss, self.args.pruning_side, self.args.non_iid, self.args.alpha, f"loss_pr_{self.args.pruning_ratio}",ifacc=False)

    def test(self) -> Tuple[float, float]:
        """Test the server model.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.root_model.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)

            logits = self.root_model(data)
            loss = F.nll_loss(logits, target)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total_samples += data.size(0)

        # calculate average accuracy and loss
        total_loss /= idx
        total_acc = total_correct / total_samples

        return total_loss, total_acc
    
    

def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("-m", "--model_name", type=str, default="vgg")
    parser.add_argument("-d", "--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])

    parser.add_argument("-i", "--non_iid", type=int, default=0)  # 0: IID, 1: Non-IID
    parser.add_argument("-a", "--alpha", type=float, default=10)
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--n_shards", type=int, default=200)
    parser.add_argument("--frac", type=float, default=0.1)

    parser.add_argument("-se", "--n_epochs", type=int, default=1500)
    parser.add_argument("-ce", "--n_client_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("-lrs", "--lr_scheduler", type=str, default="")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=1)

    parser.add_argument("--device", type=str, default="gpu")

    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="FedAvg")
    parser.add_argument("--exp_name", type=str, default="exp")

    # customized arguments
    parser.add_argument("-p", "--pruning", action="store_true", help="whether to prune")
    parser.add_argument("-ps", "--pruning_side", type=str, default="client", choices=["client", "server"])
    parser.add_argument("-pr", "--pruning_ratio", type=float, default=0.9)
    parser.add_argument("-dc", "--draw_curve", action="store_true", help="whether to draw the accuracy curve")

    return parser.parse_args()

if __name__ == "__main__":
    start_time= time.time()
    args = arg_parser()
    fed_avg = FedAvg(args)
    fed_avg.train() 
    print(f"--- Total time: {(time.time()-start_time)/60:.2f} minutes")
